"""TransE plausibility model for KG triples: a baseline defence and an extra feature for the path detector.

Trained ONLY on the clean graphs of the WebQSP train split (disjoint from the dev and test questions). The plausibility of a triple is
-||h + r - t||_1 (higher = more plausible); a triple with an unseen entity or relation has no score. A path is summarised by the minimum and
mean plausibility of its triples, the share of unscored triples, and the per-question z-score of the minimum.

  python -m kgrag.kge train        # writes runs/kg/kge_transe.pt
  python -m kgrag.kge eval         # coverage and path-level AUC (poisoned vs clean path) for each attack split
Plug-ins: defend.SCORERS["kge"] (baseline: suspicion = -min plausibility) and learned.EXTRA (extra detector features, via defended_run --kge).
"""
import argparse
import os
import time

import numpy as np

from kgrag import defend

MODEL = os.path.join(defend.RUNS, "kge_transe.pt")


def build_ids(graphs):
    """Integer ids for entities and relations and the flat (h, r, t) id arrays of a list of graphs (lists of [h, r, t])."""
    ent, rel, H, R, T = {}, {}, [], [], []
    for g in graphs:
        for h, r, t in g:
            H.append(ent.setdefault(h, len(ent)))
            T.append(ent.setdefault(t, len(ent)))
            R.append(rel.setdefault(r.strip(), len(rel)))
    return ent, rel, np.array(H, np.int64), np.array(R, np.int64), np.array(T, np.int64)


def dedupe(H, R, T, n_rel: int, n_ent: int):
    key = (H * n_rel + R) * n_ent + T
    _, first = np.unique(key, return_index=True)
    return H[first], R[first], T[first]


def train(dim: int = 64, epochs: int = 60, batch: int = 262144, lr: float = 0.01, margin: float = 1.0, seed: int = 0):
    import torch
    from datasets import load_dataset
    ds = load_dataset("rmanluo/RoG-webqsp", split="train")
    ent, rel, H, R, T = build_ids(ds["graph"])
    H, R, T = dedupe(H, R, T, len(rel), len(ent))
    n = len(H)
    print(f"{len(ds)} train questions, {n} distinct triples, {len(ent)} entities, {len(rel)} relations", flush=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ho, tr = perm[: n // 100], perm[n // 100:]
    dev = "cuda"
    torch.manual_seed(seed)
    E = torch.nn.Embedding(len(ent), dim).to(dev)
    Rl = torch.nn.Embedding(len(rel), dim).to(dev)
    torch.nn.init.uniform_(E.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
    torch.nn.init.uniform_(Rl.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
    Rl.weight.data = torch.nn.functional.normalize(Rl.weight.data, dim=1)
    opt = torch.optim.Adam(list(E.parameters()) + list(Rl.parameters()), lr=lr)
    Ht, Rt, Tt = (torch.tensor(x, device=dev) for x in (H, R, T))
    trt = torch.tensor(tr, device=dev)
    dist = lambda h, r, t: (E(h) + Rl(r) - E(t)).abs().sum(1)
    t0 = time.time()
    for ep in range(epochs):
        order = trt[torch.randperm(len(trt), device=dev)]
        tot = 0.0
        for s in range(0, len(order), batch):
            i = order[s:s + batch]
            h, r, t = Ht[i], Rt[i], Tt[i]
            flip = torch.rand(len(i), device=dev) < 0.5
            rnd = torch.randint(0, len(ent), (len(i),), device=dev)
            h2, t2 = torch.where(flip, rnd, h), torch.where(flip, t, rnd)
            loss = torch.relu(margin + dist(h, r, t) - dist(h2, r, t2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                E.weight.data = torch.nn.functional.normalize(E.weight.data, dim=1)
            tot += float(loss) * len(i)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep} loss {tot / len(order):.4f} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        i = torch.tensor(ho, device=dev)
        h, r, t = Ht[i], Rt[i], Tt[i]
        pos = -dist(h, r, t)
        neg = -dist(h, r, torch.randint(0, len(ent), (len(i),), device=dev))
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones(len(i)), np.zeros(len(i))]
        ho_auc = roc_auc_score(y, torch.cat([pos, neg]).cpu().numpy())
        fill = float(torch.median(pos).cpu())
    print(f"held-out true triples vs random-tail corruptions: AUC {ho_auc:.3f}; median plausibility {fill:.2f}", flush=True)
    torch.save({"E": E.weight.detach().cpu().numpy(), "R": Rl.weight.detach().cpu().numpy(), "ent": ent, "rel": rel, "fill": fill}, MODEL)
    print("saved", MODEL)


class Scorer:
    def __init__(self, E, R, ent, rel, fill: float):
        self.E, self.R, self.ent, self.rel, self.fill = E, R, ent, rel, fill

    @classmethod
    def load(cls, path: str = MODEL):
        import torch
        d = torch.load(path, weights_only=False)
        return cls(d["E"], d["R"], d["ent"], d["rel"], d["fill"])

    def triples(self, trs) -> np.ndarray:
        """Plausibility of each [h, r, t] (NaN when an entity or relation is unknown)."""
        out = np.full(len(trs), np.nan)
        if not trs:
            return out
        ids = np.array([(self.ent.get(h, -1), self.rel.get(r.strip(), -1), self.ent.get(t, -1)) for h, r, t in trs])
        ok = (ids >= 0).all(1)
        if ok.any():
            h, r, t = ids[ok, 0], ids[ok, 1], ids[ok, 2]
            out[ok] = -np.abs(self.E[h] + self.R[r] - self.E[t]).sum(1)
        return out

    def paths(self, recs) -> np.ndarray:
        """(n_paths, 4) features in the flat path order of `recs`: min plausibility, mean plausibility, share of unscored triples,
        per-question z-score of the min. Paths with no scored triple get the fill value."""
        rows, bounds = [], []
        for r in recs:
            start = len(rows)
            for p in r["paths"]:
                rows.append(self.triples(p["triples"]))
            bounds.append((start, len(rows)))
        feats = np.zeros((len(rows), 4))
        for i, sc in enumerate(rows):
            known = sc[~np.isnan(sc)]
            feats[i, 0] = known.min() if len(known) else self.fill
            feats[i, 1] = known.mean() if len(known) else self.fill
            feats[i, 2] = 1.0 - len(known) / max(len(sc), 1)
        for a, b in bounds:
            if b > a:
                m = feats[a:b, 0]
                feats[a:b, 3] = (m - m.mean()) / (m.std() + 1e-6)
        return feats


_SCORER = None


def scorer() -> Scorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer.load()
    return _SCORER


def extra_features(recs) -> np.ndarray:
    """Hook for learned.EXTRA: rows aligned with learned.features(recs)."""
    return scorer().paths(recs)


def score_kge(recs) -> np.ndarray:
    """Baseline scorer for defend.SCORERS (higher = more suspicious): minus the least plausible triple of the path."""
    return -scorer().paths(recs)[:, 0]


def evaluate():
    s = scorer()
    print("split        paths  poisoned  coverage(all triples scored)  AUC(-min plausibility)  AUC(-mean plausibility)")
    for split in ("dev", "test", "ad_test", "ev_test", "fp_test", "fb_test", "fs_test"):
        path = os.path.join(defend.RUNS, f"paths_{split}.jsonl")
        if not os.path.exists(path):
            continue
        recs = defend.load_paths(split)
        y = defend.flat(recs, "poisoned").astype(int)
        f = s.paths(recs)
        print(f"{split:10s} {len(y):7d} {y.mean():8.2f} {(f[:, 2] == 0).mean():12.2f} {defend.auc(y, -f[:, 0]):22.3f} {defend.auc(y, -f[:, 1]):22.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args()
    train(a.dim, a.epochs) if a.cmd == "train" else evaluate()


if __name__ == "__main__":
    main()
