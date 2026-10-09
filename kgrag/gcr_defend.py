"""Path filter in front of GCR's second-stage reasoner (the defence on a second KG-RAG system).

GCR generates reasoning paths with constrained decoding (first stage, kgrag.gcr_run, saved as beams) and then a general LLM reads the
unique beam paths and answers (second stage, kgrag.gcr_stage2). Here the beam paths are turned into the same per-path records as RoG's
retrieved paths (structural features computed on the question's own graph, labels from the attacker's inserted triples) and a conformal
removal filter drops suspicious paths before the second stage. Two detectors, both calibrated on GCR DEVELOPMENT beams only:
  transfer  the detector trained on RoG's development paths (as is), threshold calibrated on the GCR dev beams
  fitted    a detector trained on the first 100 GCR dev questions, threshold calibrated on the remaining ones (kgrag.conformal)
Variants: none, oracle (drops the inserted paths), transfer, fitted, and gtransfer = configuration D of the paper (question-level gate with the
RoG-trained detector, threshold recalibrated on GCR clean dev beams, in front of the transferred conformal filter).
Labels are STRICT (an inserted triple already present in the clean graph, in either orientation, does not make a path poisoned); run with
KG_RUNS=runs/kg_strict so the RoG training splits carry the same labels. Secondary analysis fixed in advance: questions with at least one
poisoned beam path.

  python -m kgrag.gcr_defend build --first runs/kg/gcr3_dev_attacked.jsonl --data data/poisoned_webqsp --name gcrdev
  python -m kgrag.gcr_defend run --variant fitted --split gcrtest --data data/poisoned_webqsp --out runs/kg/gcr3s2_fitted_gcrtest.jsonl
"""
import argparse
import json
import os
import warnings

import numpy as np

from kgrag import attack, conformal, defend, gate, gcr_stage2, learned, pathdata, rog

ROG_TRAIN = ["dev", "K1_dev", "K8_dev", "ad_dev"]  # RoG development paths used by the transferred detector (the "mixed" detector)


def parse_path(text: str):
    """'A -> r1 -> B -> r2 -> C' -> [[A, r1, B], [B, r2, C]]; None if the string does not have that shape."""
    toks = [t.strip() for t in text.split(" -> ")]
    if len(toks) < 3 or len(toks) % 2 == 0:
        return None
    return [[toks[i], toks[i + 1], toks[i + 2]] for i in range(0, len(toks) - 2, 2)]


def beam_record(ex: dict, beams: list[str]) -> dict:
    """Per-path record (kgrag.pathdata format) for the unique beam paths of one question."""
    ug = rog.build_graph(ex["graph"])
    nbr = {n: set(ug.neighbors(n)) for n in ug.nodes}
    ins_list = ex.get("poison_triples") or []
    clean_part = ex["graph"][: len(ex["graph"]) - len(ins_list)]
    clean_und = {(h, r.strip(), t) for h, r, t in clean_part} | {(t, r.strip(), h) for h, r, t in clean_part}
    inj = pathdata.injected_set([t for t in ins_list if (t[0], t[1].strip(), t[2]) not in clean_und])  # strict label
    paths = []
    for text in gcr_stage2.beam_paths(beams):
        tr = parse_path(text) or [[text, "", text]]  # malformed strings are kept as an inert pseudo-triple
        end = tr[-1][2]
        cn = [len(nbr.get(h, set()) & nbr.get(t, set())) for h, _, t in tr]
        jac = [c / max(len(nbr.get(h, set()) | nbr.get(t, set())), 1) for c, (h, _, t) in zip(cn, tr)]
        paths.append({"triples": tr, "text": text, "poisoned": pathdata.label_path([tuple(t) for t in tr], inj),
                      "gold_hit": any(rog.match(end, g) for g in ex["answer"]),
                      "end_deg": ug.degree(end) if end in ug else 0,
                      "min_deg": min(ug.degree(t) if t in ug else 0 for _, _, t in tr),
                      "end_rels": len({ug[end][nb]["relation"] for nb in ug.neighbors(end)}) if end in ug else 0,
                      "cn_last": cn[-1], "cn_min": min(cn), "jac_last": jac[-1], "jac_min": min(jac)})
    return {"id": ex["id"], "question": ex["question"], "gold": list(ex["answer"]),
            "adv": list(ex.get("adv_answers") or []), "paths": paths}


def build_split(first: str, data: str, name: str) -> str:
    ds = {ex["id"]: ex for ex in rog.load_data(data, "test")}
    rows = [json.loads(line) for line in open(first, encoding="utf8")]
    out = os.path.join(defend.RUNS, f"paths_{name}.jsonl")
    n_paths = n_pois = 0
    with open(out, "w", encoding="utf8") as f:
        for r in rows:
            rec = beam_record(ds[r["id"]], r["beams"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_paths += len(rec["paths"])
            n_pois += sum(p["poisoned"] for p in rec["paths"])
    print(f"{name}: {len(rows)} questions, {n_paths} unique beam paths ({n_paths / max(len(rows), 1):.1f}/q), {n_pois} poisoned", flush=True)
    return out


def transfer_mask(train_splits: list[str], cal_split: str, test_split: str, alpha: float = 0.05):
    """RoG-trained detector, conformal threshold calibrated on `cal_split` (GCR development beams). Returns (mask over test paths, threshold)."""
    warnings.filterwarnings("ignore")
    recs = [defend.load_paths(s) for s in train_splits]
    X = np.vstack([learned.features(r) for r in recs])
    y = np.concatenate([defend.flat(r, "poisoned").astype(int) for r in recs])
    model = learned._model().fit(X, y)
    cal = defend.load_paths(cal_split)
    losses = conformal.loss_matrix(cal, model.predict_proba(learned.features(cal))[:, 1])
    thr = conformal.crc_threshold(losses, alpha)
    test = defend.load_paths(test_split)
    s = model.predict_proba(learned.features(test))[:, 1]
    print(f"[crc transfer] alpha={alpha} cal_answerable_q={len(losses)} threshold={thr}", flush=True)
    return (np.zeros(len(s), dtype=bool) if thr is None else s > thr), thr


def mask_for(variant: str, split: str, alpha: float, qfpr: float = 0.05) -> np.ndarray:
    recs = defend.load_paths(split)
    if variant == "none":
        return np.zeros(sum(len(r["paths"]) for r in recs), dtype=bool)
    if variant == "oracle":
        return defend.flat(recs, "poisoned").astype(bool)
    if variant == "transfer":
        return transfer_mask(ROG_TRAIN, "gcrdev", split, alpha)[0]
    if variant == "fitted":
        return conformal.removal_mask(["gcrdev"], split, alpha)
    if variant == "gtransfer":
        base, _ = transfer_mask(ROG_TRAIN, "gcrdev", split, alpha)
        model, _ = gate.fit(ROG_TRAIN, "clean_dev", 100, qfpr)                       # RoG-trained gate detector
        cd = defend.load_paths("gcrcleandev")                                        # threshold recalibrated on GCR CLEAN dev beams
        cs = np.array([len(r["paths"]) for r in cd])
        cq = gate.question_scores(cs, model.predict_proba(learned.features(cd))[:, 1])[cs > 0]
        thr = gate.gate_threshold(cq, qfpr)
        test = defend.load_paths(split)
        ts = np.array([len(r["paths"]) for r in test])
        flagged = gate.question_scores(ts, model.predict_proba(learned.features(test))[:, 1]) >= thr
        print(f"[gate] qfpr={qfpr} threshold={thr:.3f} flagged {100 * flagged.mean():.1f}% of {len(test)} questions", flush=True)
        return gate.apply_gate(base, ts, flagged)
    raise SystemExit(f"unknown variant {variant}")


def run(variant: str, split: str, data: str, out: str, alpha: float, reasoner: str, quant: str):
    recs = defend.load_paths(split)
    mask = mask_for(variant, split, alpha)
    ds = {ex["id"]: ex for ex in rog.load_data(data, "test")}
    has_adv = any(ex.get("adv_answers") for ex in ds.values())
    done = {json.loads(line)["id"] for line in open(out, encoding="utf8")} if os.path.exists(out) else set()
    model = rog.ChatReasoner(reasoner, None if quant == "none" else quant)
    from tqdm import tqdm
    pos = 0
    with open(out, "a", encoding="utf8") as f:
        for r in tqdm(recs):
            k = len(r["paths"])
            keep = [p["text"] for p, rm in zip(r["paths"], mask[pos:pos + k]) if not rm]
            pos += k
            if r["id"] in done:
                continue
            ex = ds[r["id"]]
            pred = model.reason(ex["question"], keep) if keep else []
            row = {"id": r["id"], "n_beam_paths": k, "n_kept": len(keep), "prediction": pred, **rog.score(pred, ex["answer"])}
            if has_adv:
                row.update(attack.attack_metrics(pred, ex["adv_answers"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    rows = [json.loads(line) for line in open(out, encoding="utf8")]
    keys = [k for k in ("hit", "f1", "precision", "recall", "a_precision", "a_hit1") if k in rows[0]]
    print("RESULT", os.path.basename(out), f"n={len(rows)}", "  ".join(f"{k}={100 * np.mean([x[k] for x in rows]):.1f}" for k in keys),
          f"| kept {np.mean([x['n_kept'] / max(x['n_beam_paths'], 1) for x in rows]):.0%} of beam paths")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "run"])
    p.add_argument("--first")
    p.add_argument("--data")
    p.add_argument("--name")
    p.add_argument("--variant", choices=["none", "oracle", "transfer", "fitted", "gtransfer"])
    p.add_argument("--split")
    p.add_argument("--out")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--reasoner", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--quant", default="4bit")
    a = p.parse_args()
    if a.cmd == "build":
        build_split(a.first, a.data, a.name)
    else:
        run(a.variant, a.split, a.data, a.out, a.alpha, a.reasoner, a.quant)


if __name__ == "__main__":
    main()
