"""End-to-end test of a path filter: drop suspicious retrieved paths, let RoG reason on the rest, measure QA and attack metrics.

  python -m kgrag.defended_run --defence rels            # score name from kgrag.defend, threshold at 10% clean dev paths removed
  python -m kgrag.defended_run --defence none             # no filtering (same path set, for a fair comparison)
  python -m kgrag.defended_run --defence oracle           # removes exactly the inserted-triple paths (upper bound)
  python -m kgrag.defended_run --defence rels+verify      # sum of z-scored scores (z-scored with dev statistics)
  python -m kgrag.defended_run --defence learned:dev      # gradient-boosting detector trained on the listed dev splits (a+b = several)
Calibrated removal (conformal risk control, kgrag/conformal.py):  --defence crc:dev+K1_dev+K8_dev+ad_dev --alpha 0.05
Gated removal (question-level gate, kgrag/gate.py):  --defence gate:dev --fpr 0.02 --qfpr 0.05   (path filter learned:dev, applied only
in questions the gate flags as attacked; the gate detector uses dev + clean_dev paths).
Literature baseline (RAGDefender adapted to KG paths, kgrag/ragdefender.py):  --defence ragdefender:conc   or   ragdefender:clust
Add --kge to append TransE plausibility features (kgrag/kge.py) to the learned detector used by learned:, crc:, gate: and gatecrc:.
Gated conformal removal (dev-only choices throughout):  --defence gatecrc:dev+K1_dev+K8_dev+ad_dev --alpha 0.05 --qfpr 0.05
Adaptive-attack data: add  --prefix ad_ --data data/poisoned_webqsp_adaptive  (uses runs/kg/paths_ad_test.jsonl / paths_ad_dev.jsonl).
Output: runs/kg/defended_<prefix><name>.jsonl, one line per test question.
"""
import argparse
import json
import os

import numpy as np

from kgrag import attack, conformal, defend, gate, learned, rog


def combined_scores(names, split, recs, dev_recs=None, dev_split="dev"):
    """Sum of per-score z-values; mean/std come from the dev split so the test split is never used to fit anything."""
    out = 0.0
    for n in names:
        s_dev = defend.get_scores(n, dev_split, dev_recs or defend.load_paths(dev_split))
        s = defend.get_scores(n, split, recs)
        out = out + (s - s_dev.mean()) / (s_dev.std() + 1e-9)
    return out


def removal_mask(defence: str, test, dev, fpr: float, prefix: str = "", alpha: float = 0.05, qfpr: float = 0.05):
    y_t = defend.flat(test, "poisoned").astype(bool)
    if defence == "none":
        return np.zeros(len(y_t), dtype=bool)
    if defence == "oracle":
        return y_t
    if defence.startswith("gate:"):
        splits = defence.split(":", 1)[1].split("+")
        return gate.removal_mask(learned.removal_mask(splits, prefix + "test", fpr), splits, prefix + "test", qfpr)
    if defence.startswith("gatecrc:"):
        splits = defence.split(":", 1)[1].split("+")
        return gate.removal_mask(conformal.removal_mask(splits, prefix + "test", alpha), splits, prefix + "test", qfpr)
    if defence.startswith("ragdefender"):
        from kgrag import ragdefender
        return ragdefender.removal_mask(test, defence.split(":", 1)[1] if ":" in defence else "conc")
    if defence.startswith("crc:"):
        return conformal.removal_mask(defence.split(":", 1)[1].split("+"), prefix + "test", alpha)
    if defence.startswith("learned:"):
        return learned.removal_mask(defence.split(":", 1)[1].split("+"), prefix + "test", fpr)
    names = defence.split("+")
    s_t = combined_scores(names, prefix + "test", test, dev, prefix + "dev")
    s_d = combined_scores(names, prefix + "dev", dev, dev, prefix + "dev")
    y_d = defend.flat(dev, "poisoned").astype(int)
    return s_t > defend.threshold_at_fpr(s_d, y_d, fpr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--defence", required=True)
    p.add_argument("--fpr", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.05, help="conformal risk level for crc: defences")
    p.add_argument("--qfpr", type=float, default=0.05, help="gate: share of clean dev questions the gate may flag")
    p.add_argument("--kge", action="store_true", help="append TransE plausibility features to the learned detector")
    p.add_argument("--data", default="data/poisoned_webqsp")
    p.add_argument("--prefix", default="", help="split-name prefix of the paths files, e.g. ad_")
    p.add_argument("--quant", default="4bit")
    p.add_argument("--reasoner", default="", help="chat LLM that reads the retrieved paths instead of the RoG model")
    a = p.parse_args()
    test, dev = defend.load_paths(a.prefix + "test"), defend.load_paths(a.prefix + "dev")
    if a.kge:
        from kgrag import kge as _kge
        learned.EXTRA = _kge.extra_features
    removed = removal_mask(a.defence, test, dev, a.fpr, a.prefix, a.alpha, a.qfpr)
    tag = f"_a{a.alpha:g}" if a.defence.startswith("crc:") else ("" if abs(a.fpr - 0.10) < 1e-9 else f"_fpr{a.fpr:g}")
    if a.defence.startswith("gate:"):
        tag = f"_fpr{a.fpr:g}_q{a.qfpr:g}"
    if a.defence.startswith("gatecrc:"):
        tag = f"_a{a.alpha:g}_q{a.qfpr:g}"
    if a.kge:
        tag += "_kge"
    if a.reasoner:
        tag += "_r-" + a.reasoner.split("/")[-1]
    out_path = os.path.join(defend.RUNS, f"defended_{a.prefix}{a.defence.replace(':', '-')}{tag}.jsonl")
    done = {json.loads(line)["id"] for line in open(out_path, encoding="utf8")} if os.path.exists(out_path) else set()
    quant = None if a.quant == "none" else a.quant
    model = rog.ChatReasoner(a.reasoner, quant) if a.reasoner else rog.Rog("rmanluo/RoG", quant)
    ds = rog.load_data(a.data, "test")
    adv = {ex["id"]: ex.get("adv_answers") or [] for ex in ds}  # clean (unattacked) data has no planted answers
    pos = 0
    from tqdm import tqdm
    with open(out_path, "a", encoding="utf8") as f:
        for r in tqdm(test):
            k = len(r["paths"])
            keep = [pt["text"] for pt, rm in zip(r["paths"], removed[pos:pos + k]) if not rm]
            pos += k
            if r["id"] in done:
                continue
            pred = model.reason(r["question"], keep) if keep else []
            row = {"id": r["id"], "n_paths": k, "n_kept": len(keep), "prediction": pred, **rog.score(pred, r["gold"]),
                   **attack.attack_metrics(pred, adv[r["id"]])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    rows = [json.loads(line) for line in open(out_path, encoding="utf8")]
    keys = ("hit", "f1", "precision", "recall", "a_precision", "a_hit1", "a_mrr")
    print("RESULT", a.prefix + a.defence + tag, f"n={len(rows)}", "  ".join(f"{k}={100 * np.mean([x[k] for x in rows]):.1f}" for k in keys),
          f"| kept {np.mean([x['n_kept'] / max(x['n_paths'], 1) for x in rows]):.0%} of paths")


if __name__ == "__main__":
    main()
