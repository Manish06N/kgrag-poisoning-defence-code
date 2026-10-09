"""Tables for the defence on GCR: attacked test (1,328 questions) and clean (500), variants none / oracle / transfer / fitted.

  PYTHONPATH=. python scripts/gcr_defence_report.py
Reports F1, Hits@1, EM and the planted-answer rate, paired bootstrap differences against no defence, the share of the oracle gain recovered,
the same restricted to questions that have a poisoned beam path, and the cost on clean graphs. Reads runs/kg/gcr3s2_<variant>_<split>.jsonl.
"""
import json
import os

import numpy as np
from datasets import load_dataset

from kgrag import defend, rog

rng = np.random.default_rng(0)
N = rog.normalize
gold = {x["id"]: x["answer"] for x in load_dataset("rmanluo/RoG-webqsp", split="test")}


def load(variant, split):
    p = f"runs/kg/gcr3s2_{variant}_{split}.jsonl"
    return {(r := json.loads(line))["id"]: r for line in open(p, encoding="utf8")} if os.path.exists(p) else {}


def hits1(r, i):
    return float(bool(r[i]["prediction"]) and N(r[i]["prediction"][0]) in {N(x) for x in gold[i]})


def em(r, i):
    P = {N(p) for p in r[i]["prediction"] if p}
    return float(bool(P) and P == {N(x) for x in gold[i]})


METRICS = {"F1": lambda r, i: r[i]["f1"], "Hits@1": hits1, "EM": em, "planted top-1": lambda r, i: r[i].get("a_hit1", 0.0)}


def paired(base, other, ids, f):
    d = np.array([f(other, i) - f(base, i) for i in ids]) * 100
    m = d[rng.integers(0, len(ids), (3000, len(ids)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(split, variants, label):
    runs = {v: load(v, split) for v in variants}
    runs = {v: r for v, r in runs.items() if r}
    ids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    expected = {"gcrtest": 1328, "gcrclean": 500}[split]
    sizes = {v: len(r) for v, r in runs.items()}
    assert all(n == len(ids) == expected for n in sizes.values()), (
        f"variants do not cover the same complete question set: {sizes} (expected {expected})")
    print(f"\n== {label}: {len(ids)} questions, variants {list(runs)}")
    for name, f in METRICS.items():
        cells = []
        for v, r in runs.items():
            m = 100 * np.mean([f(r, i) for i in ids])
            c = "" if v == "none" else " (%+.1f [%+.1f, %+.1f])" % paired(runs["none"], r, ids, f)
            cells.append(f"{v}: {m:.1f}{c}")
        print(f"  {name:14s} " + " | ".join(cells))
    kept = {v: 100 * np.mean([r[i]["n_kept"] / max(r[i]["n_beam_paths"], 1) for i in ids]) for v, r in runs.items()}
    print("  beam paths kept: " + ", ".join(f"{v} {k:.0f}%" for v, k in kept.items()))
    if "oracle" in runs and "none" in runs:
        gain_o = np.mean([runs["oracle"][i]["f1"] - runs["none"][i]["f1"] for i in ids])
        for v in ("transfer", "fitted", "gtransfer"):
            if v in runs:
                g = np.mean([runs[v][i]["f1"] - runs["none"][i]["f1"] for i in ids])
                print(f"  share of the oracle F1 gain recovered by {v}: {100 * g / gain_o:.0f}% (oracle gain {100 * gain_o:+.1f})")
    return runs, ids


def main():
    runs, ids = table("gcrtest", ["none", "oracle", "transfer", "fitted", "gtransfer"], "attacked test (paper attack)")
    if os.path.exists(os.path.join(defend.RUNS, "paths_gcrtest.jsonl")):
        recs = {r["id"]: r for r in defend.load_paths("gcrtest")}
        aff = [i for i in ids if any(p["poisoned"] for p in recs[i]["paths"])]
        print(f"\n  questions with a poisoned beam path: {len(aff)} of {len(ids)}")
        for v in ("oracle", "transfer", "fitted", "gtransfer"):
            if v in runs:
                print(f"    {v:9s} F1 gain on those questions: %+.1f [%+.1f, %+.1f]" % paired(runs["none"], runs[v], aff, METRICS["F1"]))
    table("gcrclean", ["none", "transfer", "fitted", "gtransfer"], "clean graph (cost of the defence)")


if __name__ == "__main__":
    main()
