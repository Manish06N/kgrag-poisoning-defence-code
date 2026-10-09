"""Worst-case tables with unbiased point estimates.

  python scripts/worst_case_tables.py

The point estimate of the worst-case gain is the difference of the OBSERVED worst cases (worst defended F1 minus worst undefended F1).
The bootstrap is used only for the interval: a basic bootstrap interval, [2*est - q97.5, 2*est - q2.5] of the resampled differences, which corrects the upward bias that
the minimum of near-tied means gives to the resampled differences (the percentile interval and the mean of the resampled differences are biased the same way).
Defended rows come from runs/kg_strict (strict poisoned-path labels); no-defence runs are label-independent and come from runs/kg.
"""
import json
import os

import numpy as np

rng = np.random.default_rng(0)
MIX = "dev+K1_dev+K8_dev+ad_dev"
ROWS = {  # row -> run-file stem (without the attack prefix and extension)
    "A. conformal always-on, paper detector": "crc-dev_a0.05",
    "B. conformal always-on, mixed detector": f"crc-{MIX}_a0.05",
    "C. gate + conformal, paper detector": "gatecrc-dev_a0.05_q0.05",
    "D. gate + conformal, mixed detector": f"gatecrc-{MIX}_a0.05_q0.05",
    "E. always-on 2%, paper detector": "learned-dev_fpr0.02",
    "F. always-on 2%, mixed detector": f"learned-{MIX}_fpr0.02",
    "G. gate 2%, paper detector": "gate-dev_fpr0.02_q0.05",
    "H. gate 2%, mixed detector": f"gate-{MIX}_fpr0.02_q0.05",
}
ATTACKS = ["", "ad_", "ev_", "fp_", "fb_", "fs_"]
NAMES = ["paper", "adaptive", "evasive", "profile-copy", "bridge", "spread"]


def load(path):
    return {(r := json.loads(line))["id"]: r for line in open(path, encoding="utf8")} if os.path.exists(path) else None


def fmt(x):
    return f"{x:+.1f}".replace("-", "−")


def table(title, subset, rows):
    none = [load(f"runs/kg/defended_{ATTACKS[a]}none.jsonl") for a in subset]
    clean_none = load("runs/kg/defended_clean_none.jsonl")
    print(f"\n== {title}: attacks {[NAMES[a] for a in subset]}")
    for name, stem in rows.items():
        runs = [load(f"runs/kg_strict/defended_{ATTACKS[a]}{stem}.jsonl") for a in subset]
        if any(r is None or len(r) < 500 for r in runs):
            print(f"  {name:42s} (strict runs not finished)")
            continue
        ids = sorted(set.intersection(*[set(r) for r in runs + none]))
        n = len(ids)
        ix = rng.integers(0, n, (5000, n))
        F = np.array([[r[i]["f1"] for i in ids] for r in runs]) * 100
        F0 = np.array([[r[i]["f1"] for i in ids] for r in none]) * 100
        est = F.mean(1).min() - F0.mean(1).min()
        boot = np.min(np.stack([F[a][ix].mean(1) for a in range(len(subset))]), 0) - np.min(np.stack([F0[a][ix].mean(1) for a in range(len(subset))]), 0)
        lo, hi = 2 * est - np.percentile(boot, 97.5), 2 * est - np.percentile(boot, 2.5)
        worst = NAMES[subset[int(F.mean(1).argmin())]]
        planted = [np.mean([r[i]["a_precision"] for i in ids]) * 100 for r in runs]
        cl = load(f"runs/kg_strict/defended_clean_{stem}.jsonl")
        clean_txt = ""
        if cl is not None and clean_none is not None and len(cl) >= 500:
            ci = sorted(set(cl) & set(clean_none))
            d = np.array([cl[i]["f1"] - clean_none[i]["f1"] for i in ci]) * 100
            m = d[rng.integers(0, len(ci), (5000, len(ci)))].mean(1)
            clean_txt = f" | clean {100 * np.mean([cl[i]['f1'] for i in ci]):.1f} ({fmt(d.mean())} [{fmt(np.percentile(m, 2.5))}, {fmt(np.percentile(m, 97.5))}])"
        print(f"  {name:42s} worst {F.mean(1).min():.1f} ({worst}) vs {F0.mean(1).min():.1f}: gain {fmt(est)} [{fmt(lo)}, {fmt(hi)}] | worst planted {max(planted):.1f}%{clean_txt}")


table("three known attacks", [0, 1, 2], ROWS)
table("six attacks", [0, 1, 2, 3, 4, 5], {k: v for k, v in ROWS.items() if k[0] in "BCD"})
table("held-out families", [3, 4, 5], {k: v for k, v in ROWS.items() if k[0] in "BCD"})
