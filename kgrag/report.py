"""Collect every end-to-end run into one table with bootstrap confidence intervals (resampling questions).

  python -m kgrag.report            # prints the table and writes runs/kg/results_table.md
Rows: runs/kg/defended_<prefix><defence>[_fpr..|_a..][_r-..].jsonl.  Prefix -> setting:
  ''  WebQSP 500 q, paper attack | ad_ adaptive attacker | ev_ strong attacker | full_ WebQSP 1328 q | fullad_ same, adaptive
  cwq_ CWQ 500 q | cwqfull_ CWQ 3231 q (all but the 300 dev questions) | s1_..s4_ attack seeds 1-4 | clean_ NO attack (cost of the defence on unattacked data)
Paired differences against 'none' of the same setting are bootstrapped on the same resampled questions.
"""
import glob
import json
import os

import numpy as np

RUNS = os.environ.get("KG_RUNS", "runs/kg")  # set KG_RUNS to keep a different run (e.g. bf16 on the HPC) apart
SETTINGS = {"": "WebQSP-500, paper attack", "ad_": "WebQSP-500, adaptive attacker", "ev_": "WebQSP-500, strong attacker",
            "full_": "WebQSP-1328, paper attack", "fullad_": "WebQSP-1328, adaptive attacker", "cwq_": "CWQ-500, paper attack",
            "s1_": "WebQSP-500, attack seed 1", "s2_": "WebQSP-500, attack seed 2",
            "s3_": "WebQSP-500, attack seed 3", "s4_": "WebQSP-500, attack seed 4", "b8_": "WebQSP-500, stronger-budget attack (N=10, K=8), detector unseen", "fp_": "WebQSP-500, held-out family: profile-copy", "fb_": "WebQSP-500, held-out family: bridge", "fs_": "WebQSP-500, held-out family: spread", "cwqclean_": "CWQ-3231, NO attack (cost of the defence on clean data)",
            "clean_": "WebQSP-500, NO attack (cost of the defence on clean data)",
            "cwqfull_": "CWQ-3231, paper attack (all but the 300 dev questions)"}
RNG = np.random.default_rng(0)


def parse(fname: str):
    name = os.path.basename(fname)[len("defended_"):-len(".jsonl")]
    for pre in sorted(SETTINGS, key=len, reverse=True):
        if pre and name.startswith(pre):
            return pre, name[len(pre):]
    return "", name


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf8"))}


def boot(vals: np.ndarray, B: int = 2000):
    idx = RNG.integers(0, len(vals), size=(B, len(vals)))
    m = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    runs: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "defended_*.jsonl"))):
        pre, d = parse(f)
        rows = load(f)
        if len(rows) >= 100:
            runs.setdefault(pre, {})[d] = rows
    out = ["# End-to-end results (95% bootstrap CI over questions; values in %)\n"]
    for pre, name in SETTINGS.items():
        if pre not in runs:
            continue
        base = runs[pre].get("none")
        out += [f"\n## {name}\n", "| defence | n | F1 | Hit | Precision | Planted in output | F1 gain vs none | kept paths |",
                "|---|---|---|---|---|---|---|---|"]
        order = sorted(runs[pre], key=lambda d: (d != "none", d != "oracle", d))
        for d in order:
            rows = runs[pre][d]
            ids = sorted(rows)
            cells = []
            for m in ("f1", "hit", "precision", "a_precision"):
                mu, lo, hi = boot(np.array([rows[i][m] for i in ids]) * 100)
                cells.append(f"{mu:.1f} [{lo:.1f}, {hi:.1f}]")
            gain = "-"
            if base and d != "none":
                common = sorted(set(ids) & set(base))
                if len(common) >= 100:
                    diff = np.array([rows[i]["f1"] - base[i]["f1"] for i in common]) * 100
                    mu, lo, hi = boot(diff)
                    gain = f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            kept = np.mean([r["n_kept"] / max(r["n_paths"], 1) for r in rows.values()]) * 100
            out.append(f"| {d} | {len(ids)} | " + " | ".join(cells) + f" | {gain} | {kept:.0f}% |")
    text = "\n".join(out) + "\n"
    open(os.path.join(RUNS, "results_table.md"), "w", encoding="utf8").write(text)
    print(text)


if __name__ == "__main__":
    main()
