"""Figures for the paper, regenerated from the run files (no model calls).

  python scripts/make_figures.py

Writes paper/figures/fig_six_attacks.pdf (F1 per attack: no defence, C, D, oracle, with 95% bootstrap intervals and the share of the oracle gain
that D recovers) and paper/figures/fig_gcr.pdf (change in F1 on GCR attacked and clean data). C, D and the oracle are the strict-label runs in
runs/kg_strict; no-defence runs and the GCR runs are in runs/kg.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load(path):
    return {(r := json.loads(line))["id"]: r for line in open(path, encoding="utf8")}


def boot(x, n=3000):
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


# ---------------------------------------------------------------- figure: six attacks
PREFIX = ["", "ad_", "ev_", "fp_", "fb_", "fs_"]
NAMES = ["Paper", "Adaptive", "Evasive", "Profile-copy", "Bridge", "Spread"]
MIX = "gatecrc-dev+K1_dev+K8_dev+ad_dev_a0.05_q0.05"
CONF = "gatecrc-dev_a0.05_q0.05"
rows = {
    "No defence": [load(f"runs/kg/defended_{p}none.jsonl") for p in PREFIX],
    "C: gate + conformal, paper detector": [load(f"runs/kg_strict/defended_{p}{CONF}.jsonl") for p in PREFIX],
    "D: gate + conformal, mixed detector": [load(f"runs/kg_strict/defended_{p}{MIX}.jsonl") for p in PREFIX],
    "Oracle (removes inserted paths)": [load(f"runs/kg_strict/defended_{p}oracle_fpr0.02.jsonl") for p in PREFIX],
}
ids = sorted(set.intersection(*[set(r) for v in rows.values() for r in v]))
F = {k: np.array([[r[i]["f1"] for i in ids] for r in v]) * 100 for k, v in rows.items()}  # attacks x questions
clean = load("runs/kg/defended_clean_none.jsonl")
clean_f1 = 100 * np.mean([clean[i]["f1"] for i in ids if i in clean])
colors = ["#9a9a9a", "#8fb4d9", "#0b5394", "#6aa84f"]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
w = 0.2
x = np.arange(len(NAMES))
for j, (k, c) in enumerate(zip(F, colors)):
    m = F[k].mean(1)
    ci = np.array([boot(F[k][a]) for a in range(len(NAMES))])
    ax.bar(x + (j - 1.5) * w, m, w, color=c, label=k, yerr=[m - ci[:, 0], ci[:, 1] - m], error_kw={"lw": 0.7, "capsize": 1.5})
share = (F["D: gate + conformal, mixed detector"].mean(1) - F["No defence"].mean(1)) / (F["Oracle (removes inserted paths)"].mean(1) - F["No defence"].mean(1))
ax.axhline(clean_f1, color="k", lw=0.7, ls="--")
ax.text(len(NAMES) - 0.55, 71.0, f"dashed line: clean graph, no defence ({clean_f1:.1f})", ha="right", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n" + f"{100 * sh:.0f}%".replace("-", "−") for n, sh in zip(NAMES, share)])
ax.text(-0.62, 26.4, "D recovers this share\nof the oracle gain:", ha="right", va="center", fontsize=6.5, color="#0b5394", clip_on=False)
ax.set_ylabel("F1 (%)")
ax.set_ylim(30, 78)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_six_attacks.pdf")
print("six attacks: n =", len(ids), "D share of oracle gain:", np.round(100 * share).astype(int).tolist())

# ---------------------------------------------------------------- figure: GCR
def gcr(variant, split):
    return load(f"runs/kg/gcr3s2_{variant}_{split}.jsonl")


panels = [("Attacked graph (1,328 questions)", "gcrtest", ["oracle", "transfer", "fitted", "gtransfer"]),
          ("Clean graph (500 questions)", "gcrclean", ["transfer", "fitted", "gtransfer"])]
LAB = {"oracle": "Oracle", "transfer": "RoG-trained detector", "fitted": "Fitted on GCR", "gtransfer": "D (gate)"}
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=False)
for ax, (title, split, vs) in zip(axes, panels):
    base = gcr("none", split)
    for r_i, v in enumerate(vs):
        o = gcr(v, split)
        common = sorted(set(base) & set(o))
        d = np.array([o[i]["f1"] - base[i]["f1"] for i in common]) * 100
        lo, hi = boot(d)
        c = "#6aa84f" if v == "oracle" else ("#0b5394" if v == "gtransfer" else "#8fb4d9")
        ax.errorbar(d.mean(), r_i, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color=c, capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels([LAB[v] for v in vs])
    ax.invert_yaxis()
    ax.set_xlabel("change in F1 vs no defence")
    ax.set_title(title, fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_gcr.pdf")
print("gcr done")

# ---------------------------------------------------------------- shared palette for the remaining figures
BLUE, LBLUE, GREEN, ORANGE, RED, GREY, PURPLE = "#0b5394", "#8fb4d9", "#6aa84f", "#e69138", "#cc4125", "#9a9a9a", "#7b3f98"
MINUS = "−"

# ---------------------------------------------------------------- figure: baselines (values from the Section 6.3 table)
BASE = [  # name, gain, lo, hi, colour
    ("Similarity (TrustRAG-style)", -3.6, -5.4, -1.8, ORANGE), ("Perplexity filter", -3.3, -5.8, -0.9, ORANGE),
    ("LLM self-check", 0.7, -0.5, 1.8, ORANGE), ("TransE plausibility @2%", -0.0, -0.6, 0.4, ORANGE),
    ("RAGDefender, concentration", -11.9, -14.9, -8.9, RED), ("RAGDefender, clustering", -18.8, -22.5, -15.1, RED),
    ("RAGDefender, true count", -14.9, -18.3, -11.6, RED), ("Relation variety (rels)", 8.6, 5.4, 11.7, LBLUE),
    ("Learned detector @2%", 15.3, 11.9, 18.6, BLUE),
]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
for i, (n, g, lo, hi, c) in enumerate(BASE):
    ax.barh(i, g, color=c, xerr=[[g - lo], [hi - g]], error_kw={"lw": 0.8, "capsize": 2})
ax.set_yticks(range(len(BASE)))
ax.set_yticklabels([b[0] for b in BASE])
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("F1 gain over no defence (WebQSP-500, paper attack)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="text-RAG style / embedding"), Patch(color=RED, label="RAGDefender adapted to KG paths"),
                   Patch(color=LBLUE, label="structural heuristic"), Patch(color=BLUE, label="learned path detector (ours)")],
          loc="upper right", frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_baselines.pdf")

# ---------------------------------------------------------------- figure: cost vs protection and attack coverage (values from the Section 6.4 tables)
CFG = [  # label, clean change, worst-case gain, lo, hi, gate?, mixed detector?
    ("A", -6.1, 3.2, -0.2, 6.5, False, False), ("B", -5.3, 6.8, 3.3, 10.2, False, True), ("C", -0.7, 3.4, 1.6, 5.1, True, False),
    ("D", -0.5, 5.2, 2.0, 8.4, True, True), ("E", -5.9, 4.5, 1.3, 7.7, False, False), ("F", -6.1, 9.8, 6.3, 13.3, False, True),
    ("G", -0.1, 3.2, 1.4, 5.1, True, False), ("H", -0.7, 6.1, 2.9, 9.2, True, True),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
for lab, dx, g, lo, hi, gate, mixed in CFG:
    a1.errorbar(dx, g, yerr=[[g - lo], [hi - g]], fmt="s" if mixed else "o", color=BLUE if gate else ORANGE, ms=6 if lab != "D" else 8,
                capsize=2, lw=0.8, mfc=BLUE if (gate and lab == "D") else None)
    off = {"A": (-9, 0), "E": (9, 0), "B": (9, 0), "F": (9, 0), "C": (-10, -3), "D": (11, 3), "H": (-10, 3), "G": (10, -3)}[lab]
    a1.annotate(lab, (dx, g), textcoords="offset points", xytext=off, ha="center", va="center", fontsize=7, weight="bold" if lab == "D" else None)
a1.set_xlabel("change in clean-graph F1")
a1.set_ylabel("worst-case F1 gain\n(three known attacks)")
a1.set_xlim(-7, 0.8)
a1.set_ylim(-1, 15)
a1.legend(handles=[plt.Line2D([], [], marker="o", color=BLUE, ls="", label="gate"), plt.Line2D([], [], marker="o", color=ORANGE, ls="", label="always-on"),
                   plt.Line2D([], [], marker="o", color=GREY, ls="", label="paper detector"), plt.Line2D([], [], marker="s", color=GREY, ls="", label="mixed detector")],
          loc="upper center", bbox_to_anchor=(0.56, 1.0), frameon=False, fontsize=6.5, ncol=1)
a1.set_title("(a) cost against protection", fontsize=8)
COV = [("Adaptive", [("paper attack only", 9.3, 6.0, 12.4, LBLUE), ("mixed", 16.4, 13.2, 19.8, BLUE)]),
       ("Evasive", [("paper attack only", -1.4, -4.6, 1.9, LBLUE), ("mixed", 3.9, 0.9, 6.8, BLUE), ("attack-aware", 8.5, 5.4, 11.6, GREEN)])]
xpos, ticks, tl = 0, [], []
for gname, bars in COV:
    start = xpos
    for lab, g, lo, hi, c in bars:
        a2.bar(xpos, g, 0.8, color=c, yerr=[[g - lo], [hi - g]], error_kw={"lw": 0.8, "capsize": 2})
        xpos += 1
    ticks.append((start + xpos - 1) / 2)
    tl.append(gname)
    xpos += 0.7
a2.axhline(0, color="k", lw=0.7)
a2.set_xticks(ticks)
a2.set_xticklabels(tl)
a2.set_ylabel("F1 gain at 2% level")
a2.legend(handles=[Patch(color=LBLUE, label="trained on paper attack only"), Patch(color=BLUE, label="trained on paper, K=1, K=8, decoys"),
                   Patch(color=GREEN, label="also trained on the evasive attack")], loc="upper left", frameon=False, fontsize=6.3)
a2.set_ylim(-6, 27)
a2.set_title("(b) coverage of the training attacks", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_tradeoff_coverage.pdf")

# ---------------------------------------------------------------- figure: conformal validity (100-split simulation of Section 6.7, strict labels)
AL = [0.02, 0.05, 0.10, 0.20]
RISK = [0.018, 0.045, 0.096, 0.197]
WITHIN = [66, 63, 55, 58]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.2))
xs = np.arange(len(AL))
a1.bar(xs, RISK, 0.55, color=BLUE, label="realised risk (mean over 100 splits)")
a1.plot(xs, AL, "o--", color=RED, label="target level α")
a1.set_xticks(xs)
a1.set_xticklabels([f"{a:.2f}" for a in AL])
a1.set_xlabel("target level α")
a1.set_ylabel("risk of losing all clean evidence")
a1.legend(frameon=False, fontsize=6.5, loc="upper left")
a1.set_title("(a) holds on average", fontsize=8)
a2.bar(xs, WITHIN, 0.55, color=ORANGE)
for x_, v in zip(xs, WITHIN):
    a2.text(x_, v + 1.5, f"{v}%", ha="center", fontsize=7)
a2.set_xticks(xs)
a2.set_xticklabels([f"{a:.2f}" for a in AL])
a2.set_xlabel("target level α")
a2.set_ylabel("splits with realised risk ≤ α (%)")
a2.set_ylim(0, 100)
a2.set_title("(b) but not in every split", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_calibration.pdf")

# ---------------------------------------------------------------- figure: MultiGraph ablation (run files, WebQSP-500, paper attack)
def mean_f1(path, ids_=None):
    r = load(path)
    keys = sorted(r) if ids_ is None else ids_
    return 100 * np.mean([r[i]["f1"] for i in keys])


mg = {"Clean graph": ("runs/kg/defended_clean_none.jsonl", "runs/kg_mg/defended_clean_none_fpr0.02.jsonl"),
      "No defence": ("runs/kg/defended_none.jsonl", "runs/kg_mg/defended_none_fpr0.02.jsonl"),
      "D (gate + conformal)": (f"runs/kg_strict/defended_{MIX}.jsonl", f"runs/kg_mg/defended_{MIX}.jsonl"),
      "Oracle": ("runs/kg_strict/defended_oracle_fpr0.02.jsonl", "runs/kg_mg/defended_oracle_fpr0.02.jsonl")}
fig, ax = plt.subplots(figsize=(6.3, 2.5))
x = np.arange(len(mg))
for j, (lab, c) in enumerate([("Simple graph (RoG default)", LBLUE), ("MultiGraph (parallel edges kept)", PURPLE)]):
    vals = [mean_f1(v[j]) for v in mg.values()]
    bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, color=c, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(list(mg))
ax.set_ylabel("F1 (%)")
ax.set_ylim(30, 84)
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_multigraph.pdf")
print("extra figures done; MultiGraph oracle vs clean:", [round(mean_f1(mg[k][1]), 1) for k in ("Oracle", "Clean graph")])
