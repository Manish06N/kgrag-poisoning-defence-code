"""Decompose the attack damage on GCR into what a perfect path filter can reach and what it cannot, and test where the unreachable part comes from.

  KG_RUNS=runs/kg_strict PYTHONPATH=. python scripts/gcr_decompose.py

Needs the GCR clean run on the same 1,328 questions as the attacked run (split gcrcleanfull) and the second-stage outputs
runs/kg/gcr3s2_{none,oracle}_gcrtest.jsonl and runs/kg/gcr3s2_none_gcrcleanfull.jsonl.

  damage       = clean F1 - attacked F1 (no defence)
  reachable    = oracle F1 - attacked F1            (what removing exactly the poisoned beam paths gives back)
  unreachable  = clean F1 - oracle F1               (damage that remains although every poisoned path is gone)
Mechanism of the unreachable part: for each attacked question, the clean run's gold-supporting beam paths that are missing among the attacked run's
non-poisoned beam paths ("lost clean evidence"), split into paths that use an entity pair carrying an inserted triple with another relation
(candidate for overwriting in a one-relation-per-pair graph) and the rest (displacement: the planted paths took beam slots, or the changed graph changed the search).
"""
import json
import os

import numpy as np

from kgrag import defend, gcr_defend, rog

rng = np.random.default_rng(0)


def load_rows(path):
    return {(r := json.loads(line))["id"]: r for line in open(path, encoding="utf8")}


def ci(x, n=4000):
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


att = {r["id"]: r for r in defend.load_paths("gcrtest")}
cln = {r["id"]: r for r in defend.load_paths("gcrcleanfull")}
none_a = load_rows("runs/kg/gcr3s2_none_gcrtest.jsonl")
orc_a = load_rows("runs/kg/gcr3s2_oracle_gcrtest.jsonl")
none_c = load_rows("runs/kg/gcr3s2_none_gcrcleanfull.jsonl")
ids = sorted(set(att) & set(cln) & set(none_a) & set(orc_a) & set(none_c))
print(f"questions with attacked and clean runs: {len(ids)}")
f = lambda rows: np.array([rows[i]["f1"] for i in ids]) * 100
FA, FO, FC = f(none_a), f(orc_a), f(none_c)
poisoned_q = np.array([any(p["poisoned"] for p in att[i]["paths"]) for i in ids])


def line(name, x, mask=None):
    xs = x if mask is None else x[mask]
    lo, hi = ci(xs)
    print(f"  {name:34s} {xs.mean():+6.1f} [{lo:+.1f}, {hi:+.1f}]  (n={len(xs)})")


print("\nF1 means: clean %.1f | attacked %.1f | oracle %.1f" % (FC.mean(), FA.mean(), FO.mean()))
for title, mask in (("all questions", None), ("questions WITH a poisoned beam path", poisoned_q), ("questions WITHOUT one", ~poisoned_q)):
    print(f"\n{title}")
    line("damage (clean - attacked)", FC - FA, mask)
    line("reachable (oracle - attacked)", FO - FA, mask)
    line("unreachable (clean - oracle)", FC - FO, mask)
d = (FC - FA).mean()
r = (FO - FA).mean()
print(f"\nshare of the damage a perfect path filter reaches: {100 * r / d:.0f}%  (unreachable {100 * (d - r) / d:.0f}%)")

# ---- mechanism of the lost clean evidence
ds = {ex["id"]: ex for ex in rog.load_data("data/poisoned_webqsp", "test")}
lost_q = ovw_q = n_lost = n_ovw = 0
q_lost = []
for i in ids:
    clean_gold = {p["text"] for p in cln[i]["paths"] if p["gold_hit"]}
    kept = {p["text"] for p in att[i]["paths"] if not p["poisoned"]}
    lost = clean_gold - kept
    q_lost.append(bool(lost))
    if not lost:
        continue
    ins = ds[i].get("poison_triples") or []
    pair_rel = {}
    for h, r_, t in ins:
        pair_rel.setdefault(frozenset((h, t)), set()).add(r_.strip())
    ovw = 0
    for text in lost:
        tr = gcr_defend.parse_path(text) or []
        if any(frozenset((h, t)) in pair_rel and r_.strip() not in pair_rel[frozenset((h, t))] for h, r_, t in tr):
            ovw += 1
    lost_q += 1
    n_lost += len(lost)
    ovw_q += ovw > 0
    n_ovw += ovw
q_lost = np.array(q_lost)
print(f"\nquestions where a clean gold-supporting beam path is missing among the attacked non-poisoned beam paths: {lost_q} of {len(ids)} ({100 * lost_q / len(ids):.0f}%)")
print(f"  lost paths: {n_lost}; of which on an entity pair carrying an inserted triple with another relation (overwrite candidates): {n_ovw} ({100 * n_ovw / max(n_lost, 1):.0f}%);"
      f" questions with at least one such path: {ovw_q}")
print("unreachable damage (clean - oracle) in questions with lost clean evidence vs without:")
line("with lost clean evidence", FC - FO, q_lost)
line("without", FC - FO, ~q_lost)
