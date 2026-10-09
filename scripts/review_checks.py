import json, numpy as np
from datasets import load_dataset
from kgrag import rog
rng = np.random.default_rng(0)
L = lambda p: {(r := json.loads(l))["id"]: r for l in open(p, encoding="utf8")}
def boot_ratio(none, lrn, ora, B=2000):
    ids = sorted(set(none) & set(lrn) & set(ora))
    a = np.array([[none[i]["f1"], lrn[i]["f1"], ora[i]["f1"]] for i in ids]) * 100
    idx = rng.integers(0, len(ids), (B, len(ids)))
    m = a[idx].mean(1)
    r = (m[:, 1] - m[:, 0]) / (m[:, 2] - m[:, 0])
    pt = (a[:, 1].mean() - a[:, 0].mean()) / (a[:, 2].mean() - a[:, 0].mean())
    return len(ids), pt, np.percentile(r, 2.5), np.percentile(r, 97.5)
R = "runs/kg/defended_"
print("== percent of oracle gain recovered (paired bootstrap over questions)")
for tag, pre in (("WebQSP-1328", "full_"), ("CWQ-3231", "cwqfull_"), ("WebQSP-500 s4", "s4_")):
    try:
        n, pt, lo, hi = boot_ratio(L(R + pre + "none.jsonl"), L(R + pre + "learned-dev_fpr0.02.jsonl"), L(R + pre + "oracle.jsonl"))
        print(f"{tag}: n={n} recovered {100*pt:.1f}% [{100*lo:.1f}, {100*hi:.1f}]")
    except Exception as e: print(tag, "skip", e)
print("== attack strength (WebQSP-500, 4-bit RoG): clean vs attacked")
c, n_ = L(R + "clean_none.jsonl"), L(R + "none.jsonl")
ids = sorted(set(c) & set(n_))
for m in ("hit", "f1"):
    a, b = np.mean([c[i][m] for i in ids]) * 100, np.mean([n_[i][m] for i in ids]) * 100
    print(f"{m}: clean {a:.1f} -> attacked {b:.1f}  relative drop {100*(a-b)/a:.0f}%")
print("== WebQSP -> CWQ overlap")
w = load_dataset("rmanluo/RoG-webqsp", split="test"); cq = load_dataset("rmanluo/RoG-cwq", split="test")
wid = list(w["id"]); dev_ids = set(wid[1000:1300])
dev_ent = {e for x in w.select(range(1000, 1300))["q_entity"] for e in x}
print("example ids:", wid[:2], list(cq["id"][:2]))
seed = lambda s: s.split("_")[0]
cwq_ids = list(cq["id"]); cwq_ent = list(cq["q_entity"])
ov_id = np.array([seed(i) in dev_ids for i in cwq_ids]); ov_ent = np.array([bool(set(e) & dev_ent) for e in cwq_ent])
print(f"CWQ test q whose WebQSP-seed id is in detector-train dev: {ov_id.sum()} / {len(ov_id)}")
print(f"CWQ test q sharing a topic entity with detector-train dev: {ov_ent.sum()} / {len(ov_ent)}")
none, lrn, ora = L(R + "cwqfull_none.jsonl"), L(R + "cwqfull_learned-dev_fpr0.02.jsonl"), L(R + "cwqfull_oracle.jsonl")
pos = {cwq_ids[i]: i for i in range(len(cwq_ids))}
clean_mask = {i: not (ov_id[pos[i]] or ov_ent[pos[i]]) for i in none if i in pos}
for name, keep in (("all", lambda i: True), ("non-overlapping", lambda i: clean_mask.get(i, False)), ("overlapping", lambda i: not clean_mask.get(i, True))):
    sub = [i for i in none if keep(i) and i in lrn and i in ora]
    g = np.array([lrn[i]["f1"] - none[i]["f1"] for i in sub]) * 100
    m = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
    print(f"{name}: n={len(sub)} F1 gain learned-none {g.mean():+.1f} [{np.percentile(m,2.5):+.1f}, {np.percentile(m,97.5):+.1f}]")
print("== GCR scoring check (clean, 400 q)")
g = L("runs/kg/gcr_clean_400.jsonl"); gold = {x["id"]: x["answer"] for x in w}
for k in (1, 3, 10):
    sc = [rog.score(g[i]["prediction"][:k], gold[i]) for i in g]
    print(f"first {k} distinct answer string(s): hit {100*np.mean([s['hit'] for s in sc]):.1f} f1 {100*np.mean([s['f1'] for s in sc]):.1f} precision {100*np.mean([s['precision'] for s in sc]):.1f}")
lens = [len(g[i]["prediction"]) for i in g]; print("mean # predicted answer strings:", np.mean(lens), "median", np.median(lens))
print("sample:", g[next(iter(g))]["prediction"][:4], "| gold:", gold[next(iter(g))])
