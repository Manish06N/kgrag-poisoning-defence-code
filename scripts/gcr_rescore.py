import json, numpy as np
from datasets import load_from_disk
from kgrag import rog, attack
L = lambda p: {(r := json.loads(l))["id"]: r for l in open(p, encoding="utf8")}
c, p = L("runs/kg/gcr_clean_400.jsonl"), L("runs/kg/gcr_poisoned_400.jsonl")
ds = load_from_disk("data/poisoned_webqsp"); info = {x["id"]: (x["answer"], x["adv_answers"]) for x in ds}
rng = np.random.default_rng(0)
ids = sorted(set(c) & set(p))
for k in (1, 3, 10):
    f = lambda r, i: rog.score(r[i]["prediction"][:k], info[i][0])["f1"]
    d = np.array([f(p, i) - f(c, i) for i in ids]) * 100
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    a1 = np.mean([attack.attack_metrics(p[i]["prediction"][:k], info[i][1])["a_hit1"] for i in ids]) * 100
    print(f"first {k}: clean F1 {100*np.mean([f(c,i) for i in ids]):.1f} attacked F1 {100*np.mean([f(p,i) for i in ids]):.1f} diff {d.mean():+.1f} [{np.percentile(m,2.5):+.1f}, {np.percentile(m,97.5):+.1f}] planted-top1 {a1:.1f}%")
