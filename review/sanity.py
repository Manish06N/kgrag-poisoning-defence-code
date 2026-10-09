import sys, json, glob, os; sys.path.insert(0,"review")
from lib import *
for D,tag in ((K,"kg"),(S,"kg_strict"),(M,"kg_mg")):
    print("=====",tag)
    for f in sorted(glob.glob(D+"paths_*.jsonl")):
        b=os.path.basename(f)[6:-6]
        if "gcr" in b: continue
        try: rr=lines(f)
        except Exception: print(b,"UNREADABLE"); continue
        npth=sum(len(r["paths"]) for r in rr); p=sum(p["poisoned"] for r in rr for p in r["paths"]); ps=sum(p.get("poisoned_strict",p["poisoned"]) for r in rr for p in r["paths"])
        import datetime
        print(f"  {b:14s} q={len(rr):5d} paths={npth:7d} ({npth/max(len(rr),1):5.1f}/q) poisoned={p:6d} ({100*p/max(npth,1):5.1f}%) strict_field={ps:6d}  {datetime.datetime.fromtimestamp(os.path.getmtime(f)):%m-%d %H:%M}")
