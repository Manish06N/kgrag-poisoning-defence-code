import sys, json, glob, os; sys.path.insert(0,"review")
from lib import *
def ids(p):
    try: return [json.loads(l)["id"] for l in open(p,encoding="utf8")]
    except Exception as e: return []
for D,name in ((K,"kg (loose)"),(S,"kg_strict"),(M,"kg_mg")):
    print("=====",name)
    dev={os.path.basename(f)[6:-6]:ids(f) for f in glob.glob(D+"paths_*dev*.jsonl") if "gcr" not in f}
    test={os.path.basename(f)[6:-6]:ids(f) for f in glob.glob(D+"paths_*test*.jsonl") if "gcr" not in f}
    alld=set(); 
    for k,v in dev.items(): alld|=set(v)
    for k,v in sorted(dev.items()): print(f"  dev {k:12s} n={len(v)} unique={len(set(v))} first={v[0]} last={v[-1]}")
    # are all dev splits the same question set?
    base=set(dev.get("dev",[]))
    print("  all dev splits same id set as 'dev':", {k:(set(v)==base) for k,v in dev.items() if k.endswith("dev") and "cwq" not in k})
    for k,v in sorted(test.items()):
        inter=set(v)&alld
        print(f"  test {k:12s} n={len(v)} unique={len(set(v))} overlap_with_any_dev={len(inter)}")
