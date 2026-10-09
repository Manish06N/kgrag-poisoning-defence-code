import sys, json, glob, os, csv, collections; sys.path.insert(0,"review")
from lib import *
dev=set(json.loads(l)["id"] for l in open(K+"paths_dev.jsonl",encoding="utf8"))
cwqdev=set(json.loads(l)["id"] for l in open(K+"paths_cwq_dev.jsonl",encoding="utf8"))
out=[]
for D,tag in ((K,"kg"),(S,"kg_strict"),(M,"kg_mg"),(H,"bf16")):
    for f in sorted(glob.glob(D+"defended_*.jsonl")):
        rows=[]; bad=0
        for l in open(f,encoding="utf8"):
            try: rows.append(json.loads(l))
            except Exception: bad+=1
        ids=[r["id"] for r in rows]
        dup=len(ids)-len(set(ids)); ov=len(set(ids)&(dev|cwqdev))
        nopred=sum(1 for r in rows if "prediction" not in r)
        out.append((tag,os.path.basename(f),len(rows),dup,ov,bad,nopred,os.path.getmtime(f)))
w=csv.writer(open("review/run_hygiene.csv","w",newline="")); w.writerow(["dir","file","n","dup_ids","dev_overlap","unparsable_lines","no_prediction","mtime"]); w.writerows(out)
print(len(out),"files")
for o in out:
    if o[3] or o[4] or o[5] or o[6]: print("FLAG",o[:7])
c=collections.Counter((o[0],o[2]) for o in out); print(sorted(c.items()))
