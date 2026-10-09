import sys; sys.path.insert(0,"review")
from lib import *
from datasets import load_dataset
w=load_dataset("rmanluo/RoG-webqsp",split="test"); cq=load_dataset("rmanluo/RoG-cwq",split="test")
wid=list(w["id"]); dev_ids=set(wid[1000:1300])
dev_ent={e for x in w.select(range(1000,1300))["q_entity"] for e in x}
seed=lambda s:s.split("_")[0]
cid=list(cq["id"]); cent=list(cq["q_entity"])
ov_id=np.array([seed(i) in dev_ids for i in cid]); ov_ent=np.array([bool(set(e)&dev_ent) for e in cent])
print("all 3531: id-overlap",ov_id.sum(),"entity-overlap",ov_ent.sum())
none=L(K+"defended_cwqfull_none.jsonl"); pos={cid[i]:i for i in range(len(cid))}
either={i for i in none if ov_id[pos[i]] or ov_ent[pos[i]]}
print("in the 3231 evaluated: overlapping(either)",len(either),"non-overlap",len(none)-len(either))
for nm,p in (("crc strict",S+"defended_cwqfull_crc-dev_a0.05.jsonl"),("crc loose",K+"defended_cwqfull_crc-dev_a0.05.jsonl"),("learned2",K+"defended_cwqfull_learned-dev_fpr0.02.jsonl")):
    r=L(p); non=[i for i in none if i not in either]
    print(nm,"all",fmt(paired(r,none)),"non-overlap",fmt(paired(r,none,ids=non)),"overlap",fmt(paired(r,none,ids=sorted(either))))
