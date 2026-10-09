import os, sys, json, warnings; os.environ["KG_RUNS"]="runs/kg_strict"; sys.path.insert(0,"."); warnings.filterwarnings("ignore")
import numpy as np
from kgrag import gcr_defend, rog, pathdata, defend, learned
from sklearn.metrics import roc_auc_score
ds={ex["id"]:ex for ex in rog.load_data("data/poisoned_webqsp","test")}
rows=[json.loads(l) for l in open("runs/kg/gcr3_test_attacked.jsonl",encoding="utf8")]
print("n questions",len(rows))
recs=[]; yl=[]
for r in rows:
    ex=ds[r["id"]]; rec=gcr_defend.beam_record(ex,r["beams"])
    inj=pathdata.injected_set(ex.get("poison_triples") or [])
    for p in rec["paths"]: yl.append(pathdata.label_path([tuple(t) for t in p["triples"]],inj))
    recs.append(rec)
ys=np.array([p["poisoned"] for r in recs for p in r["paths"]]).astype(int); yl=np.array(yl).astype(int)
print("paths",len(ys),"per q %.2f"%(len(ys)/len(recs)),"strict poisoned %.2f%% loose %.2f%%"%(100*ys.mean(),100*yl.mean()))
qs=sum(any(p["poisoned"] for p in r["paths"]) for r in recs); print("q with strict poisoned",qs)
def fit(splits):
    rr=[defend.load_paths(s) for s in splits]
    X=np.vstack([learned.features(r) for r in rr]); y=np.concatenate([defend.flat(r,"poisoned").astype(int) for r in rr]); return learned._model().fit(X,y)
mm=fit(["dev","K1_dev","K8_dev","ad_dev"]); s=mm.predict_proba(learned.features(recs))[:,1]
print("AUC strict labels %.3f loose labels %.3f"%(roc_auc_score(ys,s),roc_auc_score(yl,s)))
n5=500
ys5=np.array([p["poisoned"] for r in recs[:n5] for p in r["paths"]]).astype(int); s5=mm.predict_proba(learned.features(recs[:n5]))[:,1]
print("first 500: poisoned %.2f%% AUC %.3f"%(100*ys5.mean(),roc_auc_score(ys5,s5)))
dv=defend.load_paths("gcrdev"); yd=defend.flat(dv,"poisoned").astype(int); print("gcr dev: q",len(dv),"paths/q %.2f"%(len(yd)/len(dv)),"poisoned %.2f%% AUC %.3f"%(100*yd.mean(),roc_auc_score(yd,mm.predict_proba(learned.features(dv))[:,1])))
# per-question mean AUC and questions-with-poison restricted
print("per-question mean AUC", defend.per_question_auc(recs,ys,s))
aff=[i for i,r in enumerate(recs) if any(p["poisoned"] for p in r["paths"])]
print("poisoned share among affected q's paths", np.mean([p["poisoned"] for i in aff for p in recs[i]["paths"]]))
