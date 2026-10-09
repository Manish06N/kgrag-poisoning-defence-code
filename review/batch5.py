import os, sys, warnings; os.environ["KG_RUNS"]=sys.argv[1]; sys.path.insert(0,"."); warnings.filterwarnings("ignore")
import numpy as np
from kgrag import defend, learned, gate
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
MIX=["dev","K1_dev","K8_dev","ad_dev"]
def fit(splits):
    recs=[defend.load_paths(s) for s in splits]
    X=np.vstack([learned.features(r) for r in recs]); y=np.concatenate([defend.flat(r,"poisoned").astype(int) for r in recs])
    return learned._model().fit(X,y)
mp=fit(["dev"]); mm=fit(MIX)
print("== path-level AUC (pooled, strict labels) paper-det / mixed-det")
for ts,nm in (("test","paper"),("ad_test","adaptive"),("ev_test","evasive"),("fp_test","profile"),("fb_test","bridge"),("fs_test","spread"),("b8_test","b8")):
    recs=defend.load_paths(ts); y=defend.flat(recs,"poisoned").astype(int); X=learned.features(recs)
    print(f"  {nm:9s} n_paths={len(y)} poisoned={y.mean():.3f} paper-det AUC {roc_auc_score(y,mp.predict_proba(X)[:,1]):.3f} mixed-det AUC {roc_auc_score(y,mm.predict_proba(X)[:,1]):.3f}")
print("== GCR beams: RoG-trained mixed detector AUC")
g=defend.load_paths("gcrtest"); y=defend.flat(g,"poisoned").astype(int); print("  gcr test q",len(g),"paths",len(y),"per q %.2f"%(len(y)/len(g)),"poisoned %.2f%%"%(100*y.mean()),"q with >=1 poisoned",sum(any(p['poisoned'] for p in r['paths']) for r in g),"AUC %.3f"%roc_auc_score(y,mm.predict_proba(learned.features(g))[:,1]))
print("== question-level attack-detection AUC (attacked test vs clean_test): gate scores")
def qs(model, ts):
    recs=defend.load_paths(ts); sz=gate._sizes(recs); return gate.question_scores(sz, model.predict_proba(learned.features(recs))[:,1])[sz>0]
for nm,model in (("paper-det",mp),("mixed-det",mm)):
    c=qs(model,"clean_test")
    for ts,an in (("test","paper"),("ad_test","adaptive"),("ev_test","evasive")):
        a=qs(model,ts); print(f"  {nm} {an}: AUC {roc_auc_score(np.r_[np.zeros(len(c)),np.ones(len(a))],np.r_[c,a]):.3f}")
print("== rels / perplexity / verify / kge path-level AUC")
for sc,sp,rs in (("rels","test","test"),("rels","ad_test","ad_test"),("perplexity","test","test"),("verify","test","test"),("kge","test","test"),("kge","ad_test","ad_test"),("kge","ev_test","ev_test")):
    f=os.path.join(os.environ["KG_RUNS"],f"scores_{sc}_{sp}.npy"); 
    if not os.path.exists(f): f=os.path.join("runs/kg",f"scores_{sc}_{sp}.npy")
    s=np.load(f); recs=defend.load_paths(rs); y=defend.flat(recs,"poisoned").astype(int)
    if sc=="kge": s=s
    print(f"  {sc:11s} {sp:8s} AUC {roc_auc_score(y,s):.3f} (n={len(y)})")
print("== 10% FPR: share of answerable questions losing all clean gold evidence (paper detector, dev threshold; test split)")
m=learned.removal_mask(["dev"],"test",0.10); recs=defend.load_paths("test"); y=defend.flat(recs,"poisoned").astype(int); gold=defend.flat(recs,"gold_hit").astype(bool); qi=defend.qindex(recs)
had=lost=0
for i in range(len(recs)):
    sel=(qi==i)&gold&(y==0)
    if sel.any(): had+=1; lost+= int(m[sel].all())
print(f"  answerable {had}, lost all {lost} = {100*lost/had:.1f}%")
print("== permutation importance (paper detector, test paths, AUC drop)")
recs=defend.load_paths("test"); X=learned.features(recs); y=defend.flat(recs,"poisoned").astype(int)
names=[f"log1p_{b}" for b in learned.BASE]+[f"z_{b}" for b in learned.BASE]+["len","ends/n","ends","relcnt/n","n"]
pi=permutation_importance(mp,X,y,scoring="roc_auc",n_repeats=5,random_state=0)
for j in np.argsort(-pi.importances_mean)[:8]: print(f"  {names[j]:14s} {pi.importances_mean[j]:.3f}")
print("n features",X.shape[1])
