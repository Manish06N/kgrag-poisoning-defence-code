import os, sys, warnings; os.environ["KG_RUNS"]=sys.argv[1]; sys.path.insert(0,"."); warnings.filterwarnings("ignore")
import numpy as np
from kgrag import defend, learned, gate
from sklearn.metrics import roc_auc_score
MIX=["dev","K1_dev","K8_dev","ad_dev"]
def fit(splits):
    recs=[defend.load_paths(s) for s in splits]
    X=np.vstack([learned.features(r) for r in recs]); y=np.concatenate([defend.flat(r,"poisoned").astype(int) for r in recs])
    return learned._model().fit(X,y)
mp=fit(["dev"]); mm=fit(MIX)
print("regime dir", sys.argv[1])
for ts,nm in (("test","paper"),("ad_test","adaptive"),("ev_test","evasive"),("fp_test","profile"),("fb_test","bridge"),("fs_test","spread")):
    recs=defend.load_paths(ts); y=defend.flat(recs,"poisoned").astype(int); X=learned.features(recs)
    sp=mp.predict_proba(X)[:,1]; sm=mm.predict_proba(X)[:,1]
    print(f"  {nm:9s} pooled paper {roc_auc_score(y,sp):.3f} mixed {roc_auc_score(y,sm):.3f} | per-question mean paper {defend.per_question_auc(recs,y,sp):.3f} mixed {defend.per_question_auc(recs,y,sm):.3f}")
recs=defend.load_paths("test"); y=defend.flat(recs,"poisoned").astype(int)
for sc in ("rels",):
    s=np.load(os.path.join(os.environ["KG_RUNS"],f"scores_{sc}_test.npy")); print("rels paper pooled %.3f perq %.3f"%(roc_auc_score(y,s),defend.per_question_auc(recs,y,s)))
    r2=defend.load_paths("ad_test"); y2=defend.flat(r2,"poisoned").astype(int); s2=np.load(os.path.join(os.environ["KG_RUNS"],f"scores_{sc}_ad_test.npy")); print("rels adaptive pooled %.3f perq %.3f"%(roc_auc_score(y2,s2),defend.per_question_auc(r2,y2,s2)))
s=np.load(os.path.join(os.environ["KG_RUNS"],"scores_perplexity_test.npy")); print("perplexity pooled %.3f perq %.3f"%(roc_auc_score(y,s),defend.per_question_auc(recs,y,s)))
g=defend.load_paths("gcrtest" if os.path.exists(os.path.join(os.environ["KG_RUNS"],"paths_gcrtest.jsonl")) else "gcr_test"); yg=defend.flat(g,"poisoned").astype(int); print("gcr: poisoned %.2f%% AUC mixed %.3f paper %.3f"%(100*yg.mean(),roc_auc_score(yg,mm.predict_proba(learned.features(g))[:,1]),roc_auc_score(yg,mp.predict_proba(learned.features(g))[:,1])))
# gate-model question-level AUC
for nm,sp in (("paper",["dev"]),("mixed",MIX)):
    model,_=gate.fit(sp,"clean_dev",100,0.05,3)
    def qs(ts):
        r=defend.load_paths(ts); sz=gate._sizes(r); return gate.question_scores(sz,model.predict_proba(learned.features(r))[:,1])[sz>0]
    c=qs("clean_test")
    print(" gate-model",nm,[ (an,round(roc_auc_score(np.r_[np.zeros(len(c)),np.ones(len(a))],np.r_[c,a]),3)) for an,a in (("paper",qs("test")),("adaptive",qs("ad_test")),("evasive",qs("ev_test")))])
m=learned.removal_mask(["dev"],"test",0.10); yy=defend.flat(recs,"poisoned").astype(int); gold=defend.flat(recs,"gold_hit").astype(bool); qi=defend.qindex(recs)
had=lost=0
for i in range(len(recs)):
    sel=(qi==i)&gold&(yy==0)
    if sel.any(): had+=1; lost+=int(m[sel].all())
print(f"10% FPR lost-all {lost}/{had} = {100*lost/had:.1f}%")
