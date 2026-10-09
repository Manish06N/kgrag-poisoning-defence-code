import os, sys, warnings; os.environ["KG_RUNS"]=sys.argv[1]; sys.path.insert(0,".")
warnings.filterwarnings("ignore")
import numpy as np
from kgrag import defend, gate, learned, conformal
MIX=["dev","K1_dev","K8_dev","ad_dev"]
def flagged(splits, test_split, qf=0.05):
    model, thr = gate.fit(splits, "clean_dev", 100, qf, 3)
    recs = defend.load_paths(test_split); sizes = gate._sizes(recs)
    qs = gate.question_scores(sizes, model.predict_proba(learned.features(recs))[:,1], 3)
    # fraction of dev questions flagged (held-out clean dev)
    held = defend.load_paths("clean_dev")[100:]; hs = gate._sizes(held)
    hq = gate.question_scores(hs, model.predict_proba(learned.features(held))[:,1], 3)[hs>0]
    return thr, float((qs>=thr)[sizes>0].mean()), int((qs>=thr).sum()), len(recs), float((hq>=thr).mean()), len(hq), int((sizes==0).sum()), int((sizes<3).sum())
for nm, splits in (("C paper det",["dev"]),("D mixed det",MIX)):
    for ts in ("clean_test","test","ad_test","ev_test","fp_test","fb_test","fs_test"):
        try: r=flagged(splits, ts)
        except Exception as e: print(nm,ts,"ERR",e); continue
        print(f"{nm:12s} {ts:10s} thr={r[0]:.4f} flagged_test={r[1]:.3f} ({r[2]}/{r[3]}) flagged_heldout_cleandev={r[4]:.3f} (n={r[5]}) zero-path q={r[6]} <3 paths={r[7]}")
