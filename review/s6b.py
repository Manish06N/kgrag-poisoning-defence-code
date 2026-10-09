import os, sys, warnings; os.environ["KG_RUNS"]="runs/kg"; sys.path.insert(0,"."); warnings.filterwarnings("ignore")
import numpy as np
from kgrag import defend, gate, learned
model, thr = gate.fit(["dev"], "clean_dev", 100, 0.05, 3)
for ts in ("cwqclean_test","cwqfull_test","clean_test","test"):
    recs=defend.load_paths(ts); sz=gate._sizes(recs); qs=gate.question_scores(sz, model.predict_proba(learned.features(recs))[:,1], 3)
    print(ts,"gate C (original labels) flagged %.1f%% of %d questions (thr %.3f)"%(100*(qs>=thr)[sz>0].mean(),(sz>0).sum(),thr))
