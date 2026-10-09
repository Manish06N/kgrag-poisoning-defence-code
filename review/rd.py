import os, sys, warnings; os.environ["KG_RUNS"]=sys.argv[1]; sys.path.insert(0,"."); warnings.filterwarnings("ignore")
import numpy as np
from kgrag import defend, ragdefender
from sentence_transformers import SentenceTransformer
ragdefender._ENC=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
recs=defend.load_paths("test")[:100]
for var in ("conc","clust","oraclen"):
    m=ragdefender.removal_mask(recs,var); y=defend.flat(recs,"poisoned").astype(bool)
    print(var, "first100: removed poisoned %.1f%% clean %.1f%% (n paths %d)"%(100*m[y].mean(),100*m[~y].mean(),len(y)))
