import sys; sys.path.insert(0, "review")
from lib import *
rng=np.random.default_rng(0)
ATT=["","ad_","ev_","fp_","fb_","fs_"]; NM=["paper","adaptive","evasive","profile","bridge","spread"]
def rowfile(stem,a,regime):
    base = S if regime=="S" else K
    return L(base+f"defended_{ATT[a]}{stem}.jsonl")
def wc(stem_runs, subset, none_runs, B=5000, seed=0):
    rng=np.random.default_rng(seed)
    ids=sorted(set.intersection(*[set(r) for r in stem_runs+none_runs])); n=len(ids)
    ix=rng.integers(0,n,(B,n))
    F=np.array([[r[i]["f1"] for i in ids] for r in stem_runs])*100; F0=np.array([[r[i]["f1"] for i in ids] for r in none_runs])*100
    est=F.mean(1).min()-F0.mean(1).min()
    boot=np.min(np.stack([F[a][ix].mean(1) for a in range(len(subset))]),0)-np.min(np.stack([F0[a][ix].mean(1) for a in range(len(subset))]),0)
    lo,hi=2*est-np.percentile(boot,97.5),2*est-np.percentile(boot,2.5)
    pct=(np.percentile(boot,2.5),np.percentile(boot,97.5))
    return n,F.mean(1),F0.mean(1),est,lo,hi,pct
ROWS={"A":"crc-dev_a0.05","B":f"crc-{MIX}_a0.05","C":"gatecrc-dev_a0.05_q0.05","D":f"gatecrc-{MIX}_a0.05_q0.05","E":"learned-dev_fpr0.02","F":f"learned-{MIX}_fpr0.02","G":"gate-dev_fpr0.02_q0.05","H":f"gate-{MIX}_fpr0.02_q0.05"}
def avail(stem,a,reg):
    r=rowfile(stem,a,reg); return r is not None and len(r)>=500
for subset,title in (([0,1,2],"known3"),([3,4,5],"heldout3"),([0,1,2,3,4,5],"six")):
    print("=====",title)
    none=[L(K+f"defended_{ATT[a]}none.jsonl") for a in subset]
    for row,stem in ROWS.items():
        for reg in ("K","S"):
            if all(avail(stem,a,reg) for a in subset):
                runs=[rowfile(stem,a,reg) for a in subset]
                n,F,F0,est,lo,hi,pct=wc(runs,subset,none)
                print(f"{row} [{reg}] n={n} F={np.round(F,1)} F0min={F0.min():.2f} Fmin={F.min():.2f} est={est:+.2f} basicCI=[{lo:+.1f},{hi:+.1f}] pctCI=[{pct[0]:+.1f},{pct[1]:+.1f}] worst={NM[subset[int(F.argmin())]]}")
