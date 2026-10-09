import sys; sys.path.insert(0,"review")
from lib import *
ATT=["","ad_","ev_","fp_","fb_","fs_"]
def runs(stem,sub): return [L(S+f"defended_{ATT[a]}{stem}.jsonl") for a in sub]
Cs="gatecrc-dev_a0.05_q0.05"; Ds=f"gatecrc-{MIX}_a0.05_q0.05"
for sub,nm in (([0,1,2],"known3"),([3,4,5],"held-out"),([0,1,2,3,4,5],"six")):
    C=runs(Cs,sub); D=runs(Ds,sub); ids=sorted(set.intersection(*[set(r) for r in C+D]))
    FC=np.array([[r[i]["f1"] for i in ids] for r in C])*100; FD=np.array([[r[i]["f1"] for i in ids] for r in D])*100
    est=FD.mean(1).min()-FC.mean(1).min(); ix=np.random.default_rng(0).integers(0,len(ids),(5000,len(ids)))
    b=np.min(np.stack([FD[a][ix].mean(1) for a in range(len(sub))]),0)-np.min(np.stack([FC[a][ix].mean(1) for a in range(len(sub))]),0)
    print(nm,"D-C worst est %+.2f basic [%+.1f,%+.1f] pct [%+.1f,%+.1f]"%(est,2*est-np.percentile(b,97.5),2*est-np.percentile(b,2.5),np.percentile(b,2.5),np.percentile(b,97.5)))
# D clean cost CI
cn=L(K+"defended_clean_none.jsonl"); print("D clean",fmt(paired(L(S+f"defended_clean_{Ds}.jsonl"),cn)),"C clean",fmt(paired(L(S+f"defended_clean_{Cs}.jsonl"),cn)))
# S8 evasive planted
n=L(K+"defended_ev_none.jsonl"); d=L(S+f"defended_ev_{Ds}.jsonl"); c=L(S+f"defended_ev_{Cs}.jsonl")
print("evasive planted: none %.1f D %.1f C %.1f ; hits1 none %.1f D %.1f"%(mean(n,'a_precision'),mean(d,'a_precision'),mean(c,'a_precision'),mean(n,'a_hit1'),mean(d,'a_hit1')))
# CWQ gate flagged share on clean + attacked is skipped (needs cwq splits)
# any-planted share
for nm,r in (("500 none",L(K+"defended_none.jsonl")),("1328 none",L(K+"defended_full_none.jsonl"))): print(nm,"any planted in output %.1f%%"%(100*np.mean([x['a_precision']>0 for x in r.values()])))
# learned@10% on paper attack
l10=L(K+"defended_learned-dev.jsonl"); print("learned@10% paper attack",round(mean(l10),1),fmt(paired(l10,L(K+"defended_none.jsonl"))), "rels",round(mean(L(K+'defended_rels.jsonl')),1))
