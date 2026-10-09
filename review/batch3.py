import sys; sys.path.insert(0,"review")
from lib import *
ATT=["","ad_","ev_","fp_","fb_","fs_"]; NM=["paper","adaptive","evasive","profile","bridge","spread"]
print("== S11 TransE feature: with - without (gatecrc mixed)")
stem=f"gatecrc-{MIX}_a0.05_q0.05"
for a,nm in zip(range(7),NM+["clean"]):
    pre=ATT[a] if a<6 else "clean_"
    w=L(K+f"defended_{pre}{stem}_kge.jsonl"); wo=L(K+f"defended_{pre}{stem}.jsonl")
    if w is None or wo is None: print(nm,"missing",w is None,wo is None); continue
    print(f"{nm:9s} with {mean(w):.1f} without {mean(wo):.1f}  with-without {fmt(paired(w,wo))}  without-with {fmt(paired(wo,w))}")
stem=f"crc-{MIX}_a0.05"
for a,nm in zip(range(7),NM+["clean"]):
    pre=ATT[a] if a<6 else "clean_"
    w=L(K+f"defended_{pre}{stem}_kge.jsonl"); wo=L(K+f"defended_{pre}{stem}.jsonl")
    if w is None or wo is None: continue
    print("always-on",nm, "with-without",fmt(paired(w,wo)))
print("== S12 shares of oracle gain recovered by D (strict), per attack; strict oracle")
for a,nm in enumerate(NM):
    none=L(K+f"defended_{ATT[a]}none.jsonl"); D=L(S+f"defended_{ATT[a]}{f'gatecrc-{MIX}_a0.05_q0.05'}.jsonl"); C=L(S+f"defended_{ATT[a]}gatecrc-dev_a0.05_q0.05.jsonl"); O=L(S+f"defended_{ATT[a]}oracle_fpr0.02.jsonl")
    Ol=L(K+f"defended_{ATT[a]}oracle.jsonl")
    def sh(r,o):
        ids=sorted(set(r)&set(o)&set(none)); aa=np.array([[none[i]['f1'],r[i]['f1'],o[i]['f1']] for i in ids])*100
        ix=np.random.default_rng(0).integers(0,len(ids),(2000,len(ids))); m=aa[ix].mean(1); s=(m[:,1]-m[:,0])/(m[:,2]-m[:,0]); pt=(aa[:,1].mean()-aa[:,0].mean())/(aa[:,2].mean()-aa[:,0].mean())
        return f"{100*pt:.1f} [{100*np.percentile(s,2.5):.0f},{100*np.percentile(s,97.5):.0f}]"
    print(f"{nm:9s} none {mean(none):.1f} oracleS {mean(O):.2f} oracleLoose {(mean(Ol) if Ol else 0):.2f} D {mean(D):.1f} shareD {sh(D,O)} shareC {sh(C,O)}")
print("== S13 b8 hit")
for f in ["defended_b8_none","defended_b8_learned-dev_fpr0.02","defended_b8_crc-dev_a0.05","defended_b8_oracle"]:
    r=L(K+f+".jsonl"); print(f, len(r), "f1 %.1f hit %.1f a_prec %.1f kept %.0f%%"%(mean(r),mean(r,'hit'),mean(r,'a_precision'),100*np.mean([x['n_kept']/max(x['n_paths'],1) for x in r.values()])))
l=L(K+"defended_b8_learned-dev_fpr0.02.jsonl"); c=L(K+"defended_b8_crc-dev_a0.05.jsonl"); print("hit identical per q:", sum(l[i]['hit']==c[i]['hit'] for i in l), "of", len(l), "hit sum", sum(x['hit'] for x in l.values()), sum(x['hit'] for x in c.values()))
print("== S15 seeds (learned-dev fpr0.02 vs none, 500 q; seed0 = main)")
for pre in ["","s1_","s2_","s3_","s4_"]:
    n=L(K+f"defended_{pre}none.jsonl"); l=L(K+f"defended_{pre}learned-dev_fpr0.02.jsonl"); print(pre or "s0", len(n), round(mean(n),1), round(mean(l),1), fmt(paired(l,n)))
