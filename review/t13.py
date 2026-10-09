import sys; sys.path.insert(0, "review")
from lib import *
def g(path): return L(path)
rows=[("paper none","defended_full_none"),("paper oracle","defended_full_oracle"),("paper crc","defended_full_crc-dev_a0.05"),("paper learned2","defended_full_learned-dev_fpr0.02"),("paper rels","defended_full_rels"),
("ad none","defended_fullad_none"),("ad oracle","defended_fullad_oracle"),("ad learned mixed","defended_fullad_learned-"+MIX+"_fpr0.02"),("ad learned paper","defended_fullad_learned-dev_fpr0.02"),
("ad crc mixed (bf16 only?)","defended_fullad_crc-"+MIX+"_a0.05")]
print("== Table 13 recompute (4-bit=runs/kg loose, bf16=HPC)")
for nm,f in rows:
    a=L(K+f+".jsonl"); b=L(H+f+".jsonl")
    if a is None or b is None: print(nm,"missing",a is None,b is None); continue
    d=paired(b,a); print(f"{nm:28s} 4bit {mean(a):.2f} (n={len(a)}) bf16 {mean(b):.2f} (n={len(b)}) diff {fmt(d)}")
# strict versions for comparison
for nm,f in [("paper oracle strict","defended_full_oracle_fpr0.02"),("paper crc strict","defended_full_crc-dev_a0.05")]:
    a=L(S+f+".jsonl"); print(nm, round(mean(a),2))
print("== evasive 1328 (bf16)")
for f in ["defended_fullev_none","defended_fullev_oracle","defended_fullev_learned-dev_fpr0.02","defended_fullev_learned-"+MIX+"_fpr0.02","defended_fullev_learned-"+MIX+"+ev_dev_fpr0.02","defended_fullev_crc-"+MIX+"_a0.05","defended_fullev_crc-"+MIX+"+ev_dev_a0.05"]:
    b=L(H+f+".jsonl"); print(f, None if b is None else (len(b), round(mean(b),2)))
n=L(H+"defended_fullev_none.jsonl")
for f in ["defended_fullev_learned-"+MIX+"_fpr0.02","defended_fullev_crc-"+MIX+"_a0.05","defended_fullev_learned-dev_fpr0.02"]:
    print("gain vs none", f[-30:], fmt(paired(L(H+f+".jsonl"),n)))
print("crc gain bf16 paper", fmt(paired(L(H+"defended_full_crc-dev_a0.05.jsonl"),L(H+"defended_full_none.jsonl"))), "4-bit loose", fmt(paired(L(K+"defended_full_crc-dev_a0.05.jsonl"),L(K+"defended_full_none.jsonl"))))
print("== S14 learned@2 / rels on 500 vs 1328")
for f in ["defended_learned-dev_fpr0.02","defended_rels","defended_none","defended_full_learned-dev_fpr0.02","defended_full_rels","defended_full_none"]:
    r=L(K+f+".jsonl"); print(f, len(r), round(mean(r),3))
# are the 500 a subset of the 1328, and identical rows?
a=L(K+"defended_rels.jsonl"); b=L(K+"defended_full_rels.jsonl")
common=[i for i in a if i in b]; print("common ids",len(common), "identical f1 on common", sum(a[i]["f1"]==b[i]["f1"] for i in common), "mean 500 on common",round(mean(b,'f1',common),2), "mean rest", round(mean(b,'f1',[i for i in b if i not in a]),2))
a=L(K+"defended_learned-dev_fpr0.02.jsonl"); b=L(K+"defended_full_learned-dev_fpr0.02.jsonl")
common=[i for i in a if i in b]; print("learned: common",len(common),"identical",sum(a[i]["f1"]==b[i]["f1"] for i in common),"mean on common (1328 file)",round(mean(b,'f1',common),2),"mean other 828",round(mean(b,'f1',[i for i in b if i not in a]),2))
