import sys; sys.path.insert(0, "review")
from lib import *
n=L(H+"defended_fullev_none.jsonl"); m=L(H+"defended_fullev_learned-"+MIX+"_fpr0.02.jsonl"); c=L(H+"defended_fullev_crc-"+MIX+"_a0.05.jsonl")
n4=L(K+"defended_ev_none.jsonl"); m4=L(K+"defended_ev_learned-"+MIX+"_fpr0.02.jsonl")
first=sorted(n4); rest=[i for i in n if i not in n4]
print("4-bit 500: none",round(mean(n4),2),"mixed",round(mean(m4),2),fmt(paired(m4,n4)))
print("bf16 first500: none",round(mean(n,'f1',first),2),"mixed",round(mean(m,'f1',first),2), fmt(paired(m,n,ids=first)))
print("bf16 other828: none",round(mean(n,'f1',rest),2),"mixed",round(mean(m,'f1',rest),2), fmt(paired(m,n,ids=rest)))
print("bf16 vs 4bit none on first500", round(mean(n,'f1',first),2), round(mean(n4),2))
print("bf16 conformal first500", fmt(paired(c,n,ids=first)), "rest", fmt(paired(c,n,ids=rest)))
