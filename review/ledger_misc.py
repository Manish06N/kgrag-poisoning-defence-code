import sys, pickle
sys.path.insert(0, "review")
from ledger1 import *

ATT = ["", "ad_", "ev_", "fp_", "fb_", "fs_"]
Ds = f"gatecrc-{MIX}_a0.05_q0.05"
Cs = "gatecrc-dev_a0.05_q0.05"
none = [L(K + f"defended_{ATT[a]}none.jsonl") for a in range(6)]


def share(r, o, nn):
    ids = sorted(set(r) & set(o) & set(nn))
    a = np.array([[nn[i]['f1'], r[i]['f1'], o[i]['f1']] for i in ids]) * 100
    return 100 * (a[:, 1].mean() - a[:, 0].mean()) / (a[:, 2].mean() - a[:, 0].mean())


cl500 = L(K + "defended_clean_none.jsonl")


def em_rel():
    sys.path.insert(0, ROOT)
    from kgrag import rog
    gold = {r["id"]: r["gold"] for r in lines(K + "paths_test.jsonl")}

    def em(r):
        P = {rog.normalize(p) for p in r["prediction"] if p}
        G = {rog.normalize(x) for x in gold[r["id"]]}
        return float(bool(P) and P == G)
    c = 100 * np.mean([em(r) for r in cl500.values()])
    a = 100 * np.mean([em(r) for r in none[0].values()])
    return 100 * (c - a) / c


chk("ABS.em", "abstract", "exact match falls 58%", 58, em_rel(), "", 500, "T1.EM entries", "relative EM drop", tol=0.5)
chk("ABS.share.web", "abstract", "recovers 73% of the gain of a perfect filter (WebQSP)", 73, 73.178, "strict/strict", 1328, "T1.share.crc.web", "configuration A (always-on conformal, paper detector), NOT the recommended D", tol=0.5)
chk("ABS.share.cwq", "abstract", "60% (CWQ)", 60, 60.23, "strict/strict", 3231, "T1.share.crc.cwq", "configuration A", tol=0.5)
chk("ABS.rd", "abstract", "RAGDefender lowers F1 by 12-19 points", 12, 11.9, "original", 500, "T2", "range 11.9 to 18.8 rounds to 12-19", tol=0.5)
dcl = paired(L(S + f"defended_clean_{Ds}.jsonl"), cl500)
chk("ABS.Dclean", "abstract", "gate clean cost -0.5 [-1.5, +0.4]", -0.5, dcl[0], "strict", 500, "kg_strict/defended_clean_gatecrc-MIX", f"CI recomputed [{dcl[1]:+.1f},{dcl[2]:+.1f}] vs paper [-1.5,+0.4]")
ATTN = ["paper", "adaptive", "evasive", "profile", "bridge", "spread"]
Fd = [mean(L(S + f"defended_{ATT[a]}{Ds}.jsonl")) for a in range(6)]
chk("ABS.D.worst", "abstract", "worst-case F1 50.8", 50.8, min(Fd), "strict", 500, "min over six attacks of D", "")
chk("ABS.none.worst", "abstract", "43.4", 43.4, min(mean(n) for n in none), "", 500, "")
sh = [share(L(S + f"defended_{ATT[a]}{Ds}.jsonl"), L(S + f"defended_{ATT[a]}oracle_fpr0.02.jsonl"), none[a]) for a in range(6)]
chk("ABS.unseen.lo", "abstract", "unseen families 33%", 33, min(sh[3:]), "strict", 500, "")
chk("ABS.unseen.hi", "abstract", "unseen families 64%", 64, max(sh[3:]), "strict", 500, "")
chk("ABS.known.lo", "abstract", "76% (paper)", 76, sh[0], "strict", 500, "")
chk("ABS.known.hi", "abstract", "82% (adaptive)", 82, sh[1], "strict", 500, "")
chk("FIG7.evasive.share", "fig 7", "evasive share -5%", -5, sh[2], "strict", 500, "", tol=0.5)
chk("ABS.bf16.max", "abstract", "bf16 within 1.1 F1", 1.1, 1.1, "original", 1328, "T13 diffs (max 1.1: adaptive no defence)", "applies to the nine rows of Table 13 only; the evasive bf16 rows have no 4-bit 1,328 counterpart")
nc = L(K + "defended_cwqfull_none.jsonl")
g = paired(L(K + "defended_cwqfull_gatecrc-dev_a0.05_q0.05.jsonl"), nc)
chk("611.cwq.gate", "6.11", "CWQ gate +7.7 [+6.4, +9.1]", 7.7, g[0], "original", 3231, "defended_cwqfull_gatecrc-dev", f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
g = paired(L(K + "defended_cwqfull_crc-dev_a0.05.jsonl"), nc)
chk("611.cwq.crc", "6.11", "CWQ always-on conformal +8.0 [+6.6, +9.4]", 8.0, g[0], "original", 3231, "", f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
g = paired(L(K + "defended_cwqfull_oracle.jsonl"), nc)
chk("611.cwq.oracle", "6.11", "CWQ oracle +11.4 [+9.8, +13.0]", 11.4, g[0], "original", 3231, "", f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
chk("611.cwq.planted", "6.11", "17.7% planted", 17.7, mean(L(K + "defended_cwqfull_gatecrc-dev_a0.05_q0.05.jsonl"), "a_precision"), "original", 3231, "")
cc = L(K + "defended_cwqclean_none.jsonl")
gc = L(K + "defended_cwqclean_gatecrc-dev_a0.05_q0.05.jsonl")
g = paired(gc, cc)
chk("611.cwq.clean", "6.11", "CWQ clean cost -1.3 [-1.8, -0.8]", -1.3, g[0], "original", 3231, "", f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
chk("611.cwq.clean48.8", "6.11", "48.8 clean through path pipeline", 48.8, mean(cc), "", 3231, "")
chk("611.cwq.clean47.5", "6.11", "47.5 with gate", 47.5, mean(gc), "", 3231, "")
stem = f"gatecrc-{MIX}_a0.05_q0.05"
for a, nm, pv in ((0, "paper", 0.4), (2, "evasive", 0.1), (3, "profile", 1.8), (4, "bridge", 4.0), (5, "spread", 0.9)):
    w = paired(L(K + f"defended_{ATT[a]}{stem}_kge.jsonl"), L(K + f"defended_{ATT[a]}{stem}.jsonl"))
    chk("611.kge." + nm, "6.11", f"TransE feature, with minus without, {nm}", pv, w[0], "original", 500, "kge vs non-kge gatecrc MIX", f"CI [{w[1]:+.1f},{w[2]:+.1f}]")
w = paired(L(K + f"defended_clean_{stem}_kge.jsonl"), L(K + f"defended_clean_{stem}.jsonl"))
chk("611.kge.clean", "6.11", "TransE feature clean -0.1", -0.1, w[0], "original", 500, "")
b8 = {k: L(K + f"defended_b8_{k}.jsonl") for k in ("none", "learned-dev_fpr0.02", "crc-dev_a0.05", "oracle")}
chk("69.share.crc", "6.9", "conformal 80.0% of oracle gain", 80.0, share(b8["crc-dev_a0.05"], b8["oracle"], b8["none"]), "original", 500, "", "CI recomputed [69.8, 91.6] vs paper [69.9, 90.7]")
chk("69.share.l2", "6.9", "2% 70.9% of oracle gain", 70.9, share(b8["learned-dev_fpr0.02"], b8["oracle"], b8["none"]), "original", 500, "")
Bs = f"crc-{MIX}_a0.05"
for a, v in ((3, 45), (4, 57), (5, 71)):
    chk(f"610.B.share.{ATTN[a]}", "6.10", f"always-on B share {ATTN[a]}", v, share(L(K + f"defended_{ATT[a]}{Bs}.jsonl"), L(S + f"defended_{ATT[a]}oracle_fpr0.02.jsonl"), none[a]), "MIXED: original B / strict oracle", 500, "", tol=0.5)
T6 = {"A": ("crc-dev_a0.05", "K"), "C": (Cs, "S"), "D": (Ds, "S")}
P6 = {"A": (60.7, 53.4, 48.8, 61.6), "C": (61.6, 49.1, 50.7, 67.0), "D": (60.8, 61.8, 50.8, 67.2)}
for row, (stem_, reg) in T6.items():
    base = S if reg == "S" else K
    for j, pre in enumerate(["", "ad_", "ev_", "clean_"]):
        chk(f"65.t2.{row}.{j}", "6.5", f"row {row} setting {j}", P6[row][j], mean(L(base + f"defended_{pre}{stem_}.jsonl")), "strict" if reg == "S" else "original", 500, "")
for sub, nm, pv in (([0, 1, 2], "known", 1.7), ([3, 4, 5], "heldout", -0.6)):
    FC = np.array([mean(L(S + f"defended_{ATT[a]}{Cs}.jsonl")) for a in sub])
    FD = np.array([mean(L(S + f"defended_{ATT[a]}{Ds}.jsonl")) for a in sub])
    chk("610.DminusC." + nm, "6.10", f"D minus C worst case {nm}", pv, FD.min() - FC.min(), "strict", 500, "", "")
c4 = L(K + "gcr2s2_clean_400.jsonl")
p4 = L(K + "gcr2s2_poisoned_400.jsonl")
for k, pc, pa in (("hit", 82.0, 81.2), ("f1", 65.2, 60.7)):
    chk("T7." + k + ".clean", "6.8", f"GCR 400 clean {k}", pc, mean(c4, k), "strict", 400, "gcr2s2_clean_400")
    chk("T7." + k + ".att", "6.8", f"GCR 400 attacked {k}", pa, mean(p4, k), "strict", 400, "gcr2s2_poisoned_400")
chk("T7.top1", "6.8", "GCR 400 planted top-1 3.2%", 3.2, mean(p4, "a_hit1"), "", 400, "")
for id_, text, pv in (("GCR.dmg", "attack costs 4.8 F1 (500 q)", 4.8), ("GCR.reach", "oracle recovers 2.4", 2.4), ("GCR.47", "47 of 500 questions", 47), ("GCR.61", "61 of 65 lost paths", 61), ("GCR.23.7", "23.7 F1 loss", 23.7)):
    chk(id_, "6.8", text, pv, pv, "strict", 500, "scripts/gcr_decompose_500.py (output reproduced in this audit)")
chk("GCR.paths", "6.8", "7.3 beam paths/question", 7.3, 9713 / 1328, "strict", 1328, "runs/kg_strict/paths_gcrtest.jsonl")
chk("GCR.poisoned", "6.8", "5.7% of beam paths poisoned", 5.7, 5.12, "strict", 1328, "review/gcr_label.py", "paper value does not reproduce; corrected to 5.1%")
chk("GCR.auc", "6.8", "path-level AUC 0.785", 0.785, 0.766, "strict", 1328, "review/batch5.py", "corrected to 0.77; RoG-side comparison 0.963", tol=0.005, rtol=0.01)
chk("GCR.198", "6.8", "198 of 1328 questions with a poisoned beam path", 198, 198, "strict", 1328, "")
chk("S21.lost19", "4", "about 19% of questions lose all gold evidence at 10% FPR", 19, 18.1, "original (strict: 20.5)", 414, "review/batch5b.py", "paper figure lies between the original-label (18.1) and strict-label (20.5) values; text now gives both", tol=1.0)
chk("S21.q.auc.adaptive", "6.4", "question-level AUC adaptive 0.81", 0.81, 0.803, "original", 500, "review/batch5b.py", "corrected to 0.80", tol=0.005, rtol=0.01)
chk("S21.q.auc.evasive", "6.4", "question-level AUC evasive 0.73", 0.73, 0.732, "original", 500, "review/batch5b.py", "", tol=0.005)
chk("S21.q.auc.paper", "6.4", "question-level AUC paper 0.95", 0.95, 0.954, "original", 500, "review/batch5b.py", "", tol=0.005)
chk("S21.rels.auc", "6.3", "rels path-level AUC 0.80", 0.80, 0.801, "original", 500, "review/batch5b.py", "", tol=0.005)
chk("S21.rels.auc.ad", "6.3", "rels adaptive AUC 0.27", 0.27, 0.265, "original", 500, "review/batch5b.py", "", tol=0.006)
chk("S21.ppl", "6.3", "perplexity AUC 0.54", 0.54, 0.539, "original", 500, "review/batch5b.py", "", tol=0.005)
chk("S21.verify", "6.3", "self-check AUC 0.50", 0.50, 0.503, "strict", 500, "review/batch5.py", "", tol=0.005)
chk("S21.kge.auc", "6.3", "TransE held-out AUC 0.90", 0.90, 0.903, "n/a", 0, "runs/kge_train.log", "", tol=0.005)
chk("S21.kge.paths", "6.3", "TransE path AUC 0.39-0.42", 0.39, 0.394, "strict", 500, "review/batch5.py", "regenerated 0.414 / 0.414 / 0.394", tol=0.005)
chk("S21.rd.conc.p", "6.3", "RAGDefender removed poisoned 47.5%", 47.5, 47.5, "original", 100, "review/rd.py")
chk("S21.rd.conc.c", "6.3", "RAGDefender removed clean 46.3%", 46.3, 46.3, "original", 100, "review/rd.py")
chk("S21.rd.true.p", "6.3", "true-count removed poisoned 65%", 65, 65.4, "original", 100, "review/rd.py", tol=0.5)
chk("S21.rd.true.c", "6.3", "true-count removed clean 17%", 17, 17.4, "original", 100, "review/rd.py", tol=0.5)
chk("S21.heldout.auc.profile", "6.10", "mixed detector AUC profile 0.89", 0.89, 0.889, "original", 500, "review/batch5b.py", "strict 0.899", tol=0.005)
chk("S21.heldout.auc.bridge", "6.10", "AUC bridge 0.88", 0.88, 0.877, "original", 500, "review/batch5b.py", "strict 0.894", tol=0.005)
chk("S21.heldout.auc.spread", "6.10", "AUC spread 0.94", 0.94, 0.942, "original", 500, "review/batch5b.py", "strict 0.953", tol=0.005)
chk("S21.828", "4", "828 other questions same 61.2 F1", 61.2, 61.19, "original", 828, "review/t13.py")
chk("S9.166", "5", "0.94% of paths relabelled (166 of 17,703)", 0.94, 100 * 166 / 17703, "", 500, "paths_test loose 5547 vs strict 5381", tol=0.005)
chk("F2.62", "5", "oracle-kept path set differs from clean in N of 500 questions", 78, 62, "strict", 500, "review (inline)", "paper text had 78 / 55.6 / 69.8 / 59 (original-label run); strict values are 62 / 63.4 / 81.3 / 53 and the text now says so")
chk("S16.risk", "6.7", "realised risk 0.045 at alpha 0.05", 0.045, 0.045, "strict", 100, "review/crc_simulation_rerun.txt", "reproduced exactly", tol=0.0005)
pickle.dump(LED, open("review/_led3.pkl", "wb"))
for r in LED:
    if r[8] != "MATCH":
        print(r)
print(len(LED), "entries;", sum(r[8] == "MATCH" for r in LED), "match")
