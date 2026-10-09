import sys, pickle
sys.path.insert(0, "review")
from ledger1 import *
sys.path.insert(0, ROOT)
from kgrag import rog

N = rog.normalize
em_gold = {}
for r in lines(K + "paths_test.jsonl") + lines(K + "paths_full_test.jsonl"):
    em_gold[r["id"]] = r["gold"]


def em(r):
    P = {N(p) for p in r["prediction"] if p}
    G = {N(x) for x in em_gold[r["id"]]}
    return float(bool(P) and P == G)


def e(p):
    return 100 * np.mean([em(r) for r in L(p).values()])


# ---- Table 1
for tag, pre, sec in (("WebQSP-1328", "full_", "6.2"), ("CWQ-3231", "cwqfull_", "6.2")):
    none = L(K + f"defended_{pre}none.jsonl")
    n = len(none)
    P = {"WebQSP-1328": dict(clean=67.2, none=45.3, learned=61.2, crc=60.1, oracle=65.6),
         "CWQ-3231": dict(clean=48.9, none=35.3, learned=43.6, crc=43.2, oracle=48.4)}[tag]
    cl = L(K + ("clean_webqsp.jsonl" if pre == "full_" else "clean_cwq.jsonl"))
    chk(f"T1.{tag}.clean", sec, f"{tag} clean F1", P["clean"], mean(cl, "f1", sorted(none)), "no attack", n, "runs/kg/clean_*.jsonl", "clean run restricted to evaluated ids", rtol=0.06)
    chk(f"T1.{tag}.none", sec, f"{tag} none", P["none"], mean(none), "n/a", n, src(K + f"defended_{pre}none.jsonl"))
    chk(f"T1.{tag}.learned2", sec, f"{tag} learned@2%", P["learned"], mean(L(K + f"defended_{pre}learned-dev_fpr0.02.jsonl")), "original", n, src(K + f"defended_{pre}learned-dev_fpr0.02.jsonl"))
    chk(f"T1.{tag}.crc", sec, f"{tag} conformal", P["crc"], mean(L(S + f"defended_{pre}crc-dev_a0.05.jsonl")), "strict", n, src(S + f"defended_{pre}crc-dev_a0.05.jsonl"))
    chk(f"T1.{tag}.oracle", sec, f"{tag} oracle", P["oracle"], mean(L(S + f"defended_{pre}oracle_fpr0.02.jsonl")), "strict", n, src(S + f"defended_{pre}oracle_fpr0.02.jsonl"))
g = paired(L(S + "defended_full_crc-dev_a0.05.jsonl"), L(K + "defended_full_none.jsonl"))
chk("T1.gain.crc.web", "6.2", "conformal gain WebQSP +14.9", 14.9, g[0], "strict", g[3], "strict crc minus none", note=f"CI recomputed [{g[1]:+.1f},{g[2]:+.1f}] vs paper [+12.8,+17.0]")
g = paired(L(S + "defended_cwqfull_crc-dev_a0.05.jsonl"), L(K + "defended_cwqfull_none.jsonl"))
chk("T1.gain.crc.cwq", "6.2", "conformal gain CWQ +7.9", 7.9, g[0], "strict", g[3], "strict crc minus none", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}] vs paper [+6.5,+9.3]")
g = paired(L(K + "defended_full_learned-dev_fpr0.02.jsonl"), L(K + "defended_full_none.jsonl"))
chk("T1.gain.l2.web", "6.2", "2% gain WebQSP +15.9", 15.9, g[0], "original", g[3], "")
g = paired(L(K + "defended_cwqfull_learned-dev_fpr0.02.jsonl"), L(K + "defended_cwqfull_none.jsonl"))
chk("T1.gain.l2.cwq", "6.2", "2% gain CWQ +8.3", 8.3, g[0], "original", g[3], "")


def share(r, o, none):
    ids = sorted(set(r) & set(o) & set(none))
    a = np.array([[none[i]['f1'], r[i]['f1'], o[i]['f1']] for i in ids]) * 100
    return 100 * (a[:, 1].mean() - a[:, 0].mean()) / (a[:, 2].mean() - a[:, 0].mean())


nw = L(K + "defended_full_none.jsonl")
nc = L(K + "defended_cwqfull_none.jsonl")
chk("T1.share.crc.web", "6.2/abstract", "73.1% (abstract 73)", 73.1, share(L(S + "defended_full_crc-dev_a0.05.jsonl"), L(S + "defended_full_oracle_fpr0.02.jsonl"), nw), "strict/strict", 1328, "strict crc, strict oracle", "point estimate 73.18")
chk("T1.share.crc.cwq", "6.2/abstract", "60.2% (abstract 60)", 60.2, share(L(S + "defended_cwqfull_crc-dev_a0.05.jsonl"), L(S + "defended_cwqfull_oracle_fpr0.02.jsonl"), nc), "strict/strict", 3231, "")
chk("T1.share.l2.web", "6.2", "78.3% (2% level)", 78.3, share(L(K + "defended_full_learned-dev_fpr0.02.jsonl"), L(S + "defended_full_oracle_fpr0.02.jsonl"), nw), "MIXED: original learned / strict oracle", 1328, "S3", "matched original/original = 78.7")
chk("T1.share.l2.cwq", "6.2", "63.1% (2% level)", 63.1, share(L(K + "defended_cwqfull_learned-dev_fpr0.02.jsonl"), L(S + "defended_cwqfull_oracle_fpr0.02.jsonl"), nc), "MIXED: original learned / strict oracle", 3231, "S3", "matched original/original = 72.8 (runs/kg/review_checks.md)")
cl1 = mean(L(K + 'clean_webqsp.jsonl'), 'f1', sorted(nw))
chk("T1.rel.recovered", "6.2", "68% of lost F1 (14.9 of 21.9)", 68, 100 * (mean(L(S + 'defended_full_crc-dev_a0.05.jsonl')) - mean(nw)) / (cl1 - mean(nw)), "strict", 1328, "", tol=0.5)
for nm, p, v in (("web none", K + "defended_full_none.jsonl", 47.4), ("web crc", S + "defended_full_crc-dev_a0.05.jsonl", 9.1), ("cwq none", K + "defended_cwqfull_none.jsonl", 48.8), ("cwq crc", S + "defended_cwqfull_crc-dev_a0.05.jsonl", 15.8)):
    r = L(p)
    chk("T1.planted." + nm, "6.2", f"planted-answer precision {nm}", v, mean(r, "a_precision"), "", len(r), src(p), "a_precision = share of predicted answers that are planted (mean over questions)")
chk("T1.EM.none500", "6.2", "EM 18.2", 18.2, e(K + "defended_none.jsonl"), "", 500, "runs/kg/defended_none.jsonl", "EM = normalised prediction set equals normalised gold set")
chk("T1.EM.crc500", "6.2", "EM 33.8 (conformal)", 33.8, e(S + "defended_crc-dev_a0.05.jsonl"), "strict", 500, "kg_strict/defended_crc-dev_a0.05")
chk("T1.EM.clean500", "6.2", "EM 43.0 clean", 43.0, e(K + "defended_clean_none.jsonl"), "", 500, "")
chk("T1.EM.oracle500", "6.2", "EM 40.0 strict oracle", 40.0, e(S + "defended_oracle_fpr0.02.jsonl"), "strict", 500, "")
# ---- 6.1
cl = L(K + "defended_clean_none.jsonl")
no = L(K + "defended_none.jsonl")
chk("61.hit.clean", "6.1", "Hit 83.2", 83.2, mean(cl, "hit"), "", 500, "defended_clean_none")
chk("61.hit.att", "6.1", "Hit 78.0", 78.0, mean(no, "hit"), "", 500, "defended_none")
chk("61.f1.clean", "6.1", "F1 67.7", 67.7, mean(cl), "", 500, "")
chk("61.f1.att", "6.1", "F1 45.9", 45.9, mean(no), "", 500, "")
chk("61.f1.rel", "6.1", "F1 -32%", 32, 100 * (mean(cl) - mean(no)) / mean(cl), "", 500, "", tol=0.5)
chk("61.hit.rel", "6.1", "Hit -6%", 6, 100 * (mean(cl, 'hit') - mean(no, 'hit')) / mean(cl, 'hit'), "", 500, "", tol=0.5)
anyp = 100 * np.mean([r['a_precision'] > 0 for r in no.values()])
chk("61.planted.500", "6.1", "planted answer appears in 47.7% of outputs", 47.7, mean(no, "a_precision"), "", 500, "defended_none a_precision", f"S4: this is precision. Share of questions whose output contains any planted answer = {anyp:.1f}%")
chk("61.EM.rel", "6.1", "EM -58%", 58, 100 * (e(K + "defended_clean_none.jsonl") - e(K + "defended_none.jsonl")) / e(K + "defended_clean_none.jsonl"), "", 500, "", tol=0.5)
b8 = L(K + "defended_b8_none.jsonl")
chk("69.f1rel", "6.9", "F1 -38%", 38, 100 * (mean(cl) - mean(b8)) / mean(cl), "", 500, "", tol=0.5)
chk("69.hitrel", "6.9", "Hit -10%", 10, 100 * (mean(cl, 'hit') - mean(b8, 'hit')) / mean(cl, 'hit'), "", 500, "", tol=0.5)
chk("69.em.b8", "6.9", "EM 18.0", 18.0, e(K + "defended_b8_none.jsonl"), "", 500, "")
# ---- Table 2 baselines
none = L(K + "defended_none.jsonl")
B = [("similarity", "similarity", 42.3, -3.6), ("perplexity", "perplexity", 42.5, -3.3), ("verify", "verify", 46.5, 0.7), ("kge_fpr0.02", "TransE@2%", 45.8, -0.0),
     ("ragdefender-conc", "RD conc", 34.0, -11.9), ("ragdefender-clust", "RD clust", 27.1, -18.8), ("ragdefender-oraclen", "RD true count", 30.9, -14.9),
     ("rels", "rels", 54.4, 8.6), ("learned-dev_fpr0.02", "learned@2%", 61.1, 15.3)]
for f, nm, pf, pg in B:
    r = L(K + f"defended_{f}.jsonl")
    g = paired(r, none)
    chk("T2." + nm, "6.3", f"{nm} F1", pf, mean(r), "original", len(r), src(K + f"defended_{f}.jsonl"))
    chk("T2.gain." + nm, "6.3", f"{nm} gain", pg, g[0], "original", g[3], "paired", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
# ---- Table 3 (6.4)
T3 = [("ad_", "learned-dev_fpr0.02", "adaptive 500 paper-det", 54.9, 9.3), ("ad_", "learned-" + MIX + "_fpr0.02", "adaptive 500 mixed", 62.0, 16.4),
      ("fullad_", "learned-dev_fpr0.02", "adaptive 1328 paper-det", 55.1, 10.0), ("fullad_", "learned-" + MIX + "_fpr0.02", "adaptive 1328 mixed", 60.7, 15.6),
      ("ev_", "learned-dev_fpr0.02", "evasive paper-det", 50.1, -1.4), ("ev_", "learned-" + MIX + "_fpr0.02", "evasive mixed", 55.4, 3.9),
      ("ev_", "learned-" + MIX + "+ev_dev_fpr0.02", "evasive attack-aware", 60.0, 8.5)]
for pre, f, nm, pf, pg in T3:
    n_ = L(K + f"defended_{pre}none.jsonl")
    r = L(K + f"defended_{pre}{f}.jsonl")
    g = paired(r, n_)
    chk("T3." + nm, "6.4", nm + " F1", pf, mean(r), "original", len(r), src(K + f"defended_{pre}{f}.jsonl"))
    chk("T3.gain." + nm, "6.4", nm + " gain", pg, g[0], "original", g[3], "paired", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
chk("T3.none.ad", "6.4", "none adaptive 45.6", 45.6, mean(L(K + "defended_ad_none.jsonl")), "", 500, "")
chk("T3.none.ev", "6.4", "none evasive 51.5", 51.5, mean(L(K + "defended_ev_none.jsonl")), "", 500, "")
pickle.dump(LED, open("review/_led1.pkl", "wb"))
for r in LED:
    if r[8] != "MATCH":
        print(r)
print(len(LED), "entries;", sum(r[8] == "MATCH" for r in LED), "match")
