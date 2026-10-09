import sys, pickle
sys.path.insert(0, "review")
from ledger1 import *

ATT = ["", "ad_", "ev_", "fp_", "fb_", "fs_"]
NM = ["paper", "adaptive", "evasive", "profile", "bridge", "spread"]
ROWS = {"A": ("crc-dev_a0.05", "K"), "B": (f"crc-{MIX}_a0.05", "K"), "C": ("gatecrc-dev_a0.05_q0.05", "S"), "D": (f"gatecrc-{MIX}_a0.05_q0.05", "S"),
        "E": ("learned-dev_fpr0.02", "K"), "F": (f"learned-{MIX}_fpr0.02", "K"), "G": ("gate-dev_fpr0.02_q0.05", "K"), "H": (f"gate-{MIX}_fpr0.02_q0.05", "K")}
# paper Table 4: clean, clean change, paper, adaptive, evasive, worst, gain, worst planted
P4 = {"A": (61.6, -6.1, 60.7, 53.4, 48.8, 48.8, 3.2, 24.8), "B": (62.4, -5.3, 62.8, 62.2, 52.4, 52.4, 6.8, 16.5),
      "C": (67.0, -0.7, 61.6, 49.1, 50.7, 49.1, 3.5, 37.2), "D": (67.2, -0.5, 60.8, 61.8, 50.8, 50.8, 5.2, 25.4),
      "E": (61.7, -5.9, 61.1, 54.9, 50.1, 50.1, 4.5, 16.7), "F": (61.6, -6.1, 63.0, 62.0, 55.4, 55.4, 9.8, 11.4),
      "G": (67.5, -0.1, 60.8, 48.8, 52.4, 48.8, 3.2, 34.7), "H": (66.9, -0.7, 62.1, 61.8, 51.7, 51.7, 6.1, 24.3)}
none = [L(K + f"defended_{ATT[a]}none.jsonl") for a in range(6)]
cn = L(K + "defended_clean_none.jsonl")
for row, (stem, reg) in ROWS.items():
    base = S if reg == "S" else K
    runs = [L(base + f"defended_{ATT[a]}{stem}.jsonl") for a in range(3)]
    cl = L(base + f"defended_clean_{stem}.jsonl")
    p = P4[row]
    rg = "strict" if reg == "S" else "original"
    chk(f"T4.{row}.clean", "6.4", f"row {row} clean F1", p[0], mean(cl), rg, len(cl), src(base + f"defended_clean_{stem}.jsonl"))
    d = paired(cl, cn)
    chk(f"T4.{row}.cleanchange", "6.4", f"row {row} clean change", p[1], d[0], rg, d[3], "paired", note=f"CI [{d[1]:+.1f},{d[2]:+.1f}]")
    for a in range(3):
        chk(f"T4.{row}.{NM[a]}", "6.4", f"row {row} {NM[a]}", p[2 + a], mean(runs[a]), rg, len(runs[a]), src(base + f"defended_{ATT[a]}{stem}.jsonl"))
    worst = min(mean(r) for r in runs)
    chk(f"T4.{row}.worst", "6.4", f"row {row} worst case", p[5], worst, rg, 500, "row minimum")
    chk(f"T4.{row}.gain", "6.4", f"row {row} worst-case gain", p[6], worst - min(mean(none[a]) for a in range(3)), rg, 500, "min defended - min undefended")
    chk(f"T4.{row}.planted", "6.4", f"row {row} worst planted", p[7], max(mean(r, "a_precision") for r in runs), rg, 500, "max over attacks of a_precision")
chk("T4.none.planted", "6.4", "no defence worst planted 47.7", 47.7, max(mean(none[a], "a_precision") for a in range(3)), "", 500, "")
T5 = [("learned-dev_fpr0.02", "K", 61.7, -5.9), ("crc-dev_a0.05", "K", 61.6, -6.1), ("learned-dev", "K", 42.1, -25.5), ("gate-dev_fpr0.02_q0.05", "K", 67.5, -0.1),
      ("gatecrc-dev_a0.05_q0.05", "S", 67.0, -0.7), (f"gatecrc-{MIX}_a0.05_q0.05", "S", 67.2, -0.5)]
for stem, reg, pf, pc in T5:
    base = S if reg == "S" else K
    r = L(base + f"defended_clean_{stem}.jsonl")
    d = paired(r, cn)
    chk(f"T5.{stem}", "6.5", f"clean {stem}", pf, mean(r), "strict" if reg == "S" else "original", len(r), src(base + f"defended_clean_{stem}.jsonl"))
    chk(f"T5.chg.{stem}", "6.5", f"clean change {stem}", pc, d[0], "", d[3], "paired", note=f"CI [{d[1]:+.1f},{d[2]:+.1f}]")


def wc(runs, nones, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    ids = sorted(set.intersection(*[set(r) for r in runs + nones]))
    n = len(ids)
    ix = rng.integers(0, n, (B, n))
    F = np.array([[r[i]["f1"] for i in ids] for r in runs]) * 100
    F0 = np.array([[r[i]["f1"] for i in ids] for r in nones]) * 100
    est = F.mean(1).min() - F0.mean(1).min()
    boot = np.min(np.stack([F[a][ix].mean(1) for a in range(len(runs))]), 0) - np.min(np.stack([F0[a][ix].mean(1) for a in range(len(runs))]), 0)
    return est, 2 * est - np.percentile(boot, 97.5), 2 * est - np.percentile(boot, 2.5), F.mean(1), F0.mean(1)


P10 = {"B": ([54.6, 56.4, 59.2], 11.6, "K"), "D": ([52.3, 56.1, 57.6], 8.9, "S"), "C": ([52.9, 56.3, 57.6], 9.5, "S")}
for row, (vals, gain, reg) in P10.items():
    stem = ROWS[row][0]
    base = S if reg == "S" else K
    runs = [L(base + f"defended_{ATT[a]}{stem}.jsonl") for a in (3, 4, 5)]
    est, lo, hi, F, F0 = wc(runs, [none[a] for a in (3, 4, 5)])
    for j, a in enumerate((3, 4, 5)):
        chk(f"T10.{row}.{NM[a]}", "6.10", f"row {row} {NM[a]} F1", vals[j], F[j], "strict" if reg == "S" else "original", 500, src(base + f"defended_{ATT[a]}{stem}.jsonl"))
    chk(f"T10.{row}.wcgain", "6.10", f"row {row} held-out worst-case gain", gain, est, "strict" if reg == "S" else "original", 500, "min-minus-min", note=f"S1. basic-bootstrap CI [{lo:+.1f},{hi:+.1f}]")
chk("T10.none.worst", "6.10", "no-defence worst 43.4", 43.4, min(mean(none[a]) for a in (3, 4, 5)), "", 500, "")
for a, v in zip((3, 4, 5), (65.4, 66.0, 65.7)):
    chk(f"T10.oracle.{NM[a]}", "6.10", f"strict oracle {NM[a]}", v, mean(L(S + f"defended_{ATT[a]}oracle_fpr0.02.jsonl")), "strict", 500, "")
for a, v in zip((3, 4, 5), (45.8, 43.5, 43.4)):
    chk(f"T10.none.{NM[a]}", "6.10", f"none {NM[a]}", v, mean(none[a]), "", 500, "")
P11 = {"B": ([62.8, 62.2, 52.4, 54.6, 56.4, 59.2], 9.0, 22.7), "C": ([61.6, 49.1, 50.7, 52.9, 56.3, 57.6], 5.7, 37.2), "D": ([60.8, 61.8, 50.8, 52.3, 56.1, 57.6], 7.4, 29.8)}
for row, (vals, gain, pl) in P11.items():
    stem = ROWS[row][0]
    reg = ROWS[row][1]
    base = S if reg == "S" else K
    runs = [L(base + f"defended_{ATT[a]}{stem}.jsonl") for a in range(6)]
    est, lo, hi, F, F0 = wc(runs, none)
    chk(f"T11.{row}.gain", "6.10", f"six-attack worst-case gain row {row}", gain, est, "strict" if reg == "S" else "original", 500, "min-minus-min", note=f"basic CI [{lo:+.1f},{hi:+.1f}]")
    chk(f"T11.{row}.planted", "6.10", f"six-attack worst planted row {row}", pl, max(mean(r, "a_precision") for r in runs), "", 500, "")
    chk(f"T11.{row}.worstF1", "6.10", f"six-attack worst F1 row {row}", min(vals), F.min(), "", 500, "")
chk("T11.none.worstplanted", "6.10", "no-defence worst planted 50.7", 50.7, max(mean(none[a], "a_precision") for a in range(6)), "", 500, "")
chk("T11.none.worst", "6.10", "no-defence worst 43.4", 43.4, min(mean(none[a]) for a in range(6)), "", 500, "")
b8 = {k: L(K + f"defended_b8_{k}.jsonl") for k in ("none", "learned-dev_fpr0.02", "crc-dev_a0.05", "oracle")}
for k, pf, ph in (("none", 42.1, 75.0), ("learned-dev_fpr0.02", 57.9, 80.2), ("crc-dev_a0.05", 60.0, 80.2), ("oracle", 64.4, 81.2)):
    chk(f"T9.{k}.f1", "6.9", f"b8 {k} F1", pf, mean(b8[k]), "original", 500, src(K + f"defended_b8_{k}.jsonl"))
    chk(f"T9.{k}.hit", "6.9", f"b8 {k} Hit", ph, mean(b8[k], "hit"), "", 500, "")
for k, pg in (("learned-dev_fpr0.02", 15.9), ("crc-dev_a0.05", 17.9), ("oracle", 22.4)):
    g = paired(b8[k], b8["none"])
    chk(f"T9.gain.{k}", "6.9", f"b8 gain {k}", pg, g[0], "original", 500, "", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}]")
mg = {"clean": L(M + "defended_clean_none_fpr0.02.jsonl"), "none": L(M + "defended_none_fpr0.02.jsonl"), "oracle": L(M + "defended_oracle_fpr0.02.jsonl"),
      "D": L(M + f"defended_gatecrc-{MIX}_a0.05_q0.05.jsonl"), "Dclean": L(M + f"defended_clean_gatecrc-{MIX}_a0.05_q0.05.jsonl")}
for k, v in (("clean", 74.6), ("none", 42.0), ("oracle", 74.6), ("D", 66.1), ("Dclean", 74.4)):
    chk(f"T12.mg.{k}", "6.12", f"MultiGraph {k}", v, mean(mg[k]), "strict", 500, "runs/kg_mg")
for k, v in (("clean", 67.7), ("none", 45.9)):
    chk(f"T12.sg.{k}", "6.12", f"simple graph {k}", v, mean({"clean": cn, "none": none[0]}[k]), "", 500, "")
chk("T12.sg.oracle", "6.12", "simple graph strict oracle 65.4", 65.4, mean(L(S + "defended_oracle_fpr0.02.jsonl")), "strict", 500, "")
chk("T12.sg.D", "6.12", "simple graph D 60.8", 60.8, mean(L(S + f"defended_gatecrc-{MIX}_a0.05_q0.05.jsonl")), "strict", 500, "")
chk("T12.sg.Dclean", "6.12", "simple graph D clean 67.2", 67.2, mean(L(S + f"defended_clean_gatecrc-{MIX}_a0.05_q0.05.jsonl")), "strict", 500, "")
chk("T12.mg.attack", "6.12", "MG attack -32.6", 32.6, mean(mg["clean"]) - mean(mg["none"]), "", 500, "")
chk("T12.sg.attack", "6.12", "simple attack -21.8", 21.8, mean(cn) - mean(none[0]), "", 500, "")
g = paired(mg["D"], mg["none"])
chk("T12.mg.Dgain", "6.12", "MG D gain +24.0", 24.0, g[0], "strict", 500, "", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}] paper [+20.9,+27.2]")
g = paired(mg["oracle"], mg["none"])
chk("T12.mg.orgain", "6.12", "MG oracle gain +32.6", 32.6, g[0], "strict", 500, "", note=f"CI [{g[1]:+.1f},{g[2]:+.1f}] paper [+29.5,+35.8]")
chk("T12.mg.share", "6.12", "MG D share ~74%", 74, 100 * (mean(mg["D"]) - mean(mg["none"])) / (mean(mg["oracle"]) - mean(mg["none"])), "strict", 500, "", tol=0.5)
for k, v, key in (("hit-sg", 83.2, "hit"), ("hit-mg", 88.6, "hit"), ("prec-sg", 77.0, "precision"), ("prec-mg", 80.9, "precision"), ("rec-sg", 72.8, "recall"), ("rec-mg", 80.0, "recall")):
    chk("T12." + k, "6.12", k, v, mean(cn if k.endswith("sg") else mg["clean"], key), "", 500, "")
rows = [("paper none", "full_none", 45.3, 46.1, 0.8), ("paper oracle", "full_oracle", 65.5, 66.3, 0.8), ("paper conformal", "full_crc-dev_a0.05", 60.3, 60.8, 0.5), ("paper learned2", "full_learned-dev_fpr0.02", 61.2, 62.1, 0.9),
        ("paper rels", "full_rels", 54.4, 55.4, 1.0), ("ad none", "fullad_none", 45.1, 46.2, 1.1), ("ad oracle", "fullad_oracle", 65.6, 66.3, 0.7), ("ad mixed", f"fullad_learned-{MIX}_fpr0.02", 60.7, 60.8, 0.2), ("ad paper", "fullad_learned-dev_fpr0.02", 55.1, 55.9, 0.8)]
for nm, f, p4, pb, pd in rows:
    a = L(K + f"defended_{f}.jsonl")
    b = L(H + f"defended_{f}.jsonl")
    d = paired(b, a)
    chk(f"T13.{nm}.4bit", "6.13", nm + " 4-bit", p4, mean(a), "original", 1328, src(K + f"defended_{f}.jsonl"))
    chk(f"T13.{nm}.bf16", "6.13", nm + " bf16", pb, mean(b), "original", 1328, "hpc_results/T1_unpacked")
    chk(f"T13.{nm}.diff", "6.13", nm + " diff", pd, d[0], "original", 1328, "paired", note=f"CI [{d[1]:+.1f},{d[2]:+.1f}]")
for f, v in (("fullev_none", 51.2), ("fullev_oracle", 66.5), ("fullev_learned-dev_fpr0.02", 49.0), (f"fullev_learned-{MIX}_fpr0.02", 51.1), (f"fullev_learned-{MIX}+ev_dev_fpr0.02", 58.4), (f"fullev_crc-{MIX}_a0.05", 50.4), (f"fullev_crc-{MIX}+ev_dev_a0.05", 58.2)):
    chk("T13.ev." + f, "6.13", "bf16 evasive " + f, v, mean(L(H + f"defended_{f}.jsonl")), "original", 1328, "hpc_results/T1_unpacked")
a = paired(L(H + "defended_full_crc-dev_a0.05.jsonl"), L(H + "defended_full_none.jsonl"))
chk("T13.crcgain.bf16", "6.13", "conformal gain bf16 +14.7", 14.7, a[0], "original", 1328, "")
a = paired(L(K + "defended_full_crc-dev_a0.05.jsonl"), L(K + "defended_full_none.jsonl"))
chk("T13.crcgain.4bit", "6.13", "conformal gain 4-bit +15.0", 15.0, a[0], "original", 1328, "")
sd = []
for pre in ["", "s1_", "s2_", "s3_", "s4_"]:
    sd.append(paired(L(K + f"defended_{pre}learned-dev_fpr0.02.jsonl"), L(K + f"defended_{pre}none.jsonl"))[0])
for v, g in zip((15.3, 15.8, 14.4, 16.1, 15.6), sd):
    chk("S15.seed", "6.6", "seed gain", v, g, "original", 500, "learned@2% paper detector, WebQSP-500")
chk("S15.mean", "6.6", "seed mean 15.4", 15.4, np.mean(sd), "original", 500, "")
chk("S15.sd", "6.6", "seed sd 0.7", 0.7, np.std(sd, ddof=1), "original", 500, "sample sd")
q1 = paired(L(K + "defended_learned-dev_fpr0.02_r-Qwen2.5-7B-Instruct.jsonl"), L(K + "defended_none_r-Qwen2.5-7B-Instruct.jsonl"))
q2 = paired(L(K + f"defended_ad_learned-{MIX}_fpr0.02_r-Qwen2.5-7B-Instruct.jsonl"), L(K + "defended_ad_none_r-Qwen2.5-7B-Instruct.jsonl"))
chk("66.qwen.paper", "6.6", "Qwen gain +10.6", 10.6, q1[0], "original", 500, "", note=f"CI [{q1[1]:+.1f},{q1[2]:+.1f}] paper [+6.6,+14.4]")
chk("66.qwen.ad", "6.6", "Qwen adaptive gain +9.1", 9.1, q2[0], "original", 500, "", note=f"CI [{q2[1]:+.1f},{q2[2]:+.1f}] paper [+5.2,+12.9]")
chk("66.qwen.planted", "6.6", "Qwen planted 32.0%", 32.0, mean(L(K + "defended_none_r-Qwen2.5-7B-Instruct.jsonl"), "a_precision"), "", 500, "")
pickle.dump(LED, open("review/_led2.pkl", "wb"))
for r in LED:
    if r[8] != "MATCH":
        print(r)
print(len(LED), "entries;", sum(r[8] == "MATCH" for r in LED), "match")
