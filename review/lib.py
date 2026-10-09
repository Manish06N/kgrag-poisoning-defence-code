"""Shared helpers for the final audit (CPU only, reads saved run files, never writes to runs/)."""
import json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"
K, S, M = ROOT + "runs/kg/", ROOT + "runs/kg_strict/", ROOT + "runs/kg_mg/"
H = ROOT + "hpc_results/T1_unpacked/runs/kg_bf16/"
MIX = "dev+K1_dev+K8_dev+ad_dev"

def L(path):
    if not os.path.exists(path): return None
    return {(r := json.loads(l))["id"]: r for l in open(path, encoding="utf8") if l.strip()}

def lines(path):
    return [json.loads(l) for l in open(path, encoding="utf8") if l.strip()]

def mean(run, key="f1", ids=None):
    ids = sorted(run) if ids is None else ids
    return 100 * float(np.mean([run[i][key] for i in ids]))

def paired(a, b, key="f1", B=2000, seed=0, ids=None):
    """mean of (a-b) in points with percentile bootstrap CI (paired over questions)"""
    ids = sorted(set(a) & set(b)) if ids is None else ids
    d = np.array([a[i][key] - b[i][key] for i in ids]) * 100
    ix = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    m = d[ix].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)

def fmt(t, sign=True):
    return (f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]" if sign else f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]")
