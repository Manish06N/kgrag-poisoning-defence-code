"""Masks of the headline defences under the current KG_RUNS labelling; run once with KG_RUNS=runs/kg (loose) and once with runs/kg_strict.
  KG_RUNS=runs/kg PYTHONPATH=. python scripts/strict_mask_diff.py out_loose.npz ; KG_RUNS=runs/kg_strict ... out_strict.npz
Then: python scripts/strict_mask_diff.py compare out_loose.npz out_strict.npz
"""
import sys

import numpy as np

MIX = "dev+K1_dev+K8_dev+ad_dev"
ATT = {"paper": ("", "data/poisoned_webqsp"), "adaptive": ("ad_", None), "evasive": ("ev_", None), "profile": ("fp_", None), "bridge": ("fb_", None), "spread": ("fs_", None), "clean": ("clean_", None)}
DEF = {"crc:dev": "crc", "gatecrc:dev": "gatecrc", f"crc:{MIX}": "crc", f"gatecrc:{MIX}": "gatecrc"}


def compute(path):
    from kgrag import defend, defended_run
    out = {}
    for an, (pre, _) in ATT.items():
        test, dev = defend.load_paths(pre + "test"), defend.load_paths(pre + "dev")
        sizes = np.array([len(r["paths"]) for r in test])
        out[f"{an}|sizes"] = sizes
        for d in DEF:
            if an == "clean" and d.startswith("crc:"):
                pass
            out[f"{an}|{d}"] = defended_run.removal_mask(d, test, dev, 0.02, pre, 0.05, 0.05)
    np.savez(path, **out)
    print("saved", path)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    print("%-10s %-34s %12s %22s" % ("attack", "defence", "paths changed", "questions with a different kept set"))
    for key in A.files:
        if key.endswith("|sizes"):
            continue
        an, d = key.split("|", 1)
        sizes = A[f"{an}|sizes"]
        diff = A[key] != B[key]
        qi = np.repeat(np.arange(len(sizes)), sizes)
        nq = len(set(qi[diff])) if diff.any() else 0
        print("%-10s %-34s %12d %14d of %d" % (an, d, int(diff.sum()), nq, len(sizes)))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        compute(sys.argv[1])
