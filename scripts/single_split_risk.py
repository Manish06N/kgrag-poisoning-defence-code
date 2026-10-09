import warnings
import numpy as np
warnings.filterwarnings("ignore")
from kgrag import conformal, defend
MIX = ["dev", "K1_dev", "K8_dev", "ad_dev"]
def risk(splits, pre, alpha=0.05):
    recs = defend.load_paths(pre + "test")
    mask = conformal.removal_mask(splits, pre + "test", alpha, verbose=False)
    pos, loss, ans = 0, 0, 0
    for r in recs:
        k = len(r["paths"]); m = mask[pos:pos + k]; pos += k
        idx = [i for i, p in enumerate(r["paths"]) if p["gold_hit"] and not p["poisoned"]]
        if idx:
            ans += 1; loss += all(m[i] for i in idx)
    return loss / max(ans, 1), ans
for name, sp in (("paper-detector (dev)", ["dev"]), ("mixed detector", MIX)):
    for pre, lab in (("", "paper"), ("ad_", "adaptive"), ("ev_", "evasive"), ("fp_", "profile"), ("fb_", "bridge"), ("fs_", "spread"), ("clean_", "clean"), ("full_", "paper 1328"), ("cwqfull_", "CWQ")):
        if name.startswith("mixed") and pre in ("full_", "cwqfull_"):
            continue
        try:
            rk, n = risk(sp, pre)
            print(f"{name:22s} {lab:11s} realised risk {rk:.3f} (answerable questions {n})", flush=True)
        except Exception as e:
            print(name, lab, "error", type(e).__name__, str(e)[:80], flush=True)
