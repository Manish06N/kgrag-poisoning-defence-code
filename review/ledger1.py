import sys, csv
sys.path.insert(0, "review")
from lib import *

LED = []


def chk(id_, sec, text, paper, regen, regime, n, source, note="", tol=0.05, rtol=0.15):
    d = abs(paper - regen)
    st = "MATCH" if d <= tol + 1e-9 else ("ROUNDING" if d <= rtol + 1e-9 else "MISMATCH")
    LED.append([id_, sec, text, paper, round(float(regen), 3), regime, n, source, st, note])
    return st


def src(p):
    return p.replace(ROOT, "")
