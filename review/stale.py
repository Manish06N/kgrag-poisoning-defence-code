import glob, os, datetime
for D in ("runs/kg_strict/","runs/kg_mg/"):
    dev_m = {n: os.path.getmtime(D+f"paths_{n}.jsonl") for n in ("dev","K1_dev","K8_dev","ad_dev","clean_dev","ev_dev") if os.path.exists(D+f"paths_{n}.jsonl")}
    for f in sorted(glob.glob(D+"defended_*.jsonl")):
        b=os.path.basename(f); m=os.path.getmtime(f)
        need=["dev"]
        if "K1_dev" in b: need=["dev","K1_dev","K8_dev","ad_dev"]
        if "+ev_dev" in b: need.append("ev_dev")
        if "gate" in b or "clean_" in b: need.append("clean_dev")
        # file mtime is last write (append); a result older than a needed input is stale
        latest=max(dev_m[n] for n in need if n in dev_m)
        flag = "STALE?" if m < latest else ""
        print(f"{b[:70]:70s} {datetime.datetime.fromtimestamp(m):%m-%d %H:%M} inputs<= {datetime.datetime.fromtimestamp(latest):%m-%d %H:%M} {flag}")
