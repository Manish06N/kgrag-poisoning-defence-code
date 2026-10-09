"""GCR second stage: a general LLM reads the reasoning paths produced by the GCR first stage and answers the question.

GCR (graph-constrained reasoning) works in two steps: (1) the KG-specialised LLM generates reasoning paths constrained to the KG
(kgrag.gcr_run, saved as beams "# Reasoning Path:\\n... \\n# Answer:\\n..."); (2) a general LLM performs inductive reasoning over those
paths. Zhao et al. run GCR with GPT-3.5-turbo for step 2. Here step 2 is a local chat LLM (default Qwen2.5-7B-Instruct, 4-bit) with GCR's
own rule instruction ("Based on the reasoning paths, please answer the given question ... return all the possible answers as a list"),
which is the instruction kgrag.rog.ChatReasoner already uses. Deviation to state in the paper: local reasoner instead of GPT-3.5-turbo.

  python -m kgrag.gcr_stage2 --first runs/kg/gcr2_clean_400.jsonl --data rmanluo/RoG-webqsp --out runs/kg/gcr2s2_clean_400.jsonl
  python -m kgrag.gcr_stage2 --first runs/kg/gcr2_poisoned_400.jsonl --data data/poisoned_webqsp --out runs/kg/gcr2s2_poisoned_400.jsonl
"""
import argparse
import json
import os
import re

import numpy as np

from kgrag import attack, rog

PATH_RE = re.compile(r"# Reasoning Path:\n(.*?)\n# Answer:", re.S)


def beam_paths(beams: list[str]) -> list[str]:
    """Unique reasoning-path strings, in beam order, taken from the first-stage beams (malformed beams are skipped)."""
    seen, out = set(), []
    for b in beams:
        m = PATH_RE.search(b)
        if not m:
            continue
        p = m.group(1).strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first", required=True, help="first-stage output of kgrag.gcr_run (must contain all beams)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reasoner", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--quant", default="4bit")
    a = p.parse_args()
    ds = rog.load_data(a.data, "test")
    info = {ex["id"]: ex for ex in ds}
    has_adv = "adv_answers" in ds.column_names
    rows_in = [json.loads(line) for line in open(a.first, encoding="utf8")]
    done = {json.loads(line)["id"] for line in open(a.out, encoding="utf8")} if os.path.exists(a.out) else set()
    model = rog.ChatReasoner(a.reasoner, None if a.quant == "none" else a.quant)
    from tqdm import tqdm
    with open(a.out, "a", encoding="utf8") as f:
        for r in tqdm(rows_in):
            if r["id"] in done:
                continue
            ex = info[r["id"]]
            paths = beam_paths(r["beams"])
            pred = model.reason(ex["question"], paths) if paths else []
            row = {"id": r["id"], "n_beam_paths": len(paths), "prediction": pred, **rog.score(pred, ex["answer"])}
            if has_adv:
                row.update(attack.attack_metrics(pred, ex["adv_answers"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    rows = [json.loads(line) for line in open(a.out, encoding="utf8")]
    keys = [k for k in ("hit", "f1", "precision", "recall", "a_precision", "a_hit1") if k in rows[0]]
    print("RESULT", os.path.basename(a.out), f"n={len(rows)}", "  ".join(f"{k}={100 * np.mean([x[k] for x in rows]):.1f}" for k in keys))


if __name__ == "__main__":
    main()
