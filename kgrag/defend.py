"""Path-level defences against KG poisoning: suspicion scores (higher = more suspicious) and their detection quality.

  python -m kgrag.defend --scores random,degree,rels,similarity,perplexity,verify
Also available: kge (TransE plausibility baseline, needs `python -m kgrag.kge train` first).
Scores are cached in runs/kg/scores_<name>_<split>.npy (one value per path, same order as the paths file).

Evaluation (labels are known: the attacker's inserted triples):
  AUC            : pooled over all paths, and mean per question (questions that have both clean and poisoned paths)
  filter at FPR  : threshold = the score that removes 10% of CLEAN dev paths (dev = other questions); on the test split report
                   the % of poisoned paths removed, % of clean paths removed, and how many questions still keep >= 1 clean
                   gold-supporting path (collateral damage).
"""
import argparse
import json
import os

import numpy as np

RUNS = os.environ.get("KG_RUNS", "runs/kg")  # set KG_RUNS to keep a different run (e.g. bf16 on the HPC) apart
FPR = 0.10


def load_paths(split: str):
    return [json.loads(line) for line in open(os.path.join(RUNS, f"paths_{split}.jsonl"), encoding="utf8")]


def flat(recs, key):
    return np.array([p[key] for r in recs for p in r["paths"]])


def qindex(recs):
    """question index for every flattened path"""
    return np.array([i for i, r in enumerate(recs) for _ in r["paths"]])


# ---------------------------------------------------------------- scorers (higher = more suspicious)
def score_random(recs, seed=0):
    return np.random.default_rng(seed).random(sum(len(r["paths"]) for r in recs))


def score_degree(recs):
    """few connections of the answer entity in the question's subgraph -> suspicious"""
    return -np.log1p(flat(recs, "end_deg").astype(float))


def score_min_degree(recs):
    return -np.log1p(flat(recs, "min_deg").astype(float))


def score_rels(recs):
    """answer entity takes part in few distinct relation types -> suspicious"""
    return -flat(recs, "end_rels").astype(float)


def score_similarity(recs):
    """TrustRAG-style: injected evidence is near-duplicate of other evidence. Mean cosine similarity to the question's other paths."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    out = []
    for r in recs:
        texts = [p["text"].replace(".", " ").replace("_", " ") for p in r["paths"]]
        if len(texts) < 2:
            out.extend([0.0] * len(texts))
            continue
        e = enc.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        s = e @ e.T
        np.fill_diagonal(s, 0.0)
        out.extend((s.sum(1) / (len(texts) - 1)).tolist())
    return np.array(out)


def score_perplexity(recs, model_name="Qwen/Qwen2.5-1.5B-Instruct", batch=16):
    """RAGuard/GMTP-style: language-model surprise (mean NLL per token) of the verbalised path"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map={"": 0}).eval()
    texts = ["Fact: " + p["text"].replace("_", " ") for r in recs for p in r["paths"]]
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        mask = enc.attention_mask[:, 1:].float()
        out[idx] = ((nll * mask).sum(1) / mask.sum(1)).cpu().numpy()
    return out


def score_verify(recs, model_name="Qwen/Qwen2.5-7B-Instruct", batch=32):
    """Arbitration with the model's own knowledge: is the path's end entity a correct answer? suspicion = log P(No) - log P(Yes)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    pairs = {}
    for r in recs:
        for p in r["paths"]:
            pairs.setdefault((r["question"], p["triples"][-1][2]), None)
    keys = list(pairs)
    prompts = [tok.apply_chat_template([{"role": "user", "content": (
        f"Question: {q}\nCandidate answer: {a}\nIs the candidate answer a correct answer to the question? Answer Yes or No.")}],
        tokenize=False, add_generation_prompt=True) for q, a in keys]
    order = np.argsort([len(p) for p in prompts])  # similar lengths together: little padding
    for n, s in enumerate(range(0, len(keys), batch)):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():  # logits_to_keep=1: only the last position (a full-vocabulary logit tensor overflowed the GPU)
            lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), dim=-1)
        for i, v in zip(idx, (lp[:, no] - lp[:, yes]).cpu().numpy()):
            pairs[keys[i]] = float(v)
        if n % 50 == 0:
            print(f"  verify {s}/{len(keys)}", flush=True)
    return np.array([pairs[(r["question"], p["triples"][-1][2])] for r in recs for p in r["paths"]])


def score_kge(recs):
    from kgrag import kge  # imported lazily: kge itself imports this module
    return kge.score_kge(recs)


SCORERS = {"random": score_random, "degree": score_degree, "min_degree": score_min_degree, "rels": score_rels,
           "similarity": score_similarity, "perplexity": score_perplexity, "verify": score_verify, "kge": score_kge}


def get_scores(name, split, recs):
    path = os.path.join(RUNS, f"scores_{name}_{split}.npy")
    if os.path.exists(path):
        return np.load(path)
    s = SCORERS[name](recs)
    np.save(path, s)
    return s


# ---------------------------------------------------------------- evaluation
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def per_question_auc(recs, y, s):
    qi, vals = qindex(recs), []
    for i in range(len(recs)):
        m = qi == i
        if 0 < y[m].sum() < m.sum():
            vals.append(auc(y[m], s[m]))
    return float(np.nanmean(vals)) if vals else float("nan")


def threshold_at_fpr(s_dev, y_dev, fpr=FPR):
    return float(np.quantile(s_dev[y_dev == 0], 1 - fpr))


def filter_stats(recs, y, s, thr):
    removed = s > thr
    gold = flat(recs, "gold_hit").astype(bool)
    qi = qindex(recs)
    had = kept = 0
    for i in range(len(recs)):
        m = (qi == i) & gold & (y == 0)  # clean gold-supporting paths of this question
        if m.any():
            had += 1
            kept += int((~removed[m]).any())
    return {"poison_removed": float(removed[y == 1].mean()), "clean_removed": float(removed[y == 0].mean()),
            "q_keep_gold": kept / max(had, 1), "poison_left_per_q": float((~removed & (y == 1)).sum() / len(recs))}


def evaluate(names):
    test, dev = load_paths("test"), load_paths("dev")
    y_t, y_d = flat(test, "poisoned").astype(int), flat(dev, "poisoned").astype(int)
    print(f"test: {len(test)} q, {len(y_t)} paths, {y_t.mean():.1%} poisoned | dev: {len(dev)} q, {len(y_d)} paths")
    print(f"{'defence':12s} {'AUC':>6s} {'AUC/q':>6s} | at 10% clean removed: {'poison out':>10s} {'clean out':>9s} {'q keep gold':>11s} {'poison left/q':>13s}")
    for n in names:
        s_t, s_d = get_scores(n, "test", test), get_scores(n, "dev", dev)
        f = filter_stats(test, y_t, s_t, threshold_at_fpr(s_d, y_d))
        print(f"{n:12s} {auc(y_t, s_t):6.3f} {per_question_auc(test, y_t, s_t):6.3f} | {'':22s}{f['poison_removed']:9.1%} {f['clean_removed']:9.1%} "
              f"{f['q_keep_gold']:11.1%} {f['poison_left_per_q']:13.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="random,degree,min_degree,rels")
    evaluate(p.parse_args().scores.split(","))
