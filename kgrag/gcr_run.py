"""GCR (graph-constrained reasoning) inference on a possibly poisoned KG, re-implemented on current transformers.

Follows the MIT-licensed reference code (third_party/gcr): index all directed paths of <= 2 hops from the topic entities in a
prefix trie, let the fine-tuned LLM generate "<PATH> ... </PATH>" only along trie paths (group beam search, k beams), then read
the answer after "# Answer:". Differences from the reference code, to be stated in the paper: 4-bit weights (16 GB card),
answers de-duplicated before F1, max_new_tokens reduced.

  python -m kgrag.gcr_run --data rmanluo/RoG-webqsp --ids 0:100 --out runs/kg/gcr_clean_100.jsonl          # clean graph
  python -m kgrag.gcr_run --data data/poisoned_webqsp --ids 0:100 --out runs/kg/gcr_poisoned_100.jsonl     # attacked graph
All k decoded beams are saved in each output row (needed by kgrag.gcr_stage2, the second-stage reasoner).
Note: a path filter that removes exactly the inserted triples is identical to running on the clean graph (oracle == clean).
"""
import argparse
import json
import os
import re
from collections import Counter

import marisa_trie
import numpy as np

from kgrag import attack, gcr, pathdata, rog

PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. It should start with <PATH> and end with </PATH>. When given a question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it. Then, use these reasoning paths to derive the answer to the question.

# Question:
{question}
# Topic entities:
{entities}
"""
START, END = "<PATH>", "</PATH>"


def _c(i: int) -> str:  # token id -> single unicode char, skipping the surrogate range
    return chr(i if i < 0xD800 else i + 0x800)


def _i(c: str) -> int:
    o = ord(c)
    return o if o < 0xD800 else o - 0x800


class PathTrie:
    def __init__(self, sequences: list[list[int]]):
        self.first = list({s[0] for s in sequences})
        self.trie = marisa_trie.Trie("".join(map(_c, s)) for s in sequences)

    def get(self, prefix: list[int]) -> list[int]:
        if not prefix:
            return self.first
        key = "".join(map(_c, prefix))
        return list({_i(e[len(key)]) for e in self.trie.keys(key) if len(e) > len(key)})


class Gcr:
    def __init__(self, path: str = "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct", quant: str | None = "4bit"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        kw = {}
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map={"": 0}, attn_implementation="sdpa", **kw).eval()
        self.start_id = self.tok.convert_tokens_to_ids(START)
        self.end_id = self.tok.convert_tokens_to_ids(END)
        self.all_tokens = list(range(len(self.tok)))

    def _allowed_fn(self, trie: PathTrie):
        torch, start, end, everything = self.torch, self.start_id, self.end_id, self.all_tokens

        def fn(batch_id, sent):
            pos = torch.where(sent == start)[0]
            if len(pos) == 0:
                return everything
            last = int(pos[-1])
            if (sent[last:] == end).any():  # path already closed
                return everything
            allowed = trie.get(sent[last:].tolist())
            return allowed if allowed else everything
        return fn

    def answer(self, ex: dict, k: int = 10, max_new_tokens: int = 512, drop: set | None = None) -> dict:
        """`drop`: set of path texts (rog.path_to_string) removed from the candidate set before the trie is built."""
        paths = gcr.dfs_paths(ex["graph"], ex["q_entity"], 2)
        texts = [rog.path_to_string(p) for p in paths]
        keep = [t for t in texts if not drop or t not in drop]
        if not keep:
            return {"n_paths": len(texts), "n_kept": 0, "prediction": [], "beams": []}
        seqs = self.tok([f"{START}{t}{END}" for t in keep], padding=False, add_special_tokens=False).input_ids
        trie = PathTrie(seqs)
        q = ex["question"] if ex["question"].endswith("?") else ex["question"] + "?"
        user = PROMPT.format(question=q, entities=",".join(ex["q_entity"]))
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        cfg = self.model.generation_config
        cfg.max_new_tokens, cfg.do_sample = max_new_tokens, False
        # GCR uses group beam search; transformers>=4.57 loads it from a Hub repo that needs trust_remote_code, which we do not
        # enable (downloaded code). Plain beam search with k beams instead: a documented deviation (less diverse beams).
        cfg.num_beams = cfg.num_return_sequences = k
        cfg.num_beam_groups = 1
        cfg.diversity_penalty = 0.0
        cfg.temperature = cfg.top_p = None
        with self.torch.inference_mode():
            out = self.model.generate(**enc, generation_config=cfg, prefix_allowed_tokens_fn=self._allowed_fn(trie),
                                      pad_token_id=self.tok.eos_token_id)
        beams = [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
        counts = Counter(beams)
        answers = [b.split("# Answer:\n")[-1].strip() for b, _ in counts.most_common()]
        answers = [re.sub(r"\s+", " ", a) for a in answers if a]
        return {"n_paths": len(texts), "n_kept": len(keep), "prediction": list(dict.fromkeys(answers)), "beams": beams}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/poisoned_webqsp")
    p.add_argument("--ids", default="0:100")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--quant", default="4bit")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ds = rog.load_data(a.data, "test")
    adv_map = {ex["id"]: ex["adv_answers"] for ex in ds} if "adv_answers" in ds.column_names else {}
    done = {json.loads(line)["id"] for line in open(a.out, encoding="utf8")} if os.path.exists(a.out) else set()
    model = Gcr(quant=None if a.quant == "none" else a.quant)
    from tqdm import tqdm
    with open(a.out, "a", encoding="utf8") as f:
        for i in tqdm(pathdata.parse_ids(a.ids, len(ds))):
            ex = ds[i]
            if ex["id"] in done:
                continue
            r = model.answer(ex, a.k)
            row = {"id": ex["id"], "n_paths": r["n_paths"], "n_kept": r["n_kept"], "prediction": r["prediction"],
                   "beams": r["beams"], **rog.score(r["prediction"], ex["answer"])}
            if adv_map:
                row.update(attack.attack_metrics(r["prediction"], adv_map[ex["id"]]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    rows = [json.loads(line) for line in open(a.out, encoding="utf8")]
    keys = [k for k in ("hit", "f1", "precision", "recall", "a_precision", "a_hit1") if k in rows[0]]
    print("RESULT", os.path.basename(a.out), f"n={len(rows)}", "  ".join(f"{k}={100 * np.mean([x[k] for x in rows]):.1f}" for k in keys))


if __name__ == "__main__":
    main()
