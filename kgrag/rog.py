"""RoG (Luo et al., ICLR 2024) planning -> retrieval -> reasoning, re-implemented on current transformers.

Logic and prompts follow the MIT-licensed reference code (third_party/rog/src/qa_prediction/*, utils/graph_utils.py).
The three stages are separate methods so that attacks/defences can act between them:

  plan(question)                      -> relation paths (beam search of the fine-tuned planner)
  retrieve(graph, q_entities, rules)  -> reasoning paths (BFS following the relation paths)
  reason(question, paths)             -> answer lines (same model, paths as context)

  python -m kgrag.rog --data rmanluo/RoG-webqsp --n 50 --quant 4bit --out runs/kg/clean_webqsp.jsonl
`--data` may also be a folder saved with datasets.save_to_disk (e.g. a poisoned copy).
"""
import argparse
import json
import os
import random
import re
import string
from collections import deque

import networkx as nx

PLAN_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"
PREDICT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\n{input} [/INST]"
PLAN_INSTRUCTION = "Please generate a valid relation path that can be helpful for answering the following question: "
SAQ_RULE_INSTRUCTION = ("Based on the reasoning paths, please answer the given question. Please keep the answer as simple "
                        "as possible and return all the possible answers as a list.")
MAX_PROMPT_TOKENS = 4096 - 100
PATH_RE = r"<PATH>(.*)<\/PATH>"


# ---------------------------------------------------------------- graph utilities (same semantics as RoG)
# Note: RoG stores the graph as an undirected nx.Graph, i.e. ONE relation per entity pair (the last triple wins). An inserted triple on an
# existing pair therefore overwrites the clean relation. build_graph(..., multi=True) keeps every relation (nx.MultiGraph) for the ablation.
def build_graph(triples, multi: bool = False):
    g = nx.MultiGraph() if multi else nx.Graph()
    for h, r, t in triples:
        g.add_edge(h, t, relation=r.strip())
    return g


def bfs_with_rule(graph, start, rule: list[str]) -> list[list[tuple]]:
    """All paths from `start` whose consecutive relations equal `rule` (every parallel edge is followed on a MultiGraph)."""
    result, queue = [], deque([(start, [])])
    multi = graph.is_multigraph()
    while queue:
        node, path = queue.popleft()
        if len(path) == len(rule):
            result.append(path)
        if len(path) < len(rule):
            if node not in graph:
                continue
            for nb in graph.neighbors(node):
                rels = [e["relation"] for e in graph[node][nb].values()] if multi else [graph[node][nb]["relation"]]
                for rel in dict.fromkeys(rels):
                    if rel != rule[len(path)]:
                        continue
                    queue.append((nb, path + [(node, rel, nb)]))
    return result


def path_to_string(path) -> str:
    out = ""
    for i, (h, r, t) in enumerate(path):
        out += f"{h} -> {r} -> {t}" if i == 0 else f" -> {r} -> {t}"
    return out.strip()


def parse_plan(text: str) -> list[str] | None:
    m = re.search(PATH_RE, text)
    if m is None:
        return None
    rel = [r.strip() for r in m.group(1).split("<SEP>") if r.strip()]
    return rel or None


# ---------------------------------------------------------------- metrics (same as RoG evaluate_results.py)
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


def score(prediction: list[str], answers: list[str]) -> dict:
    pred = list(dict.fromkeys(prediction))  # RoG de-duplicates (order kept by count; here by first occurrence)
    pred_str = " ".join(pred)
    hit = float(any(match(pred_str, a) for a in answers))
    if not pred:
        return {"hit": hit, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    matched = sum(match(pred_str, a) for a in answers)
    p, r = matched / len(pred), matched / len(answers)
    return {"hit": hit, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r), "precision": p, "recall": r}


# ---------------------------------------------------------------- the model
class Rog:
    def __init__(self, path: str = "rmanluo/RoG", quant: str | None = "4bit", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=False)
        kw = {}
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
        elif quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map={"": 0}, **kw).eval()
        self.rng = random.Random(seed)

    def n_tokens(self, text: str) -> int:
        return len(self.tok.tokenize(text))

    def _generate(self, text: str, max_new_tokens: int, **kw) -> list[str]:
        ids = self.tok.encode(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                                      pad_token_id=self.tok.eos_token_id, **kw)
        return [s.strip() for s in self.tok.batch_decode(out.sequences[:, ids.shape[1]:], skip_special_tokens=True)]

    def plan(self, question: str, n_beam: int = 3, max_new_tokens: int = 100) -> list[list[str]]:
        text = PLAN_TEMPLATE.format(instruction=PLAN_INSTRUCTION, input=question)
        outs = self._generate(text, max_new_tokens, num_beams=n_beam, num_return_sequences=n_beam, do_sample=False,
                              early_stopping=False)
        rules = [parse_plan(o) for o in outs]
        return [r for r in rules if r]

    @staticmethod
    def retrieve(graph: nx.Graph, q_entities: list[str], rules: list[list[str]]) -> list[list[tuple]]:
        paths = []
        for e in q_entities:
            for rule in rules:
                paths.extend(bfs_with_rule(graph, e, rule))
        return paths

    def _fit_paths(self, question: str, path_strings: list[str]) -> list[str]:
        """RoG's rule: if the prompt is too long, shuffle the paths and keep as many as fit."""
        base = PREDICT_TEMPLATE.format(instruction=SAQ_RULE_INSTRUCTION, input=f"Reasoning Paths:\n\n\nQuestion:\n{question}")
        if self.n_tokens(base + "\n".join(path_strings)) < MAX_PROMPT_TOKENS:
            return path_strings
        shuffled, kept = list(path_strings), []
        self.rng.shuffle(shuffled)
        for p in shuffled:
            if self.n_tokens(base + "\n".join(kept + [p])) > MAX_PROMPT_TOKENS:
                break
            kept.append(p)
        return kept

    def reason(self, question: str, path_strings: list[str], max_new_tokens: int = 512) -> list[str]:
        q = question if question.endswith("?") else question + "?"
        kept = self._fit_paths(q, path_strings)
        body = f"Reasoning Paths:\n{chr(10).join(kept)}\n\nQuestion:\n{q}"
        text = PREDICT_TEMPLATE.format(instruction=SAQ_RULE_INSTRUCTION, input=body)
        out = self._generate(text, max_new_tokens, do_sample=False)[0]
        return [x for x in out.split("\n") if x.strip()]

    def answer(self, ex: dict, triples=None, rules=None, n_beam: int = 3) -> dict:
        graph = build_graph(ex["graph"] if triples is None else triples)
        rules = self.plan(ex["question"], n_beam) if rules is None else rules
        paths = self.retrieve(graph, ex["q_entity"], rules)
        strings = [path_to_string(p) for p in paths if len(p) > 0]
        pred = self.reason(ex["question"], strings) if strings else []
        return {"rules": rules, "n_paths": len(strings), "paths": strings, "prediction": pred}


class ChatReasoner(Rog):
    """Plug-and-play reasoner (the RoG paper's own setting): any chat LLM reads the retrieved reasoning paths and answers.
    Planning and retrieval are NOT done here (they come from the RoG planner); only `reason` is used."""

    def __init__(self, path: str = "Qwen/Qwen2.5-7B-Instruct", quant: str | None = "4bit", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        kw = {}
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map={"": 0}, **kw).eval()
        self.rng = random.Random(seed)

    def reason(self, question: str, path_strings: list[str], max_new_tokens: int = 200) -> list[str]:
        q = question if question.endswith("?") else question + "?"
        kept = self._fit_paths(q, path_strings)
        msg = SAQ_RULE_INSTRUCTION + "\n\n" + f"Reasoning Paths:\n{chr(10).join(kept)}\n\nQuestion:\n{q}"
        text = self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
        out = self._generate(text, max_new_tokens, do_sample=False)[0]
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", x).strip() for x in out.split("\n")]
        return [x for x in lines if x]


# ---------------------------------------------------------------- CLI
def load_data(spec: str, split: str):
    from datasets import load_dataset, load_from_disk
    return load_from_disk(spec) if os.path.isdir(spec) else load_dataset(spec, split=split)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="rmanluo/RoG-webqsp")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--model", default="rmanluo/RoG")
    p.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ds = load_data(a.data, a.split)
    idx = list(range(len(ds)))[: a.n] if a.n else list(range(len(ds)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["id"] for line in open(a.out, encoding="utf8")}
    model = Rog(a.model, None if a.quant == "none" else a.quant)
    from tqdm import tqdm
    with open(a.out, "a", encoding="utf8") as f:
        for i in tqdm(idx):
            ex = ds[i]
            if ex["id"] in done:
                continue
            r = model.answer(ex)
            f.write(json.dumps({"id": ex["id"], "question": ex["question"], "ground_truth": ex["answer"],
                                "rules": r["rules"], "n_paths": r["n_paths"], "prediction": r["prediction"],
                                **score(r["prediction"], ex["answer"])}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
