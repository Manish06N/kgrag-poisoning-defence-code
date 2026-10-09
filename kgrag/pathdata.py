"""Retrieved reasoning paths per question, with ground-truth labels for which paths contain injected triples.

A defence acts between retrieval and reasoning: it sees the question and the retrieved paths and may drop some. Because we
know which triples the attacker inserted, we can measure how well a defence detects them WITHOUT running the LLM reasoner.

  python -m kgrag.pathdata --data data/poisoned_webqsp --plans runs/kg/clean_webqsp.jsonl --ids 0:500 --out runs/kg/paths_test.jsonl

"poisoned_strict": the path uses an inserted triple that is NOT already in the clean graph in either orientation (a reversed duplicate of a clean
triple makes "poisoned" label a clean path; the strict label does not). Used only for the oracle-gap analysis.
--multi retrieves on a MultiGraph (every parallel edge kept) instead of RoG's one-relation-per-pair graph; features still use the simple graph.
Record: {id, question, gold: [...], adv: [...], paths: [{"triples": [[h,r,t],...], "text": "h -> r -> t ...",
         "poisoned": bool (any triple was inserted), "gold_hit": bool (path ends in a gold answer)}]}
"""
import argparse
import json

from kgrag import rog


def injected_set(poison_triples) -> set:
    s = set()
    for h, r, t in poison_triples:
        s.add((h, r.strip(), t))
        s.add((t, r.strip(), h))  # the graph is undirected: a path may traverse an inserted triple backwards
    return s


def label_path(path, injected: set) -> bool:
    return any((h, r.strip(), t) in injected for h, r, t in path)


def build_record(ex: dict, rules: list, multi: bool = False) -> dict:
    graph = rog.build_graph(ex["graph"])
    ret_graph = rog.build_graph(ex["graph"], multi=True) if multi else graph
    ins_list = ex.get("poison_triples") or []
    inj = injected_set(ins_list)
    clean_part = ex["graph"][: len(ex["graph"]) - len(ins_list)]  # the poisoned graph is the clean triples followed by the inserted ones
    clean_und = {(h, r.strip(), t) for h, r, t in clean_part} | {(t, r.strip(), h) for h, r, t in clean_part}
    strict = injected_set([t for t in ins_list if (t[0], t[1].strip(), t[2]) not in clean_und])
    seen, paths = set(), []
    for p in rog.Rog.retrieve(ret_graph, ex["q_entity"], rules):
        if not p:
            continue
        text = rog.path_to_string(p)
        if text in seen:
            continue
        seen.add(text)
        end = p[-1][2]
        nodes = [t for _, _, t in p]  # every entity after the topic entity
        cn = [len(set(graph.neighbors(h)) & set(graph.neighbors(t))) for h, _, t in p]  # shared neighbours per edge
        jac = [cn_i / max(len(set(graph.neighbors(h)) | set(graph.neighbors(t))), 1) for cn_i, (h, _, t) in zip(cn, p)]
        paths.append({"triples": [list(t) for t in p], "text": text, "poisoned": label_path(p, inj), "poisoned_strict": label_path(p, strict),
                      "gold_hit": any(rog.match(end, g) for g in ex["answer"]),
                      "end_deg": graph.degree(end), "min_deg": min(graph.degree(n) for n in nodes),
                      "end_rels": len({graph[end][nb]["relation"] for nb in graph.neighbors(end)}),
                      "cn_last": cn[-1], "cn_min": min(cn), "jac_last": jac[-1], "jac_min": min(jac)})
    return {"id": ex["id"], "question": ex["question"], "gold": list(ex["answer"]), "adv": list(ex.get("adv_answers") or []),
            "paths": paths}


def load_plans(path: str) -> dict:
    plans = {}
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        plans[r["id"]] = r["rules"]
    return plans


def parse_ids(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        a, b = part.split(":")
        out.extend(range(int(a), min(int(b), n)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/poisoned_webqsp")
    p.add_argument("--plans", default="runs/kg/clean_webqsp.jsonl")
    p.add_argument("--ids", default="0:500", help="comma-separated index ranges, e.g. 0:500,1000:1300")
    p.add_argument("--out", required=True)
    p.add_argument("--multi", action="store_true", help="retrieve on a MultiGraph (ablation of the one-relation-per-pair graph)")
    a = p.parse_args()
    ds = rog.load_data(a.data, "test")
    plans = load_plans(a.plans)
    n_paths = n_pois = n_q = 0
    with open(a.out, "w", encoding="utf8") as f:
        for i in parse_ids(a.ids, len(ds)):
            ex = ds[i]
            rec = build_record(ex, plans.get(ex["id"], []), a.multi)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_q += 1
            n_paths += len(rec["paths"])
            n_pois += sum(x["poisoned"] for x in rec["paths"])
    print(f"{n_q} questions, {n_paths} paths ({n_paths / max(n_q, 1):.1f}/q), {n_pois} poisoned ({100 * n_pois / max(n_paths, 1):.1f}%)")


if __name__ == "__main__":
    main()
