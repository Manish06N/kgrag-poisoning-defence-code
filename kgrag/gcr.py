"""GCR (Luo et al., 2024) candidate paths and a path filter in front of its graph-constrained decoding.

GCR does not retrieve a few planned paths like RoG. It indexes ALL directed paths of up to `index_path_length` (=2) hops that
start at the topic entities (utils.dfs in the reference code) in a prefix trie, and the fine-tuned LLM may only generate paths
present in the trie. A poisoning defence therefore has to act on this candidate set before the trie is built.

  python -m kgrag.gcr paths --data data/poisoned_webqsp --ids 0:500 --out runs/kg/gcrpaths_test.jsonl

Records follow kgrag.pathdata (so kgrag.learned / kgrag.defend work unchanged): per path `triples`, `text`, `poisoned`, `gold_hit`
and the structural features end_deg, min_deg, end_rels, cn_last, cn_min, jac_last, jac_min (computed on the undirected graph).
"""
import argparse
import json

import networkx as nx

from kgrag import pathdata, rog


def dfs_paths(triples, q_entities, max_len: int = 2) -> list[list[tuple]]:
    """All directed paths of length <= max_len from the topic entities (same set as GCR's utils.dfs)."""
    g = nx.DiGraph()
    for h, r, t in triples:
        g.add_edge(h.strip(), t.strip(), relation=r.strip())
    out = set()

    def visit(node, path):
        if node not in g:
            return
        for nb in g.neighbors(node):
            new = path + [(node, g[node][nb]["relation"], nb)]
            out.add(tuple(new))
            if len(new) < max_len:
                visit(nb, new)

    for e in q_entities:
        visit(e, [])
    return [list(p) for p in out]


def build_record(ex: dict, max_len: int = 2) -> dict:
    ug = rog.build_graph(ex["graph"])
    nbr = {n: set(ug.neighbors(n)) for n in ug.nodes}
    inj = pathdata.injected_set(ex.get("poison_triples") or [])
    paths = []
    for p in sorted(dfs_paths(ex["graph"], ex["q_entity"], max_len)):
        end = p[-1][2]
        cn = [len(nbr.get(h, set()) & nbr.get(t, set())) for h, _, t in p]
        jac = [c / max(len(nbr.get(h, set()) | nbr.get(t, set())), 1) for c, (h, _, t) in zip(cn, p)]
        paths.append({"triples": [list(t) for t in p], "text": rog.path_to_string(p),
                      "poisoned": pathdata.label_path(p, inj),
                      "gold_hit": any(rog.match(end, g) for g in ex["answer"]),
                      "end_deg": ug.degree(end) if end in ug else 0,
                      "min_deg": min(ug.degree(t) if t in ug else 0 for _, _, t in p),
                      "end_rels": len({ug[end][nb]["relation"] for nb in ug.neighbors(end)}) if end in ug else 0,
                      "cn_last": cn[-1], "cn_min": min(cn), "jac_last": jac[-1], "jac_min": min(jac)})
    return {"id": ex["id"], "question": ex["question"], "gold": list(ex["answer"]),
            "adv": list(ex.get("adv_answers") or []), "paths": paths}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["paths"])
    p.add_argument("--data", default="data/poisoned_webqsp")
    p.add_argument("--ids", default="0:500")
    p.add_argument("--max_len", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ds = rog.load_data(a.data, "test")
    nq = npaths = npois = 0
    with open(a.out, "w", encoding="utf8") as f:
        for i in pathdata.parse_ids(a.ids, len(ds)):
            rec = build_record(ds[i], a.max_len)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            nq += 1
            npaths += len(rec["paths"])
            npois += sum(x["poisoned"] for x in rec["paths"])
    print(f"{nq} questions, {npaths} paths ({npaths / max(nq, 1):.0f}/q), {npois} poisoned ({100 * npois / max(npaths, 1):.1f}%)")


if __name__ == "__main__":
    main()
