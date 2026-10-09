from kgrag import pathdata


def example():
    clean = [["Q", "r", "A"], ["A", "r2", "G"]]
    inserted = [["A", "r", "Q"],                 # exact reverse of the clean triple (Q, r, A): an undirected duplicate
                ["A", "r3", "W"]]                # genuinely new triple towards a wrong answer
    return {"id": "q", "question": "?", "answer": ["G"], "q_entity": ["Q"], "graph": clean + inserted, "poison_triples": inserted,
            "adv_answers": ["W"]}


def test_strict_label_ignores_reversed_duplicates_but_keeps_real_insertions():
    rec = pathdata.build_record(example(), [["r", "r2"], ["r", "r3"]])
    by_text = {p["text"]: p for p in rec["paths"]}
    planted = by_text["Q -> r -> A -> r3 -> W"]
    assert planted["poisoned"] and planted["poisoned_strict"]                  # the new triple is poisoned under both labels


def test_reversed_duplicate_is_poisoned_loosely_but_not_strictly():
    ex = example()
    ex["poison_triples"] = [["A", "r", "Q"]]                                     # only the reverse duplicate
    ex["graph"] = [["Q", "r", "A"], ["A", "r2", "G"], ["A", "r", "Q"]]
    rec = pathdata.build_record(ex, [["r", "r2"]])
    p = rec["paths"][0]
    assert p["poisoned"] is True                                                 # the loose label flags the clean path
    assert p["poisoned_strict"] is False                                         # the strict label does not
