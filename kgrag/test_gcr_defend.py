from kgrag import gcr_defend


def beam(path, answer):
    return f"# Reasoning Path:\n{path}\n# Answer:\n{answer}"


def test_parse_path_shapes():
    assert gcr_defend.parse_path("A -> r -> B") == [["A", "r", "B"]]
    assert gcr_defend.parse_path("A -> r1 -> B -> r2 -> C") == [["A", "r1", "B"], ["B", "r2", "C"]]
    assert gcr_defend.parse_path("A -> r") is None                       # even number of tokens
    assert gcr_defend.parse_path("just text") is None


def toy_example():
    graph = [["Q", "r1", "A"], ["A", "r2", "G"], ["W", "r2", "X"], ["Q", "r1", "W"]]   # clean triples first, the inserted one last (as the attack builds it)
    return {"id": "q1", "question": "what?", "answer": ["G"], "graph": graph, "q_entity": ["Q"],
            "poison_triples": [["Q", "r1", "W"]], "adv_answers": ["W"]}


def test_beam_record_labels_dedupes_and_features():
    ex = toy_example()
    beams = [beam("Q -> r1 -> A -> r2 -> G", "G"), beam("Q -> r1 -> W", "W"), beam("Q -> r1 -> A -> r2 -> G", "G"), "junk"]
    rec = gcr_defend.beam_record(ex, beams)
    texts = [p["text"] for p in rec["paths"]]
    assert texts == ["Q -> r1 -> A -> r2 -> G", "Q -> r1 -> W"]            # unique, in beam order, junk skipped
    clean, pois = rec["paths"]
    assert clean["gold_hit"] and not clean["poisoned"]
    assert pois["poisoned"] and not pois["gold_hit"]
    for k in ("end_deg", "min_deg", "end_rels", "cn_last", "cn_min", "jac_last", "jac_min", "triples", "text"):
        assert k in clean
    assert rec["adv"] == ["W"] and rec["gold"] == ["G"]


def test_clean_graph_has_no_poison_labels():
    ex = toy_example()
    ex.pop("poison_triples")
    rec = gcr_defend.beam_record(ex, [beam("Q -> r1 -> W", "W")])
    assert not rec["paths"][0]["poisoned"]
