from kgrag import attack, rog

GRAPH = [
    ["Lou Seal", "sports.mascot.team", "San Francisco Giants"],
    ["Lou Seal", "sports.mascot.team", "Giants (baseball)"],
    ["San Francisco Giants", "sports.team.championships", "2014 World Series"],
    ["San Francisco Giants", "sports.team.location", "San Francisco"],
    ["Texas Rangers", "sports.team.location", "Arlington"],
]


def test_one_hop_rule_attaches_wrong_answer_to_topic_entity():
    ins = attack.poison_triples(GRAPH, ["Lou Seal"], [["sports.mascot.team"]], ["Texas Rangers"], K=4)
    assert ["Lou Seal", "sports.mascot.team", "Texas Rangers"] in ins
    assert len(ins) <= 4


def test_two_hop_rule_grounds_prefix_then_attaches_answer():
    rule = ["sports.mascot.team", "sports.team.championships"]
    ins = attack.poison_triples(GRAPH, ["Lou Seal"], [rule], ["2010 World Series"], K=4)
    assert ["San Francisco Giants", "sports.team.championships", "2010 World Series"] in ins
    assert ["Giants (baseball)", "sports.team.championships", "2010 World Series"] in ins


def test_budget_is_per_adversarial_answer():
    ins = attack.poison_triples(GRAPH, ["Lou Seal"], [["sports.mascot.team"]], ["A", "B", "C"], K=2)
    for a in "ABC":
        assert sum(t[2] == a for t in ins) <= 2
    assert len(ins) <= 6


def test_fallback_builds_complete_chains_with_bridge_entities():
    rule = ["sports.mascot.team", "sports.team.championships"]
    ins = attack.poison_triples([["x", "y", "z"]], ["Lou Seal"], [rule], ["Wrong"], K=4, seed=1)
    ends = [t for t in ins if t[2] == "Wrong"]
    assert ends, "chain must end in the adversarial answer"
    heads = {t[0] for t in ins}
    assert "Lou Seal" in heads  # chains start at the topic entity
    assert len(ins) <= 4 and len(ins) % 2 == 0  # chains are atomic (2 triples each)


def test_never_inserts_triples_that_already_exist():
    ins = attack.poison_triples(GRAPH, ["Lou Seal"], [["sports.mascot.team"]], ["San Francisco Giants"], K=4)
    assert ["Lou Seal", "sports.mascot.team", "San Francisco Giants"] not in ins


def test_candidate_cleaning_and_matching():
    assert attack.clean_candidates("1. Paris\n- Rome\n\"Berlin\"\n") == ["Paris", "Rome", "Berlin"]
    idx = {"paris": "Paris", "rome": "Rome"}
    assert attack.match_entity("PARIS", idx, []) == "Paris"
    assert attack.match_entity("Hollywoodd", idx, ["Hollywood", "Oslo"]) == "Hollywood"  # fuzzy fallback, ratio 94.7 >= 90
    assert attack.match_entity("Romee", idx, ["Rome", "Oslo"]) is None  # ratio 88.9 < the paper's 0.9 cutoff
    assert attack.match_entity("Atlantis", idx, ["Rome"]) is None


def test_cvt_nodes_are_not_entity_names():
    assert attack.is_cvt("m.05n69q3") and not attack.is_cvt("San Francisco")


def test_attack_metrics():
    m = attack.attack_metrics(["Texas Rangers", "2014 World Series"], ["Texas Rangers", "Boston Red Sox"])
    assert m["a_hit1"] == 1.0 and m["a_precision"] == 0.5 and m["a_mrr"] == 1.0
    m2 = attack.attack_metrics(["2014 World Series", "Texas Rangers"], ["Texas Rangers"])
    assert m2["a_hit1"] == 0.0 and m2["a_mrr"] == 0.5
    assert attack.attack_metrics([], ["x"])["a_precision"] == 0.0
    assert rog.normalize("The Texas Rangers!") == "texas rangers"


def test_match_strips_state_suffix_after_comma():
    idx = {"baton rouge": "Baton Rouge"}
    assert attack.match_entity("Baton Rouge, Louisiana", idx, []) == "Baton Rouge"
    assert attack.match_entity("Nowhere, Texas", idx, []) is None


def test_pick_adversarial_excludes_believed_answers():
    idx = {"pat nixon": "Pat Nixon", "betty ford": "Betty Ford"}
    out = attack.pick_adversarial([["Pat Nixon", "Betty Ford"]], idx, [], 5, exclude=["Pat Nixon"])
    assert out == ["Betty Ford"]


def test_decoys_give_adversarial_answer_many_distinct_relations_and_stay_within_budget():
    graph = [[f"e{i}", f"rel{i % 12}", f"e{i + 1}"] for i in range(40)]
    out = attack.decoy_triples(graph, [], ["WrongA", "WrongB"], D=8, seed=1)
    for a in ("WrongA", "WrongB"):
        mine = [t for t in out if a in (t[0], t[2])]
        assert 1 <= len(mine) <= 8
        assert len({t[1] for t in mine}) == len(mine)  # distinct relation types
    assert all(tuple(t) not in {tuple(g) for g in graph} for t in out)


def test_near_decoys_create_shared_neighbours_with_the_grounding_entity():
    graph = [["Lou Seal", "mascot.of", "Giants"]] + [["Giants", f"has{i}", f"fan{i}"] for i in range(30)]
    ins = attack.poison_triples(graph, ["Lou Seal"], [["mascot.of", "wins"]], ["Wrong"], K=1, seed=2)
    assert ["Giants", "wins", "Wrong"] in ins
    near = attack.decoy_triples(graph, ins, ["Wrong"], D=8, seed=2, near=True)
    giants_nb = {f"fan{i}" for i in range(30)} | {"Lou Seal"}
    partners = [t[0] if t[2] == "Wrong" else t[2] for t in near]
    assert sum(p in giants_nb for p in partners) >= len(partners) // 2  # mostly neighbours of Giants -> shared neighbours
    far = attack.decoy_triples(graph, ins, ["Wrong"], D=8, seed=2, near=False)
    assert len(far) == len(near)


def test_decoys_are_deterministic_and_handle_empty_graph():
    graph = [["a", "r1", "b"], ["b", "r2", "c"]]
    assert attack.decoy_triples(graph, [], ["X"], 4, seed=3) == attack.decoy_triples(graph, [], ["X"], 4, seed=3)
    assert attack.decoy_triples([], [], ["X"], 4) == []
