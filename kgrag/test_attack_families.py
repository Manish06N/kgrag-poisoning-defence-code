import random

from kgrag import attack


def chain_graph(n_ends=5, extra=0):
    """Q -r1-> A1..An, each Ai -r2-> G (gold). `extra` unrelated entities give the bridge sampler room."""
    g = [["Q", "r1", f"A{i}"] for i in range(n_ends)] + [[f"A{i}", "r2", "G"] for i in range(n_ends)]
    g += [[f"X{i}", "rx", f"X{i + 1}"] for i in range(extra)]
    return g


def test_paper_mode_is_the_default_and_unchanged():
    g = chain_graph()
    a = attack.poison_triples(g, ["Q"], [["r1", "r2"]], ["W1", "W2"], 4, seed=3)
    b = attack.poison_triples(g, ["Q"], [["r1", "r2"]], ["W1", "W2"], 4, seed=3, mode="paper", Ks=None)
    assert a == b
    assert all(t[1] == "r2" and t[0].startswith("A") for t in a)        # attached to grounded prefix endpoints


def test_per_answer_budgets_are_respected():
    g = chain_graph(n_ends=5)
    ins = attack.poison_triples(g, ["Q"], [["r1", "r2"]], ["W1", "W2"], 4, seed=1, Ks=[1, 3])
    assert sum(t[2] == "W1" for t in ins) == 1
    assert sum(t[2] == "W2" for t in ins) == 3


def test_spread_budgets_vary_and_never_exceed_the_total():
    seen = set()
    for s in range(200):
        ks = attack.spread_budgets(5, 4, random.Random(s))
        assert len(ks) == 5 and all(1 <= k <= 8 for k in ks) and sum(ks) <= 20
        seen.add(tuple(ks))
    assert len(seen) > 50                                               # counts are not constant across questions


def test_bridge_mode_builds_full_chains_from_inserted_triples_only():
    g = chain_graph(n_ends=3, extra=30)
    ins = attack.poison_triples(g, ["Q"], [["r1", "r2"]], ["W"], 4, seed=5, mode="bridge")
    assert ins and ins[-1][1] == "r2" and ins[-1][2] == "W"
    existing = {tuple(t) for t in g}
    assert not any(tuple(t) in existing for t in ins)                   # nothing re-uses an existing triple
    heads = {t[0] for t in ins}
    assert "Q" in heads                                                 # chains start at the topic entity


def test_profile_triples_raise_the_wrong_answer_degree_to_the_template():
    g = [["Q", "r1", "T"]] + [["T", rel, f"n{i}"] for i, rel in enumerate(["a", "b", "c", "d", "e"])] + [[f"m{i}", "z", f"m{i + 1}"] for i in range(40)]
    inserted = [["Q", "r1", "W"]]                                       # the paper-style chain already touches W once
    out = attack.profile_triples(g, inserted, ["W"], [["r1"]], ["Q"], seed=0)
    assert len(out) == 5                                                # template T has degree 6; W has 1 neighbour (Q) -> 5 more
    assert all("W" in (t[0], t[2]) for t in out)
    partners = [t[2] if t[0] == "W" else t[0] for t in out]
    assert len(set(partners)) == len(partners) and "Q" not in partners   # distinct, no overwrite of the existing edge
    assert {t[1] for t in out} <= {"a", "b", "c", "d", "e", "r1"}       # relations come from the template's own profile


def test_profile_triples_without_a_template_adds_nothing():
    assert attack.profile_triples([["A", "x", "B"]], [], ["W"], [["nope"]], ["A"]) == []
