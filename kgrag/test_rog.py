from kgrag import rog

TRIPLES = [
    ["Lou Seal", "sports.mascot.team", "San Francisco Giants"],
    ["San Francisco Giants", "sports.team.championships", "2014 World Series"],
    ["San Francisco Giants", "sports.team.championships", "2012 World Series"],
    ["San Francisco Giants", "sports.team.location", "San Francisco"],
]


def test_bfs_follows_relation_path_in_order():
    g = rog.build_graph(TRIPLES)
    paths = rog.bfs_with_rule(g, "Lou Seal", ["sports.mascot.team", "sports.team.championships"])
    ends = sorted(p[-1][2] for p in paths)
    assert ends == ["2012 World Series", "2014 World Series"]


def test_bfs_wrong_relation_gives_nothing():
    g = rog.build_graph(TRIPLES)
    assert rog.bfs_with_rule(g, "Lou Seal", ["sports.team.location"]) == []


def test_path_to_string_matches_rog_format():
    g = rog.build_graph(TRIPLES)
    p = rog.bfs_with_rule(g, "Lou Seal", ["sports.mascot.team", "sports.team.location"])[0]
    assert rog.path_to_string(p) == ("Lou Seal -> sports.mascot.team -> San Francisco Giants"
                                     " -> sports.team.location -> San Francisco")


def test_parse_plan():
    assert rog.parse_plan("<PATH>a.b<SEP>c.d</PATH>") == ["a.b", "c.d"]
    assert rog.parse_plan("no tags here") is None
    assert rog.parse_plan("<PATH></PATH>") is None


def test_score_hit_f1():
    s = rog.score(["2014 World Series", "Something Else"], ["2014 World Series"])
    assert s["hit"] == 1.0 and abs(s["precision"] - 0.5) < 1e-9 and s["recall"] == 1.0
    assert abs(s["f1"] - 2 * 0.5 * 1.0 / 1.5) < 1e-9
    assert rog.score([], ["x"])["f1"] == 0.0
    assert rog.score(["the Answer!"], ["answer"])["hit"] == 1.0  # normalisation drops articles/punctuation
