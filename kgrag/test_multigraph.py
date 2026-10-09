from kgrag import rog


TRIPLES = [["A", "r1", "B"], ["A", "r2", "B"], ["B", "r3", "C"]]     # two relations on the same entity pair (A, B)


def test_simple_graph_keeps_only_the_last_relation_per_pair():
    g = rog.build_graph(TRIPLES)
    assert rog.bfs_with_rule(g, "A", ["r1"]) == []                   # r1 was overwritten by r2
    assert rog.bfs_with_rule(g, "A", ["r2"]) == [[("A", "r2", "B")]]


def test_multigraph_keeps_every_relation():
    g = rog.build_graph(TRIPLES, multi=True)
    assert rog.bfs_with_rule(g, "A", ["r1"]) == [[("A", "r1", "B")]]
    assert rog.bfs_with_rule(g, "A", ["r2"]) == [[("A", "r2", "B")]]
    assert rog.bfs_with_rule(g, "A", ["r1", "r3"]) == [[("A", "r1", "B"), ("B", "r3", "C")]]


def test_default_graph_is_unchanged_and_not_multi():
    g = rog.build_graph(TRIPLES)
    assert not g.is_multigraph()
    assert rog.bfs_with_rule(g, "A", ["r2", "r3"]) == [[("A", "r2", "B"), ("B", "r3", "C")]]
