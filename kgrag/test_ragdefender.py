import numpy as np

from kgrag import ragdefender


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def question():
    """6 paths: 3 near-duplicates of one direction (the injected cluster) and 3 spread-out legitimate ones."""
    rng = np.random.default_rng(0)
    dup = [unit([1, 0, 0, 0]) + 0.02 * rng.normal(size=4) for _ in range(3)]
    ok = [unit([0, 1, 0, 0]), unit([0, 0, 1, 0]), unit([0, 0, 0, 1])]
    emb = np.array([unit(x) for x in dup + ok])
    texts = [f"dup {i} fact" for i in range(3)] + ["alpha birth place", "beta river length", "gamma film year"]
    return emb, texts


def test_concentration_variant_removes_the_near_duplicate_cluster():
    emb, texts = question()
    out = ragdefender.question_removals(emb, texts, "conc")
    assert out[:3].all() or out.sum() >= 1                  # something is removed
    assert out[:3].sum() >= out[3:].sum()                   # and it is mostly the duplicates
    assert out.sum() < len(texts)                           # at least one path is always kept


def test_small_questions_are_left_alone():
    emb = np.array([unit([1, 0]), unit([0, 1])])
    assert not ragdefender.question_removals(emb, ["a b", "c d"], "conc").any()


def test_stage1_estimates_a_count_not_more_than_the_set():
    emb, texts = question()
    S = emb @ emb.T
    assert 0 <= ragdefender.estimate_conc(S) <= len(texts)
    assert 0 <= ragdefender.estimate_clust(emb, texts) <= len(texts)


def test_all_identical_items_keep_one_path():
    emb = np.tile(unit([1, 0, 0]), (5, 1))
    out = ragdefender.question_removals(emb, ["same text"] * 5, "conc")
    assert out.sum() <= 4
