import numpy as np

from kgrag import conformal


def rec(paths):
    """paths: list of (poisoned, gold_hit)"""
    return {"paths": [{"poisoned": p, "gold_hit": g} for p, g in paths]}


def test_loss_is_one_only_when_all_clean_gold_paths_are_removed():
    recs = [rec([(False, True), (False, True), (True, False)]), rec([(False, False)])]
    scores = np.array([0.9, 0.2, 0.99, 0.5])
    grid = np.array([0.1, 0.5, 0.95])
    L = conformal.loss_matrix(recs, scores, grid)
    assert L.shape == (1, 3)                       # question 2 has no clean gold path -> not answerable, excluded
    assert L[0].tolist() == [1.0, 0.0, 0.0]        # lam=0.1: both gold paths removed; lam=0.5: path with score 0.2 survives


def test_poisoned_gold_paths_do_not_count_as_clean_support():
    recs = [rec([(True, True), (False, False)])]  # the only gold path is a planted one
    assert conformal.loss_matrix(recs, np.array([0.9, 0.1]), np.array([0.5])).shape == (0, 1)


def test_threshold_is_smallest_that_meets_the_bound_and_none_if_impossible():
    grid = np.array([0.1, 0.5, 0.9])
    losses = np.array([[1, 0, 0]] * 3 + [[0, 0, 0]] * 197, dtype=float)   # n=200, risk 0.015 at lam 0.1, 0 otherwise
    assert conformal.crc_threshold(losses, 0.05, grid) == 0.1              # 200/201*0.015+1/201 = 0.0199 <= 0.05
    assert conformal.crc_threshold(losses, 0.019, grid) == 0.5             # 0.0199 > 0.019 -> next lam (bound 0.005)
    assert conformal.crc_threshold(losses[:5], 0.05, grid) is None          # n=5 -> 1/6 > 0.05 can never hold
    assert conformal.crc_threshold(np.zeros((0, 3)), 0.05, grid) is None


def test_guarantee_holds_in_simulation():
    """Detector scores for a question's clean gold paths ~ Beta; true expected loss is known; mean realised loss <= alpha."""
    rng = np.random.default_rng(0)
    grid = conformal.GRID
    n_gold = 2

    def draw(n):  # n questions, each with n_gold clean gold paths with detector scores in [0,1]
        s = rng.beta(1.2, 3.0, size=(n, n_gold))
        return (s[:, :, None] > grid[None, None, :]).all(1).astype(float)

    alpha, realised = 0.1, []
    for _ in range(300):
        thr = conformal.crc_threshold(draw(150), alpha)
        k = np.searchsorted(grid, thr)
        realised.append(draw(4000)[:, k].mean())
    assert np.mean(realised) <= alpha + 0.005       # marginal guarantee (Monte-Carlo slack)
    assert np.mean(realised) > alpha * 0.5          # and it is not vacuous
