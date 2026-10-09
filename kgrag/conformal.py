"""Calibrated path removal: conformal risk control (Angelopoulos et al.) for the learned path detector.

Risk of a threshold lam on one answerable question (it has >= 1 clean gold-supporting retrieved path):
    loss = 1 if EVERY clean gold-supporting path is removed (detector probability > lam), else 0.
The loss is bounded in [0, 1] and non-increasing in lam. With n calibration questions exchangeable with a test question, the
smallest lam with   n/(n+1) * mean_loss(lam) + 1/(n+1) <= alpha   gives  E[loss on a new question] <= alpha.
The guarantee is marginal (averaged over calibration draws), concerns keeping correct evidence (not final accuracy), and holds
for the attack distribution represented in the calibration questions (calibrate on a mixture to cover several attacks).

Train / calibration questions come from the same labelled dev splits and never overlap: the first `n_train` questions of every
listed split train the detector, the remaining ones calibrate. The test split is only scored.
"""
import warnings

import numpy as np

from kgrag import defend, learned

GRID = np.concatenate([np.linspace(0.0, 0.2, 81)[1:], np.linspace(0.2, 1.0, 81)])  # candidate thresholds (probabilities)


def loss_matrix(recs, scores: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """(n_answerable_questions, len(grid)) 0/1 matrix: all clean gold-supporting paths removed at that threshold."""
    gold = defend.flat(recs, "gold_hit").astype(bool)
    y = defend.flat(recs, "poisoned").astype(int)
    qi = defend.qindex(recs)
    rows = []
    for q in range(len(recs)):
        m = (qi == q) & gold & (y == 0)
        if m.any():
            rows.append((scores[m][:, None] > grid[None, :]).all(0).astype(float))
    return np.array(rows).reshape(-1, len(grid))


def crc_threshold(losses: np.ndarray, alpha: float, grid: np.ndarray = GRID):
    """Smallest (most aggressive) threshold meeting the conformal bound, or None if even removing nothing cannot."""
    n = len(losses)
    if n == 0:
        return None
    ok = np.where(n / (n + 1) * losses.mean(0) + 1 / (n + 1) <= alpha)[0]
    return float(grid[ok[0]]) if len(ok) else None


def _split(split: str, n_train: int):
    recs = defend.load_paths(split)
    return recs[:n_train], recs[n_train:]


def removal_mask(dev_splits: list[str], test_split: str, alpha: float, n_train: int = 100, verbose: bool = True) -> np.ndarray:
    """Boolean mask over the flattened paths of `test_split`: True = remove. Falls back to removing nothing if no threshold is valid."""
    warnings.filterwarnings("ignore")
    tr, ca = [], []
    for s in dev_splits:
        a, b = _split(s, n_train)
        tr += a
        ca += b
    X = learned.features(tr)
    y = defend.flat(tr, "poisoned").astype(int)
    model = learned._model().fit(X, y)
    s_ca = model.predict_proba(learned.features(ca))[:, 1]
    losses = loss_matrix(ca, s_ca)
    thr = crc_threshold(losses, alpha)
    test = defend.load_paths(test_split)
    s_te = model.predict_proba(learned.features(test))[:, 1]
    if verbose:
        print(f"[crc] alpha={alpha} train_q={len(tr)} cal_answerable_q={len(losses)} threshold={thr}", flush=True)
    return np.zeros(len(s_te), dtype=bool) if thr is None else s_te > thr
