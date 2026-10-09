"""Learned path-suspicion detector: gradient boosting on structural features of each retrieved path.

Features per path (all computed from the question's own retrieved subgraph, no LLM): log1p of answer-entity degree, min degree
along the path, relation variety at the answer entity, shared-neighbour counts / Jaccard on the path edges; the same values
z-scored within the question; path length; and how many of the question's paths end at the same entity / use the same last relation.
Trained ONLY on dev questions (never on the test questions); the removal threshold is set from out-of-fold dev scores so that
FPR (default 10%) of clean dev paths are removed.
Optional extra features: set `learned.EXTRA` to a callable recs -> (n_paths, k) array (e.g. kgrag.kge.extra_features); it is appended to
the columns of `features` everywhere (detector, conformal calibration, gate). Default None = the 19 structural features only.
"""
import warnings
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from kgrag import defend

BASE = ["end_deg", "min_deg", "end_rels", "cn_last", "cn_min", "jac_last", "jac_min"]
EXTRA = None  # optional callable: recs -> (n_paths, k) extra feature columns (see the module docstring)


def features(recs) -> np.ndarray:
    X = []
    for r in recs:
        P = r["paths"]
        if not P:
            continue
        n = len(P)
        ends = Counter(p["triples"][-1][2] for p in P)
        rel_cnt = Counter(p["triples"][-1][1] for p in P)
        lm = np.log1p(np.array([[p[k] for k in BASE] for p in P], float))
        z = (lm - lm.mean(0)) / (lm.std(0) + 1e-6)
        for i, p in enumerate(P):
            e = p["triples"][-1][2]
            X.append(list(lm[i]) + list(z[i]) + [len(p["triples"]), ends[e] / n, ends[e], rel_cnt[p["triples"][-1][1]] / n, n])
    X = np.array(X)
    return X if EXTRA is None or len(X) == 0 else np.hstack([X, EXTRA(recs)])


def _model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5, random_state=0)


def removal_mask(train_splits: list[str], test_split: str, fpr: float = 0.10) -> np.ndarray:
    """Boolean mask over the flattened paths of `test_split`: True = remove."""
    warnings.filterwarnings("ignore")
    recs = [defend.load_paths(s) for s in train_splits]
    X = np.vstack([features(r) for r in recs])
    y = np.concatenate([defend.flat(r, "poisoned").astype(int) for r in recs])
    oof = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        oof[va] = _model().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    thr = float(np.quantile(oof[y == 0], 1 - fpr))
    model = _model().fit(X, y)
    test = defend.load_paths(test_split)
    return model.predict_proba(features(test))[:, 1] > thr
