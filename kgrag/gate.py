"""Question-level gate: remove suspicious paths only in questions that look attacked.

The always-on path filter (learned.py) costs about 6 F1 on clean graphs because it removes some correct evidence in every question.
The gate scores each QUESTION by the mean of its top-k path suspicion scores and applies the path filter only when that score
reaches a threshold chosen so that at most `q_fpr` of clean development questions are flagged.

The gate detector is trained on the first `n_train` questions of the poisoned dev splits plus the first `n_train` questions of the
clean dev split; the threshold comes from the remaining (held-out) clean dev questions, so no test question is used for fitting.

  python -m kgrag.defended_run --defence gate:dev --fpr 0.02 --qfpr 0.05            # path filter learned:dev, gated per question
"""
import warnings

import numpy as np

from kgrag import defend, learned


def question_scores(sizes: np.ndarray, scores: np.ndarray, topk: int = 3) -> np.ndarray:
    """Mean of each question's top-k path scores; 0 for a question with no paths. `scores` is flat over the questions' paths in order."""
    out = np.zeros(len(sizes))
    start = 0
    for q, n in enumerate(sizes):
        if n:
            v = np.sort(scores[start:start + n])[::-1]
            out[q] = v[:topk].mean()
        start += n
    return out


def gate_threshold(clean_question_scores: np.ndarray, q_fpr: float) -> float:
    """Smallest score such that at most about q_fpr of clean questions score at or above it."""
    return float(np.quantile(clean_question_scores, 1.0 - q_fpr))


def apply_gate(path_mask: np.ndarray, sizes: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Keep the path-level removal mask only inside flagged questions (all other questions keep every path)."""
    return np.asarray(path_mask, bool) & np.repeat(np.asarray(flagged, bool), sizes)


def _sizes(recs) -> np.ndarray:
    return np.array([len(r["paths"]) for r in recs])


def fit(poisoned_splits: list[str], clean_split: str = "clean_dev", n_train: int = 100, q_fpr: float = 0.05, topk: int = 3):
    """Train the gate detector and choose its threshold. Returns (model, threshold)."""
    warnings.filterwarnings("ignore")
    Xs, ys = [], []
    for split in poisoned_splits + [clean_split]:
        recs = defend.load_paths(split)
        m = defend.qindex(recs) < n_train
        Xs.append(learned.features(recs)[m])
        ys.append(defend.flat(recs, "poisoned").astype(int)[m])
    model = learned._model().fit(np.vstack(Xs), np.concatenate(ys))
    held = defend.load_paths(clean_split)[n_train:]
    sizes = _sizes(held)
    s = model.predict_proba(learned.features(held))[:, 1]
    q = question_scores(sizes, s, topk)[sizes > 0]
    return model, gate_threshold(q, q_fpr)


def removal_mask(path_mask: np.ndarray, poisoned_splits: list[str], test_split: str, q_fpr: float = 0.05, topk: int = 3,
                 n_train: int = 100) -> np.ndarray:
    """Gate a path-level removal mask on `test_split`: paths are removed only in questions whose top-k score reaches the threshold."""
    model, thr = fit(poisoned_splits, "clean_dev", n_train, q_fpr, topk)
    test = defend.load_paths(test_split)
    sizes = _sizes(test)
    qs = question_scores(sizes, model.predict_proba(learned.features(test))[:, 1], topk)
    return apply_gate(path_mask, sizes, qs >= thr)
