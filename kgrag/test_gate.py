import numpy as np

from kgrag import gate


def test_question_score_is_mean_of_top_k_and_zero_for_empty_questions():
    sizes = np.array([4, 0, 2])
    scores = np.array([0.1, 0.9, 0.5, 0.7,   0.2, 0.4])
    out = gate.question_scores(sizes, scores, topk=3)
    assert np.allclose(out, [(0.9 + 0.7 + 0.5) / 3, 0.0, 0.3])       # fewer than k paths -> mean of what exists


def test_threshold_flags_about_q_fpr_of_clean_questions():
    clean = np.random.default_rng(0).uniform(size=5000)
    thr = gate.gate_threshold(clean, 0.05)
    assert abs((clean >= thr).mean() - 0.05) < 0.005


def test_gate_keeps_path_removals_only_in_flagged_questions():
    sizes = np.array([3, 2, 1])
    path_mask = np.array([True, False, True,   True, True,   True])
    flagged = np.array([True, False, True])
    out = gate.apply_gate(path_mask, sizes, flagged)
    assert out.tolist() == [True, False, True,   False, False,   True]   # question 2 unflagged -> nothing removed


def test_unflagged_everywhere_removes_nothing_and_flagged_everywhere_is_the_plain_filter():
    sizes = np.array([2, 2])
    mask = np.array([True, False, False, True])
    assert not gate.apply_gate(mask, sizes, np.array([False, False])).any()
    assert gate.apply_gate(mask, sizes, np.array([True, True])).tolist() == mask.tolist()
