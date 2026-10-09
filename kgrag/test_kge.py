import numpy as np

from kgrag import kge


def toy():
    # 1-d TransE: a=0, b=1, c=5, relation r = +1  ->  (a, r, b) is perfectly plausible, (a, r, c) is not
    return kge.Scorer(np.array([[0.0], [1.0], [5.0]]), np.array([[1.0]]), {"a": 0, "b": 1, "c": 2}, {"r": 0}, fill=-9.0)


def test_triple_plausibility_and_unknowns():
    s = toy().triples([["a", "r", "b"], ["a", "r", "c"], ["a", "r", "zzz"], ["a", "unknown_rel", "b"]])
    assert s[0] == 0.0 and s[1] == -4.0
    assert np.isnan(s[2]) and np.isnan(s[3])


def test_path_features_min_mean_unknown_share_and_question_z():
    recs = [{"paths": [{"triples": [["a", "r", "b"]]}, {"triples": [["a", "r", "c"], ["a", "r", "zzz"]]}]},
            {"paths": []},
            {"paths": [{"triples": [["x", "r", "y"]]}]}]                       # nothing scored -> fill value
    f = toy().paths(recs)
    assert f.shape == (3, 4)
    assert f[0].tolist()[:3] == [0.0, 0.0, 0.0]
    assert f[1].tolist()[:3] == [-4.0, -4.0, 0.5]                              # half of its triples are unscored
    assert np.allclose(f[:2, 3], [1.0, -1.0], atol=1e-3)                       # z-score of the min inside the question
    assert f[2, 0] == -9.0 and f[2, 2] == 1.0


def test_ids_and_dedupe():
    ent, rel, H, R, T = kge.build_ids([[["a", "r", "b"], ["a", "r ", "b"], ["b", "r", "c"]]])   # "r " is stripped to "r"
    assert len(ent) == 3 and len(rel) == 1
    h, r, t = kge.dedupe(H, R, T, len(rel), len(ent))
    assert len(h) == 2                                                          # the duplicate (a, r, b) is dropped
