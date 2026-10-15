import numpy as np

from vocalocate_demo.data import SFXMeta, fit_frames, PAD_VALUE
from vocalocate_demo.evaluate import rank_metrics


def test_rank_metrics_exact_fine_and_graded():
    c = np.eye(4, dtype=np.float32)
    meta = [SFXMeta("a", frozenset(), "r0"), SFXMeta("a", frozenset(), "r0"), SFXMeta("b", frozenset(), "r1"),
            SFXMeta("a", frozenset(), "r2")]
    # Query 0 targets 0 and ranks it first; query 1 targets 3 but ranks 2 (other category) then 1 then 3.
    q = np.array([[1, 0, 0, 0], [0, 0.5, 0.9, 0.4]], dtype=np.float32)
    m = rank_metrics(q, c, np.array([0, 3]), meta)
    assert m["recall@1"] == 0.5 and m["recall@5"] == 1.0
    assert np.isclose(m["mrr"], (1 + 1 / 3) / 2)
    # Among category "a" only, query 1's target is second (behind 1).
    assert np.isclose(m["fine_mrr"], (1 + 1 / 2) / 2)
    assert 0 < m["ndcg@10"] < 1


def test_fit_frames_pads_and_crops():
    mel = np.ones((4, 5), dtype=np.float32)
    padded = fit_frames(mel, 8)
    assert padded.shape == (4, 8) and np.all(padded[:, 5:] == PAD_VALUE)
    assert fit_frames(np.arange(40, dtype=np.float32).reshape(4, 10), 3).shape == (4, 3)
