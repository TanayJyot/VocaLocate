import numpy as np

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.fusion import evaluate_fusion, fuse, tune_alpha, znorm
from vocalocate_demo.text import ClapScorer


def test_znorm_puts_scores_on_one_scale():
    z = znorm(np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]))
    assert np.allclose(z[0], z[1]) and np.allclose(z.mean(axis=1), 0)


def test_alpha_tuning_finds_the_useful_signal():
    rng = np.random.default_rng(0)
    n, m = 40, 30
    target = rng.integers(0, m, n)
    meta = [SFXMeta("a" if j % 2 else "b", frozenset(), f"r{j}") for j in range(m)]
    noise = rng.standard_normal((n, m))
    good = noise.copy()
    good[np.arange(n), target] += 5  # this signal finds the target
    # Voice is pure noise, text is good: tuning should lean on text (small alpha) and beat voice alone.
    alpha, curve = tune_alpha(noise, good, target, meta)
    assert alpha <= 0.3 and len(curve) == 11
    out = evaluate_fusion(noise, good, target, meta, noise, good, target, meta)
    assert out["test"]["fused"]["mrr"] > out["test"]["voice"]["mrr"]
    assert np.allclose(fuse(noise, good, 1.0), znorm(noise))


def test_fake_clap_is_deterministic_and_matches_words(tmp_path):
    clap = ClapScorer(backend="fake")
    a, b = clap.embed_text(["metallic short", "metallic short"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1)
    t = clap.embed_text(["metallic", "wooden"])
    assert t @ clap.embed_text(["metallic short"])[0] @ np.array([1, -1]) > 0
