"""A5 Part One: the optimised audio and feature functions match the versions they replaced.

Each `_reference_*` below is the pre-optimisation code, kept verbatim, so these tests fail if a
later "speed-up" changes what the pipeline writes to the lake. See a5/profiling/README.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocalocate_pipeline.config import FeatureConfig
from vocalocate_pipeline.flow_functions import features
from vocalocate_pipeline.flow_functions.audio import mix_to_mono, trim_silence


def _reference_log_mel(x: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    x = x.astype(np.float32)
    x = x[1:] - cfg.preemphasis * x[:-1] if x.shape[0] > 1 else x
    pad = cfg.n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect" if x.shape[0] > pad else "constant")
    window = np.hanning(cfg.win_length).astype(np.float32)
    offset = (cfg.n_fft - cfg.win_length) // 2
    full_window = np.zeros(cfg.n_fft, dtype=np.float32)
    full_window[offset : offset + cfg.win_length] = window
    n_frames = 1 + (x.shape[0] - cfg.n_fft) // cfg.hop_length
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg.n_fft)[:: cfg.hop_length][:n_frames]
    power = np.abs(np.fft.rfft(frames * full_window, axis=1)) ** 2
    fmax = cfg.fmax if cfg.fmax is not None else sr / 2
    banks = features.mel_filterbank(cfg.n_mels, cfg.n_fft, sr, cfg.fmin, fmax)
    mel = banks @ power.T
    return ((np.log(mel + 1e-5) + 4.5) / 5.0).astype(np.float32)


def _reference_trim(x: np.ndarray, threshold_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x
    loud = np.flatnonzero(np.abs(x) >= peak * 10 ** (threshold_db / 20))
    return x[loud[0] : loud[-1] + 1]


@pytest.mark.parametrize("n", [1, 600, 32_000, 97_531])
@pytest.mark.parametrize("cfg", [FeatureConfig(), FeatureConfig(fmax=None, n_mels=64)])
def test_log_mel_matches_the_pre_optimisation_version(n, cfg):
    x = (0.2 * np.random.default_rng(n).standard_normal(n)).astype(np.float32)
    got, want = features.log_mel(x, 32_000, cfg), _reference_log_mel(x, 32_000, cfg)
    assert got.shape == want.shape and got.dtype == np.float32
    # The only difference is float32 rounding order in the FFT and the matmul.
    np.testing.assert_allclose(got, want, atol=2e-4)


def test_log_mel_cache_is_read_only_and_keyed_by_settings():
    a = features._filterbank_t(128, 1024, 32_000, 0.0, 15_000.0)
    assert a is features._filterbank_t(128, 1024, 32_000, 0.0, 15_000.0)
    assert a is not features._filterbank_t(64, 1024, 32_000, 0.0, 15_000.0)
    with pytest.raises(ValueError):
        a[0, 0] = 1.0  # a caller must not be able to corrupt every later log_mel
    with pytest.raises(ValueError):
        features._full_window(1024, 800)[0] = 1.0


@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(0).standard_normal((1000, 2)).astype(np.float32),
        np.random.default_rng(1).standard_normal((1000, 6)).astype(np.float32),
        np.random.default_rng(2).standard_normal((1000, 1)).astype(np.float32),
        np.random.default_rng(3).standard_normal((1000, 2)),  # float64 input
        np.random.default_rng(4).integers(-32768, 32767, (1000, 2), dtype=np.int16),
    ],
    ids=["stereo", "5.1", "mono", "float64", "int16"],
)
def test_mix_to_mono_matches_mean(data):
    got = mix_to_mono(data)
    assert got.dtype == np.float32 and got.shape == (1000,)
    np.testing.assert_allclose(got, data.mean(axis=1).astype(np.float32), rtol=1e-6, atol=1e-6)


def test_mix_to_mono_does_not_modify_the_input():
    data = np.ones((10, 2), dtype=np.float32)
    mix_to_mono(data)
    assert np.all(data == 1)


@pytest.mark.parametrize(
    "x",
    [np.zeros(0, np.float32), np.zeros(100, np.float32), np.r_[np.zeros(50), np.ones(10), np.zeros(50)].astype(np.float32),
     np.random.default_rng(5).standard_normal(5000).astype(np.float32)],
    ids=["empty", "silent", "burst", "noise"],
)
def test_trim_silence_matches_the_pre_optimisation_version(x):
    np.testing.assert_array_equal(trim_silence(x, -50.0), _reference_trim(x, -50.0))
