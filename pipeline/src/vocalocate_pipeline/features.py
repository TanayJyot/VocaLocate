"""Log-mel features, computed the same way as the QVIM MobileNetV3 baseline (eval mode).

The baseline computes features on the GPU inside the model (with random
masking while training), so these cached features are for evaluation, for the
milestone-1 non-neural baseline and for quick checks. The training loader
still reads the cleaned audio from the silver layer.
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig

FEATURE_VERSION = "logmel-qvim-v1"


def _kaldi_mel(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filters on the Kaldi mel scale, as torchaudio's get_mel_banks builds them."""
    num_fft_bins = n_fft // 2
    bin_width = sr / n_fft
    mel_low, mel_high = _kaldi_mel(np.array(fmin)), _kaldi_mel(np.array(fmax))
    delta = (mel_high - mel_low) / (n_mels + 1)
    idx = np.arange(n_mels)[:, None]
    left = mel_low + idx * delta
    center = left + delta
    right = center + delta
    mel_f = _kaldi_mel(bin_width * np.arange(num_fft_bins))[None, :]
    up = (mel_f - left) / (center - left)
    down = (right - mel_f) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    # torchaudio pads one zero column for the Nyquist bin.
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


def log_mel(x: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    """Return a (n_mels, frames) log-mel spectrogram with the baseline's fast normalisation."""
    x = x.astype(np.float32)
    x = x[1:] - cfg.preemphasis * x[:-1] if x.shape[0] > 1 else x
    pad = cfg.n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect" if x.shape[0] > pad else "constant")

    window = np.hanning(cfg.win_length).astype(np.float32)  # symmetric, like periodic=False
    offset = (cfg.n_fft - cfg.win_length) // 2
    full_window = np.zeros(cfg.n_fft, dtype=np.float32)
    full_window[offset : offset + cfg.win_length] = window

    n_frames = 1 + (x.shape[0] - cfg.n_fft) // cfg.hop_length
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg.n_fft)[:: cfg.hop_length][:n_frames]
    power = np.abs(np.fft.rfft(frames * full_window, axis=1)) ** 2

    fmax = cfg.fmax if cfg.fmax is not None else sr / 2
    banks = mel_filterbank(cfg.n_mels, cfg.n_fft, sr, cfg.fmin, fmax)
    mel = banks @ power.T
    return ((np.log(mel + 1e-5) + 4.5) / 5.0).astype(np.float32)


def summary_vector(mel: np.ndarray) -> np.ndarray:
    """Mean and standard deviation of each mel band: a fixed-size vector for the non-neural baseline."""
    return np.concatenate([mel.mean(axis=1), mel.std(axis=1)]).astype(np.float32)
