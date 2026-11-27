"""Log-mel features, computed the same way as the QVIM MobileNetV3 baseline (eval mode).

The baseline computes features on the GPU inside the model (with random
masking while training), so these cached features are for evaluation, for the
milestone-1 non-neural baseline and for quick checks. The training loader
still reads the cleaned audio from the silver layer.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import fft as sfft

from ..config import FeatureConfig

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


# A5 Part One (profiling): log_mel used to rebuild the filterbank and the window on every call.
# Both depend only on the settings, so they are built once per setting and reused. The cached
# arrays are read-only so a caller cannot corrupt the cache.
@lru_cache(maxsize=16)
def _filterbank_t(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """The mel filterbank transposed to (n_fft // 2 + 1, n_mels), C-contiguous for `power @ banks_t`."""
    banks_t = np.ascontiguousarray(mel_filterbank(n_mels, n_fft, sr, fmin, fmax).T)
    banks_t.flags.writeable = False
    return banks_t


@lru_cache(maxsize=16)
def _full_window(n_fft: int, win_length: int) -> np.ndarray:
    window = np.hanning(win_length).astype(np.float32)  # symmetric, like periodic=False
    offset = (n_fft - win_length) // 2
    full_window = np.zeros(n_fft, dtype=np.float32)
    full_window[offset : offset + win_length] = window
    full_window.flags.writeable = False
    return full_window


def log_mel(x: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    """Return a (n_mels, frames) log-mel spectrogram with the baseline's fast normalisation."""
    x = x.astype(np.float32)
    x = x[1:] - cfg.preemphasis * x[:-1] if x.shape[0] > 1 else x
    pad = cfg.n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect" if x.shape[0] > pad else "constant")

    n_frames = 1 + (x.shape[0] - cfg.n_fft) // cfg.hop_length
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg.n_fft)[:: cfg.hop_length][:n_frames]
    # scipy.fft is ~4x faster than numpy.fft on these float32 frames.
    power = np.abs(sfft.rfft(frames * _full_window(cfg.n_fft, cfg.win_length), axis=1)) ** 2

    fmax = cfg.fmax if cfg.fmax is not None else sr / 2
    banks_t = _filterbank_t(cfg.n_mels, cfg.n_fft, sr, float(cfg.fmin), float(fmax))
    # (frames, bins) @ (bins, mels) multiplies two C-contiguous arrays. The old `banks @ power.T`
    # read power transposed, which sent numpy down a ~20x slower matmul path.
    mel = (power @ banks_t).T
    return ((np.log(mel + 1e-5) + 4.5) / 5.0).astype(np.float32)


def summary_vector(mel: np.ndarray) -> np.ndarray:
    """Mean and standard deviation of each mel band: a fixed-size vector for the non-neural baseline."""
    return np.concatenate([mel.mean(axis=1), mel.std(axis=1)]).astype(np.float32)
