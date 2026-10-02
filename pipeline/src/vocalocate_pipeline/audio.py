"""Audio cleaning: decode, check quality, resample, trim and normalise one clip."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import AudioConfig

AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}

_EPS = 1e-12


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, _EPS)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def trim_silence(x: np.ndarray, threshold_db: float) -> np.ndarray:
    """Drop leading and trailing samples quieter than (peak + threshold_db)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x
    loud = np.flatnonzero(np.abs(x) >= peak * 10 ** (threshold_db / 20))
    return x[loud[0] : loud[-1] + 1]


@dataclass
class CleanResult:
    status: str  # "ok" or "rejected"
    reject_reason: str | None = None
    audio: np.ndarray | None = None
    orig_sample_rate: int | None = None
    orig_channels: int | None = None
    orig_duration_sec: float | None = None
    duration_sec: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    truncated: bool = False
    clipped: bool = False
    flags: list[str] = field(default_factory=list)


def clean_file(path: Path, cfg: AudioConfig) -> CleanResult:
    """Decode a file and return a 32 kHz mono, trimmed, loudness-normalised clip."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several error types for bad files
        return CleanResult(status="rejected", reject_reason=f"decode_error: {type(exc).__name__}")
    return clean_array(data, sr, cfg)


def clean_array(data: np.ndarray, sr: int, cfg: AudioConfig) -> CleanResult:
    if data.ndim == 1:
        data = data[:, None]
    result = CleanResult(
        status="ok",
        orig_sample_rate=int(sr),
        orig_channels=int(data.shape[1]),
        orig_duration_sec=data.shape[0] / sr if sr else 0.0,
    )
    if data.shape[0] == 0 or not np.all(np.isfinite(data)):
        result.status, result.reject_reason = "rejected", "empty_or_invalid"
        return result

    # Measure clipping on the original channels, before mixing hides it.
    clipped_ratio = float(np.mean(np.abs(data) >= cfg.clip_threshold))
    result.clipped = clipped_ratio >= cfg.clipped_ratio_flag

    x = data.mean(axis=1).astype(np.float32)
    if dbfs(float(np.max(np.abs(x)))) < cfg.silence_peak_dbfs:
        result.status, result.reject_reason = "rejected", "silent"
        return result

    x = resample(x, sr, cfg.sample_rate)
    x = trim_silence(x, cfg.trim_db)

    max_len = int(cfg.max_duration_sec * cfg.sample_rate)
    if x.shape[0] > max_len:
        x = x[:max_len]
        result.truncated = True

    if x.shape[0] < cfg.min_duration_sec * cfg.sample_rate:
        result.status, result.reject_reason = "rejected", "too_short"
        return result

    # Loudness-normalise to a target RMS, but never push the peak past max_peak_dbfs.
    gain_db = cfg.target_rms_dbfs - dbfs(rms(x))
    peak_after = dbfs(float(np.max(np.abs(x)))) + gain_db
    if peak_after > cfg.max_peak_dbfs:
        gain_db -= peak_after - cfg.max_peak_dbfs
    x = (x * 10 ** (gain_db / 20)).astype(np.float32)

    result.audio = x
    result.duration_sec = x.shape[0] / cfg.sample_rate
    result.peak_dbfs = dbfs(float(np.max(np.abs(x))))
    result.rms_dbfs = dbfs(rms(x))
    if result.truncated:
        result.flags.append("truncated")
    if result.clipped:
        result.flags.append("clipped")
    return result


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), x, sr, subtype="PCM_16")
    tmp.replace(path)
