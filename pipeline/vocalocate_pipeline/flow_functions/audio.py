"""Audio cleaning: decode, check quality, resample, trim and normalise one clip."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..config import AudioConfig

AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}

_EPS = 1e-12

# Bit depth of each soundfile subtype. Lossy and compressed subtypes (MP3, VORBIS, OPUS, ...) have none.
_BIT_DEPTH = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}
NOISE_FRAME_SEC = 0.05
NOISE_PERCENTILE = 10.0


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


def noise_floor_dbfs(x: np.ndarray, sr: int) -> float:
    """Background-noise estimate: the 10th percentile of 50 ms frame RMS, in dBFS.

    Measured on the decoded mono signal before trimming and normalisation, so it describes the
    recording as made. The quietest tenth of the frames is usually the pauses around the sound,
    which a 10th percentile picks up without being thrown off by one dropout. A clip shorter than
    one frame uses its overall RMS.
    """
    n = max(1, int(sr * NOISE_FRAME_SEC))
    if x.shape[0] < n:
        return dbfs(rms(x))
    frames = x[: x.shape[0] // n * n].reshape(-1, n)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return dbfs(float(np.percentile(frame_rms, NOISE_PERCENTILE)))


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def mix_to_mono(data: np.ndarray) -> np.ndarray:
    """Average a (samples, channels) array to float32 mono.

    A5 Part One (profiling): `data.mean(axis=1)` reduces across the short, strided channel axis
    and took ~6 ms per 10 s stereo clip, a fifth of clean_array. Adding whole channel columns is
    a contiguous vector add per channel and gives the same result. It accumulates in the input's
    float dtype (float64 for integer input), as `mean` does, then casts to float32.
    """
    acc = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float64
    x = data[:, 0].astype(acc, copy=True)
    for c in range(1, data.shape[1]):
        x += data[:, c]
    if data.shape[1] > 1:
        x /= data.shape[1]
    return x.astype(np.float32, copy=False)


def trim_silence(x: np.ndarray, threshold_db: float) -> np.ndarray:
    """Drop leading and trailing samples quieter than (peak + threshold_db)."""
    if not x.size:
        return x
    mag = np.abs(x)  # computed once, used for both the peak and the threshold test
    peak = float(mag.max())
    if peak <= 0:
        return x
    loud = np.flatnonzero(mag >= peak * 10 ** (threshold_db / 20))
    return x[loud[0] : loud[-1] + 1]


@dataclass
class CleanResult:
    status: str  # "ok" or "rejected"
    reject_reason: str | None = None
    audio: np.ndarray | None = None
    orig_sample_rate: int | None = None
    orig_channels: int | None = None
    orig_duration_sec: float | None = None
    orig_format: str | None = None  # soundfile's container format, e.g. WAV, FLAC, MP3
    orig_bit_depth: int | None = None  # None for lossy formats
    noise_floor_dbfs: float | None = None
    duration_sec: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    truncated: bool = False
    clipped: bool = False
    flags: list[str] = field(default_factory=list)


def clean_file(path: Path, cfg: AudioConfig) -> CleanResult:
    """Decode a file and return a 32 kHz mono, trimmed, loudness-normalised clip."""
    try:
        with sf.SoundFile(str(path)) as f:
            fmt, subtype, sr = f.format, f.subtype, f.samplerate
            data = f.read(dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several error types for bad files
        return CleanResult(status="rejected", reject_reason=f"decode_error: {type(exc).__name__}")
    result = clean_array(data, sr, cfg)
    result.orig_format, result.orig_bit_depth = fmt, _BIT_DEPTH.get(subtype)
    return result


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

    x = mix_to_mono(data)
    result.noise_floor_dbfs = noise_floor_dbfs(x, sr)
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
