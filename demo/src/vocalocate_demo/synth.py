"""A procedural SFX dataset with vocal imitations and metadata, so the demo runs offline.

It stands in for the datasets in A2 Part Two until they are downloaded, and is
written in the `sfx_meta` layout so it goes through the real data pipeline
(ingest -> clean -> features -> warehouse) like any other source.

What it contains, mirroring the real data:

* 12 sound categories. Each has its own generator and three category tags;
  some tags are shared across categories (e.g. "impact", "air"), which is what
  the metadata-guided miner uses to build related pools.
* Recordings with random parameters (pitch, decay, rate, sweep). Each recording
  has a reference take plus a "distant" and a "pitched" variant, like the
  different takes of one Freesound recording. The variants are likely valid
  substitutes for the reference, which is the false-negative risk the miners
  have to handle.
* Distractor recordings with no imitations, like the extra same-class
  recordings of the Vocal Imitation Set or FSD50K used as a negative pool.
* Vocal imitations of each reference by several imitators. An imitation keeps
  the reference's loudness envelope, rhythm and pitch contour (folded into the
  voice range), but has a voice timbre, timing and pitch errors, background
  noise and a laptop-microphone sample rate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, medfilt, sosfilt

SFX_SR = 44_100
VOICE_SR = 16_000

CATEGORIES: dict[str, list[str]] = {
    "metal_impact": ["metallic", "impact", "ringing"],
    "wood_knock": ["wooden", "impact", "dry"],
    "glass_clink": ["glass", "impact", "bright"],
    "whoosh": ["air", "swoosh", "movement"],
    "wind": ["air", "ambience", "noisy"],
    "alarm_beep": ["electronic", "beep", "tonal"],
    "siren": ["tonal", "alarm", "vehicle"],
    "laser_zap": ["electronic", "sci-fi", "sweep"],
    "engine": ["mechanical", "vehicle", "rumble"],
    "rattle": ["mechanical", "rattle", "noisy"],
    "water_drip": ["water", "drip", "tonal"],
    "door_creak": ["wooden", "creak", "friction"],
}

# Vowel formants (F1, F2, F3) used for the imitators' voices.
VOWELS = {"a": (800, 1200, 2500), "o": (500, 900, 2400), "u": (350, 800, 2300), "e": (450, 1900, 2600)}


@dataclass
class Recording:
    recording_id: str
    category: str
    params: dict
    imitated: bool


def _rng_params(category: str, rng: np.random.Generator) -> dict:
    """Random parameters for one recording. Siblings in a category differ only in these."""
    u = rng.uniform
    p = {
        "dur": u(1.0, 2.4),
        "f0": float(np.exp(u(np.log(150), np.log(1500)))),
        "decay": u(4, 30),
        "rate": u(2, 9),
        "sweep": u(-1, 1),
        "n_hits": int(rng.integers(1, 4)),
    }
    if category == "engine":
        p["f0"] = u(35, 110)
    if category == "glass_clink":
        p["f0"] = u(1500, 4500)
    if category in ("wind", "whoosh"):
        p["f0"] = u(300, 3000)
    return p


def _env_hits(n: int, t: np.ndarray, starts: list[float], decay: float) -> np.ndarray:
    env = np.zeros(n)
    for s in starts:
        mask = t >= s
        env[mask] += np.exp(-(t[mask] - s) * decay)
    return env


def _bandnoise(n: int, lo: float, hi: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = max(lo, 30.0), min(hi, sr / 2 - 100)
    sos = butter(2, [lo, max(hi, lo * 1.2)], btype="band", fs=sr, output="sos")
    return sosfilt(sos, rng.standard_normal(n))


def render_sfx(category: str, p: dict, rng: np.random.Generator, sr: int = SFX_SR) -> np.ndarray:
    n = int(p["dur"] * sr)
    t = np.arange(n) / sr
    f0, decay, rate, sweep = p["f0"], p["decay"], p["rate"], p["sweep"]
    hits = [0.05 + i / rate for i in range(p["n_hits"])]
    two_pi = 2 * np.pi

    if category == "metal_impact":
        env = _env_hits(n, t, hits, decay / 4)
        x = sum(np.sin(two_pi * f0 * r * t) / (k + 1) for k, r in enumerate([1, 2.76, 5.40, 8.93])) * env
    elif category == "wood_knock":
        env = _env_hits(n, t, hits, decay * 1.2)
        x = (np.sin(two_pi * f0 * 0.5 * t) + 0.6 * _bandnoise(n, f0, f0 * 4, sr, rng)) * env
    elif category == "glass_clink":
        env = _env_hits(n, t, hits, decay)
        x = sum(np.sin(two_pi * f0 * r * t) for r in [1, 1.51, 2.33]) * env
    elif category == "whoosh":
        center = f0 * 2 ** (sweep * 2 * (t / t[-1] - 0.5))
        bell = np.exp(-(((t / t[-1]) - 0.5) ** 2) / 0.04)
        noise = rng.standard_normal(n)
        x = np.zeros(n)
        seg = n // 8
        for i in range(8):  # piecewise band-pass to follow the sweep
            sl = slice(i * seg, n if i == 7 else (i + 1) * seg)
            c = float(center[sl].mean())
            sos = butter(2, [c / 1.5, min(c * 1.5, sr / 2 - 100)], btype="band", fs=sr, output="sos")
            x[sl] = sosfilt(sos, noise[sl])
        x *= bell
    elif category == "wind":
        b, a = butter(2, min(f0, sr / 2 - 100), fs=sr)
        x = lfilter(b, a, rng.standard_normal(n)) * (0.6 + 0.4 * np.sin(two_pi * rate / 6 * t))
    elif category == "alarm_beep":
        gate = (np.sin(two_pi * rate * t) > 0).astype(float)
        x = np.sign(np.sin(two_pi * f0 * t)) * gate * 0.5
    elif category == "siren":
        f = f0 * (1 + 0.25 * np.sin(two_pi * rate / 4 * t))
        x = np.sin(two_pi * np.cumsum(f) / sr)
    elif category == "laser_zap":
        f = f0 * 2 * np.exp(-t * decay / 4) + 80
        env = _env_hits(n, t, hits, decay / 3)
        x = np.sin(two_pi * np.cumsum(f) / sr) * env
    elif category == "engine":
        ph = two_pi * np.cumsum(f0 * (1 + 0.1 * sweep * t)) / sr
        x = sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.7 + 0.3 * np.sin(two_pi * rate * t))
    elif category == "rattle":
        clicks = (rng.random(n) < rate * 8 / sr).astype(float)
        x = lfilter(*butter(2, [f0, min(f0 * 3, sr / 2 - 100)], btype="band", fs=sr), clicks) * 20
        x *= np.exp(-t * decay / 20)
    elif category == "water_drip":
        x = np.zeros(n)
        for s in hits + [s + 0.5 / rate for s in hits]:
            m = (t >= s) & (t < s + 0.08)
            tt = t[m] - s
            x[m] += np.sin(two_pi * np.cumsum(f0 * (1 + 8 * tt)) / sr) * np.exp(-tt * 40)
    elif category == "door_creak":
        f = f0 / 3 * (1 + 0.3 * np.sin(two_pi * rate / 5 * t + sweep))
        saw = 2 * ((np.cumsum(f) / sr) % 1) - 1
        x = saw * (0.5 + 0.5 * np.abs(np.sin(two_pi * rate * 3 * t)))
    else:
        raise ValueError(category)
    x = np.asarray(x, dtype=np.float64)
    fade = min(n // 20, int(0.01 * sr))
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return (0.5 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def make_variant(x: np.ndarray, kind: str, rng: np.random.Generator, sr: int = SFX_SR) -> np.ndarray:
    """Another take of the same recording: distant (filtered, reverberant) or pitched."""
    if kind == "distant":
        b, a = butter(2, 2500, fs=sr)
        y = lfilter(b, a, x)
        ir = rng.standard_normal(int(0.25 * sr)) * np.exp(-np.arange(int(0.25 * sr)) / (0.06 * sr))
        y = fftconvolve(y, ir * 0.05)[: len(x)] + 0.3 * y
    elif kind == "pitched":
        factor = 2 ** (rng.choice([-1.5, 1.5]) / 12)
        idx = np.arange(0, len(x) - 1, factor)
        y = np.interp(idx, np.arange(len(x)), x)
    else:
        raise ValueError(kind)
    return (0.5 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def _frame_contours(x: np.ndarray, sr: int, hop: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame loudness, dominant frequency and noisiness (spectral flatness) of a sound."""
    h = int(hop * sr)
    win = 2048
    xp = np.pad(x, (win // 2, win // 2))
    frames = np.lib.stride_tricks.sliding_window_view(xp, win)[::h][: len(x) // h]
    mag = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (freqs > 50) & (freqs < 8000)
    loud = np.sqrt(np.mean(frames**2, axis=1))
    dom = freqs[band][np.argmax(mag[:, band], axis=1)]
    flat = np.exp(np.mean(np.log(mag[:, band]), axis=1)) / np.mean(mag[:, band], axis=1)
    return loud, medfilt(dom, 7), medfilt(flat, 7)


def render_imitation(ref: np.ndarray, imitator: dict, rng: np.random.Generator, sr_in: int = SFX_SR) -> np.ndarray:
    """A voice-like imitation of `ref`: same envelope, rhythm and pitch shape, different timbre."""
    loud, dom, flat = _frame_contours(ref, sr_in)
    stretch = rng.uniform(0.8, 1.25)
    n_frames = max(int(len(loud) * stretch), 10)
    src = np.linspace(0, len(loud) - 1, n_frames)
    loud = np.interp(src, np.arange(len(loud)), loud)
    dom = np.interp(src, np.arange(len(dom)), dom)
    flat = np.interp(src, np.arange(len(flat)), flat)
    loud = np.convolve(loud, np.ones(4) / 4, mode="same")  # people cannot follow very fast detail

    sr = VOICE_SR
    n = int(n_frames * 0.01 * sr)
    pos = np.linspace(0, n_frames - 1, n)
    env = np.interp(pos, np.arange(n_frames), loud)
    env /= env.max() + 1e-9
    # Fold the dominant frequency into the imitator's range by octaves, keeping its shape.
    lo, hi = imitator["range"]
    f = np.maximum(dom, 40.0) * 2 ** (rng.normal(0, 1.5) / 12)
    f = lo * 2 ** (np.log2(f / lo) % np.log2(hi / lo))
    f = np.interp(pos, np.arange(n_frames), medfilt(f, 9))
    noisiness = np.clip(np.interp(pos, np.arange(n_frames), flat) * 2.5, 0.05, 0.95)

    phase = 2 * np.pi * np.cumsum(f) / sr
    if imitator["style"] == "whistle":
        voiced = np.sin(phase)
    else:
        voiced = sum(np.sin(k * phase) / k for k in range(1, 16))
        for fc in VOWELS[imitator["vowel"]]:
            fc = fc * imitator["formant_shift"]
            sos = butter(2, [fc / 1.3, min(fc * 1.3, sr / 2 - 100)], btype="band", fs=sr, output="sos")
            voiced = voiced + 2 * sosfilt(sos, voiced)
    voiced /= np.max(np.abs(voiced)) + 1e-9
    breath = _bandnoise(n, 1500, 6000, sr, rng)
    breath /= np.max(np.abs(breath)) + 1e-9
    y = ((1 - noisiness) * voiced + noisiness * breath) * env

    lead = np.zeros(int(rng.uniform(0.05, 0.3) * sr))
    y = np.concatenate([lead, y, np.zeros(int(0.1 * sr))])
    snr_db = rng.uniform(15, 35)
    noise = rng.standard_normal(len(y)) * np.sqrt(np.mean(y**2)) * 10 ** (-snr_db / 20)
    y = y + noise
    return (0.4 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def _tags(category: str, p: dict) -> list[str]:
    tags = list(CATEGORIES[category])
    tags.append("short" if p["dur"] < 1.6 else "long")
    if category not in ("wind", "whoosh"):
        tags.append("high-pitched" if p["f0"] > 600 else "low-pitched")
    return tags


def generate(
    out: Path,
    recordings_per_category: int = 14,
    distractors_per_category: int = 4,
    imitations_per_reference: int = 5,
    n_imitators: int = 8,
    categories: list[str] | None = None,
    seed: int = 490,
) -> Path:
    """Write the dataset to `out` in the sfx_meta layout and return the folder."""
    rng = np.random.default_rng(seed)
    cats = categories or list(CATEGORIES)
    imitators = []
    for i in range(n_imitators):
        style = "whistle" if i % 4 == 3 else "voice"
        rng_range = (500.0, 2000.0) if style == "whistle" else (rng.uniform(80, 140), rng.uniform(380, 600))
        imitators.append(
            {
                "id": f"imitator_{i:02d}",
                "style": style,
                "range": rng_range,
                "vowel": str(rng.choice(list(VOWELS))),
                "formant_shift": rng.uniform(0.85, 1.2),
            }
        )

    recs = []
    for cat in cats:
        for j in range(recordings_per_category + distractors_per_category):
            recs.append(Recording(f"{cat}_{j:03d}", cat, _rng_params(cat, rng), j < recordings_per_category))

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in recs:
        x = render_sfx(rec.category, rec.params, rng)
        tags = ";".join(_tags(rec.category, rec.params))
        desc = f"{rec.category.replace('_', ' ')} ({rec.params['dur']:.1f} s)"
        takes = [("reference" if rec.imitated else "library", "A", x)]
        takes += [("library", "B_distant", make_variant(x, "distant", rng))]
        takes += [("library", "C_pitched", make_variant(x, "pitched", rng))]
        for role, take, audio in takes:
            sound_id = f"{rec.recording_id}_{take}"
            path = f"sfx/{rec.category}/{sound_id}.wav"
            # Some SFX are stereo, as in real libraries; the pipeline mixes them down.
            stereo = rng.random() < 0.3
            data = np.stack([audio, audio * 0.9], axis=1) if stereo else audio
            sf.write(str(_mk(out / path)), data, SFX_SR)
            rows.append([path, role, sound_id, "", rec.category, tags, rec.recording_id, desc, ""])
        if rec.imitated:
            for k, who in enumerate(rng.choice(len(imitators), imitations_per_reference, replace=False)):
                imitator = imitators[who]
                y = render_imitation(x, imitator, rng)
                path = f"imitations/{imitator['id']}/{rec.recording_id}_{k}.wav"
                sf.write(str(_mk(out / path)), y, VOICE_SR)
                rows.append([path, "imitation", "", f"{rec.recording_id}_A", rec.category, "", "", "", imitator["id"]])

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["file", "role", "sound_id", "imitates", "category", "tags", "recording_id", "description", "imitator_id"]
        )
        w.writerows(rows)
    return out


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
