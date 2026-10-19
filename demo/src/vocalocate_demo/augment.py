"""On-the-fly augmentation for training, following the QVIM 2025 winning entry (Bhattacharjee et al.).

Their pool, applied independently to query and reference, at most one transform per sample:
time shift (±30%), gain (±10 dB), pitch shift (±3 semitones), time stretch (0.8-1.2x),
additive Gaussian noise (amplitude 0.001-0.015), frame-level corruption (duplicate, silence or
drop frames) and frequency/time masking.

They augment the waveform before the log-mel. We augment the pipeline's cached log-mel
features instead, on the training device, so training never decodes audio. Each transform is
the log-mel counterpart of theirs:

* shift: circular roll along time.
* gain: exact. Features are (ln(mel power) + 4.5) / 5, so g dB adds g * ln(10) / 50.
* pitch shift: a shift along the mel axis. Approximate: mel bins are close to log-spaced
  above ~1 kHz, where a semitone is about MEL_BINS_PER_SEMITONE bins; lower down the shift is
  slightly too large.
* time stretch: linear resampling along time, then crop or pad back to the input length.
* noise: Gaussian noise added in the feature domain, scaled from their waveform amplitude.
  Approximate: waveform noise raises a power floor, which this does not model exactly.
* frame corruption: ~10% of frames duplicated from their neighbour, silenced or dropped.
* masking: one frequency band (up to 16 bins) and one time span (up to 10%) set to silence.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import PAD_VALUE

TRANSFORMS = ("none", "shift", "gain", "pitch", "stretch", "noise", "frames", "mask")
# 128 mel bins over 0-16 kHz span about 100 semitones above ~50 Hz.
MEL_BINS_PER_SEMITONE = 1.28


def _shift(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    t = x.shape[-1]
    k = int(torch.randint(-int(0.3 * t), int(0.3 * t) + 1, (1,), generator=g))
    return torch.roll(x, k, dims=-1)


def _gain(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    db = float(torch.empty(1).uniform_(-10, 10, generator=g))
    return x + db * math.log(10) / 50


def _pitch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    semis = float(torch.empty(1).uniform_(-3, 3, generator=g))
    shift = semis * MEL_BINS_PER_SEMITONE
    m = x.shape[0]
    src = torch.arange(m, device=x.device, dtype=x.dtype) - shift
    lo = src.floor().long()
    w = (src - lo).unsqueeze(-1)
    valid_lo = (lo >= 0) & (lo < m)
    valid_hi = (lo + 1 >= 0) & (lo + 1 < m)
    pad = torch.full_like(x[0], PAD_VALUE)
    a = torch.where(valid_lo.unsqueeze(-1), x[lo.clamp(0, m - 1)], pad)
    b = torch.where(valid_hi.unsqueeze(-1), x[(lo + 1).clamp(0, m - 1)], pad)
    return a * (1 - w) + b * w


def _stretch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    rate = float(torch.empty(1).uniform_(0.8, 1.2, generator=g))
    t = x.shape[-1]
    new_t = max(int(round(t * rate)), 1)
    y = F.interpolate(x.unsqueeze(0), size=new_t, mode="linear", align_corners=False).squeeze(0)
    if new_t >= t:
        return y[:, :t]
    return torch.cat([y, torch.full((x.shape[0], t - new_t), PAD_VALUE, device=x.device, dtype=x.dtype)], dim=-1)


def _noise(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    amp = float(torch.empty(1).uniform_(0.001, 0.015, generator=g))
    return x + torch.randn(x.shape, generator=g, device="cpu").to(x) * amp * 10


def _frames(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    t = x.shape[-1]
    y = x.clone()
    picked = torch.rand(t, generator=g) < 0.1
    mode = torch.randint(0, 3, (t,), generator=g)
    keep = torch.ones(t, dtype=torch.bool)
    for i in torch.nonzero(picked).flatten().tolist():
        if mode[i] == 0 and i > 0:
            y[:, i] = y[:, i - 1]  # duplicate the previous frame
        elif mode[i] == 1:
            y[:, i] = PAD_VALUE  # silence
        else:
            keep[i] = False  # drop
    y = y[:, keep.to(y.device)]
    if y.shape[-1] < t:
        y = torch.cat([y, torch.full((x.shape[0], t - y.shape[-1]), PAD_VALUE, device=x.device, dtype=x.dtype)], -1)
    return y


def _mask(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    y = x.clone()
    m, t = x.shape
    fw = int(torch.randint(0, 17, (1,), generator=g))
    f0 = int(torch.randint(0, max(m - fw, 0) + 1, (1,), generator=g))
    y[f0 : f0 + fw] = PAD_VALUE
    tw = int(torch.randint(0, max(int(0.1 * t), 1) + 1, (1,), generator=g))
    t0 = int(torch.randint(0, max(t - tw, 0) + 1, (1,), generator=g))
    y[:, t0 : t0 + tw] = PAD_VALUE
    return y


_FNS = {"shift": _shift, "gain": _gain, "pitch": _pitch, "stretch": _stretch, "noise": _noise,
        "frames": _frames, "mask": _mask}


def augment_batch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x: (B, n_mels, T) log-mel. Each row gets one transform drawn uniformly from TRANSFORMS (incl. none)."""
    picks = torch.randint(0, len(TRANSFORMS), (x.shape[0],), generator=g).tolist()
    rows = [x[i] if TRANSFORMS[k] == "none" else _FNS[TRANSFORMS[k]](x[i], g) for i, k in enumerate(picks)]
    return torch.stack(rows)
