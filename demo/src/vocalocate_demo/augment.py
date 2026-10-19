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

Every transform works on a whole sub-batch at once with per-row parameters, so a batch costs a
few kernel launches per transform rather than one Python loop per row (and per frame).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import PAD_VALUE

TRANSFORMS = ("none", "shift", "gain", "pitch", "stretch", "noise", "frames", "mask")
# 128 mel bins over 0-16 kHz span about 100 semitones above ~50 Hz.
MEL_BINS_PER_SEMITONE = 1.28


def _uniform(n: int, lo: float, hi: float, g: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    return (torch.rand(n, generator=g) * (hi - lo) + lo).to(like)


def _gather_lerp(x: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Linear interpolation of x at fractional positions src along dim (1 = mel, 2 = time); PAD outside."""
    n = x.shape[dim]
    lo = src.floor()
    w = src - lo
    lo = lo.long()
    out = 0
    for idx, wt in ((lo, 1 - w), (lo + 1, w)):
        ok = (idx >= 0) & (idx < n)
        if dim == 1:  # src: (B, M) -> gather rows of x
            v = torch.gather(x, 1, idx.clamp(0, n - 1).unsqueeze(-1).expand(-1, -1, x.shape[2]))
            v = torch.where(ok.unsqueeze(-1), v, torch.full_like(v, PAD_VALUE))
            out = out + v * wt.unsqueeze(-1)
        else:  # src: (B, T) -> gather columns of x
            v = torch.gather(x, 2, idx.clamp(0, n - 1).unsqueeze(1).expand(-1, x.shape[1], -1))
            v = torch.where(ok.unsqueeze(1), v, torch.full_like(v, PAD_VALUE))
            out = out + v * wt.unsqueeze(1)
    return out


def _shift(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, _, t = x.shape
    k = torch.randint(-int(0.3 * t), int(0.3 * t) + 1, (b, 1), generator=g).to(x.device)
    idx = (torch.arange(t, device=x.device) - k) % t  # circular roll by k
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def _gain(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    db = _uniform(x.shape[0], -10, 10, g, x)
    return x + (db * math.log(10) / 50).view(-1, 1, 1)


def _pitch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    shift = _uniform(x.shape[0], -3, 3, g, x) * MEL_BINS_PER_SEMITONE
    src = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) - shift.unsqueeze(-1)
    return _gather_lerp(x, src, dim=1)


def _stretch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    # Output frame j of a clip stretched to new_t = t * rate frames reads input position
    # (j + 0.5) * t / new_t - 0.5, as F.interpolate(mode="linear", align_corners=False) does;
    # frames past new_t are padding, frames past t are cropped.
    t = x.shape[2]
    new_t = (_uniform(x.shape[0], 0.8, 1.2, g, x) * t).round().clamp(min=1)
    j = torch.arange(t, device=x.device, dtype=x.dtype)
    src = ((j + 0.5) * (t / new_t).unsqueeze(-1) - 0.5).clamp(min=0)
    src = torch.where(j < new_t.unsqueeze(-1), src.clamp(max=t - 1), torch.full_like(src, -2.0))
    return _gather_lerp(x, src, dim=2)


def _noise(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    amp = _uniform(x.shape[0], 0.001, 0.015, g, x)
    return x + torch.randn(x.shape, generator=g).to(x) * (amp * 10).view(-1, 1, 1)


def _frames(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, m, t = x.shape
    picked = torch.rand(b, t, generator=g) < 0.1
    mode = torch.randint(0, 3, (b, t), generator=g)
    pos = torch.arange(t)
    dup = picked & (mode == 0) & (pos > 0)  # copy the previous frame
    silence = picked & (mode == 1)
    keep = ~(picked & (mode == 2))  # dropped frames close up; the tail is padded
    src = torch.where(dup, pos - 1, pos).expand(b, t)
    order = torch.sort((~keep).int() * t + pos, dim=1, stable=True).indices  # kept frames first, in order
    src = torch.gather(src, 1, order)
    pad = torch.gather(silence, 1, order) | (torch.arange(t) >= keep.sum(1, keepdim=True))
    dev = x.device
    y = torch.gather(x, 2, src.to(dev).unsqueeze(1).expand(-1, m, -1))
    return torch.where(pad.to(dev).unsqueeze(1), torch.full_like(y, PAD_VALUE), y)


def _mask(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, m, t = x.shape
    fw = torch.randint(0, 17, (b, 1), generator=g)
    f0 = (torch.rand(b, 1, generator=g) * (m - fw + 1).clamp(min=1)).long()
    tmax = max(int(0.1 * t), 1)
    tw = torch.randint(0, tmax + 1, (b, 1), generator=g)
    t0 = (torch.rand(b, 1, generator=g) * (t - tw + 1).clamp(min=1)).long()
    fi, ti = torch.arange(m), torch.arange(t)
    band = ((fi >= f0) & (fi < f0 + fw)).unsqueeze(-1) | ((ti >= t0) & (ti < t0 + tw)).unsqueeze(1)
    return torch.where(band.to(x.device), torch.full_like(x, PAD_VALUE), x)


_FNS = {"shift": _shift, "gain": _gain, "pitch": _pitch, "stretch": _stretch, "noise": _noise,
        "frames": _frames, "mask": _mask}


def augment_batch(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x: (B, n_mels, T) log-mel. Each row gets one transform drawn uniformly from TRANSFORMS (incl. none).

    g is a CPU generator: every random draw happens on it, so results do not depend on the device.
    """
    picks = torch.randint(0, len(TRANSFORMS), (x.shape[0],), generator=g)
    out = x.clone()
    for k, name in enumerate(TRANSFORMS):
        rows = torch.nonzero(picks == k).flatten()
        if name == "none" or len(rows) == 0:
            continue
        rows = rows.to(x.device)
        out[rows] = _FNS[name](x[rows], g)
    return out
