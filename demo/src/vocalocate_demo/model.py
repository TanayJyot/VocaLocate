"""Dual-encoder QVIM model: one encoder for vocal imitations, one for SFX, one shared embedding space.

This follows the Greif et al. (2024) setup of two separate audio encoders
trained contrastively. The demo uses a small CNN so it trains in minutes on a
CPU; the pretrained MobileNetV3 from the QVIM baseline can replace `Encoder`
without changing anything else, since the rest of the code only sees
`DualEncoder.embed_imitation` and `DualEncoder.embed_sfx`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(c_in: int, c_out: int, stride: int = 1, kernel: int = 3, pool: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers, nn.MaxPool2d(2)) if pool else nn.Sequential(*layers)


class Encoder(nn.Module):
    """Log-mel (B, n_mels, T) -> L2-normalised embedding (B, dim)."""

    def __init__(self, n_mels: int = 128, dim: int = 128, width: int = 16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(1, w, stride=2, kernel=5, pool=False),  # strided stem keeps CPU training fast
            _block(w, 2 * w),
            _block(2 * w, 4 * w),
            _block(4 * w, 4 * w),
        )
        # Keep coarse frequency resolution (pitch matters for SFX), pool over time.
        self.freq_pool = nn.AdaptiveAvgPool2d((4, None))
        self.head = nn.Sequential(nn.Linear(4 * w * 4 * 2, 256), nn.ReLU(inplace=True), nn.Linear(256, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.freq_pool(self.features(x.unsqueeze(1)))  # (B, C, 4, T')
        h = h.flatten(1, 2)  # (B, C*4, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return F.normalize(self.head(h), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, n_mels: int = 128, dim: int = 128, width: int = 16, tau: float = 0.07):
        super().__init__()
        self.imitation = Encoder(n_mels, dim, width)
        self.sfx = Encoder(n_mels, dim, width)
        # Learnable temperature for the contrastive loss, as in the QVIM baseline.
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
        self.hparams = {"n_mels": n_mels, "dim": dim, "width": width, "tau": tau}

    def embed_imitation(self, x: torch.Tensor) -> torch.Tensor:
        return self.imitation(x)

    def embed_sfx(self, x: torch.Tensor) -> torch.Tensor:
        return self.sfx(x)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)
