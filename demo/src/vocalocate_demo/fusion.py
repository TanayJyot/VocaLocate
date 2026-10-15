"""Stage 7: fuse voice and text scores, s = α·z(s_voice) + (1 − α)·z(s_text).

Both scores are cosine similarities from different models, so for each query
they are z-normalised over the candidates first (A1: "after both scores are
normalized to the same scale"). α is chosen on the validation split only, then
the test split is scored once with voice only (α = 1), text only (α = 0) and
the fused score. These are the A2 evaluation components 1-3 and ablation 1.
"""

from __future__ import annotations

import numpy as np

from .data import SFXMeta
from .evaluate import score_metrics

ALPHAS = tuple(round(a, 2) for a in np.linspace(0, 1, 11))


def znorm(scores: np.ndarray) -> np.ndarray:
    """Per-query (row-wise) z-normalisation."""
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True) + 1e-9
    return (scores - mu) / sd


def fuse(voice: np.ndarray, text: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * znorm(voice) + (1 - alpha) * znorm(text)


def tune_alpha(voice: np.ndarray, text: np.ndarray, target: np.ndarray, meta: list[SFXMeta]) -> tuple[float, list]:
    """Pick α by MRR on validation. Returns (best α, [(α, MRR), ...])."""
    curve = [(a, score_metrics(fuse(voice, text, a), target, meta)["mrr"]) for a in ALPHAS]
    best = max(curve, key=lambda t: (t[1], t[0]))  # ties go to more voice weight
    return best[0], curve


def evaluate_fusion(
    voice_val, text_val, target_val, meta_val, voice_test, text_test, target_test, meta_test
) -> dict:
    alpha, curve = tune_alpha(voice_val, text_val, target_val, meta_val)
    return {
        "alpha": alpha,
        "val_curve": curve,
        "test": {
            "voice": score_metrics(voice_test, target_test, meta_test),
            "text": score_metrics(text_test, target_test, meta_test),
            "fused": score_metrics(fuse(voice_test, text_test, alpha), target_test, meta_test),
        },
    }
