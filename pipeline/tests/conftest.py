"""Shared test helpers: synthetic audio and a pipeline config pointing at a temporary lake."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocalocate_pipeline.config import PipelineConfig


def tone(freq: float, sec: float, sr: int, amp: float = 0.5, channels: int = 1) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def write(path: Path, x: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)
    return path


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    return PipelineConfig(lake_root=tmp_path / "lake", workers=1)

