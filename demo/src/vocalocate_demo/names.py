"""Model names and the model folder: plain constants, importable without torch (the app server boots on these)."""

from __future__ import annotations

from pathlib import Path

from vocalocate_pipeline.config import PipelineConfig

MODEL_NAMES = {
    "baseline": "QVIM baseline (no mining)",
    "random": "Control: random real-SFX negatives",
    "retrieval": "VocaLocate A: retrieval-driven semi-hard mining",
    "metadata": "VocaLocate B: pre-constructed metadata-guided mining",
}


def model_dir(cfg: PipelineConfig, source: str) -> Path:
    return cfg.lake_root / "models" / source
