"""Pipeline settings.

Audio and feature settings copy the QVIM challenge baseline
(github.com/qvim-aes/qvim-baseline) so our features and scores stay comparable
with the published numbers we reproduce in milestone 1.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 32_000
    # Longer clips are cut to this length and flagged, not dropped.
    max_duration_sec: float = 20.0
    min_duration_sec: float = 0.1
    # A clip whose peak is below this level is treated as silence and rejected.
    silence_peak_dbfs: float = -60.0
    # Leading/trailing audio quieter than (peak + trim_db) is trimmed.
    trim_db: float = -50.0
    target_rms_dbfs: float = -20.0
    max_peak_dbfs: float = -1.0
    # Samples at or above this absolute value count as clipped.
    clip_threshold: float = 0.999
    clipped_ratio_flag: float = 0.01


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 320
    n_mels: int = 128
    fmin: float = 0.0
    # The baseline sets fmax = sr/2 - fmax_aug_range/2 = 15 kHz when it is not augmenting.
    fmax: float | None = 15_000.0
    preemphasis: float = 0.97


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 490
    train: float = 0.8
    val: float = 0.1
    # test gets the remainder


@dataclass(frozen=True)
class PipelineConfig:
    lake_root: Path = Path("lake")
    workers: int = 1
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)

    @property
    def warehouse_path(self) -> Path:
        return self.lake_root / "warehouse" / "vocalocate.duckdb"

    def fingerprint(self) -> str:
        """Short hash of the settings that change the outputs, stored with every run."""
        payload = {
            "audio": asdict(self.audio),
            "features": asdict(self.features),
            "splits": asdict(self.splits),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(
    path: Path | None = None,
    lake_root: Path | str | None = None,
    workers: int | None = None,
) -> PipelineConfig:
    """Load settings from a TOML file (see configs/default.toml).

    `lake_root` and `workers` override the file, which is how the CLI points
    the same config at different environments.
    """
    raw: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    top = raw.get("pipeline", {})
    return PipelineConfig(
        lake_root=Path(lake_root if lake_root is not None else top.get("lake_root", "lake")),
        workers=int(workers if workers is not None else top.get("workers", 1)),
        audio=_build(AudioConfig, raw.get("audio", {})),
        features=_build(FeatureConfig, raw.get("features", {})),
        splits=_build(SplitConfig, raw.get("splits", {})),
    )
