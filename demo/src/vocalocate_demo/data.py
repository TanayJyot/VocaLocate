"""Read training pairs, candidate SFX and their cached log-mel features from the pipeline warehouse.

Everything the demo trains and evaluates on comes out of the A2 pipeline:
`v_training_pairs` gives imitation/reference pairs with their grouped split,
`v_sfx_candidates` gives every clean reference and library SFX with its
metadata, and the gold layer holds the log-mel features (QVIM baseline settings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from vocalocate_pipeline.config import PipelineConfig

# Value of a silent frame after the pipeline's log-mel normalisation: (log(1e-5) + 4.5) / 5.
PAD_VALUE = float((np.log(1e-5) + 4.5) / 5)


@dataclass
class SFXMeta:
    category: str | None
    tags: frozenset[str]
    recording_id: str | None


@dataclass
class Dataset:
    pairs: pd.DataFrame
    candidates: pd.DataFrame
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _paths: dict[str, str] = field(default_factory=dict, repr=False)
    _meta: dict[str, SFXMeta] | None = field(default=None, repr=False)

    def __post_init__(self):
        for row in self.pairs.itertuples():
            self._paths[row.imitation_clip_id] = row.imitation_feature_path
            self._paths[row.reference_clip_id] = row.reference_feature_path
        for row in self.candidates.itertuples():
            self._paths[row.clip_id] = row.feature_path

    def mel(self, clip_id: str) -> np.ndarray:
        if clip_id not in self._cache:
            self._cache[clip_id] = np.load(self._paths[clip_id]).astype(np.float32)
        return self._cache[clip_id]

    def split_pairs(self, split: str) -> pd.DataFrame:
        return self.pairs[self.pairs.split == split].reset_index(drop=True)

    def split_candidates(self, split: str) -> pd.DataFrame:
        return self.candidates[self.candidates.split == split].reset_index(drop=True)

    def meta(self) -> dict[str, SFXMeta]:
        if self._meta is not None:
            return self._meta
        out = {}
        for row in self.candidates.itertuples():
            tags = row.tags if isinstance(row.tags, (list, np.ndarray)) else []
            rec = row.recording_id if isinstance(row.recording_id, str) else None
            cat = row.category if isinstance(row.category, str) else None
            out[row.clip_id] = SFXMeta(cat, frozenset(tags), rec)
        self._meta = out
        return out


def load_dataset(cfg: PipelineConfig, sources: list[str]) -> Dataset:
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        src = ", ".join(f"'{s}'" for s in sources)
        pairs = con.execute(
            f"""SELECT * FROM v_training_pairs WHERE source IN ({src})
                AND imitation_feature_path IS NOT NULL AND reference_feature_path IS NOT NULL
                ORDER BY imitation_clip_id"""
        ).df()
        candidates = con.execute(f"SELECT * FROM v_sfx_candidates WHERE source IN ({src}) ORDER BY clip_id").df()
    finally:
        con.close()
    if pairs.empty:
        raise ValueError(f"No training pairs for {sources} in {cfg.warehouse_path}; run the dataset flow first")
    return Dataset(pairs=pairs, candidates=candidates)


def fit_frames(mel: np.ndarray, frames: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop or pad a (n_mels, T) log-mel to a fixed length. Random crop when training."""
    t = mel.shape[1]
    if t >= frames:
        start = int(rng.integers(0, t - frames + 1)) if rng is not None else 0
        return mel[:, start : start + frames]
    out = np.full((mel.shape[0], frames), PAD_VALUE, dtype=np.float32)
    out[:, :t] = mel
    return out
