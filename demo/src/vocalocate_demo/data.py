"""Read training pairs, candidate SFX and their cached log-mel features from the pipeline warehouse.

Everything the demo trains and evaluates on comes out of the A2 pipeline:
`v_training_pairs` gives imitation/reference pairs with their grouped split,
`v_sfx_candidates` gives every clean reference and library SFX with its
metadata, and the gold layer holds the log-mel features (QVIM baseline settings).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

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
    hints: dict[str, str] = field(default_factory=dict)  # imitation clip_id -> text hint (queries table)
    # Imitations kept only as negatives (v_unpaired_imitations): clip_id, split, feature_path,
    # target_clip_id, category, tags. Empty for sources that have none.
    unpaired: pd.DataFrame = field(default_factory=pd.DataFrame)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _paths: dict[str, str] = field(default_factory=dict, repr=False)
    _meta: dict[str, SFXMeta] | None = field(default=None, repr=False)

    def __post_init__(self):
        for row in self.pairs.itertuples():
            self._paths[row.imitation_clip_id] = row.imitation_feature_path
            self._paths[row.reference_clip_id] = row.reference_feature_path
        for row in self.candidates.itertuples():
            self._paths[row.clip_id] = row.feature_path
        for row in self.unpaired.itertuples():
            self._paths[row.clip_id] = row.feature_path

    def mel(self, clip_id: str) -> np.ndarray:
        if clip_id not in self._cache:
            self._cache[clip_id] = np.load(self._paths[clip_id]).astype(np.float32)
        return self._cache[clip_id]

    def preload(self, workers: int = 32) -> int:
        """Load every feature file now, in parallel. Worth it where each file read has latency
        (a network volume); otherwise features load lazily on first use. Returns the count loaded."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [c for c in self._paths if c not in self._cache]
        with ThreadPoolExecutor(workers) as pool:
            for c, x in zip(todo, pool.map(lambda c: np.load(self._paths[c]).astype(np.float32), todo)):
                self._cache[c] = x
        return len(todo)

    def split_pairs(self, split: str) -> pd.DataFrame:
        return self.pairs[self.pairs.split == split].reset_index(drop=True)

    def split_candidates(self, split: str) -> pd.DataFrame:
        return self.candidates[self.candidates.split == split].reset_index(drop=True)

    def split_unpaired(self, split: str) -> pd.DataFrame:
        if self.unpaired.empty:
            return self.unpaired
        return self.unpaired[self.unpaired.split == split].reset_index(drop=True)

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


_LAKE_SUBDIRS = {("gold", "features"), ("silver", "audio")}


def rebase_lake_path(path, lake_root: Path):
    """Re-root a lake path recorded on another machine (e.g. a Windows path) onto lake_root.

    The warehouse stores the paths the ingest wrote. When the lake is copied elsewhere (a cloud
    volume), those no longer exist; everything from gold/features or silver/audio on is kept.
    """
    if not isinstance(path, str) or Path(path).exists():
        return path
    parts = re.split(r"[\\/]", path)
    for i in range(len(parts) - 1):
        if (parts[i], parts[i + 1]) in _LAKE_SUBDIRS:
            return str(Path(lake_root).joinpath(*parts[i:]))
    return path


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
        hints = dict(con.execute(f"SELECT imitation_clip_id, text_hint FROM queries WHERE source IN ({src})").fetchall())
        try:
            unpaired = con.execute(
                f"""SELECT * FROM v_unpaired_imitations WHERE source IN ({src}) AND feature_path IS NOT NULL
                    ORDER BY clip_id"""
            ).df()
        except duckdb.CatalogException:  # warehouse built before unpaired imitations existed
            unpaired = pd.DataFrame()
    finally:
        con.close()
    for df, cols in ((pairs, ("imitation_feature_path", "reference_feature_path")), (candidates, ("feature_path",)),
                     (unpaired, ("feature_path",))):
        for c in cols:
            if c in df:
                df[c] = [rebase_lake_path(v, cfg.lake_root) for v in df[c]]
    if pairs.empty:
        raise ValueError(f"No training pairs for {sources} in {cfg.warehouse_path}; run the dataset flow first")
    return Dataset(pairs=pairs, candidates=candidates, hints=hints, unpaired=unpaired)


def fit_frames(mel: np.ndarray, frames: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop or pad a (n_mels, T) log-mel to a fixed length. Random crop when training."""
    t = mel.shape[1]
    if t >= frames:
        start = int(rng.integers(0, t - frames + 1)) if rng is not None else 0
        return mel[:, start : start + frames]
    out = np.full((mel.shape[0], frames), PAD_VALUE, dtype=np.float32)
    out[:, :t] = mel
    return out
