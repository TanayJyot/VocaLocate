"""Deployment: index a user's own SFX folders and search them with a vocal imitation.

This is the product flow from the "High-Level Product Flow" table, steps 1-6:

1. The user picks folders. The A2 library pipeline indexes them incrementally
   (unchanged files are skipped, moved files are recognised by hash).
2. Each clean library clip is embedded with the SFX encoder of the chosen model.
   Embeddings are cached in the lake's gold `embeddings` table, one part per
   library and model, so a second search embeds only new or changed files.
3. The query imitation goes through the same cleaning and log-mel steps as the
   SFX, then through the imitation encoder. It is not stored.
4. An optional text hint is scored against every clip with CLAP (audio
   embeddings cached the same way) and fused with the voice score using the α
   tuned on validation.
5-6. The fused score ranks the library and the top K are returned with their
   path and project folder.

The library files need no categories or tags: metadata was only used to choose
negatives at training time.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flow_functions.audio import clean_file
from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.flow_functions.features import log_mel
from vocalocate_pipeline.flows import run_library_flow
from vocalocate_pipeline.lake import Lake
from vocalocate_pipeline.schemas import to_table

from .data import fit_frames
from .model import DualEncoder
from .train import TrainConfig, embed

log = logging.getLogger(__name__)


class _Features:
    """Feature lookup by clip_id for `train.embed`."""

    def __init__(self, paths: dict[str, str]):
        self.paths = paths

    def mel(self, clip_id: str) -> np.ndarray:
        return np.load(self.paths[clip_id]).astype(np.float32)


@dataclass
class LibraryIndex:
    clip_ids: list[str]
    paths: list[str]
    projects: list[str | None]
    vectors: np.ndarray
    n_embedded: int  # how many clips were embedded now (the rest came from the cache)


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def index_library(
    cfg: PipelineConfig,
    roots: list[Path],
    library: str,
    model: DualEncoder,
    tcfg: TrainConfig,
    model_version: str,
    scan: bool = True,
    lock=None,
    progress=None,
) -> LibraryIndex:
    """Scan (optional), then embed every library clip, reusing cached embeddings.

    `lock` (a context manager) guards every warehouse access: the library flow then runs with
    load=False and the warehouse is rebuilt under the lock, so concurrent readers never race the
    rebuild. `progress(done, total)` is called while new clips are embedded.
    """
    guard = lock if lock is not None else nullcontext()
    if scan:
        summary = run_library_flow(roots, cfg, library=library, load=lock is None)
        if lock is not None:
            with guard:
                warehouse.load(cfg)
        log.info("library flow: %s", summary.as_dict())
    with guard:
        rows = _library_rows(cfg, library)
    lake = Lake(cfg.lake_root)
    part = f"library_{library}__{model_version}"
    cached = lake.read("embeddings", part)
    known = {r["clip_id"]: r["vector"] for r in cached.to_pylist()} if cached is not None else {}

    clip_ids = list(dict.fromkeys(r[0] for r in rows))
    missing = [cid for cid in clip_ids if cid not in known]
    if missing:
        feats = _Features({r[0]: r[3] for r in rows})
        step = 256
        for start in range(0, len(missing), step):
            chunk = missing[start : start + step]
            for cid, vec in zip(chunk, embed(model, feats, chunk, "sfx", tcfg)):
                known[cid] = vec.tolist()
            if progress is not None:
                progress(min(start + step, len(missing)), len(missing))
    # Rewrite the part with only the clips that are still in the library.
    table = to_table(
        "embeddings",
        [
            {"clip_id": cid, "source": f"library_{library}", "model_version": model_version, "tower": "sfx",
             "vector": known[cid]}
            for cid in clip_ids
        ],
    )
    lake.write("embeddings", part, table)
    vectors = np.array([known[cid] for cid in clip_ids], dtype=np.float32).reshape(len(clip_ids), -1)
    first_path = {}
    for cid, path, project, _ in rows:
        first_path.setdefault(cid, (path, project))
    return LibraryIndex(
        clip_ids=clip_ids,
        paths=[first_path[c][0] for c in clip_ids],
        projects=[first_path[c][1] for c in clip_ids],
        vectors=vectors,
        n_embedded=len(missing),
    )


def _library_rows(cfg: PipelineConfig, library: str) -> list[tuple]:
    if not cfg.warehouse_path.exists():
        return []
    return warehouse.query(
        cfg,
        "SELECT DISTINCT clip_id, path, project_id, feature_path FROM v_library_index "
        f"WHERE library = {_sql_str(library)} ORDER BY path",
    )


def cached_index(cfg: PipelineConfig, library: str, model_version: str, lock=None) -> LibraryIndex | None:
    """The library's index from cached embeddings alone (no scan, no embedding), or None.

    Only returned when the cache covers every clip of the last completed scan, so ids (positions in
    the path-ordered list) agree with every other model's index of the library.
    """
    with lock if lock is not None else nullcontext():
        rows = _library_rows(cfg, library)
    if not rows:
        return None
    cached = Lake(cfg.lake_root).read("embeddings", f"library_{library}__{model_version}")
    if cached is None:
        return None
    known = {r["clip_id"]: r["vector"] for r in cached.to_pylist()}
    clip_ids = list(dict.fromkeys(r[0] for r in rows))
    if any(cid not in known for cid in clip_ids):
        return None
    first_path = {}
    for cid, path, project, _ in rows:
        first_path.setdefault(cid, (path, project))
    return LibraryIndex(
        clip_ids=clip_ids,
        paths=[first_path[c][0] for c in clip_ids],
        projects=[first_path[c][1] for c in clip_ids],
        vectors=np.array([known[c] for c in clip_ids], dtype=np.float32).reshape(len(clip_ids), -1),
        n_embedded=0,
    )


@torch.no_grad()
def embed_query(path: Path, model: DualEncoder, tcfg: TrainConfig, cfg: PipelineConfig) -> np.ndarray:
    """Clean and featurise a recorded imitation exactly like the SFX, then embed it."""
    res = clean_file(Path(path), cfg.audio)
    if res.status != "ok":
        raise ValueError(f"query {path} rejected by cleaning: {res.reject_reason}")
    mel = fit_frames(log_mel(res.audio, cfg.audio.sample_rate, cfg.features), tcfg.frames)
    model.eval()
    device = next(model.parameters()).device
    return model.embed_imitation(torch.from_numpy(mel[None]).to(device)).cpu().numpy()[0]


def text_scores(cfg: PipelineConfig, library: str, index: LibraryIndex, text: str, clap, lock=None) -> np.ndarray:
    """CLAP text-hint scores for every library clip (step 4 of the product flow). Audio embeddings are cached."""
    from .text import cached_audio_embeddings

    with lock if lock is not None else nullcontext():
        rows = warehouse.query(
            cfg,
            f"SELECT clip_id, clean_path FROM clips WHERE source = {_sql_str('library_' + library)} AND status = 'ok'",
        )
    clean = dict(rows)
    audio = cached_audio_embeddings(
        cfg, f"library_{library}__clap", index.clip_ids, [clean[c] for c in index.clip_ids], clap
    )
    return audio @ clap.embed_text([text])[0]


def search(
    index: LibraryIndex, query_vec: np.ndarray, k: int = 5, text: np.ndarray | None = None, alpha: float = 0.5
) -> list[dict]:
    """Rank the library by voice score, or by α·z(voice) + (1−α)·z(text) when text scores are given."""
    scores = index.vectors @ query_vec
    if text is not None:
        from .fusion import fuse

        scores = fuse(scores[None], text[None], alpha)[0]
    top = np.argsort(-scores)[:k]
    return [
        {"rank": r + 1, "score": float(scores[i]), "path": index.paths[i], "project": index.projects[i]}
        for r, i in enumerate(top)
    ]
