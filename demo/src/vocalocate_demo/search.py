"""Deployment: index a user's own SFX folders and search them with a vocal imitation.

This is the product flow from the "High-Level Product Flow" table, steps 1-6:

1. The user picks folders. The A2 library pipeline indexes them incrementally
   (unchanged files are skipped, moved files are recognised by hash).
2. Each clean library clip is embedded with the SFX encoder of the chosen model.
   Embeddings are cached in the lake's gold `embeddings` table, one part per
   library and model, so a second search embeds only new or changed files.
3. The query imitation goes through the same cleaning and log-mel steps as the
   SFX, then through the imitation encoder. It is not stored.
5-6. Cosine similarity ranks the library and the top K are returned with their
   path and project folder.

The library files need no categories or tags: metadata was only used to choose
negatives at training time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.audio import clean_file
from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.features import log_mel
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
) -> LibraryIndex:
    if scan:
        summary = run_library_flow(roots, cfg, library=library)
        log.info("library flow: %s", summary.as_dict())
    rows = warehouse.query(
        cfg,
        "SELECT DISTINCT clip_id, path, project_id, feature_path FROM v_library_index "
        f"WHERE library = {_sql_str(library)} ORDER BY path",
    )
    lake = Lake(cfg.lake_root)
    part = f"library_{library}__{model_version}"
    cached = lake.read("embeddings", part)
    known = {r["clip_id"]: r["vector"] for r in cached.to_pylist()} if cached is not None else {}

    clip_ids = list(dict.fromkeys(r[0] for r in rows))
    missing = [cid for cid in clip_ids if cid not in known]
    if missing:
        feats = _Features({r[0]: r[3] for r in rows})
        for cid, vec in zip(missing, embed(model, feats, missing, "sfx", tcfg)):
            known[cid] = vec.tolist()
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


def search(index: LibraryIndex, query_vec: np.ndarray, k: int = 5) -> list[dict]:
    scores = index.vectors @ query_vec
    top = np.argsort(-scores)[:k]
    return [
        {"rank": r + 1, "score": float(scores[i]), "path": index.paths[i], "project": index.projects[i]}
        for r, i in enumerate(top)
    ]
