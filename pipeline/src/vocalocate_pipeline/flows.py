"""The two pipelines: dataset ingestion (for training and evaluation) and library indexing.

Both run the same stages: ingest -> clean -> transform -> load. Work is cached
by content hash and config fingerprint, so re-running a flow only processes
files that are new or changed.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import warehouse
from .audio import AUDIO_EXTENSIONS, clean_file, sha256_file, write_wav
from .config import PipelineConfig
from .features import FEATURE_VERSION, log_mel, summary_vector
from .ingest import ingest
from .lake import Lake
from .schemas import to_table
from .splits import assign_splits

log = logging.getLogger(__name__)

_METRIC_FIELDS = (
    "status",
    "reject_reason",
    "orig_sample_rate",
    "orig_channels",
    "orig_duration_sec",
    "duration_sec",
    "peak_dbfs",
    "rms_dbfs",
    "truncated",
    "clipped",
    "clean_path",
)
_META_FIELDS = ("source", "source_key", "role", "label", "group_key", "imitator_id", "project_id", "licence")


@dataclass
class RunSummary:
    run_id: str
    flow: str
    source: str
    n_input: int = 0
    n_processed: int = 0
    n_reused: int = 0
    n_ok: int = 0
    n_rejected: int = 0
    n_duplicate: int = 0
    n_missing: int = 0
    # Library flow only.
    n_added: int = 0
    n_changed: int = 0
    n_moved: int = 0
    n_deleted: int = 0
    n_unchanged: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id(flow: str) -> str:
    return f"{flow}-{_now():%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


# ---------------------------------------------------------------------------
# Clean + transform one clip. Runs in a worker process.
# ---------------------------------------------------------------------------


def process_clip(uri: str, clip_id: str, source: str, cfg: PipelineConfig) -> dict:
    lake = Lake(cfg.lake_root)
    res = clean_file(Path(uri), cfg.audio)
    out = {
        "status": res.status,
        "reject_reason": res.reject_reason,
        "orig_sample_rate": res.orig_sample_rate,
        "orig_channels": res.orig_channels,
        "orig_duration_sec": res.orig_duration_sec,
        "duration_sec": res.duration_sec,
        "peak_dbfs": res.peak_dbfs,
        "rms_dbfs": res.rms_dbfs,
        "truncated": res.truncated,
        "clipped": res.clipped,
        "clean_path": None,
        "feature_path": None,
        "n_frames": None,
        "summary": None,
    }
    if res.status != "ok":
        return out
    clean_path = lake.clean_audio_path(source, clip_id)
    write_wav(clean_path, res.audio, cfg.audio.sample_rate)

    mel = log_mel(res.audio, cfg.audio.sample_rate, cfg.features)
    feature_path = lake.feature_path(source, clip_id)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = feature_path.with_suffix(".tmp.npy")
    np.save(tmp, mel.astype(np.float16))
    tmp.replace(feature_path)

    out.update(
        clean_path=str(clean_path),
        feature_path=str(feature_path),
        n_frames=int(mel.shape[1]),
        summary=summary_vector(mel).tolist(),
    )
    return out


def _run_parallel(jobs: list[tuple], cfg: PipelineConfig) -> list[dict]:
    if not jobs:
        return []
    if cfg.workers <= 1:
        return [process_clip(*job, cfg) for job in jobs]
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(process_clip, *job, cfg) for job in jobs]
        return [f.result() for f in futures]


def _previous_results(lake: Lake, part: str, fingerprint: str) -> dict[str, dict]:
    """Results from an earlier run of this part that are still valid and can be reused."""
    clips = lake.read("clips", part)
    if clips is None:
        return {}
    features = lake.read("features", part)
    feats = {r["clip_id"]: r for r in features.to_pylist()} if features is not None else {}
    reusable = {}
    for row in clips.to_pylist():
        if row["config_fingerprint"] != fingerprint or row["status"] == "duplicate":
            continue
        result = {k: row[k] for k in _METRIC_FIELDS}
        if row["status"] == "ok":
            feat = feats.get(row["clip_id"])
            if feat is None or not Path(row["clean_path"]).exists() or not Path(feat["feature_path"]).exists():
                continue
            result.update(feature_path=feat["feature_path"], n_frames=feat["n_frames"], summary=feat["summary"])
        else:
            result.update(feature_path=None, n_frames=None, summary=None)
        reusable[row["clip_id"]] = result
    return reusable


def clean_and_transform(
    lake: Lake, part: str, raw_rows: list[dict], cfg: PipelineConfig, run_id: str, summary: RunSummary
) -> tuple[list[dict], list[dict]]:
    """Silver and gold stages for one batch of raw files. Returns (clip rows, feature rows)."""
    fingerprint = cfg.fingerprint()
    reusable = _previous_results(lake, part, fingerprint)

    first_key: dict[str, str] = {}
    jobs, results = [], {}
    for row in raw_rows:
        cid = row["file_sha256"]
        if cid in first_key:
            continue
        first_key[cid] = row["source_key"]
        if cid in reusable:
            results[cid] = reusable[cid]
        else:
            jobs.append((row["uri"], cid, row["source"]))

    log.info("%s: %d unique files, %d to process, %d reused", part, len(first_key), len(jobs), len(results))
    for job, result in zip(jobs, _run_parallel(jobs, cfg)):
        results[job[1]] = result
    summary.n_processed += len(jobs)
    summary.n_reused += len(first_key) - len(jobs)

    clip_rows, feature_rows = [], []
    for row in raw_rows:
        cid = row["file_sha256"]
        res = results[cid]
        clip = {k: row[k] for k in _META_FIELDS}
        clip.update({k: res[k] for k in _METRIC_FIELDS})
        clip.update(clip_id=cid, config_fingerprint=fingerprint, run_id=run_id)
        if first_key[cid] != row["source_key"]:
            clip.update({k: None for k in _METRIC_FIELDS if k not in ("truncated", "clipped")})
            clip.update(status="duplicate", reject_reason=f"duplicate_of:{first_key[cid]}")
        elif res["status"] == "ok":
            feature_rows.append(
                {
                    "clip_id": cid,
                    "source": row["source"],
                    "feature_version": FEATURE_VERSION,
                    "feature_path": res["feature_path"],
                    "n_frames": res["n_frames"],
                    "summary": res["summary"],
                }
            )
        clip_rows.append(clip)

    for clip in clip_rows:
        if clip["status"] == "ok":
            summary.n_ok += 1
        elif clip["status"] == "rejected":
            summary.n_rejected += 1
        else:
            summary.n_duplicate += 1
    return clip_rows, feature_rows


def build_pairs(clips: Iterable[dict]) -> list[dict]:
    """Pair each clean imitation with the clean reference of the same group in the same source."""
    clips = [c for c in clips if c["status"] == "ok" and c["group_key"]]
    refs = {(c["source"], c["group_key"]): c["clip_id"] for c in clips if c["role"] == "reference"}
    pairs = []
    for c in clips:
        if c["role"] != "imitation":
            continue
        ref = refs.get((c["source"], c["group_key"]))
        if ref is not None:
            pairs.append(
                {
                    "source": c["source"],
                    "imitation_clip_id": c["clip_id"],
                    "reference_clip_id": ref,
                    "group_key": c["group_key"],
                }
            )
    return pairs


def _record_run(lake: Lake, summary: RunSummary, cfg: PipelineConfig, started: datetime, status: str, message=None):
    row = {
        "run_id": summary.run_id,
        "flow": summary.flow,
        "source": summary.source,
        "started_at": started,
        "finished_at": _now(),
        "status": status,
        "n_input": summary.n_input,
        "n_processed": summary.n_processed,
        "n_reused": summary.n_reused,
        "n_ok": summary.n_ok,
        "n_rejected": summary.n_rejected,
        "n_duplicate": summary.n_duplicate,
        "config_fingerprint": cfg.fingerprint(),
        "message": message,
    }
    lake.write("pipeline_runs", summary.run_id, to_table("pipeline_runs", [row]))


# ---------------------------------------------------------------------------
# Flow 1: dataset ingestion (VimSketch, QVIM DEV, ESC-50, team recordings)
# ---------------------------------------------------------------------------


def run_dataset_flow(source: str, root: Path, cfg: PipelineConfig, load: bool = True) -> RunSummary:
    lake = Lake(cfg.lake_root)
    started = _now()
    summary = RunSummary(run_id=_new_run_id("dataset"), flow="dataset", source=source)
    try:
        result = ingest(source, root)
        summary.n_missing = len(result.missing)
        if result.missing:
            log.warning("%s: %d files listed in metadata are missing", source, len(result.missing))
        raw_rows = [f.to_row(summary.run_id, started) for f in result.files]
        summary.n_input = len(raw_rows)
        lake.write("raw_files", source, to_table("raw_files", raw_rows))

        clip_rows, feature_rows = clean_and_transform(lake, source, raw_rows, cfg, summary.run_id, summary)
        lake.write("clips", source, to_table("clips", clip_rows))
        lake.write("features", source, to_table("features", feature_rows))
        lake.write("pairs", source, to_table("pairs", build_pairs(clip_rows)))
        lake.write("splits", source, to_table("splits", assign_splits(clip_rows, cfg.splits)))

        clip_by_key = {c["source_key"]: c for c in clip_rows if c["status"] == "ok"}
        queries = [
            {
                "query_id": q["query_id"],
                "source": source,
                "imitation_clip_id": clip_by_key[q["source_key"]]["clip_id"],
                "text_hint": q["text_hint"],
                "hint_type": q["hint_type"],
                "target_group_key": q["target_group_key"],
            }
            for q in result.queries
            if q["source_key"] in clip_by_key
        ]
        lake.write("queries", source, to_table("queries", queries))
    except Exception as exc:
        _record_run(lake, summary, cfg, started, "failed", f"{type(exc).__name__}: {exc}")
        raise
    _record_run(lake, summary, cfg, started, "succeeded")
    if load:
        warehouse.load(cfg)
    return summary


# ---------------------------------------------------------------------------
# Flow 2: incremental indexing of a user's own SFX folders
# ---------------------------------------------------------------------------


def _scan(roots: list[Path]) -> dict[str, tuple[Path, Path, os.stat_result]]:
    found = {}
    for root in roots:
        root = root.resolve()
        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
                found[str(path)] = (root, path, path.stat())
    return found


def run_library_flow(roots: list[Path], cfg: PipelineConfig, library: str = "default", load: bool = True) -> RunSummary:
    lake = Lake(cfg.lake_root)
    started = _now()
    source = f"library_{library}"
    summary = RunSummary(run_id=_new_run_id("library"), flow="library", source=source)
    try:
        prev_table = lake.read("library_files", library)
        prev = {r["path"]: r for r in prev_table.to_pylist()} if prev_table is not None else {}
        prev_hashes = {r["file_sha256"] for r in prev.values() if r["status"] == "active"}

        state, raw_rows = [], []
        for key, (root, path, st) in sorted(_scan(roots).items()):
            old = prev.get(key)
            unchanged = (
                old is not None
                and old["status"] == "active"
                and old["size_bytes"] == st.st_size
                and old["mtime_ns"] == st.st_mtime_ns
            )
            sha = old["file_sha256"] if unchanged else sha256_file(path)
            if unchanged:
                summary.n_unchanged += 1
            elif old is not None and old["status"] == "active":
                summary.n_changed += 1
            elif sha in prev_hashes:
                summary.n_moved += 1
            else:
                summary.n_added += 1
            rel = path.relative_to(root)
            # The top-level folder under a library root is the project the sound belongs to.
            project = rel.parts[0] if len(rel.parts) > 1 else None
            state.append(
                {
                    "library": library,
                    "path": key,
                    "size_bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "file_sha256": sha,
                    "project_id": project,
                    "status": "active",
                    "first_seen": old["first_seen"] if old is not None else started,
                    "last_seen": started,
                }
            )
            raw_rows.append(
                {
                    "source": source,
                    "source_version": "live",
                    "source_key": key,
                    "uri": key,
                    "file_sha256": sha,
                    "size_bytes": st.st_size,
                    "role": "library",
                    "label": None,
                    "group_key": None,
                    "imitator_id": None,
                    "project_id": project,
                    "licence": None,
                    "extra_json": None,
                    "ingested_at": started,
                    "run_id": summary.run_id,
                }
            )
        seen = {s["path"] for s in state}
        current_hashes = {s["file_sha256"] for s in state}
        for key, old in prev.items():
            if key not in seen:
                # A file whose content still exists at another path was moved, not deleted.
                if old["status"] == "active" and old["file_sha256"] not in current_hashes:
                    summary.n_deleted += 1
                state.append({**old, "status": "deleted"})

        summary.n_input = len(raw_rows)
        lake.write("raw_files", source, to_table("raw_files", raw_rows))
        clip_rows, feature_rows = clean_and_transform(lake, source, raw_rows, cfg, summary.run_id, summary)
        lake.write("clips", source, to_table("clips", clip_rows))
        lake.write("features", source, to_table("features", feature_rows))
        lake.write("library_files", library, to_table("library_files", state))
    except Exception as exc:
        _record_run(lake, summary, cfg, started, "failed", f"{type(exc).__name__}: {exc}")
        raise
    _record_run(lake, summary, cfg, started, "succeeded")
    if load:
        warehouse.load(cfg)
    return summary
