"""Run a pipeline flow against the S3 lake: pull what it needs, run it locally, push the results.

This is what the ECS Fargate tasks run (see infra/). The lake bucket has the same layout as the local
lake folder, and the raw bucket holds each dataset under <source>/ exactly as downloaded.

    pull  s3://<raw>/<source>/          -> <lake>/bronze/<source>/      raw files to ingest
    pull  s3://<lake>/{tables,state,runs} + this source's clean audio and features
    run   the normal dataset flow
    push  only this source's table files, audio and features, and the run record

Clean audio and features for the source are pulled too, so the content-hash cache still skips work
that an earlier run already did. If the flow fails, only its pipeline_runs row is pushed.

An ingest never pushes another source's tables or the warehouse, so runs of different sources can
overlap without one writing stale copies over the other's results. The warehouse is rebuilt from
the lake by `load_warehouse`, which the state machine runs after every ingest.
"""

from __future__ import annotations

import logging
import os

from . import warehouse
from .config import PipelineConfig
from .flows import RunSummary, run_dataset_flow
from .lake import TABLE_LAYER, Lake
from .sync import sync

log = logging.getLogger(__name__)

TABLE_PREFIXES = ["bronze/tables", "silver/tables", "gold/tables", "state", "runs"]
WAREHOUSE_PREFIX = "warehouse"


def _source_prefixes(source: str) -> list[str]:
    return [f"silver/audio/{source}", f"gold/features/{source}"]


def _source_outputs(source: str, cfg: PipelineConfig) -> list[str]:
    """Everything an ingest of `source` writes: its file in each table, its audio and features, its run."""
    lake = Lake(cfg.lake_root)
    tables = [
        lake.table_path(name, source).relative_to(lake.root).as_posix()
        for name in TABLE_LAYER
        if name not in ("library_files", "pipeline_runs")
    ]
    return tables + _source_prefixes(source) + ["runs"]


def _env(name: str, value: str | None) -> str:
    value = value or os.environ.get(name)
    if not value:
        raise SystemExit(f"Set --{name.lower().removeprefix('vocalocate_').replace('_', '-')} or {name}")
    return value.rstrip("/")


def ingest(source: str, cfg: PipelineConfig, lake_uri: str | None = None, raw_uri: str | None = None) -> RunSummary:
    lake_uri = _env("VOCALOCATE_LAKE_URI", lake_uri)
    raw_uri = _env("VOCALOCATE_RAW_URI", raw_uri)
    local_raw = cfg.lake_root / "bronze" / source
    log.info("pull %s/%s -> %s: %s", raw_uri, source, local_raw, sync(f"{raw_uri}/{source}", local_raw))
    prefixes = TABLE_PREFIXES + _source_prefixes(source)
    log.info("pull %s -> %s: %s", lake_uri, cfg.lake_root, sync(lake_uri, cfg.lake_root, prefixes))
    try:
        summary = run_dataset_flow(source, local_raw, cfg)
    except Exception:
        sync(cfg.lake_root, lake_uri, ["runs"])  # keep the failed run's record
        raise
    log.info("push: %s", sync(cfg.lake_root, lake_uri, _source_outputs(source, cfg)))
    return summary


def load_warehouse(cfg: PipelineConfig, lake_uri: str | None = None):
    lake_uri = _env("VOCALOCATE_LAKE_URI", lake_uri)
    sync(lake_uri, cfg.lake_root, TABLE_PREFIXES)
    path = warehouse.load(cfg)
    sync(cfg.lake_root, lake_uri, [WAREHOUSE_PREFIX])
    return path


def report(cfg: PipelineConfig, sql: str, lake_uri: str | None = None):
    lake_uri = _env("VOCALOCATE_LAKE_URI", lake_uri)
    sync(lake_uri, cfg.lake_root, [WAREHOUSE_PREFIX])
    return warehouse.query(cfg, sql)
