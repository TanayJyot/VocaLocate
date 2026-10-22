"""Command-line entry point: `vocalocate-pipeline <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from . import remote, warehouse
from .config import load_config
from .flows import run_dataset_flow, run_library_flow
from .ingest import SOURCES
from .sync import sync


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vocalocate-pipeline", description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=os.environ.get("VOCALOCATE_CONFIG"),
        help="TOML settings file, e.g. configs/prod.toml (default: $VOCALOCATE_CONFIG, else built-in defaults)",
    )
    p.add_argument("--lake", type=Path, help="Lake root; overrides the config file")
    p.add_argument("--workers", type=int, help="Worker processes; overrides the config file")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("ingest-dataset", help="Ingest, clean and transform a downloaded dataset")
    d.add_argument("--source", required=True, choices=SOURCES)
    d.add_argument("--path", required=True, type=Path, help="Dataset root folder")

    lib = sub.add_parser("index-library", help="Incrementally index a user's SFX folders")
    lib.add_argument("--root", required=True, type=Path, action="append", help="Library folder (repeatable)")
    lib.add_argument("--name", default="default", help="Library name")

    sub.add_parser("load-warehouse", help="Rebuild the DuckDB warehouse from the lake")

    q = sub.add_parser("report", help="Print a data-quality summary from the warehouse")
    q.add_argument("--sql", help="Run this SQL instead of the default report")

    s = sub.add_parser("sync", help="Copy new and changed lake files between a local folder and S3")
    s.add_argument("src", help="Local path or s3:// URI")
    s.add_argument("dst", help="Local path or s3:// URI")
    s.add_argument("--only", action="append", default=[], help="Only this sub-folder, e.g. gold/features/vis")

    r = sub.add_parser("remote", help="Run a flow against the S3 lake (what the AWS pipeline tasks run)")
    r.add_argument("action", choices=["ingest", "load-warehouse", "report"])
    r.add_argument("--source", choices=SOURCES, help="Dataset to ingest (raw files under s3://<raw>/<source>/)")
    r.add_argument("--lake-uri", help="S3 lake root (default: $VOCALOCATE_LAKE_URI)")
    r.add_argument("--raw-uri", help="S3 raw-data root (default: $VOCALOCATE_RAW_URI)")
    r.add_argument("--sql", help="SQL for the report action")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(args.config, lake_root=args.lake, workers=args.workers)

    if args.command == "ingest-dataset":
        print(json.dumps(run_dataset_flow(args.source, args.path, cfg).as_dict(), indent=2))
    elif args.command == "index-library":
        print(json.dumps(run_library_flow(args.root, cfg, library=args.name).as_dict(), indent=2))
    elif args.command == "load-warehouse":
        print(warehouse.load(cfg))
    elif args.command == "report":
        for row in warehouse.query(cfg, args.sql or "SELECT * FROM v_clip_quality"):
            print(*row, sep="\t")
    elif args.command == "sync":
        print(sync(args.src, args.dst, args.only))
    elif args.command == "remote":
        if args.action == "ingest":
            if not args.source:
                raise SystemExit("remote ingest needs --source")
            summary = remote.ingest(args.source, cfg, args.lake_uri, args.raw_uri)
            print(json.dumps(summary.as_dict(), indent=2))
        elif args.action == "load-warehouse":
            print(remote.load_warehouse(cfg, args.lake_uri))
        else:
            for row in remote.report(cfg, args.sql or "SELECT * FROM v_clip_quality", args.lake_uri):
                print(*row, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
