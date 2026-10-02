"""Ingesters: turn a downloaded dataset folder into bronze manifest rows.

Each ingester only reads the dataset's own metadata and hashes the files. It
does not decode audio; that happens in the cleaning stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..audio import sha256_file


@dataclass
class RawFile:
    source: str
    source_version: str
    source_key: str
    path: Path
    role: str
    label: str | None = None
    group_key: str | None = None
    imitator_id: str | None = None
    project_id: str | None = None
    licence: str | None = None
    extra: dict = field(default_factory=dict)

    def to_row(self, run_id: str, ingested_at) -> dict:
        return {
            "source": self.source,
            "source_version": self.source_version,
            "source_key": self.source_key,
            "uri": str(self.path.resolve()),
            "file_sha256": sha256_file(self.path),
            "size_bytes": self.path.stat().st_size,
            "role": self.role,
            "label": self.label,
            "group_key": self.group_key,
            "imitator_id": self.imitator_id,
            "project_id": self.project_id,
            "licence": self.licence,
            "extra_json": json.dumps(self.extra, sort_keys=True) if self.extra else None,
            "ingested_at": ingested_at,
            "run_id": run_id,
        }


@dataclass
class IngestResult:
    files: list[RawFile]
    # Text hints that come with some imitations (team recordings only, for now).
    queries: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _registry() -> dict[str, Callable[[Path], IngestResult]]:
    from . import esc50, qvim_dev, team, vimsketch

    return {
        "esc50": esc50.ingest,
        "vimsketch": vimsketch.ingest,
        "qvim_dev": qvim_dev.ingest,
        "team": team.ingest,
    }


SOURCES = ("esc50", "vimsketch", "qvim_dev", "team")


def ingest(source: str, root: Path) -> IngestResult:
    try:
        fn = _registry()[source]
    except KeyError:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}") from None
    return fn(Path(root))
