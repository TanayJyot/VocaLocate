"""Ingesters: turn a downloaded dataset folder into bronze manifest rows.

Each ingester only reads the dataset's own metadata and hashes the files. It
does not decode audio; that happens in the cleaning stage.

What differs between datasets is how their index files say which file is which
(file-name rules, CSV columns, folder layouts), so each dataset keeps a small
module that parses its own layout. Everything else is shared here: reading the
index tables, and `IngestBuilder`, which records files, metadata, text hints and
captions and applies the common rules (missing files, repeated keys, query and
caption IDs).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..flow_functions.audio import sha256_file


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
    # Semantic metadata for SFX, keyed by source_key (sfx_meta only, for now).
    metadata: list[dict] = field(default_factory=list)
    # Free-text captions for SFX, keyed by source_key (captdure only, for now).
    captions: list[dict] = field(default_factory=list)
    # Imitator profiles (team imitators.csv; participant surveys of the Vocal Imitation Set and VocalSketch).
    imitators: list[dict] = field(default_factory=list)
    # Splits the dataset itself assigns, keyed by source_key (team recordings only: val or test).
    splits: dict[str, str] = field(default_factory=dict)
    # The reference each imitation imitates, by source key, for groups with several references
    # (ESC-50-Voice: several takes of one recording, each imitated). Other imitations pair by group.
    targets: dict[str, str] = field(default_factory=dict)
    # The group to hash a file's split by, by source key, when it is not the file's own group key:
    # a recording another source also holds takes that source's group, so both share a split.
    split_groups: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def require(path: Path, what: str) -> Path:
    """Fail early, naming the dataset, when its index file is not where its layout puts it."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}")
    return path


def read_table(path: Path, delimiter: str = ",") -> list[dict]:
    """Read a CSV or TSV index into rows with stripped keys and values (a BOM is ignored)."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def short_id(*parts) -> str:
    """A stable ID for a query or caption: the first 32 hex digits of the SHA-256 of its parts."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


class IngestBuilder:
    """Collects one dataset's files and their annotations into an IngestResult.

    A file is addressed by its `key`, its path inside the dataset (the bronze `source_key`).
    `add_file` records a file that exists and lists one that does not as missing (once).
    With `unique_keys`, a key listed again is not recorded twice, but `add_file` still
    returns True so its other rows (e.g. a second caption) are kept.
    """

    def __init__(self, root: Path, source: str, version: str, licence: str | None = None, unique_keys: bool = True):
        self.root = Path(root)
        self.source, self.version, self.licence = source, version, licence
        self.unique_keys = unique_keys
        self.files: list[RawFile] = []
        self.queries: list[dict] = []
        self.metadata: list[dict] = []
        self.captions: list[dict] = []
        self.imitators: list[dict] = []
        self.splits: dict[str, str] = {}
        self.targets: dict[str, str] = {}
        self.split_groups: dict[str, str] = {}
        self.missing: list[str] = []
        self._present: set[str] = set()
        self._missing: set[str] = set()

    def has(self, key: str) -> bool:
        """Whether this key was already passed to add_file, present or missing."""
        return key in self._present or key in self._missing

    def add_file(
        self,
        key: str,
        role: str,
        *,
        path: Path | None = None,
        label: str | None = None,
        group_key: str | None = None,
        imitator_id: str | None = None,
        project_id: str | None = None,
        extra: dict | None = None,
    ) -> bool:
        """Record a file (at root / key unless `path` is given). Returns whether the file exists."""
        if key in self._present and self.unique_keys:
            return True
        if key in self._missing:
            return False
        path = path if path is not None else self.root / key
        if not path.exists():
            self._missing.add(key)
            self.missing.append(key)
            return False
        self._present.add(key)
        self.files.append(
            RawFile(
                source=self.source,
                source_version=self.version,
                source_key=key,
                path=path,
                role=role,
                label=label,
                group_key=group_key,
                imitator_id=imitator_id,
                project_id=project_id,
                licence=self.licence,
                extra=extra or {},
            )
        )
        return True

    def add_metadata(
        self,
        key: str,
        category: str | None,
        description: str | None = None,
        tags: list[str] | None = None,
        recording_id: str | None = None,
    ) -> None:
        """Semantic metadata for an SFX file (the sfx_metadata table)."""
        self.metadata.append(
            {"source_key": key, "category": category, "description": description, "tags": tags,
             "recording_id": recording_id}
        )

    def add_query(self, key: str, text_hint: str, hint_type: str | None, target_group_key: str) -> None:
        """A text hint recorded with an imitation: a voice, text and target triple (the queries table)."""
        self.queries.append(
            {"query_id": short_id(key, text_hint), "source_key": key, "text_hint": text_hint,
             "hint_type": hint_type, "target_group_key": target_group_key}
        )

    def add_caption(self, key: str, caption: str, language: str, annotator_id: str | None) -> None:
        """A free-text caption whose target is this file (the captions table)."""
        self.captions.append(
            {"caption_id": short_id(key, annotator_id, caption), "source_key": key, "caption": caption,
             "language": language, "annotator_id": annotator_id}
        )

    def set_split(self, key: str, split: str) -> None:
        """A split the dataset itself assigns to this file, instead of the group hash."""
        self.splits[key] = split

    def set_target(self, key: str, reference_key: str) -> None:
        """The exact reference this imitation imitates, when its group holds more than one."""
        self.targets[key] = reference_key

    def set_split_group(self, key: str, group_key: str) -> None:
        """Hash this file's split by another group key (one shared with another source's copy of the recording)."""
        self.split_groups[key] = group_key

    def result(self) -> IngestResult:
        return IngestResult(
            files=self.files, queries=self.queries, metadata=self.metadata, captions=self.captions,
            imitators=self.imitators, splits=self.splits, targets=self.targets, split_groups=self.split_groups,
            missing=self.missing,
        )


def _registry() -> dict[str, Callable[[Path], IngestResult]]:
    from . import (
        captdure, esc50, esc50_voice, fine_grained_vis, qvim_dev, qvim_eval, sfx_meta, team, vimsketch, vocal_imitation_set,
    )

    return {
        "esc50": esc50.ingest,
        "esc50_voice": esc50_voice.ingest,
        "vimsketch": vimsketch.ingest,
        "qvim_dev": qvim_dev.ingest,
        "qvim_eval": qvim_eval.ingest,
        "team": team.ingest,
        "sfx_meta": sfx_meta.ingest,
        "vis": vocal_imitation_set.ingest,
        "captdure": captdure.ingest,
        "fgvis": fine_grained_vis.ingest,
        "vis_extras": vocal_imitation_set.ingest_extras,
    }


SOURCES = ("esc50", "esc50_voice", "vimsketch", "qvim_dev", "qvim_eval", "team", "sfx_meta", "vis", "captdure",
           "fgvis", "vis_extras")


def ingest(source: str, root: Path) -> IngestResult:
    try:
        fn = _registry()[source]
    except KeyError:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}") from None
    return fn(Path(root))
