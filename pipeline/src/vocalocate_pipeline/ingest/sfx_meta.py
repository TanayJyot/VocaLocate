"""A generic SFX dataset with a metadata CSV: references, extra candidate SFX and imitations.

This is the layout for datasets that come with semantic metadata (the
SFX_Metadata wish-list schema), such as the Vocal Imitation Set with its
same-class recordings or our synthetic demo set. Layout:

    metadata.csv with columns
        file, role, sound_id, imitates, category, tags, recording_id, description, imitator_id
    and the audio files it lists (paths relative to the CSV).

* role is reference (a sound with imitations), library (an extra candidate SFX
  with no imitations, e.g. another take or a same-class recording) or imitation.
* imitates is the sound_id of the reference an imitation imitates.
* tags are separated by ";".
* recording_id groups clips cut from the same original recording. All clips of
  a recording, and the imitations of its reference, share a group key and so
  always land in the same split. A recording may have at most one reference.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "sfx_meta"
VERSION = "1.0"
ROLES = {"reference", "library", "imitation"}


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def ingest(root: Path) -> IngestResult:
    meta = root / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"SFX metadata manifest not found at {meta}")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))

    recording_of: dict[str, str] = {}
    reference_of_recording: dict[str, str] = {}
    for row in rows:
        role = row["role"].strip()
        if role not in ROLES:
            raise ValueError(f"{row['file']}: unknown role {role!r}")
        if role == "imitation":
            continue
        sound_id = row["sound_id"].strip()
        recording = _clean(row.get("recording_id")) or sound_id
        recording_of[sound_id] = recording
        if role == "reference":
            other = reference_of_recording.setdefault(recording, sound_id)
            if other != sound_id:
                raise ValueError(f"recording {recording!r} has two references: {other!r} and {sound_id!r}")

    files, metadata, missing = [], [], []
    for row in rows:
        path = root / row["file"]
        if not path.exists():
            missing.append(row["file"])
            continue
        role = row["role"].strip()
        if role == "imitation":
            target = row["imitates"].strip()
            recording = recording_of.get(target)
            group = f"{SOURCE}:{recording}" if target in reference_of_recording.values() else None
            files.append(
                RawFile(
                    source=SOURCE,
                    source_version=VERSION,
                    source_key=row["file"],
                    path=path,
                    role="imitation",
                    label=_clean(row.get("category")),
                    group_key=group,
                    imitator_id=_clean(row.get("imitator_id")),
                )
            )
            continue
        sound_id = row["sound_id"].strip()
        tags = [t.strip() for t in (row.get("tags") or "").split(";") if t.strip()]
        files.append(
            RawFile(
                source=SOURCE,
                source_version=VERSION,
                source_key=row["file"],
                path=path,
                role=role,
                label=_clean(row.get("category")),
                group_key=f"{SOURCE}:{recording_of[sound_id]}",
                extra={"sound_id": sound_id},
            )
        )
        metadata.append(
            {
                "source_key": row["file"],
                "category": _clean(row.get("category")),
                "description": _clean(row.get("description")),
                "tags": tags or None,
                "recording_id": recording_of[sound_id],
            }
        )
    return IngestResult(files=files, metadata=metadata, missing=missing)
