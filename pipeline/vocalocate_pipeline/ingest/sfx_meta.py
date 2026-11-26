"""A generic SFX dataset with a metadata CSV: references, extra candidate SFX and imitations.

This is the layout for datasets that come with semantic metadata (the
SFX_Metadata wish-list schema), such as the Vocal Imitation Set with its
same-class recordings or our synthetic demo set. Layout:

    metadata.csv with columns
        file, role, sound_id, imitates, category, tags, recording_id, description, imitator_id
    and optionally text_hint, hint_type; plus the audio files it lists (paths relative to the CSV).

* role is reference (a sound with imitations), library (an extra candidate SFX
  with no imitations, e.g. another take or a same-class recording), imitation, or
  unpaired_imitation (an imitation kept only as a training negative, such as a rejected
  or practice take: it gets its target's group key and split but never forms a pair).
* imitates is the sound_id of the reference an imitation imitates.
* tags are separated by ";".
* An imitation with a text_hint becomes a voice, text and target triple in the
  queries table, like the team recordings.
* recording_id groups clips cut from the same original recording. All clips of
  a recording, and the imitations of its reference, share a group key and so
  always land in the same split. A recording may have at most one reference.
* group_id (optional, for references and library clips) replaces that group key verbatim. A
  library clip can use it to follow another source's split, e.g. a Vocal Imitation Set
  recording following its class's VimSketch reference.
"""

from __future__ import annotations

from pathlib import Path

from . import IngestBuilder, IngestResult, read_table, require

SOURCE = "sfx_meta"
VERSION = "1.0"
ROLES = {"reference", "library", "imitation", "unpaired_imitation"}
IMITATION_ROLES = {"imitation", "unpaired_imitation"}


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def ingest(root: Path) -> IngestResult:
    return ingest_rows(read_table(require(root / "metadata.csv", "SFX metadata manifest")), root)


def ingest_rows(
    rows: list[dict], root: Path, source: str = SOURCE, version: str = VERSION, licence: str | None = None
) -> IngestResult:
    """Turn metadata rows (the metadata.csv columns) into raw files, SFX metadata and text-hint queries.

    Other datasets with semantic metadata (e.g. the Vocal Imitation Set) convert their own
    metadata to these rows and call this under their own source name. `file` may be absolute.
    """
    recording_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    reference_of_recording: dict[str, str] = {}
    for row in rows:
        role = row["role"].strip()
        if role not in ROLES:
            raise ValueError(f"{row['file']}: unknown role {role!r}")
        if role in IMITATION_ROLES:
            continue
        sound_id = row["sound_id"].strip()
        recording = _clean(row.get("recording_id")) or sound_id
        recording_of[sound_id] = recording
        group_of[sound_id] = _clean(row.get("group_id")) or f"{source}:{recording}"
        if role == "reference":
            other = reference_of_recording.setdefault(recording, sound_id)
            if other != sound_id:
                raise ValueError(f"recording {recording!r} has two references: {other!r} and {sound_id!r}")

    # A file may be listed more than once: the Vocal Imitation Set lists a multi-labelled
    # recording once per class, each with its own sound ID and metadata.
    b = IngestBuilder(root, source, version, licence, unique_keys=False)
    for row in rows:
        role = row["role"].strip()
        if role in IMITATION_ROLES:
            target = row["imitates"].strip()
            group = group_of[target] if target in reference_of_recording.values() else None
            present = b.add_file(
                row["file"], role, label=_clean(row.get("category")), group_key=group,
                imitator_id=_clean(row.get("imitator_id")),
            )
            hint = _clean(row.get("text_hint"))
            if present and hint and group and role == "imitation":
                b.add_query(row["file"], hint, _clean(row.get("hint_type")), group)
            continue
        sound_id = row["sound_id"].strip()
        recording = recording_of[sound_id]
        if b.add_file(row["file"], role, label=_clean(row.get("category")), group_key=group_of[sound_id],
                      extra={"sound_id": sound_id}):
            tags = [t.strip() for t in (row.get("tags") or "").split(";") if t.strip()]
            b.add_metadata(row["file"], _clean(row.get("category")), _clean(row.get("description")), tags or None,
                           recording)
    return b.result()
