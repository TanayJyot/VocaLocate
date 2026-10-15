"""Vocal Imitation Set v1.1 (Kim et al., DCASE 2018): 302 AudioSet classes, one imitated reference each.

github.com/interactiveaudiolab/VocalImitationSet (audio on Zenodo). Layout:

    original_recordings/reference/<file>        the 302 sounds that were imitated
    original_recordings/non_reference/<file>    other recordings of the same classes (~2,700)
    original_recordings_classes.txt             tab-separated: filename, classname, category_d1..d6,
                                                is_reference, multi_labeled
    original_recordings_license.csv             filename, ..., freesound_id
    vocal_imitations/included/<file>            5,601 imitations that raters accepted
    vocal_imitations.txt                        tab-separated: imitation_filename, reference_filename,
                                                included, participant_id,
                                                participants_sound_recording_description, ...

This is the real-data source for the mining demos. It maps onto the SFX_Metadata schema:

* category is the AudioSet class; tags are its ontology path (depth 1-6), so the
  metadata-guided miner gets real related pools: same class, then siblings that
  share ancestors. Same-class non-reference recordings are candidate SFX ("library"),
  the fine-grained distractors the dataset was built for.
* recording_id is the Freesound ID, so clips of one Freesound upload share a split.
* the imitator's own description of the sound becomes the text hint, a real
  voice + text + target triple for the CLAP branch.

Only `included` imitations are used. Drafts, practice takes and rejected imitations are skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import IngestResult
from .sfx_meta import ingest_rows

SOURCE = "vis"
VERSION = "1.1"
LICENCE = "CC BY 4.0 (dataset); per-recording Freesound licences in original_recordings_license.csv"
DEPTHS = [f"category_d{i}" for i in range(1, 7)]


def _read_table(path: Path, delimiter: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f, delimiter=delimiter)]


def _true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "t"}


def _find(root: Path, folders: list[str], name: str, index: dict[str, Path]) -> Path | None:
    for folder in folders:
        p = root / folder / name
        if p.exists():
            return p
    return index.get(name)


def build_rows(root: Path) -> list[dict]:
    """Convert the dataset's own metadata into sfx_meta rows (absolute file paths)."""
    classes = _read_table(root / "original_recordings_classes.txt", "\t")
    licences = root / "original_recordings_license.csv"
    freesound = {r["filename"]: r.get("freesound_id", "") for r in _read_table(licences, ",")} if licences.exists() else {}
    # Some releases nest files one level deeper; fall back to a filename index of the whole tree.
    index = {p.name: p for p in root.rglob("*.wav")}

    rows, seen_refs = [], set()
    for c in classes:
        name, cls = c["filename"], c.get("classname", "")
        is_ref = _true(c.get("is_reference", ""))
        folders = ["original_recordings/reference", "original_recordings/non_reference"]
        path = _find(root, folders if is_ref else folders[::-1], name, index)
        if path is None:
            path = root / ("original_recordings/reference" if is_ref else "original_recordings/non_reference") / name
        ontology = [c[d] for d in DEPTHS if c.get(d)]
        role = "reference" if is_ref and name not in seen_refs else "library"
        if is_ref:
            seen_refs.add(name)
        rows.append(
            {
                "file": str(path),
                "role": role,
                # Multi-labelled recordings appear once per class; the class keeps their sound IDs apart.
                "sound_id": name if role == "reference" else f"{cls}/{name}",
                "imitates": "",
                "category": cls,
                "tags": ";".join(dict.fromkeys(ontology + [cls])),
                "recording_id": f"freesound:{freesound[name]}" if freesound.get(name) else name,
                "description": " > ".join(ontology),
                "imitator_id": "",
                "text_hint": "",
                "hint_type": "",
            }
        )

    class_of = {c["filename"]: c.get("classname", "") for c in classes if _true(c.get("is_reference", ""))}
    for im in _read_table(root / "vocal_imitations.txt", "\t"):
        if not _true(im.get("included", "")) or im.get("reference_filename") not in class_of:
            continue
        name = im["imitation_filename"]
        path = _find(root, ["vocal_imitations/included"], name, index) or root / "vocal_imitations/included" / name
        hint = im.get("participants_sound_recording_description", "")
        rows.append(
            {
                "file": str(path),
                "role": "imitation",
                "sound_id": "",
                "imitates": im["reference_filename"],
                "category": class_of[im["reference_filename"]],
                "tags": "",
                "recording_id": "",
                "description": "",
                "imitator_id": im.get("participant_id", ""),
                "text_hint": hint if hint.lower() not in {"", "nan", "none", "n/a"} else "",
                "hint_type": "participant description" if hint else "",
            }
        )
    return rows


def ingest(root: Path) -> IngestResult:
    for required in ("original_recordings_classes.txt", "vocal_imitations.txt"):
        if not (root / required).exists():
            raise FileNotFoundError(f"Vocal Imitation Set metadata not found at {root / required}")
    return ingest_rows(build_rows(root), root, source=SOURCE, version=VERSION, licence=LICENCE)
