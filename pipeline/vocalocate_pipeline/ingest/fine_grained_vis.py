"""Fine-grained Vocal Imitation Set v1.0 (zenodo.org/records/3538534): 12 classes, 9 imitated recordings each.

The Vocal Imitation Set has one imitated reference per class. This dataset takes 12 of its
classes (all animals) and collects imitations of the 9 other recordings of each class, the ones
the Vocal Imitation Set ships as un-imitated `non_reference` recordings. 108 references, 705
accepted imitations, 57 practice, draft or rejected takes. Layout:

    original_recordings/<class>/<file>          9 recordings per class, e.g. 001Animal_Domestic animals_ pets_Cat_Hiss
    original_recordings_license.csv             filename, username, user_url, file_url, freesound_id
    vocal_imitations/included/<file>            imitations that raters accepted
    vocal_imitations/excluded/<file>            practice, draft and rejected takes
    vocal_imitations.txt                        tab-separated: imitation_filename, category_d1..d6,
                                                reference_filename, included, participant_id,
                                                participants_sound_recording_description, ...

What it adds is same-class negatives that are known to be wrong: every recording of a class has
its own imitations, so an imitation of one cat hiss is a true negative for the other eight.

* Every recording is a reference with its own group, so it pairs only with its own imitations.
* category and tags are exactly what the VimSketch ingester gives the same class
  ("Animal_Domestic animals_pets_Cat_Hiss"), so in a lake that also holds VimSketch a class has
  10 imitated recordings: its 9 here plus its VimSketch reference.
* All of it goes to train (``TRAIN_SOURCES`` in splits.py): its value is the within-class
  negatives, which need the whole class in one split, and the QVIM sets are the test sets.
* recording_id is the Freesound ID. These 108 recordings are byte-identical to Vocal Imitation
  Set library clips (sources "vis" and "vis_extras"); the demo's loader keeps one copy of a clip
  and prefers it as a reference.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..flow_functions.audio import AUDIO_EXTENSIONS
from . import IngestResult
from .sfx_meta import ingest_rows
from .vimsketch import sound_metadata, vis_class_sound

SOURCE = "fgvis"
VERSION = "1.0"
LICENCE = "CC BY 4.0 (dataset); per-recording Freesound licences in original_recordings_license.csv"


def _read_table(path: Path, delimiter: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f, delimiter=delimiter)]


def _true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "t"}


def class_category(folder: str) -> str:
    """'001Animal_Domestic animals_ pets_Cat_Hiss' -> 'Animal_Domestic animals_pets_Cat_Hiss' (VimSketch's category)."""
    return sound_metadata(vis_class_sound(folder))[0]


def find_root(root: Path) -> Path:
    if (root / "vocal_imitations.txt").exists():
        return root
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        if (d / "vocal_imitations.txt").exists() and (d / "original_recordings").is_dir():
            return d
    raise FileNotFoundError(f"Fine-grained Vocal Imitation Set (vocal_imitations.txt) not found under {root}")


def build_rows(root: Path) -> list[dict]:
    """Convert the dataset's own metadata into sfx_meta rows (paths relative to `root`)."""
    fg = find_root(root)
    prefix = fg.relative_to(root).as_posix()

    def rel(*parts: str) -> str:
        return "/".join(p for p in (prefix if prefix != "." else "", *parts) if p)

    licences = fg / "original_recordings_license.csv"
    freesound = {r["filename"]: r.get("freesound_id", "") for r in _read_table(licences, ",")} if licences.exists() else {}

    rows, category_of = [], {}
    for folder in sorted(p for p in (fg / "original_recordings").iterdir() if p.is_dir()):
        category, tags, _ = sound_metadata(vis_class_sound(folder.name))
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS):
            category_of[path.name] = category
            rows.append({"file": rel("original_recordings", folder.name, path.name), "role": "reference",
                         "sound_id": path.name, "imitates": "", "category": category, "tags": ";".join(tags),
                         "recording_id": f"freesound:{freesound[path.name]}" if freesound.get(path.name) else path.name,
                         "description": " > ".join(tags), "imitator_id": "", "text_hint": "", "hint_type": ""})

    for im in _read_table(fg / "vocal_imitations.txt", "\t"):
        ref = im.get("reference_filename", "")
        if ref not in category_of:
            continue
        included = _true(im.get("included", ""))
        hint = im.get("participants_sound_recording_description", "") if included else ""
        hint = "" if hint.lower() in {"nan", "none", "n/a"} else hint
        rows.append({"file": rel("vocal_imitations", "included" if included else "excluded", im["imitation_filename"]),
                     "role": "imitation" if included else "unpaired_imitation", "sound_id": "", "imitates": ref,
                     "category": category_of[ref], "tags": "", "recording_id": "", "description": "",
                     # Participant IDs are per study, so keep them apart from the Vocal Imitation Set's.
                     "imitator_id": f"fgvis:{im['participant_id']}" if im.get("participant_id") else "",
                     "text_hint": hint, "hint_type": "participant description" if hint else ""})
    return rows


def ingest(root: Path) -> IngestResult:
    root = Path(root)
    return ingest_rows(build_rows(root), root, source=SOURCE, version=VERSION, licence=LICENCE)
