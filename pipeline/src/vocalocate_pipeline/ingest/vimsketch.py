"""VimSketch (zenodo.org/records/2596911): 542 reference sounds, 12,453 vocal imitations.

VimSketch merges the Vocal Imitation Set (sounds 001-302) and VocalSketch (303-542), one
reference per sound. It is the training set of the QVIM 2025 challenge baseline and winner.

Layout, as read by the QVIM challenge baseline:
    references/<prefix>_<reference name>.wav
    vocal_imitations/<prefix>_<reference name>.wav
    reference_file_names.csv, vocal_imitation_file_names.csv (one filename per line)
An imitation belongs to the reference with the same name after the first "_".
If the CSV lists are missing we list the folders instead.

Metadata for version B comes from the reference names themselves:

* Vocal Imitation Set sounds are named ``<class index><AudioSet ontology path>``, e.g.
  ``000Animal_Domestic animals_pets_Cat_Growling``: tags are the path's parts, as in the
  Vocal Imitation Set ingester, and the category is the sound itself (one reference per class).
* VocalSketch sounds are named ``<index>-<label>-<subset>``, e.g.
  ``002-accordian_playing-everyday``: the category is the subset (everyday, acoustic
  instruments, commercial synthesizers, single synthesizer) and the tags are the subset plus
  the label's words. This is coarser than the AudioSet ontology; it is all VocalSketch has.

VocalSketch's excluded imitations (practice takes, drafts and takes raters rejected; 9,553 in
release 1.1.2, zenodo.org/records/1251982) are what the QVIM 2025 winner used as triplet
negatives. When the VocalSketch release is unpacked next to VimSketch (or inside the folder
passed in), they are added as ``unpaired_imitation`` clips of their sound: same group and split,
never a training pair. VimSketch already holds VocalSketch's included imitations, so those are
not read again.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from ..audio import AUDIO_EXTENSIONS
from . import IngestResult
from .sfx_meta import ingest_rows

SOURCE = "vimsketch"
VERSION = "1.0"
LICENCE = "CC BY 4.0 (VimSketch); VocalSketch excluded imitations: VocalSketch 1.1.2 licence"

_VOCALSKETCH_NAME = re.compile(
    r"^\d{3}-(?P<label>.+)-(?P<subset>acoustic_instruments|commercial_synthesizers|everyday|single_synthesizer)$")
_VIS_NAME = re.compile(r"^\d{3}(?P<path>\D.*)$")
_STOPWORDS = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "the", "to", "with"}


def reference_name(filename: str) -> str:
    return "_".join(filename.split("_")[1:])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def sound_metadata(name: str) -> tuple[str, list[str], str]:
    """(category, tags, description) for a reference name (after the combined-dataset prefix)."""
    stem = Path(name).stem
    m = _VOCALSKETCH_NAME.match(stem)
    if m:
        subset = m["subset"].replace("_", " ")
        words = [w for w in m["label"].split("_") if w and w not in _STOPWORDS and not w.isdigit()]
        return subset, list(dict.fromkeys([subset, *words])), " ".join(m["label"].split("_"))
    m = _VIS_NAME.match(stem)
    parts = [p for p in (m["path"] if m else stem).split("_") if p]
    return (m["path"] if m else stem), parts, " > ".join(parts)


def _list(root: Path, folder: str, csv_name: str) -> list[str]:
    listing = root / csv_name
    if listing.exists():
        return [line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(p.name for p in (root / folder).iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def _is_vimsketch(d: Path) -> bool:
    return (d / "reference_file_names.csv").exists() or (d / "references").is_dir()


def _is_vocalsketch(d: Path) -> bool:
    return (d / "sound_recordings.csv").exists() and (d / "vocal_imitations.csv").exists()


def _children(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))) if d.is_dir() else []


def find_dirs(root: Path) -> tuple[Path, Path | None]:
    """(VimSketch folder, VocalSketch folder or None), from either folder or one that holds both."""
    vim = root if _is_vimsketch(root) else next((d for d in _children(root) if _is_vimsketch(d)), None)
    if vim is None:
        raise FileNotFoundError(f"VimSketch (references/ or reference_file_names.csv) not found under {root}")
    for d in [root, *_children(root), *_children(vim.parent)]:
        if _is_vocalsketch(d):
            return vim, d
    return vim, None


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def vocalsketch_excluded(vs: Path) -> list[dict]:
    """VocalSketch's excluded imitations: [{path, key, participant_id}], key = (label slug, subset slug).

    The key matches a VimSketch reference named <index>-<label>-<subset>. An imitation made after
    hearing a recording names it (sound_recording); one made from a text label only has the label,
    which sound_recordings.csv maps to its recording.
    """
    recording_of_label = {r["sound_label_id"]: r["filename"] for r in _read_csv(vs / "sound_recordings.csv")}
    out = []
    for table, folder in (("vocal_imitations.csv", "vocal_imitations"),
                          ("vocal_imitaitons_set2.csv", "vocal_imitations_set2"),  # sic, as released
                          ("vocal_imitations_set2.csv", "vocal_imitations_set2")):
        if not (vs / table).exists():
            continue
        for row in _read_csv(vs / table):
            if row.get("included", "").lower() == "true":
                continue
            recording = row.get("sound_recording") or recording_of_label.get(row.get("sound_label_id", ""), "")
            if not recording:
                continue
            key = (slug(Path(recording).stem), slug(row.get("audio_concept_subset", "")))
            out.append({"path": vs / folder / "excluded" / row["filename"], "key": key,
                        "participant_id": row.get("participant_id", "")})
    return out


def build_rows(root: Path) -> list[dict]:
    vim, vs = find_dirs(root)
    rows, sound_ids, by_key = [], set(), {}

    def rel(path: Path) -> str:
        return Path(os.path.relpath(path, root)).as_posix()

    for name in _list(vim, "references", "reference_file_names.csv"):
        sound = reference_name(name)
        category, tags, description = sound_metadata(sound)
        sound_ids.add(sound)
        m = _VOCALSKETCH_NAME.match(Path(sound).stem)
        if m:
            by_key[(slug(m["label"]), m["subset"])] = sound
        rows.append({"file": rel(vim / "references" / name), "role": "reference", "sound_id": sound, "imitates": "",
                     "category": category, "tags": ";".join(tags), "recording_id": sound,
                     "description": description, "imitator_id": "", "text_hint": "", "hint_type": ""})
    category_of = {r["sound_id"]: r["category"] for r in rows}
    for name in _list(vim, "vocal_imitations", "vocal_imitation_file_names.csv"):
        sound = reference_name(name)
        rows.append({"file": rel(vim / "vocal_imitations" / name), "role": "imitation", "sound_id": "",
                     # An imitation whose reference is not in the dataset gets no group, and so no pair.
                     "imitates": sound if sound in sound_ids else "", "category": category_of.get(sound, ""),
                     "tags": "", "recording_id": "", "description": "", "imitator_id": "", "text_hint": "",
                     "hint_type": ""})
    for ex in vocalsketch_excluded(vs) if vs else []:
        sound = by_key.get(ex["key"])
        if sound is None:
            continue
        rows.append({"file": rel(ex["path"]), "role": "unpaired_imitation", "sound_id": "", "imitates": sound,
                     "category": category_of[sound], "tags": "", "recording_id": "", "description": "",
                     "imitator_id": f"vocalsketch:{ex['participant_id']}" if ex["participant_id"] else "",
                     "text_hint": "", "hint_type": ""})
    return rows


def ingest(root: Path) -> IngestResult:
    return ingest_rows(build_rows(Path(root)), Path(root), source=SOURCE, version=VERSION, licence=LICENCE)
