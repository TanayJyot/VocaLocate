"""CAPTDURE (Okamoto et al., Interspeech 2023): captioned single-source sounds, held out for the text branch.

zenodo.org/records/7965763. The single-source part is 1,044 five-second clips of 14
household sound classes that the authors recorded themselves, with 4,902 crowd-sourced
captions (3 to 5 per clip, written in Japanese and translated to English). Layout:

    single_source_source/<class>/<class>_<NN>_<NNN>.wav
    single_source_caption.csv       tab-separated despite the name, UTF-8 with a BOM:
                                    worker ID, wavfile, caption (ja), caption (en);
                                    wavfile is relative to single_source_source/

A2 Part Two reserves it for testing the CLAP text-to-audio branch: the audio is not
from Freesound or AudioSet and the set was released after the CLAP models we use, so
CLAP has not seen it. Every clip goes to the test split. Each English caption becomes a
row in the captions table (a text-only query whose target is its own clip); the
Japanese caption is kept alongside.

* category is the class folder; recording_id is `<class>_<NN>`, the object or device the
  clip was recorded from, so other takes of it count as usable substitutes in nDCG.
  The naming is our reading of the file names, not documented by the dataset.
* The mixture-source part (two sources per clip) and the caption appropriateness
  scores are not used.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "captdure"
VERSION = "1.0"
LICENCE = "CC BY 4.0 (Zenodo record 7965763)"
AUDIO_DIR = "single_source_source"
CAPTIONS = "single_source_caption.csv"


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    # The header line is wrapped in one pair of quotes: "worker ID\twavfile\t...".
    if lines and lines[0].startswith('"') and lines[0].endswith('"') and "\t" in lines[0]:
        lines[0] = lines[0][1:-1]
    reader = csv.DictReader(lines, delimiter="\t")
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def _column(row: dict, *names: str) -> str:
    for name in names:
        if row.get(name):
            return row[name]
    return ""


def ingest(root: Path) -> IngestResult:
    index = root / CAPTIONS
    if not index.exists():
        raise FileNotFoundError(f"CAPTDURE caption file not found at {index}")
    audio_root = root / AUDIO_DIR

    files, metadata, captions, missing, seen = [], [], [], [], set()
    for row in _rows(index):
        wav = _column(row, "wavfile").lstrip("./")
        text = _column(row, "caption (en)")
        if not wav or not text:
            continue
        path = audio_root / wav
        if not path.exists():
            if wav not in seen:
                missing.append(wav)
                seen.add(wav)
            continue
        if wav not in seen:
            seen.add(wav)
            cls = Path(wav).parent.name or Path(wav).stem.rsplit("_", 2)[0]
            recording = Path(wav).stem.rsplit("_", 1)[0]
            files.append(
                RawFile(
                    source=SOURCE,
                    source_version=VERSION,
                    source_key=wav,
                    path=path,
                    role="reference",
                    label=cls,
                    group_key=f"{SOURCE}:{Path(wav).stem}",
                    licence=LICENCE,
                )
            )
            metadata.append(
                {"source_key": wav, "category": cls, "description": None, "tags": [cls], "recording_id": recording}
            )
        worker = _column(row, "worker ID") or None
        captions.append(
            {
                "caption_id": hashlib.sha256(f"{wav}|{worker}|{text}".encode()).hexdigest()[:32],
                "source_key": wav,
                "caption": text,
                "language": "en",
                "annotator_id": worker,
            }
        )
    return IngestResult(files=files, metadata=metadata, captions=captions, missing=missing)
