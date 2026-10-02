"""ESC-50 (github.com/karolpiczak/ESC-50): 2,000 five-second clips in 50 classes.

Layout: meta/esc50.csv (filename, fold, target, category, esc10, src_file, take)
and audio/<filename>. We use it as a small stand-in library to test the
pipeline end to end, not for training.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "esc50"
VERSION = "2.0"
LICENCE = "CC BY-NC 3.0"


def ingest(root: Path) -> IngestResult:
    meta = root / "meta" / "esc50.csv"
    if not meta.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found at {meta}")
    files, missing = [], []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            path = root / "audio" / row["filename"]
            if not path.exists():
                missing.append(row["filename"])
                continue
            files.append(
                RawFile(
                    source=SOURCE,
                    source_version=VERSION,
                    source_key=f"audio/{row['filename']}",
                    path=path,
                    role="library",
                    label=row["category"],
                    # Takes cut from the same Freesound recording share a group,
                    # so they can never be split across train and test.
                    group_key=f"{SOURCE}:{row['src_file']}",
                    licence=LICENCE,
                    extra={"fold": int(row["fold"]), "esc10": row["esc10"] == "True", "take": row["take"]},
                )
            )
    return IngestResult(files=files, missing=missing)
