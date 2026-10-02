"""QVIM challenge development set (zenodo.org/records/17453307): 121 references, 985 imitations.

Layout, as read by the QVIM challenge baseline:
    "DEV Dataset.csv" (first row skipped; columns Label, Class, Items, Query 1-3)
    Items/<Class>/<Items>, Queries/<Class>/<Query>
This is our held-out benchmark: every clip is put in the test split.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "qvim_dev"
VERSION = "dev"
LICENCE = "CC BY-SA 4.0"
QUERY_COLUMNS = ("Query 1", "Query 2", "Query 3")


def ingest(root: Path) -> IngestResult:
    meta = root / "DEV Dataset.csv"
    if not meta.exists():
        raise FileNotFoundError(f"QVIM DEV metadata not found at {meta}")
    files: list[RawFile] = []
    missing: list[str] = []
    seen: set[str] = set()

    def add(key: str, role: str, cls: str, group: str) -> None:
        if key in seen:
            return
        seen.add(key)
        path = root / key
        if not path.exists():
            missing.append(key)
            return
        files.append(
            RawFile(
                source=SOURCE,
                source_version=VERSION,
                source_key=key,
                path=path,
                role=role,
                label=cls,
                group_key=group,
                licence=LICENCE,
            )
        )

    with open(meta, newline="") as f:
        next(f)  # the file has a title row above the header
        for row in csv.DictReader(f):
            cls, item = row["Class"], row["Items"]
            if not cls or not item:
                continue
            group = f"{SOURCE}:{cls}/{item}"
            add(f"Items/{cls}/{item}", "reference", cls, group)
            for col in QUERY_COLUMNS:
                if row.get(col):
                    add(f"Queries/{cls}/{row[col]}", "imitation", cls, group)
    return IngestResult(files=files, missing=missing)
