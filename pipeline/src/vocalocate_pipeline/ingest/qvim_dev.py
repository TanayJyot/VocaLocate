"""QVIM challenge development set (zenodo.org/records/17453307): 121 references, 985 imitations.

Layout, as read by the QVIM challenge baseline:
    "DEV Dataset.csv" (first row skipped; columns Label, Class, Items, Query 1-3)
    Items/<Class>/<Items>, Queries/<Class>/<Query>
The Zenodo release (DEVDataset.zip) names the index "DEVUpdateComplete.csv" and nests
everything in a "DEVUpdatedDataset/" folder; both are found automatically.
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


def _find_index(root: Path) -> tuple[Path, Path]:
    """Return (dataset root, index CSV), accepting the baseline's and the Zenodo release's names."""
    for base in (root, root / "DEVUpdatedDataset"):
        for name in ("DEV Dataset.csv", "DEVUpdateComplete.csv"):
            if (base / name).exists():
                return base, base / name
        found = sorted(p for p in base.glob("DEV*.csv") if not p.name.startswith("._"))
        if found and (base / "Items").is_dir():
            return base, found[0]
    raise FileNotFoundError(f"QVIM DEV metadata (DEV Dataset.csv or DEVUpdateComplete.csv) not found under {root}")


def ingest(root: Path) -> IngestResult:
    root, meta = _find_index(Path(root))
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

    with open(meta, newline="", encoding="utf-8") as f:
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
