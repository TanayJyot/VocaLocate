"""Team imitation recordings (A2 Part Two, "Data we can generate").

Layout:
    recordings.csv with columns
        file, imitator_id, target_group_key, text_hint, hint_type
    and the audio files it lists (paths relative to the CSV).
`target_group_key` names the reference sound being imitated in another
source, e.g. "esc50:100032" or "vimsketch:<reference name>". Rows with a text
hint become voice, text and target triples in the queries table.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from . import IngestResult, RawFile

SOURCE = "team"
VERSION = "live"
LICENCE = "team-internal"


def ingest(root: Path) -> IngestResult:
    meta = root / "recordings.csv"
    if not meta.exists():
        raise FileNotFoundError(f"Team recordings manifest not found at {meta}")
    files, queries, missing = [], [], []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            path = root / row["file"]
            if not path.exists():
                missing.append(row["file"])
                continue
            target = row["target_group_key"].strip()
            files.append(
                RawFile(
                    source=SOURCE,
                    source_version=VERSION,
                    source_key=row["file"],
                    path=path,
                    role="imitation",
                    group_key=target,
                    imitator_id=row["imitator_id"].strip(),
                    licence=LICENCE,
                )
            )
            hint = (row.get("text_hint") or "").strip()
            if hint:
                queries.append(
                    {
                        "query_id": hashlib.sha256(f"{row['file']}|{hint}".encode()).hexdigest()[:32],
                        "source_key": row["file"],
                        "text_hint": hint,
                        "hint_type": (row.get("hint_type") or "").strip() or None,
                        "target_group_key": target,
                    }
                )
    return IngestResult(files=files, queries=queries, missing=missing)
