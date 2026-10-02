"""Data lake layout and Parquet I/O.

    lake/
      bronze/<source>/...                         raw downloads, never modified
      bronze/tables/raw_files/<source>.parquet    manifest of raw files
      silver/audio/<source>/<id[:2]>/<id>.wav     clean 32 kHz mono audio
      silver/tables/{clips,pairs,queries}/<source>.parquet
      gold/features/<source>/<id[:2]>/<id>.npy    log-mel features (float16)
      gold/tables/{features,splits}/<source>.parquet
      state/library_files/<library>.parquet       incremental-indexing state
      runs/<run_id>.parquet                       one row per pipeline run
      warehouse/vocalocate.duckdb                 query layer over all tables

Each table is stored as one Parquet file per source, so re-running a source
replaces its own file and never touches other sources. The local folder can be
swapped for S3/MinIO later without changing the layout.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import TABLES

TABLE_LAYER = {
    "raw_files": "bronze/tables",
    "clips": "silver/tables",
    "pairs": "silver/tables",
    "queries": "silver/tables",
    "features": "gold/tables",
    "splits": "gold/tables",
    "library_files": "state",
    "pipeline_runs": "runs",
}


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)

    def table_dir(self, name: str) -> Path:
        if name == "pipeline_runs":
            return self.root / "runs"
        return self.root / TABLE_LAYER[name] / name

    def table_path(self, name: str, part: str) -> Path:
        return self.table_dir(name) / f"{part}.parquet"

    def clean_audio_path(self, source: str, clip_id: str) -> Path:
        return self.root / "silver" / "audio" / source / clip_id[:2] / f"{clip_id}.wav"

    def feature_path(self, source: str, clip_id: str) -> Path:
        return self.root / "gold" / "features" / source / clip_id[:2] / f"{clip_id}.npy"

    def write(self, name: str, part: str, table: pa.Table) -> Path:
        if not table.schema.equals(TABLES[name], check_metadata=False):
            raise ValueError(f"Table {name} does not match its schema")
        path = self.table_path(name, part)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)  # atomic, so readers never see half a file
        return path

    def read(self, name: str, part: str) -> pa.Table | None:
        path = self.table_path(name, part)
        return pq.read_table(path, schema=TABLES[name]) if path.exists() else None

    def parts(self, name: str) -> list[Path]:
        d = self.table_dir(name)
        return sorted(d.glob("*.parquet")) if d.exists() else []
