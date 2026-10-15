"""Table schemas for every layer of the lake and the warehouse.

IDs line up with the A2 Part One wish-list schemas: `clip_id` plays the role
of sound_id / imitation_id (it is the SHA-256 of the raw file, so the same
file always gets the same ID), `imitator_id` links to imitator profiles and
`query_id` links text hints to imitations.
"""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

ROLES = {"reference", "imitation", "library"}
STATUSES = {"ok", "rejected", "duplicate"}
SPLITS = {"train", "val", "test"}
LIBRARY_STATUSES = {"active", "deleted"}
TOWERS = {"sfx", "imitation"}
MINING_STRATEGIES = {"retrieval", "metadata", "random"}
MINING_SELECTIONS = {"band", "fallback", "random"}


def _schema(*cols: tuple[str, pa.DataType, bool]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in cols])


# Bronze: one row per raw file we know about. Files are never modified.
RAW_FILES = _schema(
    ("source", pa.string(), False),
    ("source_version", pa.string(), False),
    ("source_key", pa.string(), False),  # path relative to the dataset root
    ("uri", pa.string(), False),  # where the raw bytes live
    ("file_sha256", pa.string(), False),
    ("size_bytes", pa.int64(), False),
    ("role", pa.string(), False),
    ("label", pa.string(), True),
    ("group_key", pa.string(), True),  # the reference sound this file is, or imitates
    ("imitator_id", pa.string(), True),
    ("project_id", pa.string(), True),
    ("licence", pa.string(), True),
    ("extra_json", pa.string(), True),  # source-specific metadata, e.g. ESC-50 fold
    ("ingested_at", pa.timestamp("us", tz="UTC"), False),
    ("run_id", pa.string(), False),
)

# Silver: one row per raw file after cleaning. Clean audio is 32 kHz mono 16-bit WAV.
CLIPS = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("source_key", pa.string(), False),
    ("role", pa.string(), False),
    ("label", pa.string(), True),
    ("group_key", pa.string(), True),
    ("imitator_id", pa.string(), True),
    ("project_id", pa.string(), True),
    ("licence", pa.string(), True),
    ("status", pa.string(), False),
    ("reject_reason", pa.string(), True),
    ("orig_sample_rate", pa.int32(), True),
    ("orig_channels", pa.int32(), True),
    ("orig_duration_sec", pa.float32(), True),
    ("duration_sec", pa.float32(), True),
    ("peak_dbfs", pa.float32(), True),
    ("rms_dbfs", pa.float32(), True),
    ("truncated", pa.bool_(), False),
    ("clipped", pa.bool_(), False),
    ("clean_path", pa.string(), True),
    ("config_fingerprint", pa.string(), False),
    ("run_id", pa.string(), False),
)

# Silver: which imitation imitates which reference.
PAIRS = _schema(
    ("source", pa.string(), False),
    ("imitation_clip_id", pa.string(), False),
    ("reference_clip_id", pa.string(), False),
    ("group_key", pa.string(), False),
)

# Silver: text hints recorded with an imitation (voice, text and target triples).
QUERIES = _schema(
    ("query_id", pa.string(), False),
    ("source", pa.string(), False),
    ("imitation_clip_id", pa.string(), False),
    ("text_hint", pa.string(), False),
    ("hint_type", pa.string(), True),
    ("target_group_key", pa.string(), False),
)

# Silver: optional semantic metadata per SFX (the SFX_Metadata wish-list schema).
# Only training-time mining and analysis read it; the deployed search never does.
SFX_METADATA = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("category", pa.string(), True),
    ("description", pa.string(), True),
    ("tags", pa.list_(pa.string()), True),
    # Clips cut from the same original recording share this ID (e.g. a Freesound ID).
    ("recording_id", pa.string(), True),
)

# Gold: cached log-mel features and a fixed-size summary vector per clean clip.
FEATURES = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("feature_version", pa.string(), False),
    ("feature_path", pa.string(), False),
    ("n_frames", pa.int32(), False),
    ("summary", pa.list_(pa.float32()), False),
)

# Gold: train/val/test assignment. Every clip of a reference sound lands in one split.
SPLITS_TABLE = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("source_key", pa.string(), False),
    ("group_key", pa.string(), False),
    ("split", pa.string(), False),
    ("strategy", pa.string(), False),
)

# Gold: encoder embeddings, cached per clip and model so re-indexing only embeds new clips.
EMBEDDINGS = _schema(
    ("clip_id", pa.string(), False),
    ("source", pa.string(), False),
    ("model_version", pa.string(), False),
    ("tower", pa.string(), False),  # "sfx" or "imitation"
    ("vector", pa.list_(pa.float32()), False),
)

# Gold: mined semi-hard negatives for fine-tuning. This is the "mined hard negatives"
# dataset from A2 Part Two: the relevance schema with a model score instead of a human rating.
HARD_NEGATIVES = _schema(
    ("strategy", pa.string(), False),  # retrieval or metadata
    ("model_version", pa.string(), False),  # the model that scored the candidates
    ("imitation_clip_id", pa.string(), False),
    ("positive_clip_id", pa.string(), False),
    ("negative_clip_id", pa.string(), False),
    ("negative_rank", pa.int32(), False),  # rank of the negative in the model's retrieval list
    ("positive_score", pa.float32(), False),
    ("negative_score", pa.float32(), False),
    ("selection", pa.string(), False),  # band, fallback or random
    ("relation", pa.string(), True),  # how metadata relates the negative to the target
    ("run_id", pa.string(), False),
)

# State for incremental indexing of a user's own library.
LIBRARY_FILES = _schema(
    ("library", pa.string(), False),
    ("path", pa.string(), False),
    ("size_bytes", pa.int64(), False),
    ("mtime_ns", pa.int64(), False),
    ("file_sha256", pa.string(), False),
    ("project_id", pa.string(), True),
    ("status", pa.string(), False),
    ("first_seen", pa.timestamp("us", tz="UTC"), False),
    ("last_seen", pa.timestamp("us", tz="UTC"), False),
)

# One row per pipeline run, for lineage and monitoring.
PIPELINE_RUNS = _schema(
    ("run_id", pa.string(), False),
    ("flow", pa.string(), False),
    ("source", pa.string(), False),
    ("started_at", pa.timestamp("us", tz="UTC"), False),
    ("finished_at", pa.timestamp("us", tz="UTC"), False),
    ("status", pa.string(), False),
    ("n_input", pa.int64(), False),
    ("n_processed", pa.int64(), False),
    ("n_reused", pa.int64(), False),
    ("n_ok", pa.int64(), False),
    ("n_rejected", pa.int64(), False),
    ("n_duplicate", pa.int64(), False),
    ("config_fingerprint", pa.string(), False),
    ("message", pa.string(), True),
)

TABLES: dict[str, pa.Schema] = {
    "raw_files": RAW_FILES,
    "clips": CLIPS,
    "pairs": PAIRS,
    "queries": QUERIES,
    "sfx_metadata": SFX_METADATA,
    "features": FEATURES,
    "splits": SPLITS_TABLE,
    "embeddings": EMBEDDINGS,
    "hard_negatives": HARD_NEGATIVES,
    "library_files": LIBRARY_FILES,
    "pipeline_runs": PIPELINE_RUNS,
}

_ENUMS: dict[str, dict[str, set[str]]] = {
    "raw_files": {"role": ROLES},
    "clips": {"role": ROLES, "status": STATUSES},
    "splits": {"split": SPLITS},
    "library_files": {"status": LIBRARY_STATUSES},
    "embeddings": {"tower": TOWERS},
    "hard_negatives": {"strategy": MINING_STRATEGIES, "selection": MINING_SELECTIONS},
}


class SchemaError(ValueError):
    pass


def to_table(name: str, rows: Iterable[dict]) -> pa.Table:
    """Build a table from row dicts and check it against the schema.

    Types are enforced by pyarrow; we also check required columns for nulls and
    enum columns for unknown values, which pyarrow does not do.
    """
    schema = TABLES[name]
    rows = list(rows)
    table = pa.Table.from_pylist(rows, schema=schema)
    for f in schema:
        if not f.nullable and table.column(f.name).null_count:
            raise SchemaError(f"{name}.{f.name} has {table.column(f.name).null_count} null values")
    for col, allowed in _ENUMS.get(name, {}).items():
        bad = set(table.column(col).to_pylist()) - allowed
        if bad:
            raise SchemaError(f"{name}.{col} has unknown values {sorted(bad)}")
    return table
