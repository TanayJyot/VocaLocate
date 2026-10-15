"""DuckDB warehouse: the query layer over the lake's Parquet tables.

The lake is the source of truth. Each load rebuilds the DuckDB file from the
Parquet tables into a temporary file and swaps it in, so a failed load never
leaves a half-built warehouse behind.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import PipelineConfig
from .lake import Lake
from .schemas import TABLES

VIEWS = {
    # Imitation/reference pairs ready for contrastive training, with their split.
    "v_training_pairs": """
        SELECT p.source, p.group_key, s.split,
               i.clip_id AS imitation_clip_id, i.clean_path AS imitation_path, i.imitator_id,
               r.clip_id AS reference_clip_id, r.clean_path AS reference_path, r.label,
               fi.feature_path AS imitation_feature_path, fr.feature_path AS reference_feature_path
        FROM pairs p
        JOIN clips i ON i.source = p.source AND i.clip_id = p.imitation_clip_id AND i.status = 'ok'
        JOIN clips r ON r.source = p.source AND r.clip_id = p.reference_clip_id AND r.status = 'ok'
        JOIN splits s ON s.source = r.source AND s.clip_id = r.clip_id
        LEFT JOIN features fi ON fi.source = i.source AND fi.clip_id = i.clip_id
        LEFT JOIN features fr ON fr.source = r.source AND fr.clip_id = r.clip_id
    """,
    # Voice, text and target triples. The target can live in any source.
    "v_triples": """
        SELECT q.query_id, q.text_hint, q.hint_type, q.target_group_key,
               i.clip_id AS imitation_clip_id, i.clean_path AS imitation_path, i.imitator_id,
               t.clip_id AS target_clip_id, t.source AS target_source, t.clean_path AS target_path
        FROM queries q
        JOIN clips i ON i.source = q.source AND i.clip_id = q.imitation_clip_id AND i.status = 'ok'
        LEFT JOIN clips t ON t.group_key = q.target_group_key AND t.role IN ('reference', 'library')
                         AND t.status = 'ok'
    """,
    # What the search app indexes: active library files with clean audio and features.
    "v_library_index": """
        SELECT lf.library, lf.path, lf.project_id, c.clip_id, c.duration_sec,
               f.feature_path, f.summary
        FROM library_files lf
        JOIN clips c ON c.source = 'library_' || lf.library AND c.source_key = lf.path AND c.status = 'ok'
        JOIN features f ON f.source = c.source AND f.clip_id = c.clip_id
        WHERE lf.status = 'active'
    """,
    # Candidate SFX for mining and evaluation: clean references and library clips of
    # dataset sources, with their split and any semantic metadata (category falls back
    # to the dataset label).
    "v_sfx_candidates": """
        SELECT c.source, c.clip_id, c.role, c.group_key, s.split, c.clean_path, f.feature_path,
               coalesce(m.category, c.label) AS category, m.tags, m.recording_id, m.description
        FROM clips c
        JOIN splits s ON s.source = c.source AND s.clip_id = c.clip_id
        JOIN features f ON f.source = c.source AND f.clip_id = c.clip_id
        LEFT JOIN sfx_metadata m ON m.source = c.source AND m.clip_id = c.clip_id
        WHERE c.status = 'ok' AND c.role IN ('reference', 'library') AND c.source NOT LIKE 'library_%'
    """,
    # Mined negatives with the metadata of the target and the negative, for analysis.
    "v_hard_negatives": """
        SELECT h.*, pm.category AS positive_category, nm.category AS negative_category,
               nm.recording_id AS negative_recording_id
        FROM hard_negatives h
        LEFT JOIN sfx_metadata pm ON pm.clip_id = h.positive_clip_id
        LEFT JOIN sfx_metadata nm ON nm.clip_id = h.negative_clip_id
    """,
    # Data-quality summary per source.
    "v_clip_quality": """
        SELECT source, role, status, coalesce(reject_reason, '') AS reason,
               count(*) AS n, round(avg(duration_sec), 2) AS avg_duration_sec,
               sum(CASE WHEN truncated THEN 1 ELSE 0 END) AS n_truncated,
               sum(CASE WHEN clipped THEN 1 ELSE 0 END) AS n_clipped
        FROM clips GROUP BY ALL ORDER BY ALL
    """,
    # The same audio file in more than one source (e.g. VimSketch and the Vocal Imitation Set).
    "v_cross_source_duplicates": """
        SELECT clip_id, list(DISTINCT source ORDER BY source) AS sources, count(*) AS n
        FROM clips WHERE status = 'ok'
        GROUP BY clip_id HAVING count(DISTINCT source) > 1
    """,
}


def load(cfg: PipelineConfig) -> Path:
    lake = Lake(cfg.lake_root)
    target = cfg.warehouse_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)

    con = duckdb.connect(str(tmp))
    try:
        for name, schema in TABLES.items():
            parts = lake.parts(name)
            if parts:
                files = ", ".join(f"'{p.as_posix()}'" for p in parts)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet([{files}], union_by_name = true)")
            else:
                empty = schema.empty_table()
                con.register("empty_tbl", empty)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM empty_tbl")
                con.unregister("empty_tbl")
        for name, sql in VIEWS.items():
            con.execute(f"CREATE VIEW {name} AS {sql}")
    finally:
        con.close()
    tmp.replace(target)
    return target


def query(cfg: PipelineConfig, sql: str):
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()
