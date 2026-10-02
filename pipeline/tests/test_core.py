from pathlib import Path

import pyarrow as pa
import pytest

from vocalocate_pipeline.config import load_config
from vocalocate_pipeline.lake import Lake
from vocalocate_pipeline.schemas import SchemaError, to_table

DEFAULT_TOML = Path(__file__).parents[1] / "configs" / "default.toml"

SPLIT_ROW = {"clip_id": "a", "source": "s", "source_key": "k", "group_key": "g", "split": "train", "strategy": "x"}


def test_default_config_file_matches_built_in_defaults():
    from_file = load_config(DEFAULT_TOML, workers=1)
    assert from_file.fingerprint() == load_config().fingerprint()
    assert from_file.audio.sample_rate == 32_000 and from_file.features.n_mels == 128


def test_cli_overrides_win_over_the_file(tmp_path):
    cfg = load_config(DEFAULT_TOML, lake_root=tmp_path, workers=2)
    assert cfg.lake_root == tmp_path and cfg.workers == 2
    assert cfg.warehouse_path == tmp_path / "warehouse" / "vocalocate.duckdb"


def test_unknown_config_key_is_an_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[audio]\nsample_rte = 16000\n")
    with pytest.raises(ValueError, match="sample_rte"):
        load_config(bad)


def test_fingerprint_changes_with_output_settings_only(tmp_path):
    custom = tmp_path / "c.toml"
    custom.write_text("[features]\nn_mels = 64\n")
    assert load_config(custom).fingerprint() != load_config().fingerprint()
    # Where the lake lives does not change what the pipeline produces.
    assert load_config(lake_root=tmp_path).fingerprint() == load_config().fingerprint()


def test_schema_rejects_nulls_and_unknown_enum_values():
    to_table("splits", [SPLIT_ROW])
    with pytest.raises(SchemaError):
        to_table("splits", [{**SPLIT_ROW, "split": "holdout"}])
    with pytest.raises(SchemaError):
        to_table("splits", [{**SPLIT_ROW, "group_key": None}])


def test_lake_round_trip_and_one_file_per_part(tmp_path):
    lake = Lake(tmp_path)
    lake.write("splits", "esc50", to_table("splits", [SPLIT_ROW]))
    lake.write("splits", "vimsketch", to_table("splits", [{**SPLIT_ROW, "source": "vimsketch"}]))
    assert lake.read("splits", "esc50").to_pylist() == [SPLIT_ROW]
    assert [p.name for p in lake.parts("splits")] == ["esc50.parquet", "vimsketch.parquet"]
    assert lake.read("splits", "missing") is None
    assert not list(tmp_path.rglob("*.tmp"))


def test_lake_refuses_a_table_with_the_wrong_schema(tmp_path):
    with pytest.raises(ValueError):
        Lake(tmp_path).write("splits", "x", pa.table({"clip_id": ["a"]}))
