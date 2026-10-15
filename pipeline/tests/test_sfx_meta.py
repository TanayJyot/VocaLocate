import csv

import pytest

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest


def test_ingest_groups_takes_and_imitations_by_recording(sfx_meta_dir):
    result = ingest("sfx_meta", sfx_meta_dir)
    groups = {f.source_key: f.group_key for f in result.files}
    assert groups["sfx/bell_A.wav"] == groups["sfx/bell_B.wav"] == groups["imitations/b/bell_1.wav"] == "sfx_meta:bell"
    assert groups["sfx/buzz.wav"] == "sfx_meta:buzz"  # no recording_id: the sound is its own recording
    meta = {m["source_key"]: m for m in result.metadata}
    assert meta["sfx/gong_B.wav"]["tags"] == ["metallic", "low"]
    assert "imitations/a/bell_0.wav" not in meta  # metadata describes SFX only


def test_two_references_for_one_recording_is_an_error(sfx_meta_dir):
    path = sfx_meta_dir / "metadata.csv"
    rows = list(csv.reader(open(path)))
    rows[2][1] = "reference"  # bell_B becomes a second reference of recording "bell"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match="two references"):
        ingest("sfx_meta", sfx_meta_dir)


def test_flow_writes_metadata_and_candidate_view(cfg, sfx_meta_dir):
    run_dataset_flow("sfx_meta", sfx_meta_dir, cfg)
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs")[0][0] == 3
    rows = warehouse.query(cfg, "SELECT role, category, recording_id, split FROM v_sfx_candidates ORDER BY ALL")
    assert len(rows) == 5
    # Both takes of a recording share a split.
    splits = warehouse.query(
        cfg, "SELECT count(DISTINCT split) FROM v_sfx_candidates WHERE recording_id = 'bell'"
    )
    assert splits == [(1,)]
    feature_paths = warehouse.query(cfg, "SELECT imitation_feature_path, reference_feature_path FROM v_training_pairs")
    assert all(a and b for a, b in feature_paths)
