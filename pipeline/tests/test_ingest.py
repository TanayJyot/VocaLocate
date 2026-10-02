import csv
from datetime import datetime, timezone

import pytest

from conftest import tone, write
from vocalocate_pipeline.audio import sha256_file
from vocalocate_pipeline.ingest import ingest
from vocalocate_pipeline.schemas import to_table


def _by_key(result):
    return {f.source_key: f for f in result.files}


def test_esc50_reads_metadata_and_reports_missing_files(esc50_dir):
    result = ingest("esc50", esc50_dir)
    files = _by_key(result)
    assert len(files) == 5 and result.missing == ["5-500-A-4.wav"]
    take_a, take_b = files["audio/1-100-A-0.wav"], files["audio/1-100-B-0.wav"]
    # Takes cut from the same recording share a group.
    assert take_a.group_key == take_b.group_key == "esc50:100"
    assert take_a.role == "library" and take_a.label == "dog" and take_a.extra["fold"] == 1


def test_vimsketch_links_imitations_to_references_by_name(vimsketch_dir):
    files = _by_key(ingest("vimsketch", vimsketch_dir))
    ref = files["references/000Animal_Cat_Growling.wav"]
    imitation = files["vocal_imitations/00001_Cat_Growling.wav"]
    assert ref.role == "reference" and imitation.role == "imitation"
    assert ref.group_key == imitation.group_key == "vimsketch:Cat_Growling.wav"
    # An imitation whose reference is not in the dataset gets no group.
    assert files["vocal_imitations/00004_Unknown_Sound.wav"].group_key is None


def test_vimsketch_falls_back_to_listing_folders(vimsketch_dir):
    (vimsketch_dir / "reference_file_names.csv").unlink()
    (vimsketch_dir / "vocal_imitation_file_names.csv").unlink()
    assert len(ingest("vimsketch", vimsketch_dir).files) == 8


def test_qvim_dev_reads_up_to_three_queries_per_reference(qvim_dir):
    files = _by_key(ingest("qvim_dev", qvim_dir))
    assert set(files) == {"Items/Dog/bark.wav", "Queries/Dog/q1.wav", "Queries/Dog/q2.wav"}
    assert {f.group_key for f in files.values()} == {"qvim_dev:Dog/bark.wav"}


def test_team_recordings_produce_text_hint_queries(tmp_path):
    root = tmp_path / "team"
    write(root / "carmen" / "creak.wav", tone(300, 1, 48_000), 48_000)
    write(root / "carmen" / "thud.wav", tone(80, 1, 48_000), 48_000)
    with open(root / "recordings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "imitator_id", "target_group_key", "text_hint", "hint_type"])
        w.writerow(["carmen/creak.wav", "carmen", "esc50:200", "wooden", "material"])
        w.writerow(["carmen/thud.wav", "carmen", "esc50:300", "", ""])
    result = ingest("team", root)
    assert {f.imitator_id for f in result.files} == {"carmen"}
    assert len(result.queries) == 1
    assert result.queries[0]["text_hint"] == "wooden" and result.queries[0]["target_group_key"] == "esc50:200"


def test_manifest_rows_match_the_raw_files_schema(esc50_dir):
    result = ingest("esc50", esc50_dir)
    now = datetime.now(timezone.utc)
    rows = [f.to_row("run-1", now) for f in result.files]
    table = to_table("raw_files", rows)
    assert table.num_rows == 5
    assert rows[0]["file_sha256"] == sha256_file(result.files[0].path)


def test_unknown_source_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown source"):
        ingest("audioset", tmp_path)
