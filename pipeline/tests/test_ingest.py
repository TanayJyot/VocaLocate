import csv
from pathlib import Path
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


def test_vimsketch_labels_both_halves_and_adds_vocalsketch_excluded_imitations(vimsketch_release):
    result = ingest("vimsketch", vimsketch_release)
    files = {f.source_key: f for f in result.files}
    meta = {m["source_key"]: m for m in result.metadata}
    vim = "Vim_Sketch_Dataset/references/"
    # Vocal Imitation Set half: tags are the AudioSet ontology path, so the two cat sounds share four.
    growl = meta[vim + "001_000Animal_Domestic animals_pets_Cat_Growling.wav"]
    hiss = meta[vim + "002_001Animal_Domestic animals_pets_Cat_Hiss.wav"]
    assert len(set(growl["tags"]) & set(hiss["tags"])) == 4
    # VocalSketch half: category is the subset, tags the subset plus the label's words.
    zipper = meta[vim + "304_002-zipper_opening-everyday.wav"]
    assert zipper["category"] == "everyday" and set(zipper["tags"]) == {"everyday", "zipper", "opening"}
    # Excluded VocalSketch takes, by recording or by label alone, become unpaired imitations of their
    # sound; included ones are already in VimSketch and are not read twice.
    unpaired = [f for f in result.files if f.role == "unpaired_imitation"]
    assert sorted(Path(f.source_key).name for f in unpaired) == ["abyss - 1.wav", "zipper - 2.wav", "zipper - 4.wav"]
    zipper_group = files[vim + "304_002-zipper_opening-everyday.wav"].group_key
    assert {f.group_key for f in unpaired if "zipper" in f.source_key} == {zipper_group}
    assert not any("zipper - 3" in k for k in files)


def test_vimsketch_without_vocalsketch_has_no_unpaired_imitations(vimsketch_release):
    import shutil

    shutil.rmtree(vimsketch_release / "interactiveaudiolab-VocalSketchDataSet-6c9f381")
    result = ingest("vimsketch", vimsketch_release / "Vim_Sketch_Dataset")
    assert len(result.files) == 8 and not any(f.role == "unpaired_imitation" for f in result.files)


def test_qvim_dev_reads_up_to_three_queries_per_reference(qvim_dir):
    files = _by_key(ingest("qvim_dev", qvim_dir))
    assert set(files) == {"Items/Dog/bark.wav", "Queries/Dog/q1.wav", "Queries/Dog/q2.wav"}
    assert {f.group_key for f in files.values()} == {"qvim_dev:Dog/bark.wav"}


def test_team_recordings_produce_text_hint_queries(tmp_path):
    root = tmp_path / "team"
    write(root / "carmen" / "creak.wav", tone(300, 1, 48_000), 48_000)
    write(root / "carmen" / "thud.wav", tone(80, 1, 48_000), 48_000)
    with open(root / "recordings.csv", "w", newline="", encoding="utf-8") as f:
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


def test_qvim_dev_accepts_the_zenodo_release_layout(qvim_dir, tmp_path):
    from vocalocate_pipeline.ingest import ingest

    # DEVDataset.zip: everything inside DEVUpdatedDataset/, index named DEVUpdateComplete.csv.
    release = tmp_path / "qvim-release"
    (qvim_dir / "DEV Dataset.csv").rename(qvim_dir / "DEVUpdateComplete.csv")
    release.mkdir()
    qvim_dir.rename(release / "DEVUpdatedDataset")
    result = ingest("qvim_dev", release)
    assert sorted(f.role for f in result.files) == ["imitation", "imitation", "reference"]
    assert not result.missing
