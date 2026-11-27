import csv

import pytest

from vocalocate_pipeline.ingest import captdure, esc50_voice, fine_grained_vis, qvim_dev, sfx_meta, team, vimsketch
from vocalocate_pipeline.ingest import vocal_imitation_set as vis


def test_captdure_skips_empty_rows_and_accepts_anonymous_caption(captdure_dir):
    with (captdure_dir / captdure.CAPTIONS).open("a", encoding="utf-8") as manifest:
        manifest.write("\t\t\tno clip\n")
        manifest.write("\tfan/fan_09_009.wav\t\tStill missing.\n")
        manifest.write("\tclock/clock_01_000.wav\t\tAn anonymous description.\n")
    result = captdure.ingest(captdure_dir)
    assert result.missing == ["fan/fan_09_009.wav"]
    [anonymous] = [row for row in result.captions if row["caption"] == "An anonymous description."]
    assert anonymous["annotator_id"] is None
    assert len(result.captions) == 5


def test_esc50_voice_reports_missing_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="speaker folders"):
        esc50_voice.ingest(tmp_path)
    (tmp_path / "vocal_imitation" / "000").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="esc50.csv"):
        esc50_voice.ingest(tmp_path)
    with pytest.raises(ValueError, match="no ESC-50 filename"):
        esc50_voice.target_name(tmp_path / "000" / "_000.wav")


def test_qvim_uses_alternate_index_and_skips_blank_item(qvim_dir):
    manifest = (qvim_dir / "DEV Dataset.csv").rename(qvim_dir / "DEV-local.csv")
    with manifest.open("a", encoding="utf-8") as output:
        output.write("Animal,Dog,,ignored.wav,,\n")
    assert len(qvim_dev.ingest(qvim_dir).files) == 3
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="QVIM index"):
        qvim_dev.ingest(qvim_dir)


def test_sfx_manifest_rejects_unknown_role(tmp_path):
    with pytest.raises(ValueError, match="bell.wav: unknown role 'noise'"):
        sfx_meta.ingest_rows([{"file": "bell.wav", "role": "noise"}], tmp_path)


def test_team_skips_missing_recording_and_reads_negative_music_training(tmp_path):
    (tmp_path / "recordings.csv").write_text(
        "file,imitator_id,target_group_key,text_hint\nmissing.wav,alice,esc50:100,short\n", encoding="utf-8"
    )
    profiles = tmp_path / "imitators.csv"
    profiles.write_text("imitator_id,music_training\nalice,No\n", encoding="utf-8")
    result = team.ingest(tmp_path)
    assert result.missing == ["missing.wav"]
    assert result.queries == []
    assert result.imitators[0]["music_training"] is False

    profiles.write_text("imitator_id,music_training\nalice,sometimes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a yes/no value"):
        team.ingest(tmp_path)


def test_vimsketch_skips_excluded_imitations_without_a_reference(tmp_path, vimsketch_release):
    with pytest.raises(FileNotFoundError, match="VimSketch"):
        vimsketch.ingest(tmp_path / "missing")
    manifest = next(vimsketch_release.glob("interactive*/vocal_imitations.csv"))
    with manifest.open("a", encoding="utf-8") as output:
        output.write("5,unmapped.wav,sound label,False,False,False,p5,unknown,L99,,,everyday\n")
        output.write("6,unmatched.wav,sound recording,False,False,False,p6,other,L99,other.wav,r99,everyday\n")
    result = vimsketch.ingest(vimsketch_release)
    assert len(result.files) == 11
    assert not result.missing


def test_vis_finds_nested_audio_and_reports_missing_files(vis_dir):
    original = next((vis_dir / "original_recordings" / "reference").glob("*.wav"))
    nested = vis_dir / "nested"
    nested.mkdir()
    moved = original.rename(nested / original.name)
    with (vis_dir / "original_recordings_classes.txt").open("a", encoding="utf-8") as manifest:
        manifest.write("absent.wav\tMissing\t\t\t\t\t\t\tTrue\tFalse\n")
    with (vis_dir / "vocal_imitations.txt").open("a", encoding="utf-8") as manifest:
        manifest.write("missing\tmissing-voice.wav\t0\tabsent.wav\tTrue\n")
        manifest.write("orphan\torphan.wav\t0\tunknown.wav\tTrue\n")

    assert any(row["file"] == str(moved) for row in vis.build_rows(vis_dir))
    result = vis.ingest(vis_dir)
    assert {path.rsplit("/", 1)[-1] for path in result.missing} == {"absent.wav", "missing-voice.wav"}


def test_vis_handles_unparseable_survey_fields_and_equivalent_numeric_ids():
    match = vis.rounded_id_matcher({"12.3", "person"}, ["12.30", "unmatched"])
    profiles = vis.survey_imitators([
        {"participant_id": "12.30", "age": "unknown", "years_actively_making_music": "2",
         "years_actively_using_music_tech": "unknown"},
        {"participant_id": "unmatched"},
    ], match)
    assert len(profiles) == 1
    assert profiles[0]["imitator_id"] == "12.3"
    assert profiles[0]["music_training"] is True
    assert profiles[0]["years_experience"] is None


def test_fine_grained_vis_accepts_dataset_root_and_ignores_missing_reference(fgvis_dir):
    root = fgvis_dir / "Fine-grained_VocalImitationSet"
    before = fine_grained_vis.ingest(root)
    index = root / "vocal_imitations.txt"
    with index.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        imitation = next(reader)
    with index.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writerow({**imitation, "imitation_filename": "orphan.wav", "reference_filename": "unknown.wav"})

    after = fine_grained_vis.ingest(root)
    assert {f.source_key for f in after.files} == {f.source_key for f in before.files}
    assert len(after.files) == 9
    assert after.missing == []


def test_fine_grained_vis_reports_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"Fine-grained Vocal Imitation Set .*not found"):
        fine_grained_vis.ingest(tmp_path)
