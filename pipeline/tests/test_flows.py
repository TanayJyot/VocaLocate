import csv
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from conftest import tone, write
from vocalocate_pipeline import flows, warehouse
from vocalocate_pipeline.cli import main
from vocalocate_pipeline.flows import run_dataset_flow, run_library_flow
from vocalocate_pipeline.lake import Lake


def _one(cfg, sql):
    return warehouse.query(cfg, sql)[0][0]


def test_esc50_flow_cleans_rejects_and_loads_warehouse(cfg, esc50_dir):
    s = run_dataset_flow("esc50", esc50_dir, cfg)
    assert (s.n_input, s.n_ok, s.n_rejected, s.n_missing) == (5, 4, 1, 1)
    assert _one(cfg, "SELECT reject_reason FROM clips WHERE status = 'rejected'") == "silent"
    assert _one(cfg, "SELECT count(*) FROM clips WHERE truncated") == 1
    assert _one(cfg, "SELECT count(*) FROM features") == 4
    # Two takes from the same Freesound recording land in the same split.
    assert _one(cfg, "SELECT count(DISTINCT split) FROM splits WHERE group_key = 'esc50:100'") == 1
    # Every clean file really is 32 kHz mono.
    import soundfile as sf

    for (path,) in warehouse.query(cfg, "SELECT clean_path FROM clips WHERE status = 'ok'"):
        info = sf.info(path)
        assert info.samplerate == 32_000 and info.channels == 1


def test_rerun_reuses_previous_work(cfg, esc50_dir):
    run_dataset_flow("esc50", esc50_dir, cfg)
    s = run_dataset_flow("esc50", esc50_dir, cfg)
    assert s.n_processed == 0 and s.n_reused == 5
    assert _one(cfg, "SELECT count(*) FROM pipeline_runs WHERE status = 'succeeded'") == 2


def test_duplicate_files_are_marked(cfg, esc50_dir):
    shutil.copy(esc50_dir / "audio" / "1-100-A-0.wav", esc50_dir / "audio" / "5-500-A-4.wav")
    s = run_dataset_flow("esc50", esc50_dir, cfg)
    assert s.n_duplicate == 1


def test_vimsketch_pairs_and_training_view(cfg, vimsketch_dir):
    run_dataset_flow("vimsketch", vimsketch_dir, cfg)
    assert _one(cfg, "SELECT count(*) FROM pairs") == 4  # the unmatched imitation gets no pair
    rows = warehouse.query(cfg, "SELECT label, split FROM v_training_pairs WHERE label = 'Cat_Growling'")
    assert len(rows) == 2 and len({r[1] for r in rows}) == 1


def test_qvim_dev_is_held_out(cfg, qvim_dir):
    run_dataset_flow("qvim_dev", qvim_dir, cfg)
    assert warehouse.query(cfg, "SELECT DISTINCT split FROM splits") == [("test",)]
    assert _one(cfg, "SELECT count(*) FROM v_training_pairs") == 2


def test_team_recordings_become_triples_linked_to_another_source(cfg, esc50_dir, tmp_path):
    run_dataset_flow("esc50", esc50_dir, cfg)
    team = tmp_path / "team"
    write(team / "tanay" / "dog_bark.wav", tone(450, 1, 48_000), 48_000)
    with open(team / "recordings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "imitator_id", "target_group_key", "text_hint", "hint_type"])
        w.writerow(["tanay/dog_bark.wav", "tanay", "esc50:200", "short", "length"])
    run_dataset_flow("team", team, cfg)
    rows = warehouse.query(cfg, "SELECT text_hint, imitator_id, target_source FROM v_triples")
    assert rows == [("short", "tanay", "esc50")]


def test_library_flow_handles_add_change_move_and_delete(cfg, tmp_path):
    lib = tmp_path / "MySFX"
    write(lib / "trailer_2025" / "SFX_0231.wav", tone(300, 2, 48_000), 48_000)
    write(lib / "trailer_2025" / "SFX_0232.wav", tone(600, 2, 48_000), 48_000)
    write(lib / "loose.wav", tone(900, 2, 48_000), 48_000)

    s = run_library_flow([lib], cfg)
    assert (s.n_added, s.n_ok) == (3, 3)
    assert _one(cfg, "SELECT count(*) FROM v_library_index WHERE project_id = 'trailer_2025'") == 2

    s = run_library_flow([lib], cfg)
    assert (s.n_unchanged, s.n_processed) == (3, 0)

    # Edit one file, move another into a new project and delete the third.
    changed = lib / "trailer_2025" / "SFX_0231.wav"
    write(changed, tone(350, 2, 48_000), 48_000)
    os.utime(changed, ns=(1, 1))
    (lib / "podcast_ep4").mkdir()
    (lib / "trailer_2025" / "SFX_0232.wav").rename(lib / "podcast_ep4" / "SFX_0232.wav")
    (lib / "loose.wav").unlink()

    s = run_library_flow([lib], cfg)
    assert (s.n_changed, s.n_moved, s.n_deleted, s.n_processed) == (1, 1, 1, 1)
    assert _one(cfg, "SELECT count(*) FROM v_library_index") == 2
    assert _one(cfg, "SELECT count(*) FROM library_files WHERE status = 'deleted'") == 2


def test_cli_end_to_end(tmp_path, esc50_dir, capsys):
    lake = tmp_path / "cli-lake"
    assert main(["--lake", str(lake), "ingest-dataset", "--source", "esc50", "--path", str(esc50_dir)]) == 0
    assert main(["--lake", str(lake), "report"]) == 0
    assert "esc50" in capsys.readouterr().out


def test_clips_record_the_original_format_bit_depth_and_noise_floor(cfg, tmp_path):
    import soundfile as sf

    lib = tmp_path / "Formats"
    lib.mkdir()
    x = tone(440, 1, 44_100)
    sf.write(str(lib / "a.wav"), x, 44_100, subtype="PCM_24")
    sf.write(str(lib / "b.flac"), tone(550, 1, 44_100), 44_100, subtype="PCM_16")
    sf.write(str(lib / "c.wav"), tone(660, 1, 44_100), 44_100, subtype="FLOAT")
    lossy = [f for f in ("MP3", "OGG") if f in sf.available_formats()]
    for i, fmt in enumerate(lossy):
        sf.write(str(lib / f"d{i}.{fmt.lower()}"), tone(770 + 50 * i, 1, 44_100), 44_100, format=fmt)
    (lib / "broken.wav").write_bytes(b"not audio")
    run_library_flow([lib], cfg)
    sql = "SELECT source_key, orig_format, orig_bit_depth, noise_floor_dbfs FROM clips"
    rows = {r[0].rsplit("/", 1)[1]: r[1:] for r in warehouse.query(cfg, sql)}
    assert rows["a.wav"][:2] == ("WAV", 24)
    assert rows["b.flac"][:2] == ("FLAC", 16)
    assert rows["c.wav"][:2] == ("WAV", 32)
    for i, fmt in enumerate(lossy):
        assert rows[f"d{i}.{fmt.lower()}"][:2] == (fmt, None)  # lossy: no bit depth
    assert rows["a.wav"][2] is not None and rows["a.wav"][2] < 0
    assert rows["broken.wav"] == (None, None, None)  # decode errors leave them empty


def test_library_roots_record_each_folder_and_files_keep_their_relative_path(cfg, tmp_path, monkeypatch):
    from vocalocate_pipeline import flows

    a, b = tmp_path / "SFX_A", tmp_path / "SFX_B"
    write(a / "proj" / "door.wav", tone(300, 1, 48_000), 48_000)
    write(b / "hit.wav", tone(600, 1, 48_000), 48_000)
    run_library_flow([a, b], cfg, library="studio")
    roots = warehouse.query(cfg, "SELECT root_id, root_path, index_status, epoch_us(last_indexed_at), last_run_id "
                                 "FROM library_roots ORDER BY root_path")
    assert [(r[1], r[2]) for r in roots] == [(str(a.resolve()), "completed"), (str(b.resolve()), "completed")]
    assert roots[0][0] == flows.root_id(a) and len(roots[0][0]) == 32 and roots[0][3] is not None
    files = warehouse.query(cfg, "SELECT root_id, rel_path, project_id FROM library_files ORDER BY path")
    assert files == [(flows.root_id(a), "proj/door.wav", "proj"), (flows.root_id(b), "hit.wav", None)]

    # A failed run marks its roots failed but keeps the last completed index time.
    first = roots[0][3]

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(flows, "clean_and_transform", boom)
    try:
        run_library_flow([a], cfg, library="studio")
    except RuntimeError:
        pass
    warehouse.load(cfg)
    rows = warehouse.query(
        cfg, "SELECT root_path, index_status, epoch_us(last_indexed_at) FROM library_roots ORDER BY root_path")
    assert rows == [(str(a.resolve()), "failed", first), (str(b.resolve()), "completed", roots[1][3])]


def _team(tmp_path, rows, header=("file", "imitator_id", "target_group_key", "text_hint", "hint_type", "split")):
    team = tmp_path / "team"
    for i, row in enumerate(rows):
        write(team / row[0], tone(400 + 37 * i, 1, 48_000), 48_000)
    with open(team / "recordings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return team


def test_team_split_comes_from_the_csv_or_a_val_test_hash_never_train(cfg, esc50_dir, tmp_path):
    from vocalocate_pipeline.flow_functions.splits import held_out_split

    run_dataset_flow("esc50", esc50_dir, cfg)
    rows = [("t/a.wav", "tanay", "esc50:200", "short", "length", "val"),
            ("t/b.wav", "tanay", "esc50:200", "", "", ""),  # same target: inherits val
            ("t/c.wav", "carmen", "esc50:400", "low", "pitch", "test")]
    rows += [(f"t/h{i}.wav", "carmen", f"esc50:9{i}", "hint", "", "") for i in range(12)]  # no split given
    run_dataset_flow("team", _team(tmp_path, rows), cfg)
    got = dict(warehouse.query(cfg, "SELECT source_key, split FROM splits WHERE source = 'team'"))
    assert got["t/a.wav"] == got["t/b.wav"] == "val" and got["t/c.wav"] == "test"
    hashed = {k: v for k, v in got.items() if "/h" in k}
    assert set(hashed.values()) == {"val", "test"}  # both halves are used, never train
    assert all(v == held_out_split(f"esc50:9{k[3:-4]}", cfg.splits) for k, v in hashed.items())
    strategies = dict(warehouse.query(cfg, "SELECT source_key, strategy FROM splits WHERE source = 'team'"))
    assert strategies["t/a.wav"] == "team" and strategies["t/h0.wav"].startswith("team_hash")
    # The query carries the team split even though its target (an ESC-50 clip) has its own split.
    assert dict(warehouse.query(cfg, "SELECT text_hint, split FROM queries WHERE text_hint <> 'hint'")) == {
        "short": "val", "low": "test"}
    assert _one(cfg, "SELECT count(*) FROM queries WHERE split IS NULL") == 0


def test_team_split_must_be_val_or_test_and_one_per_target(tmp_path):
    import pytest

    from vocalocate_pipeline.ingest import ingest

    with pytest.raises(ValueError, match="never trained"):
        ingest("team", _team(tmp_path / "x", [("a.wav", "t", "esc50:1", "", "", "train")]))
    with pytest.raises(ValueError, match="both"):
        ingest("team", _team(tmp_path / "y", [("a.wav", "t", "esc50:1", "", "", "val"),
                                             ("b.wav", "t", "esc50:1", "", "", "test")]))


def test_queries_link_to_their_target_clip_in_the_same_source(cfg, sfx_meta_dir):
    run_dataset_flow("sfx_meta", sfx_meta_dir, cfg)
    # The group holds a reference and another take; the target is the reference.
    rows = warehouse.query(cfg, """SELECT c.source_key, q.split FROM queries q
                                   JOIN clips c ON c.clip_id = q.target_clip_id""")
    assert rows == [("sfx/bell_A.wav", None)]  # no split of its own: it follows its target's


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_team_imitators(cfg, esc50_dir, tmp_path):
    run_dataset_flow("esc50", esc50_dir, cfg)
    team = _team(tmp_path, [("t/a.wav", "tanay", "esc50:200", "short", "length", "val"),
                            ("t/b.wav", "carmen", "esc50:400", "", "", "")])
    _csv(team / "imitators.csv", ["imitator_id", "role", "years_experience", "first_language", "music_training",
                                  "consent_scope"],
         [["tanay", "student", "2", "English", "yes", "course project"], ["carmen", "", "", "", "", ""]])
    run_dataset_flow("team", team, cfg)

    assert warehouse.query(cfg, "SELECT * FROM imitators ORDER BY imitator_id") == [
        ("carmen", "team", None, None, None, None, None),
        ("tanay", "team", "student", 2, "English", True, "course project")]


def test_warehouse_has_every_table_even_when_empty(cfg):
    from vocalocate_pipeline.schemas import TABLES

    warehouse.load(cfg)
    for name in ("imitators", "library_roots"):
        assert name in TABLES and _one(cfg, f"SELECT count(*) FROM {name}") == 0
    for name in ("imitation_metadata", "relevance_judgements", "search_sessions"):
        assert name not in TABLES


def test_ingest_rebuilds_missing_cached_feature(cfg, sfx_meta_dir):
    run_dataset_flow("sfx_meta", sfx_meta_dir, cfg, load=False)
    lake = Lake(cfg.lake_root)
    feature = lake.read("features", "sfx_meta").to_pylist()[0]["feature_path"]
    os.unlink(feature)

    summary = run_dataset_flow("sfx_meta", sfx_meta_dir, cfg, load=False)
    assert summary.n_ok > 0
    assert os.path.isfile(feature)

    changed = replace(cfg, audio=replace(cfg.audio, target_rms_dbfs=-18), workers=2)
    rebuilt = run_dataset_flow("sfx_meta", sfx_meta_dir, changed, load=False)
    assert rebuilt.n_processed == summary.n_ok
    assert {row["config_fingerprint"] for row in lake.read("clips", "sfx_meta").to_pylist()} == {
        changed.fingerprint()
    }


def test_ingest_omits_text_queries_for_rejected_audio(cfg, tmp_path):
    team = _team(tmp_path, [("t/a.wav", "alice", "esc50:100", "bark", "description", "val")])
    write(team / "t/a.wav", np.zeros(32_000), 32_000)
    summary = run_dataset_flow("team", team, cfg, load=False)

    assert summary.n_rejected == 1
    assert Lake(cfg.lake_root).read("queries", "team").num_rows == 0


def test_library_preserves_indexing_error_if_status_write_fails(cfg, tmp_path, monkeypatch, caplog):
    def fail_scan(_):
        raise OSError("cannot read library")

    def fail_write(*_):
        raise PermissionError("cannot update library status")

    monkeypatch.setattr(flows, "_scan", fail_scan)
    monkeypatch.setattr(flows, "_write_roots", fail_write)
    with pytest.raises(OSError, match="cannot read library"):
        run_library_flow([tmp_path], cfg, library="studio", load=False)

    assert "could not record the failed index of studio" in caplog.text
    lake = Lake(cfg.lake_root)
    [part] = lake.parts("pipeline_runs")
    [run] = lake.read("pipeline_runs", part.stem).to_pylist()
    assert (run["status"], run["message"]) == ("failed", "OSError: cannot read library")
