import csv
import os
import shutil

from conftest import tone, write
from vocalocate_pipeline import warehouse
from vocalocate_pipeline.cli import main
from vocalocate_pipeline.flows import run_dataset_flow, run_library_flow


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
    with open(team / "recordings.csv", "w", newline="") as f:
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
