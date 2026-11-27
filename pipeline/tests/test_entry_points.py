import json
import runpy
import sys

import pyarrow.fs as pafs
import pytest

from vocalocate_pipeline.cli import main
from vocalocate_pipeline.lake import Lake
from vocalocate_pipeline.sync import _needs_copy, _resolve


def test_pipeline_cli_indexes_and_syncs_a_library(tmp_path, capsys):
    lake = tmp_path / "lake"
    library = tmp_path / "sounds"
    library.mkdir()
    options = ["--lake", str(lake), "--workers", "1"]

    assert main([*options, "index-library", "--root", str(library), "--name", "studio"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source"] == "library_studio"
    assert summary["n_input"] == 0

    assert main([*options, "load-warehouse"]) == 0
    assert str(lake / "warehouse") in capsys.readouterr().out

    backup = tmp_path / "backup"
    assert main(["sync", str(lake), str(backup), "--only", "runs"]) == 0
    assert (backup / "runs").is_dir()
    assert "copied=" in capsys.readouterr().out


def test_pipeline_module_entry_point(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["vocalocate-pipeline", "--lake", str(tmp_path), "load-warehouse"])
    monkeypatch.delitem(sys.modules, "vocalocate_pipeline.cli", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("vocalocate_pipeline.cli", run_name="__main__")
    assert stopped.value.code == 0
    assert str(tmp_path / "warehouse") in capsys.readouterr().out


def test_remote_ingest_requires_source():
    with pytest.raises(SystemExit, match="remote ingest needs --source"):
        main(["remote", "ingest"])


def test_remote_ingest_requires_lake_uri(monkeypatch):
    monkeypatch.delenv("VOCALOCATE_LAKE_URI", raising=False)
    with pytest.raises(SystemExit, match="Set --lake-uri or VOCALOCATE_LAKE_URI"):
        main(["remote", "ingest", "--source", "team"])


def test_remote_ingest_syncs_failure_record(tmp_path, cfg):
    remote_lake = tmp_path / "remote"
    with pytest.raises(FileNotFoundError, match="Team recordings manifest"):
        main(["--lake", str(cfg.lake_root), "--workers", "1", "remote", "ingest", "--source", "team",
              "--lake-uri", str(remote_lake), "--raw-uri", str(tmp_path / "raw")])

    lake = Lake(remote_lake)
    [part] = lake.parts("pipeline_runs")
    [run] = lake.read("pipeline_runs", part.stem).to_pylist()
    assert run["status"] == "failed"
    assert "Team recordings manifest" in run["message"]
    assert not (remote_lake / "bronze").exists()


def test_sync_resolves_a_remote_uri_without_aws_credentials():
    # The Arrow in-memory scheme exercises URI resolution without AWS.
    filesystem, base, local = _resolve("mock:///snapshot/", None)
    assert base == "snapshot"
    assert not local
    filesystem.create_dir(base)
    with filesystem.open_output_stream(f"{base}/sound.txt") as stream:
        stream.write(b"bark")
    assert filesystem.get_file_info(f"{base}/sound.txt").size == 4


@pytest.mark.parametrize(("size", "expected"), [(10, False), (11, True)])
def test_sync_compares_sizes_when_timestamps_are_missing(size, expected):
    src = pafs.FileInfo("source", pafs.FileType.File, size=10)
    dst = pafs.FileInfo("destination", pafs.FileType.File, size=size)
    assert _needs_copy(src, dst) is expected
