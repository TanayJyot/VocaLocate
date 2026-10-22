"""Lake sync and the remote flow, with local folders standing in for the S3 buckets."""

from __future__ import annotations

import os
import shutil

import pyarrow.parquet as pq

from vocalocate_pipeline import remote
from vocalocate_pipeline.cli import main
from vocalocate_pipeline.schemas import to_table
from vocalocate_pipeline.sync import is_remote, sync


def _tree(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_sync_copies_new_and_changed_files_only(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.parquet").write_bytes(b"one")
    (src / "b.npy").write_bytes(b"two")
    (src / "a" / "half.parquet.tmp").write_bytes(b"partial")

    first = sync(src, dst)
    assert first.copied == 2 and first.bytes == 6
    assert _tree(dst) == ["a/x.parquet", "b.npy"]  # temporary files are never copied

    assert sync(src, dst).copied == 0  # nothing changed
    assert sync(dst, src).copied == 0  # a pulled file is not pushed back

    (src / "a" / "x.parquet").write_bytes(b"three")
    again = sync(src, dst)
    assert again.copied == 1 and (dst / "a" / "x.parquet").read_bytes() == b"three"


def test_sync_only_limits_to_prefixes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for rel in ["silver/tables/clips/esc50.parquet", "silver/audio/esc50/ab/c.wav", "silver/audio/vis/de/f.wav"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"x")
    sync(f"file://{src}", dst, ["silver/tables", "silver/audio/esc50/"])
    assert _tree(dst) == ["silver/audio/esc50/ab/c.wav", "silver/tables/clips/esc50.parquet"]


def test_sync_missing_source_is_empty(tmp_path):
    assert sync(tmp_path / "nope", tmp_path / "dst").copied == 0


def test_is_remote():
    assert is_remote("s3://bucket/lake")
    assert not is_remote("file:///tmp/lake")
    assert not is_remote("lake")


def test_remote_ingest_round_trip(tmp_path, cfg, esc50_dir):
    raw_bucket, lake_bucket = tmp_path / "raw-bucket", tmp_path / "lake-bucket"
    shutil.copytree(esc50_dir, raw_bucket / "esc50")

    first = remote.ingest("esc50", cfg, str(lake_bucket), str(raw_bucket))
    assert first.n_processed > 0
    pushed = _tree(lake_bucket)
    assert "warehouse/vocalocate.duckdb" not in pushed  # rebuilt by load_warehouse, not by an ingest
    assert "silver/tables/clips/esc50.parquet" in pushed
    assert any(p.startswith("gold/features/esc50/") for p in pushed)
    assert any(p.startswith("runs/") for p in pushed)
    assert not any(p.startswith("bronze/esc50/") for p in pushed)  # raw files stay in the raw bucket

    # A fresh container: the local working copy is gone, but the cache still skips finished clips.
    shutil.rmtree(cfg.lake_root)
    second = remote.ingest("esc50", cfg, str(lake_bucket), str(raw_bucket))
    assert second.n_processed == 0 and second.n_reused == first.n_processed + first.n_reused
    assert len([p for p in _tree(lake_bucket) if p.startswith("runs/")]) == 2

    shutil.rmtree(cfg.lake_root)
    remote.load_warehouse(cfg, str(lake_bucket))
    assert "warehouse/vocalocate.duckdb" in _tree(lake_bucket)
    shutil.rmtree(cfg.lake_root)
    rows = remote.report(cfg, "SELECT count(*) FROM clips WHERE source = 'esc50'", str(lake_bucket))
    assert rows[0][0] == second.n_input


def test_sync_never_replaces_a_newer_destination(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "t.parquet").write_bytes(b"stale copy")
    (dst / "t.parquet").write_bytes(b"newer result from another job")
    os.utime(src / "t.parquet", (1_000, 1_000))
    os.utime(dst / "t.parquet", (2_000, 2_000))
    assert sync(src, dst).copied == 0
    assert (dst / "t.parquet").read_bytes() == b"newer result from another job"


def test_overlapping_ingests_keep_each_others_tables(tmp_path, cfg, esc50_dir, monkeypatch):
    """A team ingest that finishes while an ESC-50 ingest is running must not be overwritten by it."""
    raw_bucket, lake_bucket = tmp_path / "raw-bucket", tmp_path / "lake-bucket"
    shutil.copytree(esc50_dir, raw_bucket / "esc50")
    team_table = lake_bucket / "silver" / "tables" / "clips" / "team.parquet"
    team_table.parent.mkdir(parents=True)
    empty_clips = to_table("clips", [])
    pq.write_table(empty_clips, team_table, compression="none")  # the team tables before
    os.utime(team_table, (1_000, 1_000))
    real_flow = remote.run_dataset_flow

    def flow_while_team_ingest_finishes(*args, **kwargs):
        summary = real_flow(*args, **kwargs)
        pq.write_table(empty_clips, team_table, compression="zstd")  # the team job pushes mid-run
        return summary

    monkeypatch.setattr(remote, "run_dataset_flow", flow_while_team_ingest_finishes)
    remote.ingest("esc50", cfg, str(lake_bucket), str(raw_bucket))
    assert pq.ParquetFile(team_table).metadata.row_group(0).column(0).compression == "ZSTD"
    assert (lake_bucket / "silver" / "tables" / "clips" / "esc50.parquet").exists()


def test_remote_cli_reads_uris_from_env(tmp_path, esc50_dir, monkeypatch, capsys):
    raw_bucket, lake_bucket = tmp_path / "raw", tmp_path / "lake-bucket"
    shutil.copytree(esc50_dir, raw_bucket / "esc50")
    monkeypatch.setenv("VOCALOCATE_LAKE_URI", str(lake_bucket))
    monkeypatch.setenv("VOCALOCATE_RAW_URI", str(raw_bucket))
    local = tmp_path / "work"
    assert main(["--lake", str(local), "remote", "ingest", "--source", "esc50"]) == 0
    assert main(["--lake", str(local), "remote", "load-warehouse"]) == 0
    os.remove(local / "warehouse" / "vocalocate.duckdb")
    assert main(["--lake", str(local), "remote", "report"]) == 0
    assert "esc50" in capsys.readouterr().out  # the report ran on the warehouse pulled from the bucket


def test_s3_from_web_identity_uses_temporary_credentials(monkeypatch):
    import sys
    import types

    import pyarrow.fs as pafs

    from vocalocate_pipeline.sync import s3_from_web_identity

    calls = {}

    class FakeSts:
        def assume_role_with_web_identity(self, **kwargs):
            calls.update(kwargs)
            return {"Credentials": {"AccessKeyId": "AKIA", "SecretAccessKey": "secret", "SessionToken": "token"}}

    fake_boto3 = types.SimpleNamespace(client=lambda service, region_name: FakeSts())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    fs = s3_from_web_identity("arn:aws:iam::123456789012:role/modal", "jwt", "ca-central-1", "test")
    assert isinstance(fs, pafs.S3FileSystem) and fs.region == "ca-central-1"
    assert calls == {"RoleArn": "arn:aws:iam::123456789012:role/modal", "RoleSessionName": "test",
                     "WebIdentityToken": "jwt"}
