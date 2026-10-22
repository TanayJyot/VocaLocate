"""Lake sync against the S3 API itself, served locally by moto (no AWS account needed)."""

from __future__ import annotations

import pytest

moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

import pyarrow.fs as pafs  # noqa: E402

from vocalocate_pipeline.sync import sync  # noqa: E402

REGION = "ca-central-1"


@pytest.fixture(scope="module")
def s3():
    server = moto_server.ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    boto3.client(
        "s3", endpoint_url=f"http://{host}:{port}", region_name=REGION,
        aws_access_key_id="test", aws_secret_access_key="test",
    ).create_bucket(Bucket="lake", CreateBucketConfiguration={"LocationConstraint": REGION})
    yield pafs.S3FileSystem(
        endpoint_override=f"{host}:{port}", scheme="http", region=REGION, access_key="test", secret_key="test"
    )
    server.stop()


def test_push_then_pull_through_s3(tmp_path, s3):
    local = tmp_path / "local"
    (local / "warehouse").mkdir(parents=True)
    (local / "warehouse" / "vocalocate.duckdb").write_bytes(b"db" * 1000)
    (local / "gold" / "features" / "vis" / "ab").mkdir(parents=True)
    (local / "gold" / "features" / "vis" / "ab" / "x.npy").write_bytes(b"f")
    (local / "gold" / "features" / "vis" / "ab" / "y.tmp.npy").write_bytes(b"partial")

    pushed = sync(local, "s3://lake", s3=s3)
    assert (pushed.copied, pushed.bytes) == (2, 2001)
    assert sync(local, "s3://lake", s3=s3).copied == 0

    fresh = tmp_path / "fresh"
    pulled = sync("s3://lake", fresh, ["warehouse", "gold/features/vis", "gold/features/missing"], s3=s3)
    assert pulled.copied == 2
    assert (fresh / "warehouse" / "vocalocate.duckdb").read_bytes() == b"db" * 1000
    assert sync(fresh, "s3://lake", s3=s3).copied == 0  # a pulled copy is not pushed back
