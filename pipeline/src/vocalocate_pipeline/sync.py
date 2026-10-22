"""Copy parts of the lake between S3 and a local working copy.

On AWS the lake's source of truth is an S3 bucket with the same layout as the local folder (see
lake.py). A pipeline job pulls the parts it needs into a local working copy, runs the normal flow on
it, and pushes back what changed. This keeps every flow, the atomic writes and DuckDB working on
plain files, and the same code runs on a laptop with no AWS at all.

A file is copied when it is missing at the destination, or when the source copy is newer and
differs in size or time. A newer copy at the destination is never replaced, so a job that pulled
a table before another job updated it cannot write its stale copy back. A local copy gets the
source's modification time, so a file that was only pulled is not pushed back again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.fs as pafs

# Temporary files from interrupted writes (lake.write, write_wav, warehouse.load) are never copied.
_SKIP_SUFFIXES = (".tmp", ".tmp.npy", ".duckdb.tmp", ".parquet.tmp")


@dataclass
class SyncResult:
    copied: int = 0
    skipped: int = 0
    bytes: int = 0


def is_remote(uri: str | Path) -> bool:
    return isinstance(uri, str) and "://" in uri and not uri.startswith("file://")


def _resolve(uri: str | Path, s3: pafs.FileSystem | None) -> tuple[pafs.FileSystem, str, bool]:
    """(filesystem, path inside it, is_local) for a local path or an s3:// URI."""
    if is_remote(uri):
        if s3 is not None and str(uri).startswith("s3://"):
            return s3, str(uri)[len("s3://"):].rstrip("/"), False
        fs, path = pafs.FileSystem.from_uri(str(uri))
        return fs, path.rstrip("/"), False
    path = str(uri)[len("file://"):] if str(uri).startswith("file://") else str(uri)
    return pafs.LocalFileSystem(), Path(path).resolve().as_posix(), True


def _listing(fs: pafs.FileSystem, base: str, prefix: str) -> dict[str, pafs.FileInfo]:
    """Files under base/prefix, keyed by their path relative to base. `prefix` may also name one file."""
    root = f"{base}/{prefix}".rstrip("/") if prefix else base
    if prefix:
        info = fs.get_file_info(root)
        if info.type == pafs.FileType.File:
            return {prefix: info}
    infos = fs.get_file_info(pafs.FileSelector(root, recursive=True, allow_not_found=True))
    out = {}
    for info in infos:
        if info.type != pafs.FileType.File or info.path.endswith(_SKIP_SUFFIXES):
            continue
        out[info.path[len(base) + 1:]] = info
    return out


_CLOCK_SLACK_NS = 1_000_000_000  # S3 keeps whole seconds


def _needs_copy(src: pafs.FileInfo, dst: pafs.FileInfo | None) -> bool:
    if dst is None:
        return True
    if src.mtime_ns is None or dst.mtime_ns is None:
        return src.size != dst.size
    if dst.mtime_ns > src.mtime_ns + _CLOCK_SLACK_NS:
        return False  # the destination was written after our copy: keep it
    return src.size != dst.size or src.mtime_ns > dst.mtime_ns + _CLOCK_SLACK_NS


def sync(
    src: str | Path, dst: str | Path, only: list[str] | tuple[str, ...] = (), s3: pafs.FileSystem | None = None
) -> SyncResult:
    """Copy new and changed files under `src` to `dst`. `only` limits it to these sub-folders or files.

    `src` and `dst` are local paths or s3:// URIs. `s3` is an S3 filesystem to use instead of the
    default credential chain (the Modal launcher passes one built from its OIDC role).
    Nothing is ever deleted at the destination.
    """
    src_fs, src_base, _ = _resolve(src, s3)
    dst_fs, dst_base, dst_local = _resolve(dst, s3)
    result = SyncResult()
    for prefix in only or ("",):
        prefix = prefix.strip("/")
        theirs = _listing(dst_fs, dst_base, prefix)
        for rel, info in sorted(_listing(src_fs, src_base, prefix).items()):
            if not _needs_copy(info, theirs.get(rel)):
                result.skipped += 1
                continue
            target = f"{dst_base}/{rel}"
            if dst_local:  # S3 has no folders to create
                dst_fs.create_dir(target.rsplit("/", 1)[0], recursive=True)
            tmp = f"{target}.tmp" if dst_local else target  # an S3 upload is already all-or-nothing
            with src_fs.open_input_stream(info.path) as fin, dst_fs.open_output_stream(tmp) as fout:
                while chunk := fin.read(8 << 20):
                    fout.write(chunk)
            if dst_local:
                os.replace(tmp, target)
                if info.mtime_ns is not None:
                    os.utime(target, ns=(info.mtime_ns, info.mtime_ns))
            result.copied += 1
            result.bytes += info.size
    return result


def s3_from_web_identity(role_arn: str, token: str, region: str, session_name: str = "vocalocate") -> pafs.S3FileSystem:
    """An S3 filesystem with temporary credentials for `role_arn`, from an OIDC token (e.g. Modal's)."""
    import boto3  # only needed outside AWS, where no role is attached to the machine

    creds = boto3.client("sts", region_name=region).assume_role_with_web_identity(
        RoleArn=role_arn, RoleSessionName=session_name, WebIdentityToken=token
    )["Credentials"]
    return pafs.S3FileSystem(
        access_key=creds["AccessKeyId"],
        secret_key=creds["SecretAccessKey"],
        session_token=creds["SessionToken"],
        region=region,
    )
