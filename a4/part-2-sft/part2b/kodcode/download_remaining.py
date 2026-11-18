"""Download and verify the missing KodCode-V1 Parquet shards for the 40K audit."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REVISION = "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5"
ROOT = Path("/private/tmp")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    info = HfApi().dataset_info("KodCode/KodCode-V1", revision=REVISION, files_metadata=True)
    expected = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    if info.sha != REVISION:
        raise RuntimeError(f"Revision changed: {info.sha}")
    for i in range(15):
        filename = f"data/train-{i:05d}-of-00015.parquet"
        target = ROOT / f"kodcode_shard{i}_humaneval_audit.parquet"
        wanted = expected[filename]
        if not target.exists():
            downloaded = Path(hf_hub_download(
                repo_id="KodCode/KodCode-V1", repo_type="dataset",
                revision=REVISION, filename=filename,
            ))
            if sha256(downloaded) != wanted:
                raise RuntimeError(f"Downloaded hash mismatch: {filename}")
            target.symlink_to(downloaded)
        actual = sha256(target)
        if actual != wanted:
            raise RuntimeError(f"Local hash mismatch: {target}")
        print(f"verified shard {i}/14: {actual}", flush=True)
    (ROOT / "kodcode_v1_revision.json").write_text(json.dumps({
        "repo": "KodCode/KodCode-V1", "revision": REVISION,
        "shards": {str(i): expected[f"data/train-{i:05d}-of-00015.parquet"] for i in range(15)},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
