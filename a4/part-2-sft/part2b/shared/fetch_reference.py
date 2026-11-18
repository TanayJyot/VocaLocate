#!/usr/bin/env python3
"""Fetch current Hub Parquet exports for an approximate original-SFT overlap audit.

The baseline did not record Hub revisions, so these files do not establish exact
identity with the original Modal cache. Dataset-repository commits are recorded.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / "reference_raw"
SOURCES = [
    ("smoltalk_train", "HuggingFaceTB/smol-smoltalk", "default", "train"),
    ("mmlu_auxiliary_train", "cais/mmlu", "all", "auxiliary_train"),
    ("gsm8k_train", "openai/gsm8k", "main", "train"),
    ("gsm8k_test", "openai/gsm8k", "main", "test"),
]


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"warning": "Current Hub Parquet exports approximate unpinned Part 2a data; do not claim exact baseline identity.", "sources": {}}
    for name, repo, subset, split in SOURCES:
        info = json.loads(get(f"https://huggingface.co/api/datasets/{repo}"))
        urls = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/parquet/{subset}/{split}"))
        folder = ROOT / name
        folder.mkdir(exist_ok=True)
        files = []
        for i, url in enumerate(urls):
            path = folder / f"{i:05d}.parquet"
            if not path.exists():
                tmp = path.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                import pyarrow.parquet as pq
                pq.ParquetFile(tmp)
                tmp.replace(path)
            files.append({"file": str(path.relative_to(ROOT)), "url": url, "bytes": path.stat().st_size, "sha256": sha(path)})
        manifest["sources"][name] = {"repository": repo, "current_repo_sha": info.get("sha"),
                                     "subset": subset, "split": split, "files": files}
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(files)} shards", flush=True)


if __name__ == "__main__":
    main()
