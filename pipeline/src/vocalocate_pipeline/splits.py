"""Train/val/test splits grouped by reference sound.

A reference and all of its imitations always share a split, so the model is
never tested on a sound it was trained on (A1 milestone M2). The split comes
from a seeded hash of the group key, so it does not change when new files are
added or when the pipeline is re-run.
"""

from __future__ import annotations

import hashlib

from .config import SplitConfig

# Sources whose every clip is held out for evaluation.
BENCHMARK_SOURCES = {"qvim_dev"}


def split_for_group(group_key: str, cfg: SplitConfig) -> str:
    digest = hashlib.sha256(f"{cfg.seed}:{group_key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < cfg.train:
        return "train"
    if u < cfg.train + cfg.val:
        return "val"
    return "test"


def assign_splits(clips: list[dict], cfg: SplitConfig) -> list[dict]:
    """Return split rows for clean clips. Clips without a group use their own ID as the group."""
    rows = []
    for clip in clips:
        if clip["status"] != "ok":
            continue
        group = clip["group_key"] or f"{clip['source']}:clip:{clip['clip_id']}"
        if clip["source"] in BENCHMARK_SOURCES:
            split, strategy = "test", "benchmark"
        else:
            split, strategy = split_for_group(group, cfg), f"group_hash(seed={cfg.seed})"
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "source": clip["source"],
                "source_key": clip["source_key"],
                "group_key": group,
                "split": split,
                "strategy": strategy,
            }
        )
    return rows
