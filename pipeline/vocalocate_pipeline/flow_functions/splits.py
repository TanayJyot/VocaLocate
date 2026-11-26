"""Train/val/test splits grouped by reference sound.

A reference and all of its imitations always share a split, so the model is
never tested on a sound it was trained on (A1 milestone M2). The split comes
from a seeded hash of the group key, so it does not change when new files are
added or when the pipeline is re-run.
"""

from __future__ import annotations

import hashlib

from ..config import SplitConfig

# Sources whose every clip is held out for evaluation.
BENCHMARK_SOURCES = {"qvim_dev", "qvim_eval", "captdure"}
# Sources never used for training (A2 Part Two: team recordings). Their target sounds are split
# before recording into a validation group, for tuning the fusion weight, and a test group.
HELD_OUT_SOURCES = {"team"}
HELD_OUT_SPLITS = {"val", "test"}
# Sources whose every clip is training data: the fine-grained set's value is its within-class
# negatives, which need each whole class in the training split.
TRAIN_SOURCES = {"fgvis"}


def split_for_group(group_key: str, cfg: SplitConfig) -> str:
    digest = hashlib.sha256(f"{cfg.seed}:{group_key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < cfg.train:
        return "train"
    if u < cfg.train + cfg.val:
        return "val"
    return "test"


def held_out_split(group_key: str, cfg: SplitConfig) -> str:
    """val or test, 50/50, from a seeded hash of the group key: for held-out sources with no split given."""
    digest = hashlib.sha256(f"{cfg.seed}:held_out:{group_key}".encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < 0.5 else "test"


def assign_splits(
    clips: list[dict], cfg: SplitConfig, given: dict[str, str] | None = None,
    split_groups: dict[str, str] | None = None,
) -> list[dict]:
    """Return split rows for clean clips. Clips without a group use their own ID as the group.

    `given` maps source keys of a held-out source to the split its dataset assigns (e.g. the
    team's recordings.csv); held-out clips without one get `held_out_split` of their group.
    `split_groups` maps source keys to the group to hash by instead of the clip's own
    (IngestResult.split_groups); the row records the group that was hashed.
    """
    given, split_groups = given or {}, split_groups or {}
    rows = []
    for clip in clips:
        if clip["status"] != "ok":
            continue
        group = (split_groups.get(clip["source_key"]) or clip["group_key"]
                 or f"{clip['source']}:clip:{clip['clip_id']}")
        if clip["source"] in BENCHMARK_SOURCES:
            split, strategy = "test", "benchmark"
        elif clip["source"] in TRAIN_SOURCES:
            split, strategy = "train", "train_only"
        elif clip["source"] in HELD_OUT_SOURCES and clip["source_key"] in given:
            split, strategy = given[clip["source_key"]], clip["source"]
        elif clip["source"] in HELD_OUT_SOURCES:
            split, strategy = held_out_split(group, cfg), f"{clip['source']}_hash(seed={cfg.seed})"
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
