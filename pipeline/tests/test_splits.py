from collections import Counter

from vocalocate_pipeline.config import SplitConfig
from vocalocate_pipeline.splits import assign_splits, split_for_group


def _clip(i, group, source="vimsketch", status="ok"):
    return {"clip_id": f"c{i}", "source": source, "source_key": f"k{i}", "group_key": group, "status": status}


def test_split_is_deterministic_and_roughly_proportional():
    cfg = SplitConfig()
    groups = [f"g{i}" for i in range(5000)]
    first = [split_for_group(g, cfg) for g in groups]
    assert first == [split_for_group(g, cfg) for g in groups]
    counts = Counter(first)
    assert 0.77 < counts["train"] / 5000 < 0.83
    assert 0.08 < counts["val"] / 5000 < 0.12


def test_all_clips_of_a_reference_share_a_split():
    clips = [_clip(i, f"g{i % 40}") for i in range(400)]
    by_group = {}
    for row in assign_splits(clips, SplitConfig()):
        by_group.setdefault(row["group_key"], set()).add(row["split"])
    assert all(len(s) == 1 for s in by_group.values())


def test_benchmark_source_is_always_test_and_rejected_clips_are_skipped():
    clips = [_clip(i, f"g{i}", source="qvim_dev") for i in range(20)] + [_clip(99, "x", status="rejected")]
    rows = assign_splits(clips, SplitConfig())
    assert len(rows) == 20
    assert {r["split"] for r in rows} == {"test"}

