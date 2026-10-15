import numpy as np

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.mining import MiningConfig, mine, relation


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _setup():
    """One query whose positive is `pos`. Candidates at controlled cosines to the query."""
    rng = np.random.default_rng(0)
    q = _unit([1.0, 0, 0, 0, 0, 0])
    # (id, cosine to query, category, recording)
    cands = [
        ("pos", 0.90, "metal", "r0"),
        ("take", 0.95, "metal", "r0"),  # another take of the target's recording: a likely substitute
        ("above", 0.93, "wood", "r1"),  # scores above the positive: too risky
        ("band_same_cat", 0.80, "metal", "r2"),
        ("band_other", 0.75, "glass", "r3"),
        ("far_same_cat", 0.20, "metal", "r4"),
        ("far_other", 0.10, "wind", "r5"),
    ]
    pool = []
    for _, cos, _, _ in cands:
        rest = _unit(rng.standard_normal(5)) * np.sqrt(1 - cos**2)
        pool.append(np.concatenate([[cos], rest]))
    meta = {
        cid: SFXMeta(cat, frozenset({cat, "impact"} if cat in ("metal", "wood", "glass") else {cat}), rec)
        for cid, _, cat, rec in cands
    }
    ids = [c[0] for c in cands]
    return q[None], ids, np.array(pool, dtype=np.float32), meta


def _mine(strategy, **kw):
    q, ids, pool, meta = _setup()
    cfg = MiningConfig(**{"band": 0.25, "n_neg": 2, "dup_quantile": 1.0, "min_shared_tags": 2, **kw})
    return mine(strategy, q, ["q"], ["pos"], pool, ids, meta, cfg, "test-model", "run-1")


def test_relation_levels():
    t = SFXMeta("metal", frozenset({"metal", "impact"}), "r0")
    assert relation(t, SFXMeta("metal", frozenset(), "r0")) == "same_recording"
    assert relation(t, SFXMeta("metal", frozenset({"metal", "impact"}), "r1")) == "same_category+tags"
    assert relation(t, SFXMeta("metal", frozenset(), "r1")) == "same_category"
    assert relation(t, SFXMeta("wood", frozenset({"metal", "impact"}), "r2")) == "shared_tags"
    assert relation(t, SFXMeta("wind", frozenset(), "r3")) == "unrelated"


def test_retrieval_driven_keeps_semi_hard_band_and_filters_false_negatives():
    mined = _mine("retrieval")
    negs = set(mined.negatives["q"])
    assert negs == {"band_same_cat", "band_other"}
    assert mined.stats["filtered_same_recording"] == 1
    assert mined.stats["above_positive"] == 1
    assert all(r["selection"] == "band" for r in mined.rows)
    assert all(r["positive_score"] - 0.25 < r["negative_score"] < r["positive_score"] for r in mined.rows)


def test_retrieval_driven_without_metadata_filter_can_pick_other_takes():
    mined = _mine("retrieval", use_metadata_filter=False, band=0.5)
    # "take" scores above the positive, so the band still excludes it.
    assert "take" not in mined.negatives["q"]
    assert mined.stats.get("filtered_same_recording", 0) == 0


def test_metadata_guided_prefers_related_sfx_even_if_not_top_mistakes():
    mined = _mine("metadata", n_neg=1)
    # Highest tier first: same category and shared tags, in the band.
    assert mined.negatives["q"] == ["band_same_cat"]
    assert mined.rows[0]["relation"] == "same_category+tags"
    assert "take" not in mined.negatives["q"]  # other takes are never in the pool


def test_metadata_guided_falls_back_below_band():
    mined = _mine("metadata", n_neg=3, band=0.05)
    # Nothing in the narrow band: the related SFX below the positive are used instead, closest first.
    # Only the two same-category SFX are related (wood and glass share one tag, fewer than min_shared_tags).
    assert [r["selection"] for r in mined.rows] == ["fallback"] * 2
    assert mined.negatives["q"] == ["band_same_cat", "far_same_cat"]


def test_random_control_never_picks_target_or_its_takes():
    mined = _mine("random", n_neg=4)
    assert not {"pos", "take"} & set(mined.negatives["q"])
    assert len(mined.negatives["q"]) == 4
