"""A5 Part One: the optimised evaluation, mining and batching code matches the versions it replaced.

Each `_reference_*` below is the pre-optimisation code, kept verbatim, so these tests fail if a
later "speed-up" changes a metric, a mined negative or a training batch. See a5/profiling/README.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.evaluate import score_metrics
from vocalocate_demo.mining import MiningConfig, _PoolRelations, mine, relation
from vocalocate_demo.train import _unique_reference_batches


def _reference_fine_and_ndcg(sims, target, cand_meta):
    n_q, n_c = sims.shape
    t_score = sims[np.arange(n_q), target]
    cats = np.array([m.category for m in cand_meta], dtype=object)
    recs = np.array([m.recording_id for m in cand_meta], dtype=object)
    fine = []
    for i, t in enumerate(target):
        same = cats == cats[t]
        fine.append(1.0 / (1 + int((sims[i, same] > t_score[i]).sum())))
    order = np.argsort(-sims, axis=1)[:, :10]
    discounts = 1.0 / np.log2(np.arange(2, 12))
    ndcg = []
    for i, t in enumerate(target):
        rel = np.zeros(n_c)
        rel[cats == cats[t]] = 1
        if recs[t] is not None:
            rel[recs == recs[t]] = 2
        rel[t] = 3
        gains = (2 ** rel[order[i]] - 1) @ discounts[: order.shape[1]]
        ideal = (2 ** np.sort(rel)[::-1][: order.shape[1]] - 1) @ discounts[: order.shape[1]]
        ndcg.append(gains / ideal)
    return float(np.mean(fine)), float(np.mean(ndcg))


def _reference_batches(pairs, batch_size, rng):
    refs = pairs.reference_clip_id.to_numpy()
    batches = []
    for i in rng.permutation(len(pairs)):
        for rows, seen in batches:
            if len(rows) < batch_size and refs[i] not in seen:
                rows.append(int(i))
                seen.add(refs[i])
                break
        else:
            batches.append(([int(i)], {refs[i]}))
    return [rows for rows, _ in batches if len(rows) > 1]


def _meta(rng, n, none_rate=0.15):
    """Metadata with missing categories and recordings, shared recordings across categories, empty tags."""
    out = []
    for _ in range(n):
        cat = None if rng.random() < none_rate else f"c{rng.integers(5)}"
        rec = None if rng.random() < none_rate else f"r{rng.integers(n // 3 + 1)}"
        tags = frozenset(f"t{t}" for t in rng.choice(8, size=int(rng.integers(0, 4)), replace=False))
        out.append(SFXMeta(cat, tags, rec))
    return out


@pytest.mark.parametrize("n_q,n_c", [(40, 200), (25, 7), (5, 10), (1, 1)])
def test_score_metrics_matches_the_pre_optimisation_loop(n_q, n_c):
    rng = np.random.default_rng(n_q * 1000 + n_c)
    sims = rng.standard_normal((n_q, n_c))
    target = rng.integers(0, n_c, size=n_q)
    meta = _meta(rng, n_c)
    got = score_metrics(sims, target, meta)
    fine, ndcg = _reference_fine_and_ndcg(sims, target, meta)
    assert got["fine_mrr"] == pytest.approx(fine, abs=1e-12)
    assert got["ndcg@10"] == pytest.approx(ndcg, abs=1e-12)


def test_perfect_ranking_scores_one():
    meta = [SFXMeta("a", frozenset(), "r1"), SFXMeta("a", frozenset(), "r1"), SFXMeta("a", frozenset(), None),
            SFXMeta(None, frozenset(), None)]
    sims = np.array([[4.0, 3.0, 2.0, 1.0]])  # target, other take, same category, unrelated
    out = score_metrics(sims, np.array([0]), meta)
    assert out["mrr"] == out["fine_mrr"] == out["ndcg@10"] == pytest.approx(1.0)


@pytest.mark.parametrize("min_shared_tags", [0, 1, 2])
def test_pool_relations_equal_relation_for_every_pair(min_shared_tags):
    pool = _meta(np.random.default_rng(min_shared_tags), 60, none_rate=0.3)
    rels = _PoolRelations(pool, min_shared_tags)
    for p in range(len(pool)):
        want = [relation(pool[p], m, min_shared_tags) for m in pool]
        got = rels.of(p)
        assert list(got[np.arange(len(pool)) != p]) == [w for j, w in enumerate(want) if j != p]


@pytest.mark.parametrize("strategy", ["retrieval", "metadata", "random"])
def test_mining_is_deterministic_and_never_picks_the_positive(strategy):
    rng = np.random.default_rng(7)
    n = 80
    pool = rng.standard_normal((n, 16)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    ids = [f"s{i}" for i in range(n)]
    meta = dict(zip(ids, _meta(rng, n)))
    pos = rng.integers(0, n, size=50)
    q = pool[pos] + 0.1 * rng.standard_normal((50, 16)).astype(np.float32)
    args = (strategy, q, [f"q{i}" for i in range(50)], [ids[p] for p in pos], pool, ids, meta, MiningConfig(), "v", "r")
    a, b = mine(*args), mine(*args)
    assert a.rows == b.rows and a.rows
    for r in a.rows:
        assert r["negative_clip_id"] != r["positive_clip_id"]
        assert r["relation"] != "same_recording"


@pytest.mark.parametrize("n,n_refs,bs", [(500, 120, 32), (300, 5, 8), (50, 50, 1), (0, 1, 4), (97, 40, 2)])
def test_unique_reference_batches_match_the_pre_optimisation_version(n, n_refs, bs):
    refs = np.random.default_rng(n).integers(0, n_refs, size=n)
    pairs = pd.DataFrame({"reference_clip_id": [f"r{r}" for r in refs]})
    got = _unique_reference_batches(pairs, bs, np.random.default_rng(1))
    assert got == _reference_batches(pairs, bs, np.random.default_rng(1))
    for rows in got:
        assert len(rows) <= bs and len({refs[i] for i in rows}) == len(rows)
