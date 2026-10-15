"""Candidate-side semi-hard negative mining: the two VocaLocate approaches, plus a random control.

Both approaches keep the same imitation as the anchor and choose a *real SFX*
as the negative (unlike thatsoundslikeme, whose negatives are other vocal
imitations). They differ in where the candidate negatives come from:

Retrieval-driven mining ("learn from real retrieval mistakes")
    1. Score every training-pool SFX for each training imitation with the
       baseline and take the top-K that are not the target.
    2. Filter likely false negatives: other takes of the target's recording
       (metadata) and SFX whose baseline embedding is a near-duplicate of the
       target's, i.e. in the top 1% of SFX-SFX similarities in the pool (audio
       only, so it works without metadata too).
    3. Semi-hard band: keep candidates scored below the positive but within
       `band` of it:  s_pos - band < s_neg < s_pos.  Sample from the band,
       not just the top wrong result.

Pre-constructed metadata-guided mining ("deliberately teach known related SFX")
    1. Before fine-tuning, build a pool of related non-target SFX for each
       target from training metadata, in priority tiers: same category and
       sharing tags, then same category, then sharing tags across categories.
       Other takes of the target's recording are never in the pool.
    2. Score the pool with the baseline and apply the same semi-hard band,
       filling from the highest tier first. These negatives need not be
       current top retrieval mistakes.

Random control: a random non-target SFX from the training pool, so we can tell
whether mining helps beyond simply training for longer (A2 stage 5).

Metadata is only used here, at training time. The deployed search never reads it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .data import SFXMeta


@dataclass
class MiningConfig:
    top_k: int = 20  # retrieval-driven: how deep into the ranked list to look
    band: float = 0.25  # semi-hard band below the positive score
    n_neg: int = 3  # negatives kept per imitation
    # A candidate is a near-duplicate of the target when their SFX-SFX cosine is in the top
    # `dup_quantile` of all pool pairs. Relative, because embedding spaces differ in spread.
    dup_quantile: float = 0.99
    use_metadata_filter: bool = True  # retrieval-driven: drop other takes of the target's recording
    min_shared_tags: int = 2  # metadata-guided: tags needed to count as related across categories
    seed: int = 490


def relation(target: SFXMeta, cand: SFXMeta, min_shared_tags: int = 2) -> str:
    if target.recording_id is not None and target.recording_id == cand.recording_id:
        return "same_recording"
    same_cat = target.category is not None and target.category == cand.category
    shared = len(target.tags & cand.tags) >= min_shared_tags
    if same_cat and shared:
        return "same_category+tags"
    if same_cat:
        return "same_category"
    if shared:
        return "shared_tags"
    return "unrelated"


METADATA_TIERS = ("same_category+tags", "same_category", "shared_tags")


@dataclass
class Mined:
    rows: list[dict]
    negatives: dict[str, list[str]]  # imitation clip_id -> negative clip_ids
    stats: dict


def _row(strategy, model_version, qid, pid, nid, rank, s_pos, s_neg, selection, rel, run_id):
    return {
        "strategy": strategy,
        "model_version": model_version,
        "imitation_clip_id": qid,
        "positive_clip_id": pid,
        "negative_clip_id": nid,
        "negative_rank": int(rank),
        "positive_score": float(s_pos),
        "negative_score": float(s_neg),
        "selection": selection,
        "relation": rel,
        "run_id": run_id,
    }


def _finish(strategy: str, rows: list[dict], n_queries: int, counters: Counter, cfg: MiningConfig) -> Mined:
    negatives: dict[str, list[str]] = {}
    for r in rows:
        negatives.setdefault(r["imitation_clip_id"], []).append(r["negative_clip_id"])
    gaps = [r["positive_score"] - r["negative_score"] for r in rows]
    stats = {
        "strategy": strategy,
        "config": asdict(cfg),
        "n_queries": n_queries,
        "n_queries_with_negative": len(negatives),
        "n_negatives": len(rows),
        "selection": dict(Counter(r["selection"] for r in rows)),
        "relation": dict(Counter(r["relation"] for r in rows)),
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "mean_rank": float(np.mean([r["negative_rank"] for r in rows])) if rows else None,
        **{k: int(v) for k, v in counters.items()},
    }
    return Mined(rows=rows, negatives=negatives, stats=stats)


def mine(
    strategy: str,
    q_emb: np.ndarray,
    q_ids: list[str],
    pos_ids: list[str],
    pool_emb: np.ndarray,
    pool_ids: list[str],
    meta: dict[str, SFXMeta],
    cfg: MiningConfig,
    model_version: str,
    run_id: str,
) -> Mined:
    """Choose negatives for each (imitation, positive) pair from the training pool."""
    if strategy not in ("retrieval", "metadata", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(cfg.seed)
    index = {cid: i for i, cid in enumerate(pool_ids)}
    pool_meta = [meta.get(cid, SFXMeta(None, frozenset(), None)) for cid in pool_ids]
    sims = q_emb @ pool_emb.T
    pool_sims = pool_emb @ pool_emb.T
    off_diag = pool_sims[~np.eye(len(pool_ids), dtype=bool)]
    dup_threshold = float(np.quantile(off_diag, cfg.dup_quantile)) if off_diag.size else 1.0
    order = np.argsort(-sims, axis=1)
    rank_of = np.empty_like(order)
    rank_of[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1] + 1)

    rows, counters = [], Counter()
    for i, (qid, pid) in enumerate(zip(q_ids, pos_ids)):
        if pid not in index:
            counters["skipped_positive_not_in_pool"] += 1
            continue
        p = index[pid]
        s_pos = sims[i, p]
        target_meta = pool_meta[p]
        rels = [relation(target_meta, m, cfg.min_shared_tags) for m in pool_meta]
        dup = pool_sims[p] > dup_threshold

        def ok(j: int) -> bool:
            return j != p and not dup[j] and rels[j] != "same_recording"

        def add(j: int, selection: str):
            rows.append(
                _row(strategy, model_version, qid, pid, pool_ids[j], rank_of[i, j], s_pos, sims[i, j], selection,
                     rels[j], run_id)
            )

        def in_band(j: int) -> bool:
            return s_pos - cfg.band < sims[i, j] < s_pos

        if strategy == "random":
            choices = [j for j in range(len(pool_ids)) if j != p and rels[j] != "same_recording"]
            for j in rng.choice(choices, size=min(cfg.n_neg, len(choices)), replace=False):
                add(int(j), "random")
            continue

        if strategy == "retrieval":
            top = [int(j) for j in order[i] if j != p][: cfg.top_k]
            kept = []
            for j in top:
                if dup[j]:
                    counters["filtered_near_duplicate"] += 1
                elif cfg.use_metadata_filter and rels[j] == "same_recording":
                    counters["filtered_same_recording"] += 1
                else:
                    kept.append(j)
            counters["above_positive"] += sum(sims[i, j] >= s_pos for j in kept)
            band = [j for j in kept if in_band(j)]
            for j in rng.permutation(band)[: cfg.n_neg]:
                add(int(j), "band")
            if len(band) < cfg.n_neg:
                # Closest-below-positive candidates from the top-K that missed the band.
                below = [j for j in kept if sims[i, j] <= s_pos - cfg.band]
                for j in below[: cfg.n_neg - len(band)]:
                    add(j, "fallback")
            continue

        # Metadata-guided: the pool is fixed by metadata before the model scores anything.
        tiers = [[j for j in range(len(pool_ids)) if rels[j] == t and ok(j)] for t in METADATA_TIERS]
        pool = [j for tier in tiers for j in tier]
        counters["metadata_pool_size"] += len(pool)
        if not pool:
            counters["empty_metadata_pool"] += 1
            continue
        chosen: list[int] = []
        for tier in tiers:
            band = [j for j in tier if in_band(j)]
            chosen += [int(j) for j in rng.permutation(band)[: cfg.n_neg - len(chosen)]]
            if len(chosen) == cfg.n_neg:
                break
        for j in chosen:
            add(j, "band")
        if len(chosen) < cfg.n_neg:
            below = sorted((j for j in pool if sims[i, j] <= s_pos - cfg.band), key=lambda j: -sims[i, j])
            fill = below[: cfg.n_neg - len(chosen)]
            for j in fill:
                add(j, "fallback")
            if not chosen and not fill:
                # Everything related scores above the positive: take one at random rather than nothing.
                add(int(rng.choice(pool)), "random")
    if strategy == "metadata" and q_ids:
        counters["mean_metadata_pool_size"] = counters.pop("metadata_pool_size", 0) // len(q_ids)
    counters["dup_threshold_x1000"] = int(dup_threshold * 1000)
    return _finish(strategy, rows, len(q_ids), counters, cfg)
