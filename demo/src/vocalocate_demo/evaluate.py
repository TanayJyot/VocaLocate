"""Retrieval metrics from A2: Recall@K, MRR, a fine-grained MRR and graded nDCG.

For each query imitation we rank every candidate SFX in the same split.

* Recall@K and MRR use the exact target only, as in the QVIM benchmark.
* Fine-grained MRR ranks the target only among SFX of its own category, like the
  Greif et al. (2024) fine test with same-class hard negatives. This is where
  candidate-side mining should help most.
* nDCG@10 uses graded relevance from the Relevance_Judgement wish-list schema:
  3 exact target, 2 usable substitute (another take of the same recording),
  1 same category, 0 unrelated. Here the grades come from metadata, not raters.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset, SFXMeta


def rank_metrics(
    q: np.ndarray,
    c: np.ndarray,
    target: np.ndarray,
    cand_meta: list[SFXMeta],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """q: (Nq, d) queries, c: (Nc, d) candidates, target: (Nq,) index of each query's target in c."""
    if len(q) == 0:
        return {"n_queries": 0}
    return score_metrics(q @ c.T, target, cand_meta, ks)


def score_metrics(
    sims: np.ndarray,
    target: np.ndarray,
    cand_meta: list[SFXMeta],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Metrics from a (Nq, Nc) score matrix, so fused voice + text scores are ranked the same way."""
    if len(sims) == 0:
        return {"n_queries": 0}
    n_q, n_c = sims.shape
    t_score = sims[np.arange(n_q), target]
    ranks = 1 + (sims > t_score[:, None]).sum(axis=1)
    out: dict[str, float] = {"n_queries": int(n_q), "n_candidates": int(n_c)}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
    out["mrr"] = float(np.mean(1.0 / ranks))

    # A5 Part One (profiling): fine MRR and nDCG used a Python loop over queries that rebuilt an
    # n_c relevance vector and sorted it for every query, and a full argsort of every row to read
    # 10 entries. Both are now whole-matrix numpy operations; the metrics are unchanged.
    # Categories and recordings become integer codes so equality is a vectorised int compare.
    # A None category is a value of its own (None == None, as before); a None recording matches nothing.
    cat_code = value_codes([m.category for m in cand_meta])
    rec_code = value_codes([m.recording_id for m in cand_meta], none_is_unique=True)
    same_cat = cat_code[None, :] == cat_code[target][:, None]  # (n_q, n_c)
    same_rec = rec_code[None, :] == rec_code[target][:, None]
    rows = np.arange(n_q)

    out["fine_mrr"] = float(np.mean(1.0 / (1 + (same_cat & (sims > t_score[:, None])).sum(axis=1))))

    top = min(10, n_c)
    # argpartition finds the top 10 in O(n_c); only those 10 are then sorted.
    part = np.argpartition(-sims, top - 1, axis=1)[:, :top] if top < n_c else np.tile(np.arange(n_c), (n_q, 1))
    order = np.take_along_axis(part, np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1), axis=1)
    discounts = 1.0 / np.log2(np.arange(2, 2 + top))

    rel_top = np.where(np.take_along_axis(same_cat, order, axis=1), 1.0, 0.0)
    rel_top[np.take_along_axis(same_rec, order, axis=1)] = 2.0
    rel_top[order == target[:, None]] = 3.0
    gains = (2**rel_top - 1) @ discounts

    # The ideal ranking is the target (3), then the other takes (2), then the rest of the category (1).
    n_rec = same_rec.sum(axis=1) - same_rec[rows, target]
    n_cat = (same_cat & ~same_rec).sum(axis=1) - (same_cat & ~same_rec)[rows, target]
    pos = np.arange(top)[None, :]
    ideal_rel = np.where(pos == 0, 3.0, np.where(pos <= n_rec[:, None], 2.0, np.where(pos <= (n_rec + n_cat)[:, None], 1.0, 0.0)))
    ideal = (2**ideal_rel - 1) @ discounts
    out["ndcg@10"] = float(np.mean(gains / ideal))
    return out


def value_codes(values: list, none_is_unique: bool = False) -> np.ndarray:
    """Integer code per value. With none_is_unique, every None gets its own code (it equals nothing else)."""
    table: dict = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if v is None and none_is_unique:
            out[i] = -1 - i
        else:
            out[i] = table.setdefault(v, len(table))
    return out


def evaluate_split(model, ds: Dataset, split: str, cfg) -> dict[str, float]:
    """Embed a split's imitations and candidate SFX with `model` and score them."""
    from .train import embed  # local import: train imports this module

    pairs = ds.split_pairs(split)
    cands = ds.split_candidates(split)
    index = {cid: i for i, cid in enumerate(cands.clip_id)}
    pairs = pairs[pairs.reference_clip_id.isin(index)]
    if pairs.empty:
        return {"n_queries": 0, "mrr": 0.0}
    meta = ds.meta()
    q = embed(model, ds, list(pairs.imitation_clip_id), "imitation", cfg)
    c = embed(model, ds, list(cands.clip_id), "sfx", cfg)
    target = np.array([index[r] for r in pairs.reference_clip_id])
    return rank_metrics(q, c, target, [meta[cid] for cid in cands.clip_id])
