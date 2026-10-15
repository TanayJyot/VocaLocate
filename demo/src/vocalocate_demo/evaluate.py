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

    cats = np.array([m.category for m in cand_meta], dtype=object)
    recs = np.array([m.recording_id for m in cand_meta], dtype=object)
    fine = []
    for i, t in enumerate(target):
        same = cats == cats[t]
        fine.append(1.0 / (1 + int((sims[i, same] > t_score[i]).sum())))
    out["fine_mrr"] = float(np.mean(fine))

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
    out["ndcg@10"] = float(np.mean(ndcg))
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
