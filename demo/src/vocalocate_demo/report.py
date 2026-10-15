"""Write the demo results as a short markdown report."""

from __future__ import annotations

from pathlib import Path

from .experiment import MODEL_NAMES

METRICS = ("recall@1", "recall@5", "mrr", "fine_mrr", "ndcg@10")


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def write_report(results: dict, path: Path) -> None:
    lines = [
        "# VocaLocate semi-hard negative mining demo",
        "",
        f"Source `{results['source']}`, device `{results['device']}`, total {results.get('total_sec', '?')} s.",
        "",
        "## Data (from the A2 pipeline warehouse)",
        "",
        "| split | imitation/reference pairs | references | candidate SFX |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, d in results["data"].items():
        lines.append(f"| {split} | {d['pairs']} | {d['references']} | {d['candidate_sfx']} |")

    lines += [
        "",
        "## Held-out test results",
        "",
        "Every imitation in the test split ranks all test-split SFX (references plus other takes and distractors).",
        "Fine MRR ranks the target only among SFX of its own category. nDCG@10 gives partial credit for",
        "other takes of the same recording (2) and same-category SFX (1). Each model keeps its best epoch",
        "by validation MRR; the test split is used once, here.",
        "",
        "| model | " + " | ".join(METRICS) + " | val MRR | kept epoch |",
        "| --- | " + " | ".join("---:" for _ in METRICS) + " | ---: | ---: |",
    ]
    for name, m in results["models"].items():
        row = [_fmt(m["test"].get(k)) for k in METRICS]
        lines.append(
            f"| {MODEL_NAMES.get(name, name)} | " + " | ".join(row) + f" | {_fmt(m['val']['mrr'])} | {m['kept_epoch']} |"
        )

    lines += [
        "",
        "## Mined negatives",
        "",
        "| strategy | imitations with a negative | negatives | band / fallback / random | mean rank | mean gap s+ − s− | relation to target |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for name, s in results["mining"].items():
        sel = s["selection"]
        rel = ", ".join(f"{k} {v}" for k, v in sorted(s["relation"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {name} | {s['n_queries_with_negative']}/{s['n_queries']} | {s['n_negatives']} | "
            f"{sel.get('band', 0)} / {sel.get('fallback', 0)} / {sel.get('random', 0)} | {_fmt(s['mean_rank'])} | "
            f"{_fmt(s['mean_gap'])} | {rel} |"
        )
    r = results["mining"].get("retrieval", {})
    if r:
        lines += [
            "",
            f"Retrieval-driven filtering removed {r.get('filtered_same_recording', 0)} other takes of the target's "
            f"recording (metadata) and {r.get('filtered_near_duplicate', 0)} audio near-duplicates from the top-"
            f"{r['config']['top_k']} lists (band {r['config']['band']}, near-duplicate quantile {r['config']['dup_quantile']}); {r.get('above_positive', 0)} candidates scored above the positive and were "
            "left out as too risky.",
        ]
    m = results["mining"].get("metadata", {})
    if m:
        lines.append(f"The metadata-guided pools held {m.get('mean_metadata_pool_size', 0)} related SFX per target on average.")

    if results.get("sweep") and len(results["sweep"]) > len(results["mining"]):
        lines += ["", "## Validation sweep (stage 5)", "", "| strategy | band | dup quantile | val MRR |",
                  "| --- | ---: | ---: | ---: |"]
        for row in results["sweep"]:
            lines.append(f"| {row['strategy']} | {row['band']} | {row['dup_quantile']} | {row['val_mrr']:.3f} |")

    demo = results.get("search_demo")
    if demo:
        lines += [
            "",
            "## Searching a messy library",
            "",
            f"Held-out SFX were copied into `library/<project>/SFX_0001.wav ...` ({next(iter(demo['library'].values()))['files']} files), "
            "indexed by the A2 library flow, embedded once per model and cached in the lake. Rank of the true target:",
            "",
            "| query | category | " + " | ".join(demo["queries"][0]["results"]) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in demo["queries"][0]["results"]) + " |",
        ]
        for q in demo["queries"]:
            ranks = " | ".join(str(v["target_rank"]) for v in q["results"].values())
            lines.append(f"| `{Path(q['query']).name}` | {q['category']} | {ranks} |")
    path.write_text("\n".join(lines) + "\n")
