"""Average results.json files from several seeds into one table (mean ± std)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiment import MODEL_NAMES
from .recipe import RECIPE_NAMES
from .report import METRICS

LABELS = {**MODEL_NAMES, **RECIPE_NAMES}


def _on_benchmark(runs: list[dict], names: list[str], ref: str, bench: str, seeds: list, missing: list[str]) -> str:
    """For sources with no test split of their own (VimSketch): the benchmark is the test.

    Two numbers per model: the epoch picked on our validation split (clean), and the epoch that
    scored best on the benchmark itself (the protocol the QVIM 2025 winners reported).
    """
    def col(name: str, key: str, metric: str = "mrr") -> np.ndarray:
        return np.array([r["models"][name].get(key, {}).get(bench, {}).get(metric, np.nan) for r in runs])

    def delta(name: str, key: str) -> str:
        if name == ref:
            return "–"
        d = col(name, key) - col(ref, key)
        return f"{np.nanmean(d):+.3f} ({int((d > 0).sum())}/{len(runs)} seeds better)"

    lines = [
        f"# VocaLocate recipe comparison on {bench}: {len(runs)} seeds ({', '.join(map(str, seeds))})",
        "",
        f"{bench} test metrics, mean ± std over seeds. \"Selected on validation\" keeps the epoch with the best "
        f"validation MRR (the benchmark stays unseen); \"selected on {bench}\" keeps the epoch that scored best on "
        f"{bench} itself, as the QVIM 2025 winners did. Δ is each model minus {LABELS.get(ref, ref)!r} of the same seed.",
        "",
        f"| model | MRR, selected on validation | Δ | MRR, selected on {bench} | Δ | R@1 | R@5 | nDCG@10 | val MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        v, b = col(name, "benchmark"), col(name, "benchmark_selected_on_benchmark")
        extra = " | ".join(f"{np.nanmean(col(name, 'benchmark', k)):.3f}" for k in ("recall@1", "recall@5", "ndcg@10"))
        val = np.array([r["models"][name]["val"]["mrr"] for r in runs])
        lines.append(f"| {LABELS[name]} | {np.nanmean(v):.3f} ± {np.nanstd(v):.3f} | {delta(name, 'benchmark')} | "
                     f"{np.nanmean(b):.3f} ± {np.nanstd(b):.3f} | {delta(name, 'benchmark_selected_on_benchmark')} | "
                     f"{extra} | {val.mean():.3f} |")
    if missing:
        lines += ["", f"Skipped {len(missing)} run(s) without results.json: " + ", ".join(missing)]
    return "\n".join(lines) + "\n"


def summarize(paths: list[Path]) -> str:
    runs, missing = [], []
    for p in paths:
        p = p / "results.json" if p.is_dir() else p
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            missing.append(str(p))
    if not runs:
        raise ValueError(f"no results.json found in {[str(p) for p in paths]}")
    seeds = [r["train_config"]["seed"] for r in runs]
    ref = runs[0].get("reference_model", "baseline")
    names = [n for n in LABELS if all(n in r["models"] for r in runs)]
    benches = sorted({b for r in runs for m in r["models"].values() for b in m.get("benchmark", {})})
    title = "recipe comparison" if runs[0].get("kind") == "recipe" else "mining demo"
    if benches and all(m["test"].get("n_queries", 1) == 0 for r in runs for m in r["models"].values()):
        return _on_benchmark(runs, names, ref, benches[0], seeds, missing)
    lines = [
        f"# VocaLocate {title}: {len(runs)} seeds ({', '.join(map(str, seeds))})",
        "",
        f"Held-out test metrics, mean ± std over seeds. Δ MRR is each model minus {LABELS.get(ref, ref)!r} "
        "of the same seed." + (f" Benchmark columns: MRR on the {', '.join(benches)} test split." if benches else ""),
        "",
        "| model | " + " | ".join(METRICS) + " | Δ MRR vs reference | val MRR | "
        + "".join(f"{b} MRR | " for b in benches),
        "| --- | " + " | ".join("---:" for _ in METRICS) + " | ---: | ---: | " + "".join("---: | " for _ in benches),
    ]
    for name in names:
        cells = []
        for m in METRICS:
            v = np.array([r["models"][name]["test"][m] for r in runs])
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        d = np.array([r["models"][name]["test"]["mrr"] - r["models"][ref]["test"]["mrr"] for r in runs])
        wins = int((d > 0).sum())
        val = np.array([r["models"][name]["val"]["mrr"] for r in runs])
        delta = "–" if name == ref else f"{d.mean():+.3f} ({wins}/{len(runs)} seeds better)"
        bench_cells = ""
        for b in benches:
            v = np.array([r["models"][name].get("benchmark", {}).get(b, {}).get("mrr", np.nan) for r in runs])
            bench_cells += f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f} | "
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {delta} | {val.mean():.3f} | " + bench_cells)
    with_text = [r for r in runs if r.get("text", {}).get("models")]
    if with_text:
        lines += [
            "",
            f"Text branch ({with_text[0]['text']['clap']}), test imitations with a hint, mean ± std over "
            f"{len(with_text)} seeds:",
            "",
            "| voice model | α | MRR voice only | MRR text only | MRR voice + text | Δ MRR from adding text |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]

        def ms(values) -> str:
            v = np.array(values)
            return f"{v.mean():.3f} ± {v.std():.3f}"

        for name in MODEL_NAMES:
            rows = [r["text"]["models"][name] for r in with_text if name in r["text"]["models"]]
            if len(rows) != len(with_text):
                continue
            gain = [m["test"]["fused"]["mrr"] - m["test"]["voice"]["mrr"] for m in rows]
            wins = sum(g > 0 for g in gain)
            lines.append(
                f"| {MODEL_NAMES[name]} | {ms([m['alpha'] for m in rows])} | {ms([m['test']['voice']['mrr'] for m in rows])} | "
                f"{ms([m['test']['text']['mrr'] for m in rows])} | {ms([m['test']['fused']['mrr'] for m in rows])} | "
                f"{np.mean(gain):+.3f} ({wins}/{len(rows)} seeds better) |"
            )
    sweeps = [row for r in runs for row in r.get("sweep", [])]
    if sweeps:
        lines += ["", "Mining settings picked by validation, per seed:", ""]
        for r in runs:
            picked = {k: (v["config"]["band"], v["config"]["dup_quantile"]) for k, v in r["mining"].items()}
            lines.append(f"* seed {r['train_config']['seed']}: " + ", ".join(
                f"{k} band {b} dup {d}" for k, (b, d) in picked.items()))
    if missing:
        lines += ["", f"Skipped {len(missing)} run(s) without results.json: " + ", ".join(missing)]
    return "\n".join(lines) + "\n"
