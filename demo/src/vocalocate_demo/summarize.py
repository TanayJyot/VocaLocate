"""Average results.json files from several seeds into one table (mean ± std)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiment import MODEL_NAMES
from .report import METRICS


def summarize(paths: list[Path]) -> str:
    runs = []
    for p in paths:
        p = p / "results.json" if p.is_dir() else p
        runs.append(json.loads(p.read_text()))
    seeds = [r["train_config"]["seed"] for r in runs]
    lines = [
        f"# VocaLocate mining demo: {len(runs)} seeds ({', '.join(map(str, seeds))})",
        "",
        "Held-out test metrics, mean ± std over seeds. Δ MRR is each model minus the baseline of the same seed.",
        "",
        "| model | " + " | ".join(METRICS) + " | Δ MRR vs baseline | val MRR |",
        "| --- | " + " | ".join("---:" for _ in METRICS) + " | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        if not all(name in r["models"] for r in runs):
            continue
        cells = []
        for m in METRICS:
            v = np.array([r["models"][name]["test"][m] for r in runs])
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        d = np.array([r["models"][name]["test"]["mrr"] - r["models"]["baseline"]["test"]["mrr"] for r in runs])
        wins = int((d > 0).sum())
        val = np.array([r["models"][name]["val"]["mrr"] for r in runs])
        delta = "–" if name == "baseline" else f"{d.mean():+.3f} ({wins}/{len(runs)} seeds better)"
        lines.append(f"| {MODEL_NAMES[name]} | " + " | ".join(cells) + f" | {delta} | {val.mean():.3f} |")
    sweeps = [row for r in runs for row in r.get("sweep", [])]
    if sweeps:
        lines += ["", "Mining settings picked by validation, per seed:", ""]
        for r in runs:
            picked = {k: (v["config"]["band"], v["config"]["dup_quantile"]) for k, v in r["mining"].items()}
            lines.append(f"* seed {r['train_config']['seed']}: " + ", ".join(
                f"{k} band {b} dup {d}" for k, (b, d) in picked.items()))
    return "\n".join(lines) + "\n"
