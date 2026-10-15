"""Average results.json files from several seeds into one table (mean ± std)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiment import MODEL_NAMES
from .report import METRICS


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
