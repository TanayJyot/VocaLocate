"""Synthetic data -> A2 pipeline -> baseline -> both mining approaches -> fine-tuning -> search."""

import json

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig, SplitConfig
from vocalocate_demo.cli import main
from vocalocate_demo.demo_flow import run_demo
from vocalocate_demo.mining import MiningConfig
from vocalocate_demo.synth import generate
from vocalocate_demo.train import TrainConfig


def test_demo_runs_end_to_end(tmp_path, monkeypatch):
    import vocalocate_demo.demo_flow as flow

    tiny = dict(recordings_per_category=4, distractors_per_category=1, imitations_per_reference=3,
                categories=["metal_impact", "whoosh", "alarm_beep", "engine"])
    monkeypatch.setattr(flow, "generate", lambda out, **_: generate(out, **tiny))
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1, splits=SplitConfig(train=0.5, val=0.25))
    tcfg = TrainConfig(epochs=2, ft_epochs=1, frames=128, batch_size=8, device="cpu")
    from vocalocate_demo.text import ClapScorer

    results = run_demo(tmp_path, cfg, tcfg, MiningConfig(), n_queries=2, clap=ClapScorer(backend="fake"))

    assert set(results["models"]) == {"baseline", "retrieval", "metadata", "random"}
    for m in results["models"].values():
        assert 0 <= m["test"]["mrr"] <= 1
    # The mined negatives are in the warehouse, with the metadata of both sides.
    rows = warehouse.query(cfg, "SELECT strategy, count(*) FROM v_hard_negatives GROUP BY ALL ORDER BY ALL")
    assert {r[0] for r in rows} <= {"metadata", "random", "retrieval"} and rows
    assert warehouse.query(cfg, "SELECT count(*) FROM v_hard_negatives WHERE positive_category IS NULL")[0][0] == 0
    mining_runs = warehouse.query(cfg, "SELECT count(*) FROM pipeline_runs WHERE flow = 'mining'")[0][0]
    assert mining_runs == 3
    # Search over the messy library: every query's target is somewhere in the ranking.
    demo = results["search_demo"]
    assert len(demo["queries"]) == 2
    assert all(r["target_rank"] >= 1 for q in demo["queries"] for r in q["results"].values())
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# VocaLocate") and "## Text branch" in report
    # Text branch: hints came through the pipeline's queries table and were fused with every voice model.
    text = results["text"]
    assert text["n_queries"]["test"] > 0 and set(text["models"]) == set(results["models"])
    for m in text["models"].values():
        assert 0 <= m["alpha"] <= 1 and {"voice", "text", "fused"} == set(m["test"])
    assert warehouse.query(cfg, "SELECT count(*) FROM embeddings WHERE tower = 'clap_audio'")[0][0] > 0

    # Second search via the CLI reuses the cached embeddings and the library index.
    out = tmp_path / "search.json"
    query = demo["queries"][0]["query"]
    argv = ["--lake", str(cfg.lake_root), "search", "--root", str(tmp_path / "library"), "--name", "demo",
            "--query", query, "--model", "metadata", "-k", "3", "--json", "--text", "metallic", "--clap-backend", "fake"]
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(argv) == 0
    hits = json.loads(buf.getvalue())["metadata"][query]
    assert len(hits) == 3 and hits[0]["score"] >= hits[-1]["score"]

    # External benchmark: score the trained models on a QVIM DEV-style set they never saw.
    import csv

    import numpy as np
    import soundfile as sf
    from vocalocate_pipeline.flows import run_dataset_flow
    from vocalocate_demo.experiment import run_benchmark

    qvim = tmp_path / "qvim-dev"
    rows = []
    for i, cls in enumerate(["Dog", "Bell", "Engine"]):
        t = np.arange(32_000) / 32_000
        for kind, sub, freq in (("item", "Items", 300 + 200 * i), ("q", "Queries", 320 + 200 * i)):
            path = qvim / sub / cls / f"{kind}{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32"), 32_000)
        rows.append(["X", cls, f"item{i}.wav", f"q{i}.wav", "", ""])
    with open(qvim / "DEV Dataset.csv", "w", newline="", encoding="utf-8") as f:
        f.write("DEV set,,,,,\n")
        w = csv.writer(f)
        w.writerow(["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"])
        w.writerows(rows)
    run_dataset_flow("qvim_dev", qvim, cfg)
    bench = run_benchmark(cfg, "sfx_meta", "qvim_dev", device="cpu")
    assert set(bench["models"]) == {"baseline", "random", "retrieval", "metadata"}
    assert all(m["n_queries"] == 3 and m["n_candidates"] == 3 for m in bench["models"].values())
    assert bench["published_mrr"]["challenge winner"] == 0.293

