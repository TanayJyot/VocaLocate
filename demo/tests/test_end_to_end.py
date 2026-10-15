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
    results = run_demo(tmp_path, cfg, tcfg, MiningConfig(), n_queries=2)

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
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# VocaLocate")

    # Second search via the CLI reuses the cached embeddings and the library index.
    out = tmp_path / "search.json"
    query = demo["queries"][0]["query"]
    argv = ["--lake", str(cfg.lake_root), "search", "--root", str(tmp_path / "library"), "--name", "demo",
            "--query", query, "--model", "metadata", "-k", "3", "--json"]
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(argv) == 0
    hits = json.loads(buf.getvalue())["metadata"][query]
    assert len(hits) == 3 and hits[0]["score"] >= hits[-1]["score"]
