import json

import pytest

from vocalocate_demo.cli import main
from vocalocate_demo.report import METRICS
from vocalocate_demo.summarize import summarize


def _run(seed, mrr):
    metrics = dict.fromkeys(METRICS, mrr) | {"n_queries": 3}
    return {
        "train_config": {"seed": seed},
        "models": {"baseline": {"test": metrics, "val": {"mrr": 0.5},
                                "benchmark": {"qvim_dev": {"mrr": 0.25}}}},
        "text": {"clap": "fixture-clap", "models": {
            "baseline": {"alpha": 0.5, "test": {"voice": {"mrr": 0.5}, "text": {"mrr": 0.25},
                                               "fused": {"mrr": 0.75}}}
        }},
        "sweep": [{"strategy": "metadata", "band": 0.25, "dup_quantile": 0.99, "val_mrr": 0.5}],
        "mining": {"metadata": {"config": {"band": 0.25, "dup_quantile": 0.99}}},
    }


def test_summarize_cli_renders_benchmarks_text_and_sweep_results(tmp_path, capsys):
    runs = []
    for seed, mrr in ((490, 0.25), (491, 0.75)):
        folder = tmp_path / f"run-{seed}"
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps(_run(seed, mrr)), encoding="utf-8")
        runs.append(str(folder))

    output = tmp_path / "summary.md"
    missing = tmp_path / "missing.json"
    assert main(["summarize", *runs, str(missing), "--out", str(output)]) == 0
    rendered = output.read_text(encoding="utf-8")
    assert rendered.strip() == capsys.readouterr().out.strip()
    assert "mining demo: 2 seeds (490, 491)" in rendered
    assert "qvim_dev MRR" in rendered
    assert "0.500 ± 0.250" in rendered
    assert "Text branch (fixture-clap)" in rendered
    assert "+0.250 (2/2 seeds better)" in rendered
    assert "* seed 491: metadata band 0.25 dup 0.99" in rendered
    assert f"Skipped 1 run(s) without results.json: {missing}" in rendered


def test_summarize_requires_at_least_one_result(tmp_path):
    with pytest.raises(ValueError, match="no results.json found"):
        summarize([tmp_path])
