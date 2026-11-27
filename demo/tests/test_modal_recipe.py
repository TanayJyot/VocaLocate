"""Exercise the Modal launcher without a Modal account, GPU, or AWS calls."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vocalocate_demo import cli
from vocalocate_pipeline import sync


@pytest.fixture
def recipe_app(tmp_path, monkeypatch):
    app = Mock()
    app.function.return_value = lambda fn: fn
    app.local_entrypoint.return_value = lambda fn: fn
    modal = SimpleNamespace(
        is_local=lambda: True,
        Image=Mock(),
        App=Mock(return_value=app),
        Volume=Mock(),
        Secret=Mock(),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    path = Path(__file__).resolve().parents[1] / "modal" / "recipe_app.py"
    spec = importlib.util.spec_from_file_location("test_recipe_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "LAKE", str(tmp_path / "lake"))
    return launcher


def test_sync_copies_training_inputs_and_publishes_only_requested_source(recipe_app, monkeypatch):
    for name, value in {
        "VOCALOCATE_AWS_ROLE_ARN": "test-role",
        "MODAL_IDENTITY_TOKEN": "test-identity",
        "AWS_REGION": "ca-central-1",
        "VOCALOCATE_LAKE_URI": "s3://training-lake",
        "VOCALOCATE_MODELS_URI": "s3://model-outputs",
    }.items():
        monkeypatch.setenv(name, value)

    s3 = Mock(return_value=object())
    transfer = Mock(return_value=SimpleNamespace(copied=3, bytes=2_000_000_000, skipped=2))
    monkeypatch.setattr(sync, "s3_from_web_identity", s3)
    monkeypatch.setattr(sync, "sync", transfer)

    assert recipe_app.pull_lake("vis", "qvim_dev", "qvim_eval,") == "pulled 3 files (2.00 GB), 2 already up to date"
    assert recipe_app.publish("vis") == "published 3 files (2.00 GB) to s3://model-outputs/vis"
    assert transfer.call_args_list == [
        call(
            "s3://training-lake", recipe_app.LAKE,
            ["warehouse", "gold/features/vis", "gold/features/qvim_dev", "gold/features/qvim_eval"],
            s3=s3.return_value,
        ),
        call(f"{recipe_app.LAKE}/models/vis", "s3://model-outputs/vis", s3=s3.return_value),
    ]
    assert s3.call_args_list == [
        call("test-role", "test-identity", "ca-central-1", session_name="modal-vocalocate-recipe")
    ] * 2
    assert recipe_app.lake.method_calls == [call.reload(), call.commit(), call.reload()]


@pytest.mark.parametrize("n_queries", [3, 0])
def test_training_reports_metrics_and_skips_completed_run(recipe_app, monkeypatch, n_queries):
    output = Path(recipe_app.LAKE) / "models" / "vis" / "recipe-490" / "shared.json"

    def train(args):
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({
            "history": [{"sec": 2}, {"sec": 4}],
            "gpu": "test GPU",
            "kept_epoch": 2,
            "test": {"n_queries": n_queries, "mrr": 0.5},
            "benchmark": {"qvim_dev": {"mrr": 0.6}, "qvim_eval": {"mrr": 0.7}},
            "benchmark_selected_on_benchmark": {"qvim_dev": {"mrr": 0.8}},
        }), encoding="utf-8")
        return 0

    run = Mock(side_effect=train)
    monkeypatch.setattr(cli, "main", run)
    job = ("vis", "shared", 490, 8, 2, "qvim_dev", "qvim_eval", "vis=2")
    result = recipe_app.train_one(*job)
    assert isinstance(result.pop("wall_sec"), int)
    assert result == {
        "seed": 490, "name": "shared", "status": "done",
        "epoch_sec": 3.0, "gpu": "test GPU", "kept_epoch": 2,
        "test_mrr": 0.5 if n_queries else None,
        "bench_mrr": 0.6, "bench_mrr_selected_on_bench": 0.8,
        "extra_bench_mrr": {"qvim_eval": 0.7},
    }
    run.assert_called_once_with([
        "--lake", recipe_app.LAKE, "-v", "recipe", "--source", "vis", "--models", "shared",
        "--seed", "490", "--encoder", "mobilenetv3", "--device", "cuda", "--frames", "1000",
        "--batch-size", "8", "--bench", "qvim_dev", "--no-results-file", "--preload",
        "--extra-bench", "qvim_eval", "--recipe-epochs", "2", "--oversample", "vis=2",
    ])

    assert recipe_app.train_one(*job) == {"seed": 490, "name": "shared", "status": "already done"}
    assert run.call_count == 1
    assert recipe_app.lake.method_calls == [call.reload(), call.commit(), call.reload()]


def test_failed_training_does_not_commit_recipe_output(recipe_app, monkeypatch):
    run = Mock(return_value=2)
    monkeypatch.setattr(cli, "main", run)

    with pytest.raises(RuntimeError, match="exited with 2"):
        recipe_app.train_one("vis", "shared", 490, 8, 0, "qvim_dev")

    assert "--recipe-epochs" not in run.call_args.args[0]
    assert "--oversample" not in run.call_args.args[0]
    recipe_app.lake.commit.assert_not_called()


def test_collect_writes_aggregate_results_on_cpu(recipe_app, monkeypatch):
    run = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", run)

    recipe_app.collect("vis", 490, "greif,shared", "qvim_dev", "qvim_eval")

    run.assert_called_once_with([
        "--lake", recipe_app.LAKE, "-v", "recipe", "--source", "vis", "--models", "greif,shared",
        "--seed", "490", "--encoder", "mobilenetv3", "--device", "cpu", "--frames", "1000",
        "--bench", "qvim_dev", "--extra-bench", "qvim_eval",
    ])
    assert recipe_app.lake.method_calls == [call.reload(), call.commit()]


@pytest.mark.parametrize("sync_enabled, fail_first", [(True, False), (False, True)])
def test_launcher_collects_only_complete_seeds(recipe_app, monkeypatch, sync_enabled, fail_first):
    pull = Mock()
    publish = Mock()
    collect = Mock()
    training_results = [{"status": "done"} for _ in range(4)]
    if fail_first:
        training_results[0] = RuntimeError("training interrupted")
    train = Mock(return_value=training_results)
    for name, replacement in {
        "pull_lake": SimpleNamespace(remote=pull),
        "publish": SimpleNamespace(remote=publish),
        "collect": SimpleNamespace(remote=collect),
        "train_one": SimpleNamespace(starmap=train),
    }.items():
        monkeypatch.setattr(recipe_app, name, replacement)

    kwargs = dict(seeds="490, 491,", models="greif, shared, ", epochs=2, sync=sync_enabled, oversample="vis=2")
    if fail_first:
        with pytest.raises(SystemExit, match="1 trainings failed"):
            recipe_app.main(**kwargs)
    else:
        recipe_app.main(**kwargs)

    train.assert_called_once_with(
        [("vis", model, seed, 64, 2, "qvim_dev", "", "vis=2")
         for seed in (490, 491) for model in ("greif", "shared")],
        return_exceptions=True, order_outputs=True,
    )
    completed_seeds = (491,) if fail_first else (490, 491)
    assert collect.call_args_list == [call("vis", seed, "greif,shared", "qvim_dev", "") for seed in completed_seeds]
    if sync_enabled:
        pull.assert_called_once_with("vis", "qvim_dev", "")
        publish.assert_called_once_with("vis")
    else:
        pull.assert_not_called()
        publish.assert_not_called()
