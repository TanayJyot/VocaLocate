"""Check the demo entry points and CPU-safe benchmark preprocessing."""

import csv
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from vocalocate_demo import app, demo_flow, experiment, model, recipe, text
from vocalocate_demo.cli import main
from vocalocate_demo.data import SFXMeta
from vocalocate_demo.synth import CATEGORIES


def test_app_uses_workdir_and_indexes_explicit_library(tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    library = workdir / "library"
    library.mkdir(parents=True)
    user_library = tmp_path / "sounds"
    user_library.mkdir()
    create_state = Mock()
    server = Mock()
    server.serve_forever.side_effect = KeyboardInterrupt
    serve = Mock(return_value=server)
    monkeypatch.setattr(app, "AppState", create_state)
    monkeypatch.setattr(app, "serve", serve)

    assert main(["app", "--workdir", str(workdir), "--root", str(user_library),
                 "--name", "studio", "--source", "vis", "--port", "8766",
                 "--text", "--clap-backend", "fake"]) == 0

    cfg, source = create_state.call_args.args
    assert cfg.lake_root == workdir / "lake"
    assert source == "vis"
    assert create_state.call_args.kwargs["examples"] == workdir / "queries"
    assert create_state.call_args.kwargs["clap"].backend == "fake"
    state = create_state.return_value
    assert state.add_library.call_args_list == [
        call("demo", [str(library)]), call("studio", [str(user_library)])
    ]
    serve.assert_called_once_with(state, "127.0.0.1", 8766)
    server.serve_forever.assert_called_once_with()


def test_search_reports_missing_checkpoint(tmp_path):
    lake = tmp_path / "lake"
    with pytest.raises(SystemExit, match=r"no trained model.*baseline\.pt"):
        main(["--lake", str(lake), "search", "--root", str(tmp_path),
              "--query", str(tmp_path / "query.wav"), "--model", "baseline"])


@pytest.mark.parametrize("extra_axis", [False, True])
def test_onnx_benchmark_ranks_matching_audio_and_writes_report(tmp_path, monkeypatch, capsys, extra_axis):
    rate = 16_000

    def write(name, sr, sign):
        path = tmp_path / f"{name}.wav"
        stereo = np.tile(np.array([0.25, 0.5], dtype=np.float32) * sign, (sr // 100, 1))
        sf.write(path, stereo, sr, subtype="FLOAT")
        return str(path)

    references = pd.DataFrame({
        "clip_id": ["a", "b"],
        "clean_path": [write("ref-a", rate, 1), write("ref-b", rate, -1)],
    })
    queries = pd.DataFrame({
        "reference_clip_id": ["b", "a"],
        "imitation_path": [write("query-b", rate // 2, -1), write("query-a", rate // 2, 1)],
    })
    dataset = SimpleNamespace(
        split_pairs=Mock(return_value=queries),
        split_candidates=Mock(return_value=references),
        meta=lambda: {clip: SFXMeta(clip, frozenset(), clip) for clip in ("a", "b")},
    )
    monkeypatch.setattr(experiment, "load_dataset", Mock(return_value=dataset))

    def embed(outputs, inputs):
        assert outputs == ["embedding"]
        # One clip per call, [1, samples] (or [1, 1, samples] for a model with an extra leading axis).
        clip = inputs["audio"]
        shape = (1, rate * 10)
        assert clip.shape == ((1, *shape) if extra_axis else shape)
        assert clip.dtype == np.float32
        np.testing.assert_allclose(np.abs(clip[..., 80]), 0.375, atol=1e-3)
        assert not np.any(clip[..., 200:])
        anchor = clip[..., 80]
        return [np.stack([anchor > 0, anchor < 0], axis=-1).astype(np.float32)]

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="audio", shape=[None] * (3 if extra_axis else 2))],
        get_outputs=lambda: [SimpleNamespace(name="embedding")],
        get_providers=lambda: ["CPUExecutionProvider"],
        run=Mock(side_effect=embed),
    )
    inference = Mock(return_value=session)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"], InferenceSession=inference,
    ))

    lake, model = tmp_path / "lake", tmp_path / "fixture.onnx"
    assert main(["--lake", str(lake), "benchmark-onnx", "--model", str(model), "--label", "fixture"]) == 0
    inference.assert_called_once_with(str(model), providers=["CPUExecutionProvider"])
    dataset.split_pairs.assert_called_once_with("test")
    dataset.split_candidates.assert_called_once_with("test")
    assert session.run.call_count == 4  # 2 queries + 2 references, one clip each

    report = json.loads((lake / "benchmarks" / "fixture_qvim_dev_16k.json").read_text(encoding="utf-8"))
    assert report["model"] == str(model)
    assert report["providers"] == ["CPUExecutionProvider"]
    assert report["metrics"]["n_queries"] == 2
    assert report["metrics"]["mrr"] == report["metrics"]["recall@1"] == 1.0
    rendered = capsys.readouterr().out
    assert "MRR 1.000" in rendered
    assert "published:" in rendered


def test_generate_module_entry_point_builds_all_sound_categories(tmp_path, monkeypatch, capsys):
    output = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["vocalocate-demo", "generate", "--small", "--out", str(output)])
    monkeypatch.delitem(sys.modules, "vocalocate_demo.cli", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("vocalocate_demo.cli", run_name="__main__")
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == str(output)

    with (output / "metadata.csv").open(newline="", encoding="utf-8") as manifest:
        rows = list(csv.DictReader(manifest))
    refs = [row for row in rows if row["role"] == "reference"]
    assert {row["category"] for row in refs} == set(CATEGORIES)
    assert len(refs) == 4 * len(CATEGORIES)
    for row in refs:
        audio, _ = sf.read(output / row["file"])
        assert np.isfinite(audio).all()
        assert 0.49 < np.max(np.abs(audio)) < 0.51


def test_run_cli_uses_small_defaults_and_prints_report(tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "run"
    workdir.mkdir()

    def run(workdir, cfg, train_cfg, mining_cfg, *, small, sweep, clap):
        assert cfg.lake_root == workdir / "lake"
        assert small and sweep
        assert (train_cfg.epochs, train_cfg.ft_epochs) == (6, 3)
        assert mining_cfg.n_neg == 4
        assert clap.backend == "fake"
        (workdir / "report.md").write_text("# Small demo\n", encoding="utf-8")

    monkeypatch.setattr(demo_flow, "run_demo", run)
    assert main(["run", "--workdir", str(workdir), "--small", "--sweep", "--n-neg", "4",
                 "--text-branch", "--clap-backend", "fake"]) == 0
    assert "# Small demo" in capsys.readouterr().out


def test_train_cli_reads_explicit_config_and_writes_sweep_report(tmp_path, monkeypatch, capsys):
    config = tmp_path / "override.toml"
    config.write_text("[pipeline]\nworkers = 2\n", encoding="utf-8")
    lake = tmp_path / "lake"
    (lake / "models" / "vis").mkdir(parents=True)
    run = Mock(return_value={
        "source": "vis", "device": "cpu", "data": {}, "models": {}, "mining": {},
        "sweep": [{"strategy": "retrieval", "band": 0.25, "dup_quantile": 0.99, "val_mrr": 0.75}],
    })
    monkeypatch.setattr(experiment, "run_experiment", run)

    assert main(["--config", str(config), "--lake", str(lake), "train", "--source", "vis",
                 "--sweep", "--device", "cpu"]) == 0
    cfg, source, train_cfg, _ = run.call_args.args
    assert (cfg.lake_root, cfg.workers, source, train_cfg.device) == (lake, 2, "vis", "cpu")
    assert run.call_args.kwargs == {"sweep": True, "clap": None}
    report = (lake / "models" / "vis" / "report.md").read_text(encoding="utf-8")
    assert "Validation sweep" in report
    assert "| retrieval | 0.25 | 0.99 | 0.750 |" in report
    assert report.strip() in capsys.readouterr().out


def test_recipe_cli_forwards_candidate_count_and_disables_amp(tmp_path, monkeypatch):
    run = Mock(return_value={"models": {}})
    monkeypatch.setattr(recipe, "run_recipe_suite", run)

    assert main(["--lake", str(tmp_path), "recipe", "--models", "greif", "--n-cands", "5",
                 "--no-amp", "--no-results-file"]) == 0
    assert run.call_args.args[3] == ["greif"]
    assert run.call_args.kwargs["overrides"] == {"n_cands": 5, "amp": False}
    assert run.call_args.kwargs["write_results"] is False


def test_fetch_models_checks_embedding_shapes_on_cpu(monkeypatch, capsys):
    image_encoder = Mock(return_value=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(128 * 256, 2)))
    clap = Mock()
    clap.return_value.embed_text.return_value = np.eye(2)
    monkeypatch.setattr(model, "MobileNetEncoder", image_encoder)
    monkeypatch.setattr(text, "ClapScorer", clap)

    assert main(["fetch-models", "--pretrained", "mn10_as", "--clap-model", "test-model"]) == 0
    image_encoder.assert_called_once_with(pretrained="mn10_as")
    clap.assert_called_once_with("test-model", backend="hf", device="cpu")
    clap.return_value.embed_text.assert_called_once_with(["metallic", "a dog barking"])
    rendered = capsys.readouterr().out
    assert "embedding (1, 2)" in rendered
    assert "text embedding (2, 2)" in rendered
