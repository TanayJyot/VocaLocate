import duckdb
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vocalocate_demo import experiment, synth
from vocalocate_demo.data import Dataset, load_dataset, rebase_lake_path
from vocalocate_demo.demo_flow import demo_searches
from vocalocate_demo.evaluate import rank_metrics, score_metrics
from vocalocate_demo.mining import MiningConfig
from vocalocate_demo.search import embed_query
from vocalocate_demo.text import ClapScorer
from vocalocate_demo.train import TrainConfig
from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig


def test_existing_warehouse_without_unpaired_view_can_still_load_pairs(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    cfg.warehouse_path.parent.mkdir(parents=True)
    with duckdb.connect(str(cfg.warehouse_path)) as con:
        con.execute("""
            CREATE TABLE v_training_pairs AS SELECT
                'vis' AS source, 'voice' AS imitation_clip_id, 'sound' AS reference_clip_id,
                'train' AS split, 'features/voice.npy' AS imitation_feature_path,
                'features/sound.npy' AS reference_feature_path;
            CREATE TABLE v_sfx_candidates AS SELECT
                'vis' AS source, 'sound' AS clip_id, 'reference' AS role, 'features/sound.npy' AS feature_path;
            CREATE TABLE queries AS SELECT 'vis' AS source, 'voice' AS imitation_clip_id, 'bark' AS text_hint;
        """)
    dataset = load_dataset(cfg, ["vis"])
    assert dataset.split_pairs("train").imitation_clip_id.tolist() == ["voice"]
    assert dataset.split_unpaired("train").empty
    assert dataset.hints == {"voice": "bark"}
    assert rebase_lake_path("features/voice.npy", cfg.lake_root) == "features/voice.npy"


def test_dataset_loading_reports_an_empty_source(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    warehouse.load(cfg)
    with pytest.raises(ValueError, match=r"No training pairs for \['vis'\]"):
        load_dataset(cfg, ["vis"])


def test_metrics_report_empty_query_set():
    targets = np.empty(0, dtype=int)
    assert rank_metrics(np.empty((0, 2)), np.eye(2), targets, []) == {"n_queries": 0}
    assert score_metrics(np.empty((0, 2)), targets, []) == {"n_queries": 0}


def test_search_rejects_silent_query_before_embedding(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    query = tmp_path / "silent.wav"
    sf.write(query, np.zeros(32_000), 32_000)
    with pytest.raises(ValueError, match="silent.wav rejected by cleaning: silent"):
        embed_query(query, None, TrainConfig(), cfg)


def test_recipe_text_requires_at_least_one_requested_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: Dataset(pd.DataFrame(), pd.DataFrame()))
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    with pytest.raises(FileNotFoundError, match="no recipe checkpoints"):
        experiment.run_recipe_text_branch(cfg, "vis", [tmp_path], ClapScorer(backend="fake"), names=["greif"])


def test_text_branch_skips_splits_without_hints(tmp_path, caplog):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    clip = tmp_path / "sound.wav"
    signal = 0.5 * np.sin(2 * np.pi * 400 * np.arange(16_000) / 16_000)
    sf.write(clip, signal, 16_000)
    dataset = Dataset(
        pairs=pd.DataFrame([
            {"imitation_clip_id": f"{split}-voice", "reference_clip_id": f"{split}-sound", "split": split,
             "imitation_feature_path": None, "reference_feature_path": None}
            for split in ("val", "test")
        ]),
        candidates=pd.DataFrame([
            {"clip_id": f"{split}-sound", "clean_path": str(clip), "feature_path": None, "split": split,
             "tags": ["metal"], "category": "bell", "recording_id": f"{split}-recording"}
            for split in ("val", "test")
        ]),
        hints={"val-voice": "bell"},
    )
    clap = ClapScorer(backend="fake")
    result = experiment.run_text_branch(cfg, "vis", dataset, {}, TrainConfig(device="cpu"), clap)

    assert result == {"clap": clap.version, "n_queries": {"val": 1, "test": 0}, "models": {}}
    assert "no val or test imitations with hints" in caplog.text


def test_demo_and_benchmark_handle_missing_models(tmp_path, monkeypatch):
    cfg = PipelineConfig(lake_root=tmp_path / "lake")
    assert demo_searches(cfg, tmp_path, []) == {"library": {}, "queries": []}
    monkeypatch.setattr(experiment, "load_dataset", lambda cfg, sources: Dataset(pd.DataFrame(), pd.DataFrame()))

    (cfg.lake_root / "models" / "vis").mkdir(parents=True)
    result = experiment.run_benchmark(cfg, "vis", names=("baseline",), device="cpu")
    assert result["models"] == {}
    assert (cfg.lake_root / "models/vis/benchmark_qvim_dev.json").is_file()


def test_sweep_grid_keeps_controls_fixed_and_covers_all_band_cutoffs():
    base = MiningConfig(n_neg=4, seed=51)
    options = experiment.sweep_grid(base)
    assert {(cfg.band, cfg.dup_quantile) for cfg in options} == {
        (band, cutoff) for band in experiment.SWEEP_BANDS for cutoff in experiment.SWEEP_DUP_QUANTILES
    }
    assert all(cfg.n_neg == 4 and cfg.seed == 51 for cfg in options)


def test_synthetic_renderer_rejects_unknown_category_and_variant():
    rng = np.random.default_rng(490)
    with pytest.raises(ValueError, match="whale"):
        synth.render_sfx("whale", synth._rng_params("whale", rng), rng)
    with pytest.raises(ValueError, match="stretched"):
        synth.make_variant(np.ones(100), "stretched", rng)
