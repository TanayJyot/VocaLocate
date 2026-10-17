from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest


def test_captdure_reads_tab_separated_captions_with_bom_and_quoted_header(captdure_dir):
    result = ingest("captdure", captdure_dir)
    assert sorted(f.source_key for f in result.files) == [
        "clock/clock_01_000.wav", "clock/clock_01_001.wav", "fan/fan_02_003.wav"]
    assert {f.role for f in result.files} == {"reference"}
    meta = {m["source_key"]: m for m in result.metadata}
    assert meta["clock/clock_01_001.wav"]["category"] == "clock"
    assert meta["clock/clock_01_001.wav"]["recording_id"] == "clock_01"
    assert len(result.captions) == 4
    assert {c["caption"] for c in result.captions if c["source_key"] == "clock/clock_01_000.wav"} == {
        "A clock ticking steadily.", "Tick tock of a wall clock."}
    assert result.missing == ["fan/fan_09_009.wav"]


def test_captdure_flow_holds_everything_out_and_builds_caption_view(cfg, captdure_dir):
    run_dataset_flow("captdure", captdure_dir, cfg)
    assert warehouse.query(cfg, "SELECT DISTINCT split FROM v_sfx_candidates WHERE source = 'captdure'") == [("test",)]
    rows = warehouse.query(cfg, "SELECT category, count(*) FROM v_captions GROUP BY ALL ORDER BY ALL")
    assert rows == [("clock", 3), ("fan", 1)]
