from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest


def test_vis_maps_ontology_freesound_ids_and_descriptions(vis_dir):
    result = ingest("vis", vis_dir)
    roles = {f.source_key.split("/")[-1]: f.role for f in result.files}
    assert roles == {"ref_dog.wav": "reference", "other_dog.wav": "library", "ref_cat.wav": "reference",
                     "other_cat.wav": "library", "im1.wav": "imitation", "im2.wav": "imitation", "im3.wav": "imitation",
                     "im4.wav": "unpaired_imitation"}
    meta = {m["source_key"].split("/")[-1]: m for m in result.metadata}
    assert meta["other_dog.wav"]["category"] == "Bark"
    assert meta["other_dog.wav"]["tags"] == ["Animal", "Domestic animals, pets", "Dog", "Bark"]
    assert meta["ref_cat.wav"]["recording_id"] == "freesound:1002"
    # Participants' own descriptions become text hints; empty ones are skipped.
    assert sorted(q["text_hint"] for q in result.queries) == ["a dog barking", "cat meowing"]


def test_vis_flow_builds_pairs_and_related_candidates(cfg, vis_dir):
    run_dataset_flow("vis", vis_dir, cfg)
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'vis'")[0][0] == 3
    rows = warehouse.query(cfg, "SELECT role, category FROM v_sfx_candidates WHERE source = 'vis' ORDER BY ALL")
    assert rows == [("library", "Bark"), ("library", "Meow"), ("reference", "Bark"), ("reference", "Meow")]
    assert warehouse.query(cfg, "SELECT count(*) FROM queries WHERE source = 'vis'")[0][0] == 2


def test_vis_keeps_rejected_imitations_as_unpaired_negatives(cfg, vis_dir):
    run_dataset_flow("vis", vis_dir, cfg)
    # The rejected take never forms a pair...
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'vis'")[0][0] == 3
    # ...but is available as a negative, linked to the sound it imitates and in that sound's split.
    rows = warehouse.query(cfg, """SELECT u.category, u.split = s.split FROM v_unpaired_imitations u
                                   JOIN splits s ON s.clip_id = u.target_clip_id""")
    assert rows == [("Meow", True)]
