from vocalocate_pipeline import warehouse
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_pipeline.ingest import ingest
from vocalocate_pipeline.ingest.fine_grained_vis import class_category


def _name(key: str) -> str:
    return key.replace("\\", "/").split("/")[-1]


def test_class_names_match_vimsketch():
    assert class_category("001Animal_Domestic animals_ pets_Cat_Hiss") == "Animal_Domestic animals_pets_Cat_Hiss"
    assert (class_category("009Animal_Livestock_ farm animals_ working animals_Fowl_Chicken_ rooster_Cluck")
            == "Animal_Livestock_farm animals_working animals_Fowl_Chicken_rooster_Cluck")


def test_every_recording_is_a_reference_with_its_own_imitations(fgvis_dir):
    result = ingest("fgvis", fgvis_dir)
    files = {_name(f.source_key): f for f in result.files}
    assert sorted(n for n, f in files.items() if f.role == "reference") == ["caw_a.wav", "caw_b.wav", "hiss_a.wav",
                                                                             "hiss_b.wav"]
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-3.wav"].role == "unpaired_imitation"
    # Each imitation pairs with its own recording, not with the class.
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-2.wav"].group_key == files["hiss_b.wav"].group_key
    assert files["hiss_a.wav"].group_key != files["hiss_b.wav"].group_key
    assert files["001Animal_Domestic animals_ pets_Cat_Hiss-1.wav"].imitator_id == "fgvis:7"
    meta = {_name(m["source_key"]): m for m in result.metadata}
    assert meta["hiss_a.wav"]["category"] == meta["hiss_b.wav"]["category"] == "Animal_Domestic animals_pets_Cat_Hiss"
    assert meta["hiss_a.wav"]["tags"] == ["Animal", "Domestic animals", "pets", "Cat", "Hiss"]
    assert meta["caw_a.wav"]["recording_id"] == "freesound:2002"
    assert sorted(q["text_hint"] for q in result.queries) == ["a crow", "air leaking"]


def test_flow_puts_everything_in_train_and_builds_pairs(cfg, fgvis_dir):
    run_dataset_flow("fgvis", fgvis_dir, cfg)
    assert warehouse.query(cfg, "SELECT DISTINCT split FROM splits WHERE source = 'fgvis'") == [("train",)]
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'fgvis'")[0][0] == 4
    # The practice take is a negative linked to the recording it imitates.
    rows = warehouse.query(cfg, "SELECT category FROM v_unpaired_imitations WHERE source = 'fgvis'")
    assert rows == [("Animal_Domestic animals_pets_Cat_Hiss",)]


def test_with_vimsketch_a_class_shares_one_category(cfg, fgvis_dir, vimsketch_release):
    run_dataset_flow("vimsketch", vimsketch_release, cfg)
    run_dataset_flow("fgvis", fgvis_dir, cfg)
    rows = warehouse.query(cfg, """SELECT source, count(*) FROM v_sfx_candidates
                                   WHERE category = 'Animal_Domestic animals_pets_Cat_Hiss' GROUP BY ALL ORDER BY ALL""")
    assert rows == [("fgvis", 2), ("vimsketch", 1)]
    assert warehouse.query(cfg, "SELECT count(*) FROM v_cross_source_duplicates")[0][0] == 0


def test_vis_class_names_follow_vimsketch():
    from vocalocate_pipeline.ingest.vimsketch import vis_class_sound

    assert (vis_class_sound("017Animal_Livestock_ farm animals_ working animals_Horse_Snort (horse)")
            == "017Animal_Livestock_farm animals_working animals_Horse_Snort (horse).wav")
    assert (vis_class_sound("282Source-ambiguous sounds_Deformable shell_Crumpling_ crinkling")
            == "282Source_ambiguous sounds_Deformable shell_Crumpling_crinkling.wav")
    assert (vis_class_sound("029Animal_Wild animals_Rodents_ rats_ mice_Mouse___3Samples")
            == "029Animal_Wild animals_Rodents_rats_mice_Mouse_3Samples.wav")


def _vis_release(root):
    """A Vocal Imitation Set whose class is VimSketch's 001 (Cat Hiss): a reference and two extras."""
    import csv

    from conftest import _mk, tone, write

    cls = "001Animal_Domestic animals_ pets_Cat_Hiss"
    with open(_mk(root / "original_recordings_classes.txt"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "classname", "category_d1", "category_d2", "category_d3", "category_d4",
                    "category_d5", "category_d6", "is_reference", "multi_labeled"])
        for i, (name, ref) in enumerate([("hiss_ref.wav", "TRUE"), ("hiss_x.wav", "FALSE"), ("hiss_y.wav", "FALSE")]):
            write(root / "original_recordings" / ("reference" if ref == "TRUE" else "non_reference") / name,
                  tone(500 + 120 * i, 1.5, 44_100), 44_100)
            w.writerow([name, cls, "Animal", "Domestic animals, pets", "Cat", "Hiss", "", "", ref, "FALSE"])
    return root


def test_vis_extras_are_library_clips_in_their_vimsketch_class_and_split(cfg, tmp_path, vimsketch_release):
    from vocalocate_pipeline.flow_functions.splits import split_for_group

    vis = _vis_release(tmp_path / "VocalImitationSet")
    result = ingest("vis_extras", vis)
    assert sorted((_name(f.source_key), f.role) for f in result.files) == [("hiss_x.wav", "library"),
                                                                          ("hiss_y.wav", "library")]
    key = "vimsketch:001Animal_Domestic animals_pets_Cat_Hiss.wav"
    assert {f.group_key for f in result.files} == {key}
    run_dataset_flow("vimsketch", vimsketch_release, cfg)
    run_dataset_flow("vis_extras", vis, cfg)
    rows = warehouse.query(cfg, """SELECT source, role, split FROM v_sfx_candidates
                                   WHERE category = 'Animal_Domestic animals_pets_Cat_Hiss' ORDER BY ALL""")
    split = split_for_group(key, cfg.splits)
    assert rows == [("vimsketch", "reference", split), ("vis_extras", "library", split), ("vis_extras", "library", split)]
    # No pairs: the extras are candidates only.
    assert warehouse.query(cfg, "SELECT count(*) FROM v_training_pairs WHERE source = 'vis_extras'")[0][0] == 0
