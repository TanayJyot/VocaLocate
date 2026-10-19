"""The QVIM 2025 winning recipe, end to end on a tiny Vocal Imitation Set-shaped dataset (CPU, CNN)."""

import csv
import json

import numpy as np
import soundfile as sf
import torch

from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.flows import run_dataset_flow
from vocalocate_demo.augment import TRANSFORMS, _FNS, augment_batch
from vocalocate_demo.cli import RECIPE_ORDER, main
from vocalocate_demo.data import load_dataset
from vocalocate_demo.recipe import RECIPES, _semi_hard, excluded_pools
from vocalocate_demo.summarize import summarize


def _tone(freq, sec=1.2, sr=16_000, amp=0.4):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr


def _vis_dir(root):
    """12 classes in 3 ontology families: reference + same-class recording, 3 included and 2 excluded imitations."""
    classes, imits = [], []
    for k in range(12):
        fam = ["Animal", "Vehicle", "Tools"][k // 4]
        cls = f"{fam}_class{k}"
        for name, is_ref, f in ((f"ref{k}.wav", "True", 200 + 60 * k), (f"other{k}.wav", "False", 215 + 60 * k)):
            folder = "reference" if is_ref == "True" else "non_reference"
            p = root / "original_recordings" / folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), *_tone(f, sr=44_100))
            classes.append([name, cls, fam, f"{fam} sub", cls, "", "", "", is_ref, "False"])
        for j in range(5):
            included = "True" if j < 3 else "False"
            name = f"im{k}_{j}.wav"
            folder = "included" if included == "True" else "excluded"
            p = root / "vocal_imitations" / folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), *_tone(200 + 60 * k + 7 * j))
            imits.append([len(imits), name, k, f"ref{k}.wav", included, "False", "False", f"p{j}", ""])
    with open(root / "original_recordings_classes.txt", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "classname", "category_d1", "category_d2", "category_d3", "category_d4",
                    "category_d5", "category_d6", "is_reference", "multi_labeled"])
        w.writerows(classes)
    with open(root / "vocal_imitations.txt", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["imitation_id", "imitation_filename", "category_id", "reference_filename", "included",
                    "draft", "training", "participant_id", "participants_sound_recording_description"])
        w.writerows(imits)
    return root


def test_augmentations_keep_shape_and_change_input():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 32, 50)
    for name, fn in _FNS.items():
        y = fn(x, g)
        assert y.shape == x.shape, name
        assert torch.isfinite(y).all(), name
        assert not torch.equal(y, x), name
    assert augment_batch(x, g).shape == x.shape
    assert set(_FNS) | {"none"} == set(TRANSFORMS)


def test_batched_augmentations_match_their_reference_ops():
    from vocalocate_demo.augment import _gain, _shift, _stretch

    x = torch.randn(3, 8, 40)
    # shift is a per-row circular roll, gain a per-row constant offset
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    k = torch.randint(-12, 13, (3, 1), generator=g2).flatten().tolist()
    assert torch.equal(_shift(x, g1), torch.stack([torch.roll(r, s, dims=-1) for r, s in zip(x, k)]))
    d = _gain(x, torch.Generator().manual_seed(2)) - x
    assert torch.allclose(d, d[:, :1, :1].expand_as(d))
    # stretch matches F.interpolate (linear, align_corners=False) then crop/pad
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    rates = (torch.rand(3, generator=g2) * 0.4 + 0.8).tolist()
    y = _stretch(x, g1)
    for r, rate, row in zip(y, rates, x):
        n = max(int(round(40 * rate)), 1)
        ref = torch.nn.functional.interpolate(row.unsqueeze(0), size=n, mode="linear", align_corners=False)[0]
        m = min(n, 40)
        assert torch.allclose(r[:, :m], ref[:, :m], atol=1e-4)


def test_semi_hard_picks_hardest_negative_inside_the_band():
    s_pos = torch.tensor([0.8, 0.8, 0.8])
    s_neg = torch.tensor([[0.5, 0.7],    # both semi-hard: hardest is 0.7
                          [0.9, 0.1],    # 0.9 above the positive, 0.1 outside the band: none active
                          [0.6, 0.75]])  # last one is padding
    valid = torch.tensor([[True, True], [True, True], [True, False]])
    loss, active = _semi_hard(s_pos, s_neg, valid, margin=0.6)
    assert active.tolist() == [True, False, True]
    assert torch.allclose(loss, torch.tensor([0.6 - 0.8 + 0.7, 0.0, 0.6 - 0.8 + 0.6]))


def test_recipe_suite_runs_every_negative_source(tmp_path):
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1)
    run_dataset_flow("vis", _vis_dir(tmp_path / "VocalImitationSet"), cfg)
    ds = load_dataset(cfg, ["vis"])
    assert not ds.unpaired.empty and set(ds.unpaired.split) <= {"train", "val", "test"}

    # Excluded-imitation pools never contain imitations of the anchor's own sound.
    pools = excluded_pools(ds, RECIPES["excluded"], np.random.default_rng(0))
    target_of = dict(zip(ds.unpaired.clip_id, ds.unpaired.target_clip_id))
    assert pools and all(target_of[c] != ref for ref, cands in pools.items() for c in cands)

    args = ["--lake", str(cfg.lake_root), "recipe", "--source", "vis", "--encoder", "cnn", "--frames", "32",
            "--batch-size", "4", "--recipe-epochs", "2", "--bench", "", "--device", "cpu", "--seed", "490"]
    assert main(args) in (0, None)
    out = cfg.lake_root / "models" / "vis" / "recipe-490"
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert list(res["models"]) == list(RECIPE_ORDER)
    assert res["recipes"]["greif"]["shared"] is False and res["recipes"]["excluded"]["negatives"] == "excluded"
    trip = res["models"]["excluded"]["history"]
    assert len(trip) == 2 and all("active" in h for h in trip)

    # A re-run resumes: every condition is already done, so nothing retrains.
    stamp = (out / "excluded.pt").stat().st_mtime
    assert main(args) in (0, None)
    assert (out / "excluded.pt").stat().st_mtime == stamp

    # Parallel mode: one recipe per job writes only its own <name>.json; a later full call collects.
    par = args[:-1] + ["491", "--models", "greif", "--no-results-file"]
    assert main(par) in (0, None)
    out491 = cfg.lake_root / "models" / "vis" / "recipe-491"
    assert (out491 / "greif.json").exists() and not (out491 / "results.json").exists()
    assert json.loads((out491 / "greif.json").read_text(encoding="utf-8"))["batch_size"] == 4

    table = summarize([out])
    assert "recipe comparison" in table and "C: winner's method" in table


def test_lake_paths_from_another_machine_are_rebased(tmp_path):
    from vocalocate_demo.data import rebase_lake_path

    win = r"C:\Users\me\runs_real\lake\gold\features\vis\ab\abc.npy"
    assert rebase_lake_path(win, tmp_path) == str(tmp_path / "gold" / "features" / "vis" / "ab" / "abc.npy")
    here = tmp_path / "x.npy"
    here.write_bytes(b"")
    assert rebase_lake_path(str(here), tmp_path / "elsewhere") == str(here)  # existing paths are kept
    assert rebase_lake_path(None, tmp_path) is None
