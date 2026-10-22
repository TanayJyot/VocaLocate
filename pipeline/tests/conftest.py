"""Shared test helpers, plus synthetic datasets laid out exactly like the real ones so tests run offline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocalocate_pipeline.config import PipelineConfig


def tone(freq: float, sec: float, sr: int, amp: float = 0.5, channels: int = 1) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def write(path: Path, x: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)
    return path


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    return PipelineConfig(lake_root=tmp_path / "lake", workers=1)


@pytest.fixture
def esc50_dir(tmp_path) -> Path:
    root = tmp_path / "ESC-50"
    rows = [
        # filename, fold, target, category, esc10, src_file, take
        ("1-100-A-0.wav", tone(440, 5, 44_100), 44_100, "dog", "100", "A"),
        ("1-100-B-0.wav", tone(660, 5, 44_100), 44_100, "dog", "100", "B"),  # same source recording
        ("2-200-A-1.wav", tone(1000, 5, 44_100, channels=2), 44_100, "rooster", "200", "A"),
        ("3-300-A-2.wav", np.zeros(5 * 44_100, dtype=np.float32), 44_100, "pig", "300", "A"),  # silent
        ("4-400-A-3.wav", tone(300, 25, 22_050), 22_050, "cow", "400", "A"),  # too long
    ]
    with open(_mk(root / "meta" / "esc50.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        for i, (name, x, sr, cat, src, take) in enumerate(rows):
            write(root / "audio" / name, x, sr)
            w.writerow([name, name[0], i, cat, "False", src, take])
        w.writerow(["5-500-A-4.wav", 5, 4, "frog", "False", "500", "A"])  # listed but missing
    return root


@pytest.fixture
def vimsketch_dir(tmp_path) -> Path:
    root = tmp_path / "Vim_Sketch_Dataset"
    refs = ["000Animal_Cat_Growling.wav", "001Vehicle_Car_Horn.wav", "002Tools_Hammer.wav"]
    imits = [
        "00000_Cat_Growling.wav",
        "00001_Cat_Growling.wav",
        "00002_Car_Horn.wav",
        "00003_Hammer.wav",
        "00004_Unknown_Sound.wav",  # no matching reference
    ]
    for i, name in enumerate(refs):
        write(root / "references" / name, tone(300 + 200 * i, 3, 48_000), 48_000)
    for i, name in enumerate(imits):
        write(root / "vocal_imitations" / name, tone(250 + 150 * i, 2, 16_000), 16_000)
    (root / "reference_file_names.csv").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (root / "vocal_imitation_file_names.csv").write_text("\n".join(imits) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def vimsketch_release(tmp_path) -> Path:
    """VimSketch and VocalSketch unpacked side by side, named as in the real releases."""
    root = tmp_path / "datasets"
    vim = root / "Vim_Sketch_Dataset"
    refs = ["001_000Animal_Domestic animals_pets_Cat_Growling.wav", "002_001Animal_Domestic animals_pets_Cat_Hiss.wav",
            "303_001-abyss_of_despair-commercial_synthesizers.wav", "304_002-zipper_opening-everyday.wav"]
    imits = ["00001_000Animal_Domestic animals_pets_Cat_Growling.wav", "00002_001Animal_Domestic animals_pets_Cat_Hiss.wav",
             "05602_001-abyss_of_despair-commercial_synthesizers.wav", "05603_002-zipper_opening-everyday.wav"]
    for i, name in enumerate(refs):
        write(vim / "references" / name, tone(300 + 100 * i, 2, 44_100), 44_100)
    for i, name in enumerate(imits):
        write(vim / "vocal_imitations" / name, tone(250 + 90 * i, 1, 44_100), 44_100)
    (vim / "reference_file_names.csv").write_text("\n".join(refs), encoding="utf-8")  # no trailing newline, as released
    (vim / "vocal_imitation_file_names.csv").write_text("\n".join(imits), encoding="utf-8")

    vs = root / "interactiveaudiolab-VocalSketchDataSet-6c9f381"
    with open(_mk(vs / "sound_recordings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sound_label", "sound_label_id", "filename", "audio_concept_subset"])
        w.writerow(["r1", "abyss of despair", "L1", "abyss_of_despair.wav", "commercial synthesizers"])
        w.writerow(["r2", "zipper", "L2", "zipper_opening.wav", "everyday"])
    cols = ["id", "filename", "stimulus_type", "included", "draft", "training", "participant_id", "sound_label",
            "sound_label_id", "sound_recording", "sound_recording_id", "audio_concept_subset"]
    set1 = [  # an excluded imitation made from a recording, one made from a text label only, and an included one
        ["1", "abyss - 1.wav", "sound recording", "False", "True", "False", "p1", "abyss of despair", "L1",
         "abyss_of_despair.wav", "r1", "commercial synthesizers"],
        ["2", "zipper - 2.wav", "sound label", "False", "False", "True", "p2", "zipper", "L2", "", "", "everyday"],
        ["3", "zipper - 3.wav", "sound label", "True", "False", "False", "p3", "zipper", "L2", "", "", "everyday"],
    ]
    set2 = [["4", "zipper - 4.wav", "sound recording", "False", "False", "False", "p4", "zipper", "L2",
             "zipper_opening.wav", "r2", "everyday"]]
    for table, folder, rows in (("vocal_imitations.csv", "vocal_imitations", set1),
                                ("vocal_imitaitons_set2.csv", "vocal_imitations_set2", set2)):
        with open(_mk(vs / table), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        for i, row in enumerate(rows):
            sub = "included" if row[3] == "True" else "excluded"
            write(vs / folder / sub / row[1], tone(600 + 40 * i, 1, 44_100), 44_100)
    return root


@pytest.fixture
def qvim_dir(tmp_path) -> Path:
    root = tmp_path / "qvim-dev"
    write(root / "Items" / "Dog" / "bark.wav", tone(500, 2, 32_000), 32_000)
    write(root / "Queries" / "Dog" / "q1.wav", tone(520, 1, 32_000), 32_000)
    write(root / "Queries" / "Dog" / "q2.wav", tone(540, 1, 32_000), 32_000)
    with open(_mk(root / "DEV Dataset.csv"), "w", newline="", encoding="utf-8") as f:
        f.write("DEV set,,,,,\n")
        w = csv.writer(f)
        w.writerow(["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"])
        w.writerow(["Animal", "Dog", "bark.wav", "q1.wav", "q2.wav", ""])
    return root


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def sfx_meta_dir(tmp_path) -> Path:
    """Two recordings with a reference and an extra take each, one distractor, three imitations."""
    root = tmp_path / "sfx_meta"
    rows = [
        # file, role, sound_id, imitates, category, tags, recording_id
        ("sfx/bell_A.wav", "reference", "bell_A", "", "bell", "metallic;ringing", "bell"),
        ("sfx/bell_B.wav", "library", "bell_B", "", "bell", "metallic;ringing", "bell"),
        ("sfx/gong_A.wav", "reference", "gong_A", "", "bell", "metallic;low", "gong"),
        ("sfx/gong_B.wav", "library", "gong_B", "", "bell", "metallic;low", "gong"),
        ("sfx/buzz.wav", "library", "buzz", "", "buzzer", "electronic", ""),
        ("imitations/a/bell_0.wav", "imitation", "", "bell_A", "bell", "", ""),
        ("imitations/b/bell_1.wav", "imitation", "", "bell_A", "bell", "", ""),
        ("imitations/a/gong_0.wav", "imitation", "", "gong_A", "bell", "", ""),
    ]
    with open(_mk(root / "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "role", "sound_id", "imitates", "category", "tags", "recording_id", "description",
                    "imitator_id", "text_hint", "hint_type"])
        for i, (name, role, sid, imit, cat, tags, rec) in enumerate(rows):
            sr = 16_000 if role == "imitation" else 44_100
            write(root / name, tone(300 + 97 * i, 1.5, sr), sr)
            imitator = name.split("/")[1] if role == "imitation" else ""
            hint = ("metallic", "quality") if name == "imitations/a/bell_0.wav" else ("", "")
            w.writerow([name, role, sid, imit, cat, tags, rec, "", imitator, *hint])
    return root


@pytest.fixture
def vis_dir(tmp_path) -> Path:
    """Vocal Imitation Set layout: 2 classes, a reference + a same-class non-reference each, 3 imitations."""
    root = tmp_path / "VocalImitationSet"
    classes = [
        # filename, classname, d1, d2, d3, is_reference
        ("ref_dog.wav", "Bark", "Animal", "Domestic animals, pets", "Dog", "True"),
        ("other_dog.wav", "Bark", "Animal", "Domestic animals, pets", "Dog", "False"),
        ("ref_cat.wav", "Meow", "Animal", "Domestic animals, pets", "Cat", "True"),
        ("other_cat.wav", "Meow", "Animal", "Domestic animals, pets", "Cat", "False"),
    ]
    with open(_mk(root / "original_recordings_classes.txt"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "classname", "category_d1", "category_d2", "category_d3", "category_d4",
                    "category_d5", "category_d6", "is_reference", "multi_labeled"])
        for i, (name, cls, d1, d2, d3, ref) in enumerate(classes):
            folder = "reference" if ref == "True" else "non_reference"
            write(root / "original_recordings" / folder / name, tone(300 + 80 * i, 1.5, 44_100), 44_100)
            w.writerow([name, cls, d1, d2, d3, "", "", "", ref, "False"])
    with open(_mk(root / "original_recordings_license.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "username", "user_url", "file_url", "freesound_id"])
        for i, (name, *_rest) in enumerate(classes):
            w.writerow([name, "u", "", "", str(1000 + i)])
    imits = [
        ("im1.wav", "ref_dog.wav", "True", "p1", "a dog barking"),
        ("im2.wav", "ref_dog.wav", "True", "p2", ""),
        ("im3.wav", "ref_cat.wav", "True", "p1", "cat meowing"),
        ("im4.wav", "ref_cat.wav", "False", "p3", "rejected"),  # not included: kept as an unpaired negative
    ]
    with open(_mk(root / "vocal_imitations.txt"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["imitation_id", "imitation_filename", "category_id", "reference_filename", "included",
                    "draft", "training", "participant_id", "participants_sound_recording_description"])
        for i, (name, ref, inc, pid, desc) in enumerate(imits):
            folder = "included" if inc == "True" else "excluded"
            write(root / "vocal_imitations" / folder / name, tone(250 + 60 * i, 1.2, 16_000), 16_000)
            w.writerow([i, name, i, ref, inc, "False", "False", pid, desc])
    return root


@pytest.fixture
def captdure_dir(tmp_path) -> Path:
    """CAPTDURE layout: tab-separated "CSV" with a BOM and a quoted header, wavfile relative to the audio folder."""
    root = tmp_path / "captdure"
    clips = [("clock/clock_01_000.wav", 500), ("clock/clock_01_001.wav", 520), ("fan/fan_02_003.wav", 900)]
    for i, (name, freq) in enumerate(clips):
        write(root / "single_source_source" / name, tone(freq, 5, 48_000), 48_000)
    rows = [
        ("90", "clock/clock_01_000.wav", "時計", "A clock ticking steadily."),
        ("12", "clock/clock_01_000.wav", "時計", "Tick tock of a wall clock."),
        ("90", "clock/clock_01_001.wav", "時計", "Faster clock ticks."),
        ("33", "fan/fan_02_003.wav", "扇風機", "A fan blowing air."),
        ("33", "fan/fan_09_009.wav", "扇風機", "Listed but missing."),
    ]
    header = '"worker ID\twavfile\tcaption (ja)\tcaption (en)"'
    body = "\n".join("\t".join(r) for r in rows)
    (root / "single_source_caption.csv").write_text("﻿" + header + "\n" + body + "\n", encoding="utf-8")
    return root
