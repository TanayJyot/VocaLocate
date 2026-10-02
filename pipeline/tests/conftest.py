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
    with open(_mk(root / "meta" / "esc50.csv"), "w", newline="") as f:
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
    (root / "reference_file_names.csv").write_text("\n".join(refs) + "\n")
    (root / "vocal_imitation_file_names.csv").write_text("\n".join(imits) + "\n")
    return root


@pytest.fixture
def qvim_dir(tmp_path) -> Path:
    root = tmp_path / "qvim-dev"
    write(root / "Items" / "Dog" / "bark.wav", tone(500, 2, 32_000), 32_000)
    write(root / "Queries" / "Dog" / "q1.wav", tone(520, 1, 32_000), 32_000)
    write(root / "Queries" / "Dog" / "q2.wav", tone(540, 1, 32_000), 32_000)
    with open(_mk(root / "DEV Dataset.csv"), "w", newline="") as f:
        f.write("DEV set,,,,,\n")
        w = csv.writer(f)
        w.writerow(["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"])
        w.writerow(["Animal", "Dog", "bark.wav", "q1.wav", "q2.wav", ""])
    return root


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
