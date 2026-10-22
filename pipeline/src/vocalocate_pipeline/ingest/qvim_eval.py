"""QVIM challenge evaluation set (MR20VI, zenodo.org/records/17453307): 100 references, 1,062 imitations.

The set the challenge's objective results were computed on, released after the challenge as
EVALDataset.zip. Same layout as the DEV set (see qvim_dev): "EVAL Dataset - EVAL Dataset.csv"
(title row, then Label, Class, Items, Query 1-3) with Items/<Class>/ and Queries/<Class>/, nested
in an "EVALCheck2/" folder. Like DEV it covers 20 classes; 4 of its 100 reference files also
appear in DEV. A held-out benchmark: every clip is put in the test split.
"""

from __future__ import annotations

from pathlib import Path

from . import IngestResult
from .qvim_dev import _find_index, read_challenge_set

SOURCE = "qvim_eval"
VERSION = "eval"


def ingest(root: Path) -> IngestResult:
    root, meta = _find_index(Path(root), ("EVAL Dataset - EVAL Dataset.csv", "EVAL Dataset.csv"), "EVALCheck2",
                             "EVAL*.csv")
    return read_challenge_set(root, meta, SOURCE, VERSION)
