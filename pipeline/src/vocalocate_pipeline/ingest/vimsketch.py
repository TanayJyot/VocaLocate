"""VimSketch (zenodo.org/records/2596911): 542 reference sounds, 12,543 vocal imitations.

Layout, as read by the QVIM challenge baseline:
    references/<prefix>_<reference name>.wav
    vocal_imitations/<prefix>_<reference name>.wav
    reference_file_names.csv, vocal_imitation_file_names.csv (one filename per line)
An imitation belongs to the reference with the same name after the first "_".
If the CSV lists are missing we list the folders instead.
"""

from __future__ import annotations

from pathlib import Path

from ..audio import AUDIO_EXTENSIONS
from . import IngestResult, RawFile

SOURCE = "vimsketch"
VERSION = "1.0"
LICENCE = "CC BY 4.0"


def reference_name(filename: str) -> str:
    return "_".join(filename.split("_")[1:])


def _list(root: Path, folder: str, csv_name: str) -> list[str]:
    listing = root / csv_name
    if listing.exists():
        return [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    return sorted(p.name for p in (root / folder).iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def ingest(root: Path) -> IngestResult:
    files, missing = [], []
    known_refs = set()
    for name in _list(root, "references", "reference_file_names.csv"):
        path = root / "references" / name
        if not path.exists():
            missing.append(f"references/{name}")
            continue
        ref = reference_name(name)
        known_refs.add(ref)
        files.append(
            RawFile(
                source=SOURCE,
                source_version=VERSION,
                source_key=f"references/{name}",
                path=path,
                role="reference",
                label=Path(ref).stem,
                group_key=f"{SOURCE}:{ref}",
                licence=LICENCE,
            )
        )
    for name in _list(root, "vocal_imitations", "vocal_imitation_file_names.csv"):
        path = root / "vocal_imitations" / name
        if not path.exists():
            missing.append(f"vocal_imitations/{name}")
            continue
        ref = reference_name(name)
        files.append(
            RawFile(
                source=SOURCE,
                source_version=VERSION,
                source_key=f"vocal_imitations/{name}",
                path=path,
                role="imitation",
                label=Path(ref).stem,
                # Imitations with no matching reference keep no group, and so get no pair.
                group_key=f"{SOURCE}:{ref}" if ref in known_refs else None,
                licence=LICENCE,
            )
        )
    return IngestResult(files=files, missing=missing)
