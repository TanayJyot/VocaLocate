"""Vocal Imitation Set v1.1 (Kim et al., DCASE 2018): 302 AudioSet classes, one imitated reference each.

github.com/interactiveaudiolab/VocalImitationSet (audio on Zenodo). Layout:

    original_recordings/reference/<file>        the 302 sounds that were imitated
    original_recordings/non_reference/<file>    other recordings of the same classes (~2,700)
    original_recordings_classes.txt             tab-separated: filename, classname, category_d1..d6,
                                                is_reference, multi_labeled
    original_recordings_license.csv             filename, ..., freesound_id
    vocal_imitations/included/<file>            5,601 imitations that raters accepted
    vocal_imitations.txt                        tab-separated: imitation_filename, reference_filename,
                                                included, participant_id,
                                                participants_sound_recording_description, ...
    participant_survey.csv                      optional: participant_id, age, gender,
                                                years_actively_using_music_tech,
                                                years_actively_making_music, ...

This is the real-data source for the mining demos. It maps onto the SFX_Metadata schema:

* category is the AudioSet class; tags are its ontology path (depth 1-6), so the
  metadata-guided miner gets real related pools: same class, then siblings that
  share ancestors. Same-class non-reference recordings are candidate SFX ("library"),
  the fine-grained distractors the dataset was built for.
* recording_id is the Freesound ID, so clips of one Freesound upload share a split.
* the imitator's own description of the sound becomes the text hint, a real
  voice + text + target triple for the CLAP branch.

Only `included` imitations form training pairs. Drafts, practice takes and imitations raters
rejected (`vocal_imitations/excluded/`) are kept as `unpaired_imitation` clips: they share their
target's group and split but never pair with it, so training can use them as negatives for
*other* sounds, the way the QVIM 2025 winner used VocalSketch's excluded imitations.

The participant survey fills part of the imitator profile (see `survey_imitators`). In
vocal_imitations.txt the participant IDs are rounded to six significant digits
(e.g. 5734240000000000.0), so survey rows are matched to them by the rounded value.
Nothing in the release records the microphone, the vocalization type, the participants'
first language or their consent terms.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from . import IngestResult, read_table, require
from .sfx_meta import ingest_rows

SOURCE = "vis"
VERSION = "1.1"
LICENCE = "CC BY 4.0 (dataset); per-recording Freesound licences in original_recordings_license.csv"
DEPTHS = [f"category_d{i}" for i in range(1, 7)]
SURVEY = "participant_survey.csv"
# Both datasets were recorded by Amazon Mechanical Turk workers through the VocalSketch interface.
CROWD_ROLE = "crowd worker (Amazon Mechanical Turk)"


def _true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "t"}


def _find(root: Path, folders: list[str], name: str, index: dict[str, Path]) -> Path | None:
    for folder in folders:
        p = root / folder / name
        if p.exists():
            return p
    return index.get(name)


def build_rows(root: Path) -> list[dict]:
    """Convert the dataset's own metadata into sfx_meta rows (absolute file paths)."""
    classes = read_table(root / "original_recordings_classes.txt", "\t")
    licences = root / "original_recordings_license.csv"
    freesound = {r["filename"]: r.get("freesound_id", "") for r in read_table(licences, ",")} if licences.exists() else {}
    # Some releases nest files one level deeper; fall back to a filename index of the whole tree.
    index = {p.name: p for p in root.rglob("*.wav")}

    rows, seen_refs = [], set()
    for c in classes:
        name, cls = c["filename"], c.get("classname", "")
        is_ref = _true(c.get("is_reference", ""))
        folders = ["original_recordings/reference", "original_recordings/non_reference"]
        path = _find(root, folders if is_ref else folders[::-1], name, index)
        if path is None:
            path = root / ("original_recordings/reference" if is_ref else "original_recordings/non_reference") / name
        ontology = [c[d] for d in DEPTHS if c.get(d)]
        role = "reference" if is_ref and name not in seen_refs else "library"
        if is_ref:
            seen_refs.add(name)
        rows.append(
            {
                "file": str(path),
                "role": role,
                # Multi-labelled recordings appear once per class; the class keeps their sound IDs apart.
                "sound_id": name if role == "reference" else f"{cls}/{name}",
                "imitates": "",
                "category": cls,
                "tags": ";".join(dict.fromkeys(ontology + [cls])),
                "recording_id": f"freesound:{freesound[name]}" if freesound.get(name) else name,
                "description": " > ".join(ontology),
                "imitator_id": "",
                "text_hint": "",
                "hint_type": "",
            }
        )

    class_of = {c["filename"]: c.get("classname", "") for c in classes if _true(c.get("is_reference", ""))}
    for im in read_table(root / "vocal_imitations.txt", "\t"):
        if im.get("reference_filename") not in class_of:
            continue
        included = _true(im.get("included", ""))
        name = im["imitation_filename"]
        folders = ["vocal_imitations/included", "vocal_imitations/excluded"]
        path = _find(root, folders if included else folders[::-1], name, index)
        if path is None:
            path = root / folders[0 if included else 1] / name
        hint = im.get("participants_sound_recording_description", "") if included else ""
        rows.append(
            {
                "file": str(path),
                "role": "imitation" if included else "unpaired_imitation",
                "sound_id": "",
                "imitates": im["reference_filename"],
                "category": class_of[im["reference_filename"]],
                "tags": "",
                "recording_id": "",
                "description": "",
                "imitator_id": im.get("participant_id", ""),
                "text_hint": hint if hint.lower() not in {"", "nan", "none", "n/a"} else "",
                "hint_type": "participant description" if hint else "",
            }
        )
    return rows


def _years(value: str | None, age: str | None) -> int | None:
    """Whole years from a survey answer; None when missing or longer than the participant has lived."""
    try:
        years = float(value or "")
    except ValueError:
        return None
    try:
        if years > float(age or ""):
            return None  # e.g. 72 "years" at age 19: probably months, but we cannot tell
    except ValueError:
        pass
    return int(years) if years >= 0 else None


def survey_imitators(survey: list[dict], imitator_id: Callable[[str], str | None]) -> list[dict]:
    """Imitator profiles from the rows of a VocalSketch-style participant_survey.csv.

    `imitator_id` maps a survey participant_id to the imitator_id our clips use, or None to skip
    the participant (e.g. one who only took part in a listening task). Only the profile fields the
    survey answers are filled:

    * role: the crowd-worker role both datasets were recorded with.
    * years_experience: years_actively_using_music_tech (audio/music production technology).
    * music_training: years_actively_making_music > 0. This is the nearest question the survey
      asks ("how long have you been actively creating, practicing, or performing music"); it is
      not formal training.
    * first_language and consent_scope are not asked.
    """
    out, seen = [], set()
    for row in survey:
        iid = imitator_id(row.get("participant_id", ""))
        if iid is None or iid in seen:
            continue
        seen.add(iid)
        making = _years(row.get("years_actively_making_music"), row.get("age"))
        out.append(
            {
                "imitator_id": iid,
                "role": CROWD_ROLE,
                "years_experience": _years(row.get("years_actively_using_music_tech"), row.get("age")),
                "first_language": None,
                "music_training": None if making is None else making > 0,
                "consent_scope": None,
            }
        )
    return out


def rounded_id_matcher(ids: set[str], survey_ids: list[str]) -> Callable[[str], str | None]:
    """Match full survey IDs to the imitator IDs of vocal_imitations.txt, which are rounded floats.

    An exact match wins; otherwise the survey ID is rounded to six significant digits. A rounded
    value shared by two survey IDs is ambiguous and matches nothing.
    """

    def value(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None

    def rounded(v: float) -> float:
        return float(f"{v:.6g}")

    by_value = {value(i): i for i in ids if value(i) is not None}
    counts = Counter(rounded(v) for v in {value(p) for p in survey_ids} if v is not None)

    def match(pid: str) -> str | None:
        if pid in ids:
            return pid
        v = value(pid)
        if v is None:
            return None
        if v in by_value:
            return by_value[v]
        return by_value.get(rounded(v)) if counts[rounded(v)] == 1 else None

    return match


def ingest(root: Path) -> IngestResult:
    for required in ("original_recordings_classes.txt", "vocal_imitations.txt"):
        require(root / required, "Vocal Imitation Set metadata")
    rows = build_rows(root)
    result = ingest_rows(rows, root, source=SOURCE, version=VERSION, licence=LICENCE)
    if (root / SURVEY).exists():
        survey = read_table(root / SURVEY, ",")
        ids = {r["imitator_id"] for r in rows if r["imitator_id"]}
        match = rounded_id_matcher(ids, [r.get("participant_id", "") for r in survey])
        result.imitators = survey_imitators(survey, match)
    return result


# --------------------------------------------------------------------------- extras for a VimSketch lake

EXTRAS_SOURCE = "vis_extras"


def build_extra_rows(root: Path) -> list[dict]:
    """Only the un-imitated non-reference recordings, as library clips named for a VimSketch lake.

    VimSketch already holds each class's imitated reference and its imitations; what it lacks is
    the ~9 other recordings per class (2,633 in all). Category and tags are the VimSketch
    ingester's for the class, so a class's extras sit in the same category as its reference, and
    each extra takes the group key of that reference so it lands in the same split.
    """
    # Imported here: vimsketch imports survey_imitators from this module.
    from .vimsketch import SOURCE as VIMSKETCH
    from .vimsketch import sound_metadata, vis_class_sound

    classes = read_table(root / "original_recordings_classes.txt", "\t")
    licences = root / "original_recordings_license.csv"
    freesound = {r["filename"]: r.get("freesound_id", "") for r in read_table(licences, ",")} if licences.exists() else {}
    index = {p.name: p for p in root.rglob("*.wav")}
    rows = []
    for c in classes:
        if _true(c.get("is_reference", "")):
            continue
        name, cls = c["filename"], c.get("classname", "")
        path = _find(root, ["original_recordings/non_reference", "original_recordings/reference"], name, index)
        sound = vis_class_sound(cls)
        category, tags, description = sound_metadata(sound)
        rows.append({"file": str(path or root / "original_recordings/non_reference" / name), "role": "library",
                     "sound_id": f"{cls}/{name}", "imitates": "", "category": category, "tags": ";".join(tags),
                     "recording_id": f"freesound:{freesound[name]}" if freesound.get(name) else name,
                     "group_id": f"{VIMSKETCH}:{sound}", "description": description, "imitator_id": "",
                     "text_hint": "", "hint_type": ""})
    return rows


def ingest_extras(root: Path) -> IngestResult:
    require(root / "original_recordings_classes.txt", "Vocal Imitation Set metadata")
    return ingest_rows(build_extra_rows(root), root, source=EXTRAS_SOURCE, version=VERSION, licence=LICENCE)
