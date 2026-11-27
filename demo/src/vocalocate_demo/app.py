"""The VocaLocate app: a local web UI over the library pipeline, the trained models and CLAP.

It follows the "High-Level Product Flow":

1. Select library folders (absolute paths on this machine; nothing is uploaded).
2. Index them with the A2 library flow (incremental) and embed every clip with
   the chosen model's SFX encoder, cached in the lake. Progress shows each
   folder and the current step.
3. Record an imitation in the browser (or upload one, or pick a demo example).
4. Optionally add a text hint, scored with CLAP and fused with the voice score.
5-6. Ranked results on their own page: file name, project folder, path,
   similarity, a 5-second preview and native drag-and-drop out of the browser.
7. Refine: change the hint or model and search again. The query embedding,
   text scores and library index are all cached, so nothing is recomputed.

The server is standard-library only (ThreadingHTTPServer) and serves the static
page in ./ui. As a desktop app (`vocalocate`, see desktop.py) it runs from a
per-user data dir: libraries persist in settings.json and are restored from
cached embeddings at boot. `vocalocate-demo app --workdir demo_out` still runs
it against a demo run, in memory only. The HTTP API is in demo/APP_CONTRACT.md.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import logging
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import numpy as np

from vocalocate_pipeline.config import PipelineConfig

from . import __version__
from . import desktop
from .names import MODEL_NAMES, model_dir

if TYPE_CHECKING:
    from .search import LibraryIndex

log = logging.getLogger(__name__)


# -- the engine (torch, scipy, the pipeline flows) -------------------------------------------------
# Imported lazily so the server binds and serves the window, /api/health and /api/status within a
# second or two of process start; a cold `import torch` alone can take ~30 s. The boot thread imports
# it in the background (AppState._boot); request threads that need it meanwhile get a 503.
_engine_lock = threading.Lock()
_engine_loaded = threading.Event()


class WarmingUp(Exception):
    """The engine is still being imported by the boot thread (HTTP 503)."""


WARMING_MESSAGE = "VocaLocate is still warming up… try again in a few seconds"


def load_engine(wait: bool = True) -> None:
    """Import the heavy modules. With wait=False, raise WarmingUp instead of blocking on another thread's import."""
    if _engine_loaded.is_set():
        return
    if not _engine_lock.acquire(blocking=wait):
        raise WarmingUp(WARMING_MESSAGE)
    try:
        from . import search as _search, train as _train  # noqa: F401
        from vocalocate_pipeline.flow_functions import audio as _audio  # noqa: F401
        _engine_loaded.set()
    finally:
        _engine_lock.release()


def _train():
    from . import train

    return train


# Module-level names (tests monkeypatch `app.index_library`); each imports the engine on first use.
def index_library(*args, **kwargs):
    from .search import index_library as fn

    return fn(*args, **kwargs)


def cached_index(*args, **kwargs):
    from .search import cached_index as fn

    return fn(*args, **kwargs)


def embed_query(*args, **kwargs):
    from .search import embed_query as fn

    return fn(*args, **kwargs)


def search(*args, **kwargs):
    from .search import search as fn

    return fn(*args, **kwargs)


def text_scores(*args, **kwargs):
    from .search import text_scores as fn

    return fn(*args, **kwargs)

UI_DIR = Path(__file__).parent / "ui"
# Order shown in the UI; the first one present is the default. `shared` is the recipe's shared
# encoder + augmentation (no mining), the model chosen for the app: it ties the QVIM 2025 winners'
# released model on QVIM DEV and EVAL and, unlike it, can be retrained (demo/README.md, "Choosing
# the app's model").
UI_MODELS = ("shared", "metadata", "retrieval", "baseline", "random")
MODEL_LABELS = {**MODEL_NAMES, "shared": "VocaLocate: shared encoder + augmentation (recommended)"}
HINT_SUGGESTIONS = ("metallic", "wooden", "short", "long", "high-pitched", "low-pitched", "distant", "electronic")


@dataclass
class Library:
    name: str
    roots: list[str]
    state: str = "queued"  # queued, scanning, embedding, ready, error
    message: str = ""
    folders: list[dict] = field(default_factory=list)  # per root: path, files
    indexes: dict[str, LibraryIndex] = field(default_factory=dict)  # model name -> index
    started: float = field(default_factory=time.time)
    finished: float | None = None
    summary: dict = field(default_factory=dict)
    saved: bool = False  # persisted in settings.json
    progress: dict | None = None  # {"done", "total"} while new clips are embedded

    def public(self) -> dict:
        any_index = next(iter(self.indexes.values()), None)
        return {
            "name": self.name,
            "roots": self.roots,
            "state": self.state,
            "message": self.message,
            "folders": self.folders,
            "files": len(any_index.clip_ids) if any_index else None,
            "models_ready": sorted(self.indexes),
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "summary": self.summary,
            "progress": self.progress,
            "saved": self.saved,
        }


class AppState:
    """Everything the server knows. With `home` (the desktop data dir) libraries and settings persist.

    `models` is the folder of trained .pt files; default lake/models/<source>. Point it at a recipe run
    (e.g. lake/models/vimsketch/recipe-492) to serve its `shared` model.
    """

    def __init__(self, cfg: PipelineConfig, source: str, clap=None, examples: Path | None = None,
                 models: Path | None = None, home: Path | None = None, text_auto: bool | None = None,
                 autostart_dir: Path | None = None):
        self.cfg, self.source, self.clap = cfg, source, clap
        self.models_dir = Path(models) if models else model_dir(cfg, source)
        self.home = Path(home) if home is not None else None
        self.autostart_dir = autostart_dir  # tests point this at a fake Startup folder
        self.lock = threading.RLock()  # libraries dict and index swaps
        self.model_lock = threading.RLock()  # model construction (ours and CLAP's)
        self.index_lock = threading.Lock()  # one indexing job at a time (the pipeline writes shared lake parts)
        self.wh_lock = threading.RLock()  # warehouse reads vs the warehouse rebuild after a scan
        self.libraries: dict[str, Library] = {}
        self.models: dict[str, tuple] = {}
        self.query_cache: dict[tuple[str, str], np.ndarray] = {}
        self.text_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self.alphas = self._read_alphas()
        self.examples = _load_examples(examples) if examples else []
        self.settings = desktop.load_settings(self.home) if self.home else dict(desktop.DEFAULT_SETTINGS)
        self.text_state = "ready" if clap is not None else "off"
        self.text_error = ""
        # Load CLAP in the background only for the desktop app (or when asked), never in plain tests.
        self.text_auto = (self.home is not None and bool(self.settings.get("text"))) if text_auto is None else text_auto
        self.boot = {"state": "starting", "message": "Starting"}
        self.pick_folder = _pick_folder  # replaceable in tests
        self.reveal_file = _reveal
        for entry in self.settings["libraries"]:
            self.libraries[entry["name"]] = Library(
                name=entry["name"], roots=entry["roots"], saved=True,
                message="Restoring from the last session",
            )

    # -- boot ---------------------------------------------------------------
    def start(self) -> None:
        """Background boot: preload models, attach cached indexes, rescan libraries, load CLAP."""
        threading.Thread(target=self._boot, name="vocalocate-boot", daemon=True).start()

    def _boot(self) -> None:
        restored = list(self.libraries.values())
        try:
            # The server is already answering; the window shows boot.message while this runs.
            self.boot = {"state": "starting", "message": "Loading the sound engine"}
            load_engine()
            models = self.available_models()
            self.boot = {"state": "loading_models", "message": f"Loading {len(models)} model(s)"}
            for name in models:
                self.boot = {"state": "loading_models", "message": f"Loading {MODEL_LABELS[name]}"}
                _, _, version = self.model(name)
                for lib in restored:
                    self._attach_cached(lib, name, version)
            self.boot = {"state": "ready", "message": "Ready"}
        except Exception as exc:
            log.exception("boot failed")
            self.boot = {"state": "ready", "message": f"Could not load the models: {type(exc).__name__}: {exc}"}
        if not self.available_models():
            # CLAP waits for the first model import, which would otherwise queue behind its download.
            for lib in restored:
                lib.message = "Waiting for models: import a models folder to index this library"
            return
        self._maybe_load_clap()
        for lib in restored:
            if self.libraries.get(lib.name) is lib:
                if lib.indexes:
                    lib.state, lib.message = "ready", "Restored; checking for new or changed files"
                self._queue_index(lib)

    def _attach_cached(self, lib: Library, name: str, version: str) -> None:
        if name in lib.indexes:
            return
        try:
            idx = cached_index(self.cfg, lib.name, version, lock=self.wh_lock)
        except Exception:
            log.exception("could not restore %s for %s", lib.name, name)
            return
        if idx is not None:
            with self.lock:
                lib.indexes[name] = idx

    def _maybe_load_clap(self) -> None:
        if self.text_auto and self.clap is None and self.text_state == "off" and not self.text_error:
            self.text_state = "loading"
            threading.Thread(target=self._load_clap, name="vocalocate-clap", daemon=True).start()

    def _load_clap(self) -> None:
        if self.clap is not None:
            self.text_state = "ready"
            return
        if importlib.util.find_spec("transformers") is None:
            self.text_state, self.text_error = "off", "text hints need the `transformers` package (pip install transformers)"
            return
        self.text_state = "loading"
        try:
            from .text import ClapScorer

            # transformers builds models with a process-wide "empty weights" patch; building one of our
            # checkpoints at the same time would get meta tensors. Model construction is serialized.
            with self.model_lock:
                self.clap = ClapScorer()
            self.text_state, self.text_error = "ready", ""
        except Exception as exc:
            log.exception("loading CLAP failed")
            self.text_state, self.text_error = "error", f"{type(exc).__name__}: {exc}"

    # -- settings -----------------------------------------------------------
    def _save_settings(self) -> None:
        if self.home is not None:
            desktop.save_settings(self.home, self.settings)

    def _persist_library(self, lib: Library) -> None:
        if self.home is None:
            return
        libs = [e for e in self.settings["libraries"] if e["name"] != lib.name]
        libs.append({"name": lib.name, "roots": lib.roots})
        self.settings["libraries"] = libs
        self._save_settings()
        lib.saved = True

    def autostart_supported(self) -> bool:
        return self.home is not None and (self.autostart_dir is not None or desktop.autostart_supported())

    def public_settings(self) -> dict:
        supported = self.autostart_supported()
        return {
            "model": self.settings.get("model"),
            "text": bool(self.settings.get("text")),
            "autostart": supported and desktop.autostart_enabled(self.autostart_dir),
            "autostart_supported": supported,
        }

    def update_settings(self, body: dict) -> dict:
        if "model" in body:
            if body["model"] not in UI_MODELS:
                raise ValueError(f"unknown model {body['model']!r}; choose one of {', '.join(UI_MODELS)}")
            self.settings["model"] = body["model"]
        if "autostart" in body:
            if not self.autostart_supported():
                raise ValueError("starting at sign-in is only available in the Windows desktop app (vocalocate)")
            if body["autostart"]:
                desktop.enable_autostart(self.home, self.autostart_dir)
            else:
                desktop.disable_autostart(self.autostart_dir)
            self.settings["autostart"] = bool(body["autostart"])
        if "text" in body:
            self.settings["text"] = bool(body["text"])
            if self.settings["text"] and self.clap is None and self.text_state in ("off", "error"):
                threading.Thread(target=self._load_clap, name="vocalocate-clap", daemon=True).start()
        self._save_settings()
        return self.public_settings()

    # -- models -------------------------------------------------------------
    def _read_alphas(self) -> dict:
        fusion = self.models_dir / "fusion.json"
        try:
            return json.loads(fusion.read_text(encoding="utf-8"))["alpha"] if fusion.exists() else {}
        except (OSError, ValueError, KeyError):
            return {}

    def available_models(self) -> list[str]:
        return [m for m in UI_MODELS if (self.models_dir / f"{m}.pt").exists()]

    def default_model(self) -> str:
        available = self.available_models()
        if not available:
            raise KeyError(f"no trained model in {self.models_dir}")
        return available[0]

    def model(self, name: str):
        hit = self.models.get(name)
        if hit is not None:  # no lock: a CLAP load may hold it for minutes
            return hit
        with self.model_lock:
            if name not in self.models:
                path = self.models_dir / f"{name}.pt"
                if not path.exists():
                    raise KeyError(f"no trained model {name!r} in {path.parent}")
                model, tcfg, ckpt = _train().load(path)
                self.models[name] = (model, tcfg, ckpt["model_version"])
            return self.models[name]

    def public_models(self) -> list[dict]:
        return [{"name": m, "label": MODEL_LABELS[m]} for m in self.available_models()]

    def import_models(self, folder: str) -> dict:
        """Copy *.pt (+ fusion.json) from a folder into this app's model dir, load them, re-index libraries."""
        src = Path(str(folder)).expanduser()
        if not src.is_dir():
            raise ValueError(f"not a folder on this machine: {folder}")
        found = [m for m in UI_MODELS if (src / f"{m}.pt").is_file()]
        if not found:
            raise ValueError(
                f"no VocaLocate models in {src}: pick the folder that holds metadata.pt, retrieval.pt, "
                "baseline.pt, random.pt or shared.pt (and optionally fusion.json)"
            )
        dest = self.models_dir
        dest.mkdir(parents=True, exist_ok=True)
        same = dest.resolve() == src.resolve()
        changed = []
        for name in found:
            target = dest / f"{name}.pt"
            if not same:
                tmp = dest / f".{name}.pt.importing"
                shutil.copyfile(src / f"{name}.pt", tmp)
                try:
                    with self.model_lock:
                        _, _, ckpt = _train().load(tmp, device="cpu")
                    if "model_version" not in ckpt:
                        raise ValueError("no model_version")
                except Exception as exc:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(f"{src / (name + '.pt')} is not a VocaLocate model checkpoint ({exc})") from exc
                os.replace(tmp, target)
            changed.append(name)
        if (src / "fusion.json").is_file() and not same:
            shutil.copyfile(src / "fusion.json", dest / ".fusion.json.importing")
            os.replace(dest / ".fusion.json.importing", dest / "fusion.json")
        self.alphas = self._read_alphas()
        old = {n: self.models[n][2] for n in changed if n in self.models}
        with self.model_lock:
            for n in changed:
                self.models.pop(n, None)
        for n in changed:
            _, _, version = self.model(n)
            if n in old and old[n] != version:
                with self.lock:
                    for lib in self.libraries.values():
                        lib.indexes.pop(n, None)
        self._maybe_load_clap()
        for lib in list(self.libraries.values()):
            missing = [n for n in self.available_models() if n not in lib.indexes]
            if missing:
                try:
                    self.rescan(lib.name)
                except ValueError as exc:  # e.g. the folder is gone; the library reports it
                    lib.state, lib.message = "error", str(exc)
        return {"models": self.public_models()}

    # -- libraries ----------------------------------------------------------
    def add_library(self, name: str, roots: list[str], persist: bool = True) -> Library:
        clean = []
        for r in roots:
            p = Path(r).expanduser()
            if not p.is_dir():
                raise ValueError(f"not a folder on this machine: {r}")
            clean.append(str(p.resolve()))
        lib = Library(name=name, roots=clean)
        with self.lock:
            old = self.libraries.get(name)
            if old is not None and old.roots == clean:
                # Keep searching the previous index while the incremental re-index runs.
                lib.indexes = dict(old.indexes)
                lib.saved = old.saved
            self.libraries[name] = lib
        if persist:
            self._persist_library(lib)
        self._queue_index(lib)
        return lib

    def delete_library(self, name: str) -> None:
        with self.lock:
            if self.libraries.pop(name, None) is None:
                raise KeyError(f"unknown library {name!r}")
            self.text_cache = {k: v for k, v in self.text_cache.items() if k[0] != name}
        if self.home is not None:
            self.settings["libraries"] = [e for e in self.settings["libraries"] if e["name"] != name]
            self._save_settings()

    def rescan(self, name: str) -> Library:
        lib = self.libraries.get(name)
        if lib is None:
            raise KeyError(f"unknown library {name!r}")
        return self.add_library(name, lib.roots, persist=False)

    def _queue_index(self, lib: Library) -> None:
        threading.Thread(target=self._index, args=(lib,), name=f"index-{lib.name}", daemon=True).start()

    def _current(self, lib: Library) -> bool:
        return self.libraries.get(lib.name) is lib

    def _index(self, lib: Library) -> None:
        with self.index_lock:
            if not self._current(lib):  # deleted or replaced while waiting
                return
            try:
                lib.started = time.time()
                gone = [r for r in lib.roots if not Path(r).is_dir()]
                if gone:
                    # An unplugged drive must not scan as "everything deleted" and wipe the cached index.
                    lib.state, lib.finished = "error", time.time()
                    lib.message = (f"folder not found: {gone[0]} (reconnect the drive, or remove and re-add the "
                                   "library); searching the last index" if lib.indexes else f"folder not found: {gone[0]}")
                    return
                load_engine()
                from vocalocate_pipeline.flow_functions.audio import AUDIO_EXTENSIONS

                lib.state, lib.message = "scanning", "Counting audio files"
                lib.folders = [
                    {"path": r, "files": sum(1 for p in Path(r).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)}
                    for r in lib.roots
                ]
                n_files = sum(f["files"] for f in lib.folders)
                models = self.available_models()
                if not models:
                    raise RuntimeError("no models installed yet: import a models folder first (Import models…)")
                # First index: publish each model as soon as it is ready. Re-index: build a fresh set and swap it
                # in at the end, so ids (positions in the path-ordered list) always agree across models.
                incremental = not lib.indexes
                fresh: dict[str, LibraryIndex] = {}
                for i, name in enumerate(models):
                    model, tcfg, version = self.model(name)
                    lib.state = "scanning" if i == 0 else "embedding"
                    lib.message = (
                        f"Scanning {n_files:,} files: reading, cleaning and featurising new or changed ones"
                        if i == 0
                        else f"Embedding with {MODEL_LABELS[name]}"
                    )
                    lib.progress = None
                    t0 = time.time()

                    def progress(done, total, lib=lib):
                        lib.progress = {"done": done, "total": total}

                    idx = index_library(self.cfg, [Path(r) for r in lib.roots], lib.name, model, tcfg, version,
                                        scan=i == 0, lock=self.wh_lock, progress=progress)
                    lib.progress = None
                    if not self._current(lib):
                        return
                    fresh[name] = idx
                    if incremental:
                        with self.lock:
                            lib.indexes[name] = idx
                    lib.summary[name] = {"embedded": idx.n_embedded, "cached": len(idx.clip_ids) - idx.n_embedded,
                                         "sec": round(time.time() - t0, 1)}
                with self.lock:
                    lib.indexes = fresh
                    self.text_cache = {k: v for k, v in self.text_cache.items() if k[0] != lib.name}
                lib.state, lib.message, lib.finished = "ready", "Ready to search", time.time()
            except Exception as exc:  # report to the UI instead of killing the thread silently
                log.exception("indexing %s failed", lib.name)
                lib.progress = None
                lib.state, lib.message, lib.finished = "error", f"{type(exc).__name__}: {exc}", time.time()

    def default_library(self) -> str:
        if "default" in self.libraries or len(self.libraries) != 1:
            return "default"
        return next(iter(self.libraries))

    # -- search -------------------------------------------------------------
    def resolve_model(self, lib: Library, requested: str | None) -> str:
        if requested:
            return requested
        for name in [self.settings.get("model"), *UI_MODELS]:
            if name in lib.indexes:
                return name
        return self.settings.get("model") or "metadata"

    def search(self, library: str, model_name: str | None, audio: bytes, text: str, k: int) -> dict:
        lib = self.libraries.get(library)
        if lib is None:
            raise ValueError(f"library {library!r} is not indexed yet: add a sound folder first")
        model_name = self.resolve_model(lib, model_name)
        if model_name not in lib.indexes:
            raise ValueError(f"library {library!r} is not indexed for model {model_name!r} yet")
        idx = lib.indexes[model_name]
        model, tcfg, version = self.model(model_name)
        t0 = time.time()
        audio_key = hashlib.sha256(audio).hexdigest()
        reused = []
        vec = self.query_cache.get((audio_key, version))
        if vec is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                tmp = Path(f.name)
            try:
                vec = embed_query(tmp, model, tcfg, self.cfg)
            finally:
                tmp.unlink(missing_ok=True)
            self.query_cache[(audio_key, version)] = vec
        else:
            reused.append("query embedding")
        text = text.strip()
        scores_text, alpha, ignored = None, None, None
        if text and self.clap is None:
            # Never fail a search over the hint: rank by voice and say why the hint was not used.
            ignored = ("text hints are still loading; showing voice-only results" if self.text_state == "loading"
                       else self.text_error or "text hints are off; start the app with --text to use them")
        elif text:
            key = (library, model_name, text.lower())
            scores_text = self.text_cache.get(key)
            if scores_text is None or len(scores_text) != len(idx.clip_ids):
                scores_text = text_scores(self.cfg, library, idx, text, self.clap, lock=self.wh_lock)
                self.text_cache[key] = scores_text
            else:
                reused.append("text scores")
            alpha = float(self.alphas.get(model_name, 0.5))
        hits = search(idx, vec, k=min(k, len(idx.clip_ids)), text=scores_text, alpha=alpha if alpha is not None else 0.5)
        position = {p: i for i, p in enumerate(idx.paths)}
        voice = idx.vectors @ vec
        for h in hits:
            i = position[h["path"]]
            h["id"] = i
            h["name"] = Path(h["path"]).name
            h["voice_score"] = float(voice[i])
            if scores_text is not None:
                h["text_score"] = float(scores_text[i])
        out = {
            "library": library,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "text": text,
            "alpha": alpha,
            "results": hits,
            "searched": len(idx.clip_ids),
            "folders": lib.folders,
            "ms": round(1000 * (time.time() - t0), 1),
            "reused": reused,
        }
        if ignored:
            out["text_ignored"] = ignored
        return out

    def library_file(self, library: str, i: int) -> Path:
        lib = self.libraries.get(library)
        idx = next(iter(lib.indexes.values()), None) if lib else None
        if idx is None or not 0 <= i < len(idx.paths):
            raise KeyError("unknown file")
        # Every model's index of a library comes from the same ORDER BY path query, so ids agree
        # across models. Only files the library index knows about are ever served.
        return Path(idx.paths[i])

    def needs(self) -> list[str]:
        out = []
        if not self.available_models():
            out.append("models")
        if not self.libraries:
            out.append("library")
        return out

    def status(self) -> dict:
        libs = list(self.libraries.values())
        return {
            "source": self.source,
            "lake": str(self.cfg.lake_root),
            "models": self.public_models(),
            "text": self.clap is not None,
            "clap": getattr(self.clap, "version", None),
            "alphas": self.alphas,
            "libraries": [lib.public() for lib in libs],
            "examples": [
                {**{k: e[k] for k in ("id", "name", "category", "hint")}, "target": Path(e["target"]).name}
                for e in self.examples
            ],
            "hints": HINT_SUGGESTIONS,
            "ready": any(lib.indexes for lib in libs),
            "boot": dict(self.boot),
            "text_state": self.text_state,
            "text_error": self.text_error,
            "needs": self.needs(),
            "settings": self.public_settings(),
            "home": str(self.home) if self.home is not None else None,
            "version": __version__,
        }


def _load_examples(folder: Path) -> list[dict]:
    """Example imitations from a demo run's queries/queries.csv (held-out test imitations)."""
    csv_path = Path(folder) / "queries.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {"id": i, "name": Path(r["query"]).name, "path": r["query"], "category": r.get("category", ""),
         "hint": r.get("hint", ""), "target": r.get("target", "")}
        for i, r in enumerate(rows)
        if Path(r["query"]).exists()
    ]


# -- native helpers -------------------------------------------------------------------------------
class _FolderPicker:
    """Native folder dialog. One long-lived thread owns every Tk call; None when there is no GUI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def __call__(self, title: str = "Choose a folder") -> str | None:
        if sys.platform == "darwin":  # Tk must own the main thread on macOS; ask AppleScript instead
            return _pick_folder_macos(title)
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="folder-picker", daemon=True)
                self._thread.start()
            reply: queue.Queue = queue.Queue()
            self._jobs.put((title, reply))
            return reply.get()

    def _run(self) -> None:
        try:
            import tkinter
            from tkinter import filedialog

            root = tkinter.Tk()
            root.withdraw()
        except Exception:
            log.info("no GUI for the folder picker", exc_info=True)
            root = filedialog = None
        while True:
            title, reply = self._jobs.get()
            if root is None:
                reply.put(None)
                continue
            try:
                root.attributes("-topmost", True)
                root.lift()
                root.update()
                path = filedialog.askdirectory(parent=root, title=title, mustexist=True)
                root.update()
                reply.put(str(Path(path)) if path else None)
            except Exception:
                log.exception("folder picker failed")
                reply.put(None)


def _pick_folder_macos(title: str) -> str | None:
    script = f'POSIX path of (choose folder with prompt "{title.replace(chr(34), "")}")'
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


_pick_folder = _FolderPicker()


def _reveal(path: Path) -> None:
    """Show a file in the OS file manager, selected where the platform supports it."""
    if sys.platform == "win32":
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(Path(path).parent)])


# -- HTTP -----------------------------------------------------------------------------------------
def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VocaLocate/0.1"

        def log_message(self, fmt, *args):  # route through logging instead of stderr spam
            log.debug("%s - %s", self.address_string(), fmt % args)

        # -- helpers --------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, status: int, message: str):
            self._json({"error": message}, status)

        def _file(self, path: Path, download_name: str | None = None):
            if not path.is_file():
                return self._error(404, "file not found")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            extra = {"Content-Disposition": f'inline; filename="{download_name or path.name}"'}
            self._send(200, path.read_bytes(), ctype, extra)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 50 * 1024 * 1024:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("send a JSON object")
            return body

        def _allowed(self, write: bool) -> bool:
            """Host must be this server on localhost; writes must be JSON from our own origin (or no Origin).

            Blocks drive-by POSTs from web pages and DNS-rebinding reads of the always-on local server.
            """
            bind_host, port = self.server.server_address[:2]
            host = (self.headers.get("Host") or "").strip().lower()
            if bind_host not in ("0.0.0.0", "::", ""):
                allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}", f"{bind_host}:{port}".lower()}
                if host not in allowed:
                    return False
            if not write:
                return True
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin.strip().lower() == f"http://{host}"

        def _forbidden(self):
            # Read (and discard) a small body first: replying while the client is still sending makes
            # Windows reset the connection, and the client sees ConnectionAborted instead of the 403.
            n = int(self.headers.get("Content-Length") or 0)
            if 0 < n <= 1024 * 1024:
                self.rfile.read(n)
            else:
                self.close_connection = True
            return self._error(403, "forbidden: VocaLocate only accepts JSON requests from its own window "
                                    "(http://127.0.0.1:<port>/)")

        # -- routes ---------------------------------------------------------
        def do_GET(self):
            if not self._allowed(write=False):
                return self._forbidden()
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if not parts:
                    return self._file(UI_DIR / "index.html")
                if parts[0] == "static" and len(parts) == 2 and "/" not in parts[1] and ".." not in parts[1]:
                    return self._file(UI_DIR / parts[1])
                if parts == ["api", "health"]:
                    return self._json({"ok": True, "version": __version__, "app": desktop.APP_NAME})
                if parts == ["api", "status"]:
                    return self._json(state.status())
                if parts[:2] == ["api", "libraries"] and len(parts) == 3:
                    lib = state.libraries.get(parts[2])
                    return self._json(lib.public()) if lib else self._error(404, "unknown library")
                if parts[:2] == ["api", "audio"] and len(parts) == 4:
                    return self._file(state.library_file(parts[2], int(parts[3])))
                if parts[:2] == ["api", "example"] and len(parts) == 3:
                    ex = next((e for e in state.examples if e["id"] == int(parts[2])), None)
                    return self._file(Path(ex["path"])) if ex else self._error(404, "unknown example")
                return self._error(404, "not found")
            except (KeyError, ValueError) as exc:
                return self._error(404, str(exc))

        def do_DELETE(self):
            if not self._allowed(write=True):
                return self._forbidden()
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            try:
                if parts[:2] == ["api", "libraries"] and len(parts) == 3:
                    state.delete_library(parts[2])
                    return self._json({"deleted": parts[2]})
                return self._error(404, "not found")
            except KeyError as exc:
                return self._error(404, str(exc).strip("'\""))

        def do_POST(self):
            if not self._allowed(write=True):
                return self._forbidden()
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                body = self._body()
                if url.path == "/api/libraries":
                    name = (body.get("name") or "default").strip() or "default"
                    roots = [r for r in body.get("roots", []) if str(r).strip()]
                    if not roots:
                        raise ValueError("add at least one folder")
                    return self._json(state.add_library(name, roots).public(), 202)
                if parts[:2] == ["api", "libraries"] and len(parts) == 4 and parts[3] == "rescan":
                    if parts[2] not in state.libraries:
                        return self._error(404, f"unknown library {parts[2]!r}")
                    return self._json(state.rescan(parts[2]).public(), 202)
                if url.path == "/api/search":
                    if body.get("example") is not None:
                        ex = next((e for e in state.examples if e["id"] == int(body["example"])), None)
                        if ex is None:
                            raise ValueError("unknown example")
                        audio = Path(ex["path"]).read_bytes()
                    else:
                        audio = base64.b64decode(body.get("audio") or "")
                    if not audio:
                        raise ValueError("record or choose an imitation first")
                    load_engine(wait=False)  # 503 while the boot thread is still importing torch
                    out = state.search(
                        body.get("library") or state.default_library(), body.get("model") or None, audio,
                        body.get("text") or "", int(body.get("k") or 10),
                    )
                    return self._json(out)
                if url.path == "/api/pick-folder":
                    return self._json({"path": state.pick_folder(str(body.get("title") or "Choose a folder"))})
                if url.path == "/api/models/import":
                    if not str(body.get("path") or "").strip():
                        raise ValueError("choose the folder that holds the models (*.pt)")
                    load_engine()  # first run: may wait for the boot thread's import to finish
                    return self._json(state.import_models(body["path"]))
                if url.path == "/api/settings":
                    return self._json(state.update_settings(body))
                if url.path == "/api/reveal":
                    path = state.library_file(str(body.get("library") or ""), int(body.get("id")))
                    state.reveal_file(path)
                    return self._json({"path": str(path)})
                return self._error(404, "not found")
            except WarmingUp as exc:
                return self._error(503, str(exc))
            except KeyError as exc:
                return self._error(400 if url.path != "/api/reveal" else 404, str(exc).strip("'\""))
            except (ValueError, TypeError) as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                log.error("request failed: %s", traceback.format_exc())
                return self._error(500, f"{type(exc).__name__}: {exc}")

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use, which breaks
    # "next free port" and single-instance detection.
    allow_reuse_address = sys.platform != "win32"


def serve(state: AppState, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create the server (call .serve_forever() on it). Binds to localhost only by default."""
    return _Server((host, port), make_handler(state))


def serve_free(state: AppState, host: str = "127.0.0.1", port: int = 8765, tries: int = 50) -> ThreadingHTTPServer:
    """Like serve(), on the first free port from `port` up."""
    last: OSError | None = None
    for p in range(port, port + tries):
        try:
            return serve(state, host, p)
        except OSError as exc:
            last = exc
    raise OSError(f"no free port in {port}-{port + tries - 1}") from last
