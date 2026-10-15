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
page in ./ui. Run it with `vocalocate-demo app --workdir demo_out`.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import mimetypes
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from vocalocate_pipeline.audio import AUDIO_EXTENSIONS
from vocalocate_pipeline.config import PipelineConfig

from . import train as T
from .experiment import MODEL_NAMES, model_dir
from .search import LibraryIndex, embed_query, index_library, search, text_scores

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"
# Order shown in the UI: the two VocaLocate versions first.
UI_MODELS = ("metadata", "retrieval", "baseline", "random")
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
        }


class AppState:
    def __init__(self, cfg: PipelineConfig, source: str, clap=None, examples: Path | None = None):
        self.cfg, self.source, self.clap = cfg, source, clap
        self.lock = threading.RLock()
        self.libraries: dict[str, Library] = {}
        self.models: dict[str, tuple] = {}
        self.query_cache: dict[tuple[str, str], np.ndarray] = {}
        self.text_cache: dict[tuple[str, str, str], np.ndarray] = {}
        fusion = model_dir(cfg, source) / "fusion.json"
        self.alphas = json.loads(fusion.read_text(encoding="utf-8"))["alpha"] if fusion.exists() else {}
        self.examples = _load_examples(examples) if examples else []

    # -- models -------------------------------------------------------------
    def available_models(self) -> list[str]:
        d = model_dir(self.cfg, self.source)
        return [m for m in UI_MODELS if (d / f"{m}.pt").exists()]

    def model(self, name: str):
        with self.lock:
            if name not in self.models:
                path = model_dir(self.cfg, self.source) / f"{name}.pt"
                if not path.exists():
                    raise KeyError(f"no trained model {name!r} in {path.parent}")
                model, tcfg, ckpt = T.load(path)
                self.models[name] = (model, tcfg, ckpt["model_version"])
            return self.models[name]

    # -- libraries ----------------------------------------------------------
    def add_library(self, name: str, roots: list[str]) -> Library:
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
            self.libraries[name] = lib
        threading.Thread(target=self._index, args=(lib,), daemon=True).start()
        return lib

    def _index(self, lib: Library) -> None:
        try:
            lib.folders = [
                {"path": r, "files": sum(1 for p in Path(r).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)}
                for r in lib.roots
            ]
            models = self.available_models()
            if not models:
                raise RuntimeError("no trained models; run `vocalocate-demo run` first")
            for i, name in enumerate(models):
                model, tcfg, version = self.model(name)
                lib.state = "scanning" if i == 0 else "embedding"
                lib.message = (
                    "Reading, cleaning and featurising new or changed files (library pipeline)"
                    if i == 0
                    else f"Embedding with {MODEL_NAMES[name]}"
                )
                t0 = time.time()
                idx = index_library(self.cfg, [Path(r) for r in lib.roots], lib.name, model, tcfg, version, scan=i == 0)
                with self.lock:
                    lib.indexes[name] = idx
                lib.summary[name] = {"embedded": idx.n_embedded, "cached": len(idx.clip_ids) - idx.n_embedded,
                                     "sec": round(time.time() - t0, 1)}
            lib.state, lib.message, lib.finished = "ready", "Ready to search", time.time()
        except Exception as exc:  # report to the UI instead of killing the thread silently
            log.exception("indexing %s failed", lib.name)
            lib.state, lib.message, lib.finished = "error", f"{type(exc).__name__}: {exc}", time.time()

    # -- search -------------------------------------------------------------
    def search(self, library: str, model_name: str, audio: bytes, text: str, k: int) -> dict:
        lib = self.libraries.get(library)
        if lib is None or model_name not in lib.indexes:
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
        scores_text, alpha = None, None
        if text:
            if self.clap is None:
                raise ValueError("text hints need CLAP: start the app with --text")
            key = (library, model_name, text.lower())
            scores_text = self.text_cache.get(key)
            if scores_text is None:
                scores_text = text_scores(self.cfg, library, idx, text, self.clap)
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
        return {
            "library": library,
            "model": model_name,
            "model_label": MODEL_NAMES[model_name],
            "text": text,
            "alpha": alpha,
            "results": hits,
            "searched": len(idx.clip_ids),
            "folders": lib.folders,
            "ms": round(1000 * (time.time() - t0), 1),
            "reused": reused,
        }

    def library_file(self, library: str, i: int) -> Path:
        lib = self.libraries.get(library)
        idx = next(iter(lib.indexes.values()), None) if lib else None
        if idx is None or not 0 <= i < len(idx.paths):
            raise KeyError("unknown file")
        # Every model's index of a library comes from the same ORDER BY path query, so ids agree
        # across models. Only files the library index knows about are ever served.
        return Path(idx.paths[i])

    def status(self) -> dict:
        return {
            "source": self.source,
            "lake": str(self.cfg.lake_root),
            "models": [{"name": m, "label": MODEL_NAMES[m]} for m in self.available_models()],
            "text": self.clap is not None,
            "clap": getattr(self.clap, "version", None),
            "alphas": self.alphas,
            "libraries": [lib.public() for lib in self.libraries.values()],
            "examples": [
                {**{k: e[k] for k in ("id", "name", "category", "hint")}, "target": Path(e["target"]).name}
                for e in self.examples
            ],
            "hints": HINT_SUGGESTIONS,
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
            return json.loads(self.rfile.read(n) or b"{}")

        # -- routes ---------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if not parts:
                    return self._file(UI_DIR / "index.html")
                if parts[0] == "static" and len(parts) == 2 and "/" not in parts[1] and ".." not in parts[1]:
                    return self._file(UI_DIR / parts[1])
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

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/libraries":
                    name = (body.get("name") or "default").strip() or "default"
                    roots = [r for r in body.get("roots", []) if str(r).strip()]
                    if not roots:
                        raise ValueError("add at least one folder")
                    return self._json(state.add_library(name, roots).public(), 202)
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
                    out = state.search(
                        body.get("library") or "default", body.get("model") or "metadata", audio,
                        body.get("text") or "", int(body.get("k") or 10),
                    )
                    return self._json(out)
                return self._error(404, "not found")
            except (KeyError, ValueError) as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                log.error("request failed: %s", traceback.format_exc())
                return self._error(500, f"{type(exc).__name__}: {exc}")

    return Handler


def serve(state: AppState, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create the server (call .serve_forever() on it). Binds to localhost only by default."""
    return ThreadingHTTPServer((host, port), make_handler(state))
