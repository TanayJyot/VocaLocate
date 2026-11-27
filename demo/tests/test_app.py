"""The web app's API, against a tiny demo run: index, search (example and uploaded audio), text hints, files."""

import base64
import http.client
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vocalocate_pipeline.config import PipelineConfig, SplitConfig
from vocalocate_demo import app
from vocalocate_demo.app import AppState, Library, serve
from vocalocate_demo.demo_flow import run_demo
from vocalocate_demo.experiment import model_dir
from vocalocate_demo.mining import MiningConfig
from vocalocate_demo.synth import generate
from vocalocate_demo.text import ClapScorer
from vocalocate_demo.train import TrainConfig


def _call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _tiny_demo(tmp_path, monkeypatch):
    import vocalocate_demo.demo_flow as flow

    tiny = dict(recordings_per_category=4, distractors_per_category=1, imitations_per_reference=3,
                categories=["metal_impact", "whoosh", "alarm_beep", "engine"])
    monkeypatch.setattr(flow, "generate", lambda out, **_: generate(out, **tiny))
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1, splits=SplitConfig(train=0.5, val=0.25))
    clap = ClapScorer(backend="fake")
    run_demo(tmp_path, cfg, TrainConfig(epochs=1, ft_epochs=1, frames=128, batch_size=8, device="cpu"),
             MiningConfig(), n_queries=2, clap=clap)
    return cfg, clap


def _wait_ready(base, name):
    for _ in range(300):
        lib = _call(base, f"/api/libraries/{name}")[1]
        if lib["state"] in ("ready", "error"):
            return lib
        time.sleep(0.1)
    return lib


def test_app_serves_a_recipe_folder_and_defaults_to_its_shared_model(tmp_path, monkeypatch):
    """--models points the app at a recipe run's folder; its shared model comes first and is the default."""
    cfg, clap = _tiny_demo(tmp_path, monkeypatch)
    recipe = tmp_path / "recipe-492"
    recipe.mkdir()
    # Stand-ins for a recipe run's checkpoints: same format (train.save), different names.
    shutil.copy(model_dir(cfg, "sfx_meta") / "baseline.pt", recipe / "shared.pt")
    shutil.copy(model_dir(cfg, "sfx_meta") / "metadata.pt", recipe / "metadata.pt")

    state = AppState(cfg, "sfx_meta", clap=clap, examples=tmp_path / "queries", models=recipe)
    assert state.available_models() == ["shared", "metadata"] and state.default_model() == "shared"
    server = serve(state, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        st = _call(base, "/api/status")[1]
        assert [m["name"] for m in st["models"]] == ["shared", "metadata"]
        assert "recommended" in st["models"][0]["label"]
        _call(base, "/api/libraries", {"name": "demo", "roots": [str(tmp_path / "library")]})
        lib = _wait_ready(base, "demo")
        assert lib["state"] == "ready" and sorted(lib["models_ready"]) == ["metadata", "shared"]
        status, res = _call(base, "/api/search", {"library": "demo", "example": 0, "k": 3})  # no model given
        assert status == 200 and res["model"] == "shared" and len(res["results"]) == 3
    finally:
        server.shutdown()


def test_app_api_end_to_end(tmp_path, monkeypatch):
    cfg, clap = _tiny_demo(tmp_path, monkeypatch)

    state = AppState(cfg, "sfx_meta", clap=clap, examples=tmp_path / "queries")
    server = serve(state, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, page = _call(base, "/")
        assert status == 200 and b"VocaLocate" in page
        status, st = _call(base, "/api/status")
        assert status == 200 and {m["name"] for m in st["models"]} == {"metadata", "retrieval", "baseline", "random"}
        assert st["text"] and len(st["examples"]) == 2
        assert _call(base, "/static/app.js")[0] == 200
        assert _call(base, "/static/missing.js") == (404, {"error": "file not found"})
        status, example = _call(base, "/api/example/0")
        assert status == 200 and example[:4] == b"RIFF"
        assert _call(base, "/api/example/99") == (404, {"error": "unknown example"})
        assert _call(base, "/api/search", {"example": 99}) == (400, {"error": "unknown example"})
        assert _call(base, "/api/libraries", {"roots": []}) == (400, {"error": "add at least one folder"})
        assert _call(base, "/not-a-route", {}) == (404, {"error": "not found"})

        oversized = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            oversized.request("POST", "/api/search", headers={"Content-Length": str(50 * 1024 * 1024 + 1),
                                                                "Content-Type": "application/json"})
            response = oversized.getresponse()
            assert (response.status, json.loads(response.read())) == (400, {"error": "request too large"})
        finally:
            oversized.close()

        assert _call(base, "/api/libraries", {"name": "x", "roots": [str(tmp_path / "nope")]})[0] == 400
        status, lib = _call(base, "/api/libraries", {"name": "demo", "roots": [str(tmp_path / "library")]})
        assert status == 202 and lib["roots"] == [str((tmp_path / "library").resolve())]
        for _ in range(300):
            lib = _call(base, "/api/libraries/demo")[1]
            if lib["state"] in ("ready", "error"):
                break
            time.sleep(0.1)
        assert lib["state"] == "ready", lib["message"]
        assert lib["folders"][0]["files"] == lib["files"] > 0 and len(lib["models_ready"]) == 4

        status, res = _call(base, "/api/search", {"library": "demo", "model": "metadata", "example": 0, "k": 5})
        assert status == 200 and len(res["results"]) == 5 and res["reused"] == []
        scores = [h["score"] for h in res["results"]]
        assert scores == sorted(scores, reverse=True)

        # Refining with a hint reuses the query embedding; the same hint again reuses text scores too.
        status, res2 = _call(base, "/api/search", {"library": "demo", "model": "metadata", "example": 0, "text": "metallic"})
        assert status == 200 and "query embedding" in res2["reused"] and "text_score" in res2["results"][0]
        res3 = _call(base, "/api/search", {"library": "demo", "model": "metadata", "example": 0, "text": "metallic"})[1]
        assert "text scores" in res3["reused"]

        state.clap = None  # the hint is ignored (and said so), never a failed search
        status, res_off = _call(base, "/api/search", {"library": "demo", "example": 0, "text": "metallic"})
        assert status == 200 and "text_score" not in res_off["results"][0]
        assert "start the app with --text" in res_off["text_ignored"]
        state.clap = clap

        # Uploaded/recorded audio arrives as base64 WAV.
        wav = (tmp_path / "queries").glob("*.wav").__next__().read_bytes()
        status, res4 = _call(base, "/api/search", {"library": "demo", "model": "retrieval",
                                                   "audio": base64.b64encode(wav).decode()})
        assert status == 200 and res4["results"]

        # Files: only ones in the index are served.
        status, data = _call(base, f"/api/audio/demo/{res['results'][0]['id']}")
        assert status == 200 and data[:4] == b"RIFF"
        assert _call(base, "/api/audio/demo/99999")[0] == 404
        assert _call(base, "/static/../app.py")[0] == 404
        assert _call(base, "/api/search", {"library": "demo", "model": "metadata"})[0] == 400
        monkeypatch.setattr(state, "search", Mock(side_effect=OSError("audio unavailable")))
        assert _call(base, "/api/search", {"example": 0}) == (500, {"error": "OSError: audio unavailable"})
    finally:
        server.shutdown()
        server.server_close()


def test_app_reports_missing_model_and_unindexed_library(tmp_path, caplog):
    state = AppState(PipelineConfig(lake_root=tmp_path / "lake"), "sfx_meta", examples=tmp_path / "queries")
    assert state.examples == []
    assert state.available_models() == []
    with pytest.raises(KeyError, match="no trained model in"):
        state.default_model()
    with pytest.raises(KeyError, match="no trained model 'baseline'"):
        state.model("baseline")
    with pytest.raises(ValueError, match="library 'demo' is not indexed yet"):
        state.search("demo", "metadata", b"", "", 1)

    library = Library(name="demo", roots=[str(tmp_path)])
    state.libraries["demo"] = library  # _index skips libraries that were removed or replaced
    with pytest.raises(ValueError, match="is not indexed for model 'metadata'"):
        state.search("demo", "metadata", b"", "", 1)
    state._index(library)
    assert library.state == "error"
    assert "no models installed yet" in library.message
    assert library.finished is not None
    assert "indexing demo failed" in caplog.text


def test_reindex_keeps_previous_results_visible_until_background_work_starts(tmp_path, monkeypatch):
    state = AppState(PipelineConfig(lake_root=tmp_path / "lake"), "sfx_meta")
    previous = Library(name="demo", roots=[str(tmp_path.resolve())])
    previous.indexes["baseline"] = Mock()
    state.libraries["demo"] = previous
    thread = Mock()
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=thread))

    current = state.add_library("demo", [str(tmp_path)])
    assert current.indexes == previous.indexes
    assert current.indexes is not previous.indexes
    thread.assert_called_once_with(target=state._index, args=(current,), name="index-demo", daemon=True)
    thread.return_value.start.assert_called_once_with()
