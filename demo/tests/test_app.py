"""The web app's API, against a tiny demo run: index, search (example and uploaded audio), text hints, files."""

import base64
import json
import threading
import time
import urllib.error
import urllib.request

from vocalocate_pipeline.config import PipelineConfig, SplitConfig
from vocalocate_demo.app import AppState, serve
from vocalocate_demo.demo_flow import run_demo
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


def test_app_api_end_to_end(tmp_path, monkeypatch):
    import vocalocate_demo.demo_flow as flow

    tiny = dict(recordings_per_category=4, distractors_per_category=1, imitations_per_reference=3,
                categories=["metal_impact", "whoosh", "alarm_beep", "engine"])
    monkeypatch.setattr(flow, "generate", lambda out, **_: generate(out, **tiny))
    cfg = PipelineConfig(lake_root=tmp_path / "lake", workers=1, splits=SplitConfig(train=0.5, val=0.25))
    clap = ClapScorer(backend="fake")
    run_demo(tmp_path, cfg, TrainConfig(epochs=1, ft_epochs=1, frames=128, batch_size=8, device="cpu"),
             MiningConfig(), n_queries=2, clap=clap)

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
    finally:
        server.shutdown()
