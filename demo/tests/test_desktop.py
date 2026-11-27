"""The desktop app: data dir, settings, restore-on-boot, the new routes, the security check, autostart."""

import base64
import http.client
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocalocate_pipeline.config import PipelineConfig
from vocalocate_demo import app as app_mod
from vocalocate_demo import desktop
from vocalocate_demo import train as T
from vocalocate_demo.app import AppState, serve, serve_free
from vocalocate_demo.model import DualEncoder


# -- fixtures -------------------------------------------------------------------------------------
@pytest.fixture
def models_dir(tmp_path):
    """Two tiny untrained checkpoints in the format train.save writes, plus fusion.json."""
    out = tmp_path / "trained"
    tcfg = T.TrainConfig(frames=64, device="cpu")
    for i, name in enumerate(("metadata", "retrieval")):
        T.save(DualEncoder(dim=16, width=4, frames=64), out / f"{name}.pt", tcfg, {"model_version": f"test-{name}-{i}"})
    (out / "fusion.json").write_text(json.dumps({"alpha": {"metadata": 0.7, "retrieval": 0.6}}), encoding="utf-8")
    return out


@pytest.fixture
def sounds(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "My Sounds"
    for i in range(10):
        sr, n = 32_000, 32_000 // 2 + 800 * i
        t = np.arange(n) / sr
        wave = 0.3 * np.sin(2 * np.pi * (200 + 90 * i) * t) * np.exp(-t * (1 + i)) + 0.02 * rng.standard_normal(n)
        sub = root / ("impacts" if i % 2 else "beeps")
        sub.mkdir(parents=True, exist_ok=True)
        sf.write(sub / f"clip_{i:02d}.wav", wave.astype(np.float32), sr)
    return root


def _state(home: Path, **kw) -> AppState:
    cfg = PipelineConfig(lake_root=home / "lake", workers=1)
    kw.setdefault("text_auto", False)  # never load (or download) CLAP in tests
    return AppState(cfg, desktop.SOURCE, home=home, autostart_dir=home / "fake_startup", **kw)


def _start(state):
    server = serve(state, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _call(base, path, body=None, method=None, headers=None):
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_ready(base, name, timeout=180):
    lib = None
    for _ in range(int(timeout * 10)):
        lib = _call(base, "/api/libraries/" + urllib.parse.quote(name))[1]
        if lib["state"] in ("ready", "error"):
            break
        time.sleep(0.1)
    assert lib["state"] == "ready", lib["message"]
    return lib


# -- data dir and files -----------------------------------------------------------------------------
def test_home_env_override_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("VOCALOCATE_HOME", str(tmp_path / "vh"))
    assert desktop.home() == tmp_path / "vh"
    monkeypatch.delenv("VOCALOCATE_HOME")
    assert desktop.home().name.lower() == "vocalocate"


def test_settings_round_trip_and_corrupt_file(tmp_path):
    assert desktop.load_settings(tmp_path) == desktop.DEFAULT_SETTINGS
    s = {**desktop.DEFAULT_SETTINGS, "libraries": [{"name": "sfx", "roots": ["/a"]}], "model": "retrieval"}
    desktop.save_settings(tmp_path, s)
    assert desktop.load_settings(tmp_path) == s
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]  # atomic write leaves no temp file
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    assert desktop.load_settings(tmp_path) == desktop.DEFAULT_SETTINGS


def test_packaged_config_matches_source_checkout():
    from vocalocate_demo.cli import _default_config, _parser

    source = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"
    assert desktop.PACKAGED_CONFIG.read_bytes() == source.read_bytes()
    args = _parser().parse_args(["generate", "--out", "x"])
    assert _default_config(args) == desktop.PACKAGED_CONFIG.resolve()


def test_autostart_with_fake_startup_dir(tmp_path):
    folder = tmp_path / "Startup"
    folder.mkdir()
    assert not desktop.autostart_enabled(folder)
    path = desktop.enable_autostart(tmp_path / "home", folder)
    text = path.read_text(encoding="utf-8")
    assert path.name == "VocaLocate.vbs" and desktop.autostart_enabled(folder)
    assert "--serve" in text and "CreateObject(\"WScript.Shell\")" in text and str(tmp_path / "home") in text
    assert text.rstrip().endswith(", 0, False")  # hidden window, do not wait
    desktop.disable_autostart(folder)
    assert not desktop.autostart_enabled(folder) and not path.exists()
    if sys.platform != "win32":
        with pytest.raises(RuntimeError):
            desktop.enable_autostart(tmp_path / "home")


def test_autostart_script_quotes_paths_with_spaces(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "backend_command",
                        lambda: [r"C:\Program Files\Python 311\pythonw.exe", "-m", "vocalocate_demo.desktop", "--serve"])
    text = desktop.autostart_script(tmp_path / "My Home")
    # VBS doubles the quotes that wrap the exe path inside the string literal.
    expected = 'shell.Run "' + r'""C:\Program Files\Python 311\pythonw.exe"" -m vocalocate_demo.desktop --serve' + '", 0, False'
    assert expected in text
    assert '""' + str(tmp_path / "My Home") not in text and str(tmp_path / "My Home") in text
    assert desktop._vbs_quote('a "b" c') == '"a ""b"" c"'


def test_autostart_replaces_and_removes_legacy_cmd(tmp_path):
    folder = tmp_path / "Startup"
    folder.mkdir()
    legacy = folder / "VocaLocate.cmd"
    legacy.write_text("@echo off", encoding="utf-8")
    assert desktop.autostart_enabled(folder)  # an old .cmd still counts as enabled
    path = desktop.enable_autostart(tmp_path / "home", folder)
    assert path.exists() and not legacy.exists()
    legacy.write_text("@echo off", encoding="utf-8")
    desktop.disable_autostart(folder)
    assert not legacy.exists() and not path.exists() and not desktop.autostart_enabled(folder)


def test_server_json_probe_and_free_port(tmp_path):
    state = _state(tmp_path)
    server, base = _start(state)
    try:
        port = server.server_address[1]
        assert desktop.running_port(tmp_path) is None
        desktop.write_server_info(tmp_path, port)
        assert desktop.running_port(tmp_path) == port
        assert _call(base, "/api/health")[1]["ok"] is True
        # The recorded port is busy, so the next server takes the next free one.
        other = serve_free(state, port=port)
        try:
            assert other.server_address[1] > port
        finally:
            other.server_close()
        desktop.clear_server_info(tmp_path)
        assert desktop.read_server_info(tmp_path) is None
    finally:
        server.shutdown()
        server.server_close()
    assert not desktop.probe(port, timeout=0.5)


# -- API --------------------------------------------------------------------------------------------
def test_security_rejects_cross_site_requests(tmp_path):
    server, base = _start(_state(tmp_path))
    port = server.server_address[1]
    try:
        body = {"model": "retrieval"}
        assert _call(base, "/api/settings", body)[0] == 200
        assert _call(base, "/api/settings", body, headers={"Content-Type": "application/json; charset=utf-8"})[0] == 200
        assert _call(base, "/api/settings", body, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
        assert _call(base, "/api/settings", body, headers={"Content-Type": "text/plain"})[0] == 403
        assert _call(base, "/api/settings", body, headers={"Origin": "https://evil.example"})[0] == 403
        assert _call(base, "/api/settings", body, headers={"Origin": "null"})[0] == 403
        assert _call(base, "/api/libraries/x", method="DELETE", headers={"Content-Type": "text/plain"})[0] == 403
        for method, path in (("POST", "/api/settings"), ("GET", "/api/status")):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request(method, path, body=b"{}", headers={"Host": f"evil.example:{port}",
                                                             "Content-Type": "application/json"})
            assert conn.getresponse().status == 403
            conn.close()
    finally:
        server.shutdown()


def test_onboarding_import_index_search_and_manage(tmp_path, models_dir, sounds):
    home = tmp_path / "home"
    state = _state(home)
    revealed = []
    state.reveal_file = revealed.append
    state.pick_folder = lambda title: str(sounds)
    server, base = _start(state)
    try:
        st = _call(base, "/api/status")[1]
        assert st["needs"] == ["models", "library"] and st["ready"] is False and st["home"] == str(home)
        assert st["settings"]["autostart_supported"] and st["settings"]["autostart"] is False
        assert st["text_state"] == "off"

        empty = tmp_path / "empty"
        empty.mkdir()
        status, err = _call(base, "/api/models/import", {"path": str(empty)})
        assert status == 400 and "metadata.pt" in err["error"]
        status, res = _call(base, "/api/models/import", {"path": str(models_dir)})
        assert status == 200 and [m["name"] for m in res["models"]] == ["metadata", "retrieval"]
        assert (home / "lake" / "models" / "installed" / "fusion.json").exists()
        assert _call(base, "/api/status")[1]["alphas"]["metadata"] == 0.7

        assert _call(base, "/api/pick-folder", {"title": "Sounds"})[1] == {"path": str(sounds)}
        status, lib = _call(base, "/api/libraries", {"name": "My Sounds", "roots": [str(sounds)]})
        assert status == 202 and lib["saved"] is True
        assert desktop.load_settings(home)["libraries"] == [{"name": "My Sounds", "roots": [str(sounds.resolve())]}]
        lib = _wait_ready(base, "My Sounds")
        assert lib["files"] == 10 and lib["models_ready"] == ["metadata", "retrieval"] and lib["progress"] is None
        st = _call(base, "/api/status")[1]
        assert st["needs"] == [] and st["ready"] is True

        status, s = _call(base, "/api/settings", {"model": "retrieval", "autostart": True})
        assert status == 200 and s["model"] == "retrieval" and s["autostart"] is True
        assert (home / "fake_startup" / desktop.AUTOSTART_FILE).exists()
        assert _call(base, "/api/settings", {"autostart": False})[1]["autostart"] is False
        assert _call(base, "/api/settings", {"model": "nope"})[0] == 400

        wav = base64.b64encode(next(sounds.rglob("*.wav")).read_bytes()).decode()
        status, res = _call(base, "/api/search", {"library": "My Sounds", "audio": wav, "k": 3})
        assert status == 200 and res["model"] == "retrieval" and len(res["results"]) == 3
        # No CLAP loaded: the hint is ignored, the search still answers.
        status, res = _call(base, "/api/search", {"library": "My Sounds", "audio": wav, "text": "metallic"})
        assert status == 200 and res["text_ignored"] and "query embedding" in res["reused"]

        hit = res["results"][0]
        status, out = _call(base, "/api/reveal", {"library": "My Sounds", "id": hit["id"]})
        assert status == 200 and revealed == [Path(hit["path"])]
        assert _call(base, "/api/reveal", {"library": "My Sounds", "id": 999})[0] == 404

        assert _call(base, "/api/libraries/My%20Sounds/rescan", {})[0] == 202
        lib = _wait_ready(base, "My Sounds")
        assert lib["summary"]["metadata"]["embedded"] == 0  # nothing changed: all cached
        assert _call(base, "/api/libraries/nope/rescan", {})[0] == 404

        assert _call(base, "/api/libraries/My%20Sounds", method="DELETE")[0] == 200
        assert _call(base, "/api/libraries/My%20Sounds", method="DELETE")[0] == 404
        assert desktop.load_settings(home)["libraries"] == []
        assert _call(base, "/api/status")[1]["needs"] == ["library"]
    finally:
        server.shutdown()


def test_restore_on_boot_searches_before_rescan(tmp_path, models_dir, sounds, monkeypatch):
    home = tmp_path / "home"
    first = _state(home)
    first.import_models(str(models_dir))
    first.add_library("My Sounds", [str(sounds)])
    server, base = _start(first)
    try:
        _wait_ready(base, "My Sounds")
    finally:
        server.shutdown()
        server.server_close()

    # A new process on the same data dir: the rescan is held, the cached index must already answer.
    release = threading.Event()
    real = app_mod.index_library

    def held(*a, **kw):
        release.wait(60)
        return real(*a, **kw)

    monkeypatch.setattr(app_mod, "index_library", held)
    second = _state(home)
    assert second.status()["libraries"][0]["saved"] is True
    second.start()
    server, base = _start(second)
    try:
        for _ in range(600):
            st = _call(base, "/api/status")[1]
            if st["ready"] and st["boot"]["state"] == "ready":
                break
            time.sleep(0.1)
        assert st["ready"], st
        lib = st["libraries"][0]
        assert lib["models_ready"] == ["metadata", "retrieval"] and lib["files"] == 10
        wav = base64.b64encode(next(sounds.rglob("*.wav")).read_bytes()).decode()
        status, res = _call(base, "/api/search", {"audio": wav, "k": 2})
        assert status == 200 and res["library"] == "My Sounds" and res["model"] == "metadata"
        assert _call(base, f"/api/audio/My%20Sounds/{res['results'][0]['id']}")[1][:4] == b"RIFF"
        release.set()
        _wait_ready(base, "My Sounds")
    finally:
        release.set()
        server.shutdown()


def test_pick_folder_without_gui_returns_null(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert app_mod._FolderPicker()("x") is None


def test_port_probe_ignores_other_servers():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert not desktop.probe(s.getsockname()[1], timeout=0.3)


def test_missing_folder_at_boot_keeps_the_cached_index(tmp_path, models_dir, sounds):
    home = tmp_path / "home"
    first = _state(home)
    first.import_models(str(models_dir))
    first.add_library("My Sounds", [str(sounds)])
    server, base = _start(first)
    try:
        _wait_ready(base, "My Sounds")
    finally:
        server.shutdown()
        server.server_close()
    sounds.rename(sounds.with_name("unplugged"))  # e.g. an external drive that is not connected

    second = _state(home)
    second.start()
    for _ in range(600):
        lib = second.libraries["My Sounds"]
        if lib.state == "error":
            break
        time.sleep(0.1)
    assert lib.state == "error" and "folder not found" in lib.message
    assert sorted(lib.indexes) == ["metadata", "retrieval"] and second.status()["ready"]
    third = _state(home)  # the embeddings part was not rewritten empty
    third.start()
    for _ in range(600):
        if third.libraries["My Sounds"].indexes:
            break
        time.sleep(0.1)
    assert len(next(iter(third.libraries["My Sounds"].indexes.values())).clip_ids) == 10


# -- instant window -----------------------------------------------------------------------------------
def test_app_imports_without_torch():
    """The server must bind before torch loads: importing app/desktop may not pull in the engine."""
    import subprocess

    code = ("import sys, vocalocate_demo.app, vocalocate_demo.desktop; "
            "print([m for m in ('torch', 'scipy', 'vocalocate_demo.search', 'vocalocate_demo.train') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_search_is_503_while_the_engine_is_warming(tmp_path, monkeypatch):
    loaded, lock = threading.Event(), threading.Lock()
    monkeypatch.setattr(app_mod, "_engine_loaded", loaded)
    monkeypatch.setattr(app_mod, "_engine_lock", lock)
    lock.acquire()  # the boot thread is mid-import
    server, base = _start(_state(tmp_path))
    try:
        assert _call(base, "/api/health")[0] == 200
        assert _call(base, "/api/status")[0] == 200
        status, err = _call(base, "/api/search", {"audio": base64.b64encode(b"RIFF").decode()})
        assert status == 503 and "warming up" in err["error"]
    finally:
        lock.release()
        server.shutdown()


def test_stale_server_json_is_ignored_without_probing(tmp_path, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    desktop.write_server_info(tmp_path, dead)
    monkeypatch.setattr(desktop, "probe", lambda *a, **k: pytest.fail("probed a dead port"))
    assert desktop.running_port(tmp_path) is None


# -- Start-menu shortcut ------------------------------------------------------------------------------
def test_shortcut_target_and_script_quoting(tmp_path, monkeypatch):
    exe = tmp_path / "env" / "python.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(exe))
    target, args = desktop.shortcut_target()
    assert args == "-m vocalocate_demo.desktop" and Path(target).parent == exe.parent
    (exe.parent / "Scripts").mkdir()
    (exe.parent / "Scripts" / "vocalocate.exe").write_bytes(b"")
    assert desktop.shortcut_target() == (str(exe.parent / "Scripts" / "vocalocate.exe"), "")
    script = desktop.shortcut_script(tmp_path / "it's here" / "VocaLocate.lnk", r"C:\Program Files\vocalocate.exe")
    assert "CreateShortcut('" + str(tmp_path / "it''s here" / "VocaLocate.lnk") + "')" in script
    assert r"$s.TargetPath = 'C:\Program Files\vocalocate.exe'" in script and "$s.Save()" in script


def test_shortcut_is_windows_only(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="only"):
        desktop.install_shortcut(tmp_path)
    assert desktop.main(["--install-shortcut"]) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="creates a real .lnk through WScript.Shell")
def test_install_and_remove_shortcut_in_a_fake_programs_dir(tmp_path):
    import subprocess

    folder = tmp_path / "Start Menu" / "Programs"
    path = desktop.install_shortcut(folder)
    assert path == folder / desktop.SHORTCUT_FILE and path.exists()
    exe, args = desktop.shortcut_target()
    read = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + desktop._ps_quote(str(path)) + "); "
            "$s.TargetPath + '|' + $s.Arguments")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", read],
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.strip().lower() == f"{exe}|{args}".lower()
    assert desktop.remove_shortcut(folder) is True and not path.exists()
    assert desktop.remove_shortcut(folder) is False
