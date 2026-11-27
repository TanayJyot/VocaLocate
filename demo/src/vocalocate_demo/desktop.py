"""The VocaLocate desktop app: per-user data dir, settings, single-instance launcher, autostart.

    vocalocate             start the backend in the background if it is not running, open the app window
    vocalocate --serve     run the backend in this process (headless; what the launcher and autostart start)
    vocalocate --background  start the backend in the background if needed, open no window
    vocalocate --install-shortcut [--desktop]   add VocaLocate to the Start menu (and the Desktop); Windows only
    vocalocate --remove-shortcut [--desktop]    remove those shortcuts

Everything lives in one data dir (`VOCALOCATE_HOME`, else the platform default):

    lake/                      pipeline lake (library index, features, cached embeddings)
    lake/models/installed/     imported models (*.pt + fusion.json)
    settings.json              libraries, preferred model, text hints, autostart
    server.json                {"pid", "port"} of the running backend
    logs/server.log

Nothing here imports Windows-only modules at import time, so it is testable on Linux.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import __version__

log = logging.getLogger(__name__)

APP_NAME = "VocaLocate"
SOURCE = "installed"  # models live in lake/models/installed
DEFAULT_PORT = 8765
DEFAULT_SETTINGS = {"libraries": [], "model": "shared", "text": True, "autostart": False}
PACKAGED_CONFIG = Path(__file__).parent / "configs" / "demo.toml"


# -- data dir and files -------------------------------------------------------------------------
def home() -> Path:
    env = os.environ.get("VOCALOCATE_HOME")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "vocalocate"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_settings(home_dir: Path) -> dict:
    """settings.json merged over the defaults; a missing or corrupt file gives the defaults."""
    raw = _read_json(Path(home_dir) / "settings.json") or {}
    out = {**DEFAULT_SETTINGS, **{k: v for k, v in raw.items() if k in DEFAULT_SETTINGS}}
    libs = []
    for lib in out["libraries"] if isinstance(out["libraries"], list) else []:
        if isinstance(lib, dict) and lib.get("name") and isinstance(lib.get("roots"), list):
            libs.append({"name": str(lib["name"]), "roots": [str(r) for r in lib["roots"]]})
    out["libraries"] = libs
    return out


def save_settings(home_dir: Path, settings: dict) -> None:
    _atomic_write(Path(home_dir) / "settings.json", json.dumps(settings, indent=2))


def write_server_info(home_dir: Path, port: int, pid: int | None = None) -> None:
    _atomic_write(Path(home_dir) / "server.json", json.dumps({"pid": pid or os.getpid(), "port": port}))


def read_server_info(home_dir: Path) -> dict | None:
    return _read_json(Path(home_dir) / "server.json")


def clear_server_info(home_dir: Path) -> None:
    """Remove server.json if it is ours (another instance may have replaced it)."""
    info = read_server_info(home_dir)
    if info and info.get("pid") == os.getpid():
        (Path(home_dir) / "server.json").unlink(missing_ok=True)


# -- single instance ----------------------------------------------------------------------------
def probe(port: int, timeout: float = 1.0) -> bool:
    """True when a VocaLocate backend answers /api/health on this port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            data = json.loads(r.read())
        return bool(data.get("ok")) and data.get("app") == APP_NAME
    except Exception:
        return False


def running_port(home_dir: Path) -> int | None:
    info = read_server_info(home_dir)
    port = info.get("port") if info else None
    if not isinstance(port, int):
        return None
    # A stale server.json (killed backend) points at a dead port: binding it is instant, while an HTTP
    # probe of a closed port on Windows waits out its timeout.
    if port_free(port):
        return None
    return port if probe(port) else None


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


# -- backend ------------------------------------------------------------------------------------
def _pythonw() -> str:
    exe = Path(sys.executable)
    if sys.platform == "win32":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


def backend_command() -> list[str]:
    return [_pythonw(), "-m", "vocalocate_demo.desktop", "--serve"]


def start_backend(home_dir: Path) -> subprocess.Popen:
    """Start `--serve` detached from this process, with no console window, logging to logs/server.log."""
    logs = Path(home_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = open(logs / "server.log", "ab")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT, "close_fds": True,
                    "env": {**os.environ, "VOCALOCATE_HOME": str(home_dir)}}
    cmd = backend_command()
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        # pythonw has no console at all; python.exe gets a hidden one (not DETACHED_PROCESS: its
        # console-less parent would make every console child, e.g. pipeline workers, pop a window).
        if not cmd[0].lower().endswith("pythonw.exe"):
            flags |= subprocess.CREATE_NO_WINDOW
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    finally:
        out.close()


def ensure_backend(home_dir: Path, timeout: float = 180.0) -> int:
    """Port of the running backend, starting one first if needed (a cold start imports torch first, ~30 s)."""
    port = running_port(home_dir)
    if port:
        return port
    proc = start_backend(home_dir)
    deadline = time.time() + timeout
    while time.time() < deadline:
        port = running_port(home_dir)
        if port:
            return port
        if proc.poll() is not None:
            raise RuntimeError(f"the VocaLocate backend exited (code {proc.returncode}); see {home_dir / 'logs' / 'server.log'}")
        time.sleep(0.25)
    raise RuntimeError(f"the VocaLocate backend did not start in {timeout:.0f} s; see {home_dir / 'logs' / 'server.log'}")


# -- app window ---------------------------------------------------------------------------------
def _browser_candidates() -> list[str]:
    found: list[str] = []
    if sys.platform == "win32":
        bases = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        rels = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
        for rel in rels:  # Edge first: it ships with Windows
            for base in bases:
                if base and (Path(base) / rel).exists():
                    found.append(str(Path(base) / rel))
    elif sys.platform == "darwin":
        for app in ("Microsoft Edge", "Google Chrome", "Chromium"):
            p = Path("/Applications") / f"{app}.app" / "Contents" / "MacOS" / app
            if p.exists():
                found.append(str(p))
    for name in ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            found.append(p)
    return list(dict.fromkeys(found))


def open_window(url: str, home_dir: Path) -> str:
    """Open the UI as an app window (Edge/Chrome --app, own profile so mic permission sticks), else a tab."""
    profile = Path(home_dir) / "browser"
    for exe in _browser_candidates():
        try:
            subprocess.Popen(
                [exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
                 "--window-size=1200,860"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
            )
            return exe
        except OSError:
            continue
    import webbrowser

    webbrowser.open(url)
    return "webbrowser"


def launch(home_dir: Path | None = None, window: bool = True) -> int:
    home_dir = Path(home_dir or home())
    try:
        port = ensure_backend(home_dir)
    except RuntimeError as exc:
        _show_error(str(exc))
        return 1
    if window:
        open_window(f"http://127.0.0.1:{port}/", home_dir)
    return 0


def _show_error(message: str) -> None:
    """Report a launcher failure: there is no console under pythonw."""
    log.error(message)
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
        except Exception:
            pass


# -- autostart (Windows Startup folder) ---------------------------------------------------------
AUTOSTART_FILE = "VocaLocate.vbs"  # runs the backend hidden: a .cmd would flash a console window
LEGACY_AUTOSTART_FILES = ("VocaLocate.cmd",)  # written by older versions; removed on disable


def autostart_supported() -> bool:
    return sys.platform == "win32"


def startup_dir() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _autostart_paths(folder: Path | None) -> list[Path]:
    if folder is None:
        if not autostart_supported():
            raise RuntimeError("starting VocaLocate at sign-in is only supported on Windows")
        folder = startup_dir()
    return [Path(folder) / name for name in (AUTOSTART_FILE, *LEGACY_AUTOSTART_FILES)]


def _vbs_quote(text: str) -> str:
    """A VBScript string literal: wrapped in quotes, embedded quotes doubled."""
    return '"' + text.replace('"', '""') + '"'


def autostart_script(home_dir: Path) -> str:
    """VBScript that runs `pythonw -m vocalocate_demo.desktop --serve` with a hidden window (style 0)."""
    exe, *args = backend_command()
    command = f'"{exe}" {" ".join(args)}'  # the shell needs the exe path quoted when it has spaces
    lines = [
        "' Starts the VocaLocate backend hidden at sign-in. Delete this file to turn it off.",
        'Set shell = CreateObject("WScript.Shell")',
        f'shell.Environment("Process")("VOCALOCATE_HOME") = {_vbs_quote(str(home_dir))}',
        f"shell.Run {_vbs_quote(command)}, 0, False",
    ]
    return "\r\n".join(lines) + "\r\n"


def enable_autostart(home_dir: Path, folder: Path | None = None) -> Path:
    path, *legacy = _autostart_paths(folder)
    _atomic_write(path, autostart_script(Path(home_dir)))
    for old in legacy:
        old.unlink(missing_ok=True)
    return path


def disable_autostart(folder: Path | None = None) -> None:
    for path in _autostart_paths(folder):
        path.unlink(missing_ok=True)


def autostart_enabled(folder: Path | None = None) -> bool:
    try:
        return any(p.exists() for p in _autostart_paths(folder))
    except RuntimeError:
        return False


# -- Start-menu / Desktop shortcuts (Windows) ---------------------------------------------------
SHORTCUT_FILE = "VocaLocate.lnk"


def shortcuts_supported() -> bool:
    return sys.platform == "win32"


def programs_dir() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def desktop_dir() -> Path:
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(profile) / "Desktop"


def shortcut_target() -> tuple[str, str]:
    """(exe, arguments) the shortcut runs: the `vocalocate` gui-script if installed, else pythonw -m."""
    exe = Path(sys.executable)
    for folder in (exe.parent, exe.parent / "Scripts"):  # venv: Scripts\python.exe; system install: python.exe + Scripts\
        script = folder / "vocalocate.exe"
        if script.is_file():
            return str(script), ""
    return _pythonw(), "-m vocalocate_demo.desktop"


def _ps_quote(text: str) -> str:
    """A PowerShell single-quoted literal (no expansion; embedded quotes doubled)."""
    return "'" + text.replace("'", "''") + "'"


def shortcut_script(path: Path, exe: str, args: str = "") -> str:
    """PowerShell that writes a .lnk via WScript.Shell (no pywin32 needed)."""
    workdir = str(Path(exe).parent)
    return "; ".join([
        "$ErrorActionPreference = 'Stop'",
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps_quote(str(path)) + ")",
        "$s.TargetPath = " + _ps_quote(exe),
        "$s.Arguments = " + _ps_quote(args),
        "$s.WorkingDirectory = " + _ps_quote(workdir),
        "$s.Description = " + _ps_quote("Search your sound effects by voice"),
        "$s.IconLocation = " + _ps_quote(exe + ",0"),
        "$s.Save()",
    ])


def install_shortcut(folder: Path | None = None) -> Path:
    """Create VocaLocate.lnk in `folder` (default: the user's Start-menu Programs folder)."""
    if not shortcuts_supported():
        raise RuntimeError("shortcuts are only created on Windows; run `vocalocate` to start the app")
    path = Path(folder or programs_dir()) / SHORTCUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    exe, args = shortcut_target()
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         shortcut_script(path, exe, args)],
        capture_output=True, text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if out.returncode != 0 or not path.exists():
        raise RuntimeError(f"could not create {path}: {(out.stderr or out.stdout).strip()}")
    return path


def remove_shortcut(folder: Path | None = None) -> bool:
    """Delete VocaLocate.lnk from `folder` (default: Start-menu Programs). True if one was removed."""
    if folder is None and not shortcuts_supported():
        raise RuntimeError("shortcuts are only created on Windows")
    path = Path(folder or programs_dir()) / SHORTCUT_FILE
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def _shortcut_command(install: bool, desktop: bool) -> int:
    """--install-shortcut / --remove-shortcut. Messages go to stdout and, without a console, a message box."""
    folders = [("Start menu", None)] + ([("Desktop", desktop_dir())] if desktop else [])
    lines = []
    try:
        for label, folder in folders:
            if install:
                lines.append(f"{label} shortcut created: {install_shortcut(folder)}")
            else:
                gone = remove_shortcut(folder)
                lines.append(f"{label} shortcut {'removed' if gone else 'was not installed'}")
    except RuntimeError as exc:
        _show_error(str(exc))
        return 1
    _show_info("\n".join(lines))
    return 0


def _show_info(message: str) -> None:
    if sys.stdout is not None:
        print(message)
    elif sys.platform == "win32":  # the gui-script has no console
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x40)
        except Exception:
            pass


# -- headless server ----------------------------------------------------------------------------
def build_state(home_dir: Path, workers: int | None = None, text: bool | None = None):
    """The AppState for a data dir: packaged pipeline config, lake under the data dir, source 'installed'."""
    from vocalocate_pipeline.config import load_config

    from .app import AppState

    cfg = load_config(PACKAGED_CONFIG if PACKAGED_CONFIG.exists() else None, lake_root=Path(home_dir) / "lake",
                      workers=workers)
    return AppState(cfg, SOURCE, home=Path(home_dir), text_auto=text)


def serve_main(home_dir: Path | None = None, port: int | None = None, host: str = "127.0.0.1") -> int:
    from .app import serve_free

    home_dir = Path(home_dir or home())
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "logs").mkdir(exist_ok=True)
    log_file = open(home_dir / "logs" / "server.log", "a", encoding="utf-8", buffering=1)
    # pythonw has no stdout/stderr; stray prints (tqdm, warnings) must not crash the server.
    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    existing = running_port(home_dir)
    if existing:
        log.info("VocaLocate is already running on port %d", existing)
        return 0
    state = build_state(home_dir)
    server = serve_free(state, host, port or DEFAULT_PORT)
    bound = server.server_address[1]
    write_server_info(home_dir, bound)
    log.info("VocaLocate %s serving %s on http://%s:%d", __version__, home_dir, host, bound)
    state.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        clear_server_info(home_dir)
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vocalocate", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true", help="run the backend in this process (headless)")
    mode.add_argument("--background", action="store_true", help="start the backend if needed, open no window")
    mode.add_argument("--install-shortcut", action="store_true", help="add VocaLocate to the Start menu (Windows)")
    mode.add_argument("--remove-shortcut", action="store_true", help="remove the Start-menu shortcut (Windows)")
    p.add_argument("--desktop", action="store_true", help="with --install/--remove-shortcut: also the Desktop shortcut")
    p.add_argument("--home", type=Path, help="data dir (default: VOCALOCATE_HOME or the per-user app data dir)")
    p.add_argument("--port", type=int, help=f"--serve: first port to try (default {DEFAULT_PORT})")
    args = p.parse_args(argv)
    if args.home:
        os.environ["VOCALOCATE_HOME"] = str(args.home)
    if args.serve:
        return serve_main(args.home, args.port)
    if args.install_shortcut or args.remove_shortcut:
        return _shortcut_command(args.install_shortcut, args.desktop)
    return launch(args.home, window=not args.background)


if __name__ == "__main__":
    raise SystemExit(main())
