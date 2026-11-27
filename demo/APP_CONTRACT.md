# VocaLocate desktop app: contract between frontend (`ui/*`) and backend (`app.py` + launcher)

Working agreement for the redesign. Backend owns `src/vocalocate_demo/app.py`, `cli.py`, a new
`src/vocalocate_demo/desktop.py` (launcher, data dir, autostart), `pyproject.toml`, tests. Frontend
owns `src/vocalocate_demo/ui/*` only. Neither edits the other's files.

## Product flow (what "spontaneous" means)

1. User installs once (`pip install` / installer) and launches **VocaLocate** (Start-menu / desktop
   shortcut, or `vocalocate` gui-script). No terminal, no flags. The launcher starts the backend in the
   background if it is not already running (single instance), waits for `/api/health`, then opens an
   app window (Edge/Chrome `--app=` mode, falling back to the default browser). The server binds and
   answers `/`, `/static/*`, `/api/health` and `/api/status` within ~1-2 s; torch and the pipeline are
   imported by the background boot thread, and the window's warm-up screen shows `boot.message`.
   Shortcuts: `vocalocate --install-shortcut [--desktop]` / `--remove-shortcut [--desktop]` write or
   delete `VocaLocate.lnk` in `%APPDATA%\Microsoft\Windows\Start Menu\Programs` (and the Desktop)
   via PowerShell `WScript.Shell`, targeting `Scriptsocalocate.exe` (else `pythonw -m
   vocalocate_demo.desktop`). Windows only; elsewhere they exit 1 with a message.
2. Optional: "Start VocaLocate when I sign in" toggle (writes `VocaLocate.vbs` to the Windows Startup
   folder; it runs `pythonw -m vocalocate_demo.desktop --serve` hidden; legacy `VocaLocate.cmd` is
   removed). Off by default; toggled from the UI.
3. First run, nothing set up: the window shows a one-screen onboarding. If no models are installed it
   offers "Import models…" (native folder picker → a folder holding `*.pt` + optional `fusion.json`).
   Then "Add your sound folder" (native folder picker). Indexing starts immediately and the search
   screen is usable as soon as the first model's index is ready.
4. Every later launch: libraries are restored from disk instantly (cached embeddings), a background
   incremental rescan picks up new/changed files. The user lands directly on search.
5. Search: press Space (or click the big record button) → speak → release/stop → results appear in
   place automatically (no Search button needed). Typing a hint re-ranks with ~300 ms debounce.
   Drag a result into an editor; click to preview.

## Per-user data dir

`VOCALOCATE_HOME` env var, else `%LOCALAPPDATA%\VocaLocate` (Windows) / `~/Library/Application Support/VocaLocate`
(macOS) / `$XDG_DATA_HOME/vocalocate` or `~/.local/share/vocalocate`. Contains:

```
lake/                       pipeline lake (library index, features, cached embeddings)
lake/models/<source>/*.pt   installed models (+ fusion.json); source default "installed"
settings.json               {"libraries": [{"name", "roots"}], "model": "metadata", "text": true, "autostart": false}
server.json                 {"pid", "port", "token"?} written by the running backend
logs/server.log
```

Pipeline config: bundled with the package (do not rely on `parents[2]/configs/demo.toml`), so an
installed copy uses the same audio/feature settings the checkpoints were trained with.

## HTTP API (localhost only; port 8765 default, next free port if busy, recorded in server.json)

Existing routes keep their shapes (tests depend on them):

- `GET  /` → index.html; `GET /static/<file>` → flat files in ui/ (no subfolders).
- `GET  /api/status` → existing fields **plus**:
  `ready` (bool: at least one library ready for at least one model),
  `boot` ({"state": "starting"|"loading_models"|"ready", "message"}; "starting" covers the engine
  import, even with no models installed),
  `text_state` ("off"|"loading"|"ready"|"error"), `text_error` (string),
  `needs` (list of onboarding steps still missing: subset of ["models", "library"]),
  `settings` ({"model", "text", "autostart", "autostart_supported"}),
  `home` (data dir path or null), `version`.
  Each library in `libraries` also gets `progress` ({"done", "total"} while embedding new clips, else
  null; the scan phase has no per-file progress: `state` "scanning" + `folders[].files` counts) and
  `saved` (bool).
- `GET  /api/libraries/<name>` → as before.
- `GET  /api/audio/<library>/<id>` → as before (only indexed files).
- `GET  /api/example/<id>` → as before.
- `POST /api/libraries` {"name", "roots"} → 202, as before; now also persisted to settings.json.
- `POST /api/search` {"library", "model", "audio"(b64 wav) | "example", "text", "k"} → as before.
  `model` may be omitted → settings.model, falling back to the first available. A `text` hint with no
  CLAP loaded still returns 200 (voice-only ranking) plus `text_ignored` (reason string).
  503 `{"error": "VocaLocate is still warming up…"}` while the boot thread is importing the engine.

New:

- `GET  /api/health` → `{"ok": true, "version", "app": "VocaLocate"}` (launcher readiness probe;
  answers before torch is loaded).
- `POST /api/pick-folder` {"title"?} → `{"path": "C:\\..."} | {"path": null}` (native dialog on the
  server's machine, topmost; null if cancelled or no GUI available → UI falls back to a text field).
- `DELETE /api/libraries/<name>` → 200; forgets it (settings + in-memory; cached files may stay).
- `POST /api/libraries/<name>/rescan` → 202.
- `POST /api/models/import` {"path"} → 200 `{"models": [...]}`; copies `*.pt` (+fusion.json) into the
  data dir and loads them (waits for the engine import if boot is still running). 400 with a clear
  message if the folder has no models.
- `POST /api/settings` {"model"?, "autostart"?, "text"?} → 200 new settings.
- `POST /api/reveal` {"library", "id"} → opens the file's folder with the file selected (Explorer
  `/select,`), only for indexed files.

Security: every request, GETs included, must have `Host` `127.0.0.1:<port>`, `localhost:<port>` or
`[::1]:<port>` (blocks DNS rebinding). Every non-GET request must also carry `Content-Type:
application/json` and, if present, an `Origin` equal to the server's own origin. Otherwise 403.
(Blocks drive-by POSTs from web pages to an always-on local server.)

Errors: `{"error": "<human sentence that names the fix>"}` with 4xx/5xx.
