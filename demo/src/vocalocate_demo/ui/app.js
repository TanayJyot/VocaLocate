/* VocaLocate local UI. Plain JS (classic script), no build step, no network beyond this origin.
   Coded against demo/APP_CONTRACT.md; degrades to the older server where routes/fields are missing. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const MAX_REC_SEC = 15;
const PREVIEW_SEC = 5;
const HINT_DEBOUNCE_MS = 300;
const POLL_BUSY_MS = 1000;
const POLL_IDLE_MS = 6000;
const HEALTH_MS = 1500;
const K = 30;

const S = {
  status: null, statusKey: "",
  online: true, everOnline: false,
  lib: store("vl.lib") || "",
  model: store("vl.model") || "",
  hint: "",
  query: null,          // {kind:"audio"|"example", b64?, exampleId?, url, label, samples, seconds}
  waiting: false,       // a query is held until a model is ready
  inflight: false, pending: false,
  results: null, resultsQuery: null,
  sel: -1,
  rec: null, recStarting: false,
  player: null, playing: null, playTok: 0, playTimer: null,
  kbd: false,           // last interaction was keyboard (vs pointer)
  spaceHandled: false,
  missing: new Set(),   // contract routes this server does not have
  pollT: null, hintT: null,
  busyFolder: null,     // "drawer" | "models" | "library" while a native picker is open
  fallback: {},         // context -> true when the path text field is shown
  confirmRemove: null,
  sweepDone: false,
};

// ---------------------------------------------------------------- small utilities
function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k);
    if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v);
  } catch (e) { /* storage unavailable: conveniences only */ }
  return null;
}
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : String(kid));
  return n;
}
function icon(name, cls = "ico") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}
function setIcon(node, name) { const u = node.querySelector("use"); if (u) u.setAttribute("href", "#i-" + name); }
function fmt(n, d = 2) { return Number(n).toFixed(d); }
function num(n) { return Number(n).toLocaleString(); }
function plural(n, word) { return `${num(n)} ${word}${n === 1 ? "" : "s"}`; }
function announce(msg) { const l = $("#live"); l.textContent = ""; requestAnimationFrame(() => { l.textContent = msg; }); }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function reducedMotion() { return matchMedia("(prefers-reduced-motion: reduce)").matches; }
function isAbs(p) { return /^([A-Za-z]:[\\/]|\\\\|\/|~)/.test(p); }
function baseName(p) { const parts = String(p).split(/[\\/]+/).filter(Boolean); return parts.length ? parts[parts.length - 1].replace(/:$/, "") : "Library"; }
function shortModel(label) { const i = String(label).indexOf(":"); return i > 0 ? label.slice(0, i) : label; }
function modelDesc(label) { const i = String(label).indexOf(":"); return i > 0 ? label.slice(i + 1).trim() : ""; }

function showNotice(msg, kind = "error") {
  const n = $("#notice");
  n.textContent = msg || "";
  n.className = "notice" + (kind === "info" ? " info" : "");
  n.setAttribute("role", kind === "info" ? "status" : "alert");
  n.hidden = !msg;
}
function setErr(id, msg) { const e = $(id); e.textContent = msg || ""; e.hidden = !msg; }

// ---------------------------------------------------------------- API
class ApiError extends Error {}
async function api(path, opts = {}) {
  const method = opts.method || (opts.body !== undefined ? "POST" : "GET");
  const init = { method };
  if (method !== "GET") {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.body === undefined ? {} : opts.body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    const err = new ApiError("Can't reach VocaLocate. Reconnecting…");
    err.offline = true;
    throw err;
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) {
    const msg = data && data.error;
    const err = new ApiError(msg ? sentence(msg) : `Request failed (HTTP ${res.status}).`);
    err.status = res.status;
    // Older servers: unknown routes are 404 "not found"; no DELETE handler is 501; some 405.
    err.missing = res.status === 405 || res.status === 501 || (res.status === 404 && (!msg || msg === "not found"));
    throw err;
  }
  return data;
}
function sentence(s) { s = String(s).trim(); if (!s) return s; s = s[0].toUpperCase() + s.slice(1); return /[.!?…]$/.test(s) ? s : s + "."; }

// ---------------------------------------------------------------- status + polling
function normalize(st) {
  st.libraries = st.libraries || [];
  st.models = st.models || [];
  st.examples = st.examples || [];
  st.hints = st.hints || [];
  if (!st.boot) st.boot = { state: "ready", message: "" };
  if (typeof st.ready !== "boolean") st.ready = st.libraries.some((l) => l.models_ready && l.models_ready.length);
  if (!st.text_state) st.text_state = st.text ? "ready" : "off";
  if (!Array.isArray(st.needs)) {
    st.needs = [];
    if (!st.models.length) st.needs.push("models");
    if (!st.libraries.length) st.needs.push("library");
  }
  st.settings = Object.assign({ model: null, autostart: false, autostart_supported: false }, st.settings || {});
  return st;
}
function anyBusy(st) {
  return !st || st.boot.state !== "ready" || st.text_state === "loading" ||
    st.libraries.some((l) => l.state !== "ready" && l.state !== "error");
}
async function poll() {
  clearTimeout(S.pollT);
  if (!S.online) return;
  try {
    const st = normalize(await api("/api/status"));
    applyStatus(st);
  } catch (e) {
    if (e.offline) return goOffline();
  }
  S.pollT = setTimeout(poll, anyBusy(S.status) ? POLL_BUSY_MS : POLL_IDLE_MS);
}
function pollSoon() { clearTimeout(S.pollT); S.pollT = setTimeout(poll, 150); }

function goOffline() {
  if (!S.online) return;
  S.online = false;
  clearTimeout(S.pollT);
  stopRecording(true);
  stopPlayback();
  render();
  healthLoop();
}
async function healthLoop() {
  if (S.online) return;
  let reachable = false;
  try { await fetch(S.missing.has("health") ? "/api/status" : "/api/health", { cache: "no-store" }); reachable = true; } catch (e) { /* still down */ }
  if (reachable) {
    S.online = true;
    S.statusKey = "";
    announce("Reconnected to VocaLocate.");
    return poll();
  }
  setTimeout(healthLoop, HEALTH_MS);
}

function applyStatus(st) {
  const key = JSON.stringify(st);
  S.everOnline = true;
  if (key === S.statusKey) return;
  const prev = S.status;
  S.status = st;
  S.statusKey = key;
  // Selection: keep the user's library if it still exists.
  if (!st.libraries.some((l) => l.name === S.lib)) {
    const pick = st.libraries.find((l) => l.models_ready && l.models_ready.length) || st.libraries[0];
    S.lib = pick ? pick.name : "";
  }
  if (st.settings.model && !S.missing.has("settings")) S.model = st.settings.model;
  if (!st.models.some((m) => m.name === S.model)) S.model = st.models.length ? st.models[0].name : "";
  render();
  const textNow = textReady(), textBefore = prev ? prev.text_state === "ready" || (prev.text && !prev.text_state) : false;
  if (S.waiting && canSearch()) requestSearch();
  else if (textNow && !textBefore && S.hint.trim() && S.query) requestSearch();
}

// ---------------------------------------------------------------- derived state
function currentLib() { return S.status ? S.status.libraries.find((l) => l.name === S.lib) || null : null; }
function canSearch() { const l = currentLib(); return !!(l && l.models_ready && l.models_ready.length); }
function effModel(lib) {
  if (!lib || !lib.models_ready || !lib.models_ready.length) return S.model;
  return lib.models_ready.includes(S.model) ? S.model : lib.models_ready[0];
}
function textReady() { return !!S.status && S.status.text_state === "ready"; }
function modelLabel(name) { const m = S.status && S.status.models.find((x) => x.name === name); return m ? m.label : name; }
function libBusy(l) { return l.state !== "ready" && l.state !== "error"; }
// While the pipeline scans there is no per-file progress: say how many files, never a frozen 0%.
function scanText(l) {
  const n = (l.folders || []).reduce((a, f) => a + (f.files || 0), 0);
  return l.state === "scanning" && n ? `Scanning ${plural(n, "file")}…` : null;
}
function audioUrl(lib, id) { return `/api/audio/${encodeURIComponent(lib)}/${id}`; }

// ---------------------------------------------------------------- top-level render
function render() {
  const st = S.status;
  const gate = $("#gate"), app = $("#app");
  let mode = "app";
  if (!S.online) mode = S.everOnline ? "offline" : "connecting";
  else if (!st) mode = "connecting";
  // Boot only blocks while nothing is searchable yet; restored libraries can be used while it finishes.
  else if (st.boot.state !== "ready" && !st.ready) mode = "boot";
  else if (st.needs.length) mode = "setup";
  gate.hidden = mode === "app";
  app.hidden = mode !== "app";
  if (mode !== "app") { closeDrawer(); closeSettings(); renderGate(mode); return; }
  if (S.gateMode && S.gateMode !== "app") { S.gateMode = "app"; requestAnimationFrame(() => { drawPad(); sweep(); }); }
  S.gateMode = "app";
  renderBar();
  renderStrip();
  renderConsole();
  if (!S.results) renderEmpty();
  if (!$("#drawer").hidden) renderLibs();
  if (!$("#settings").hidden && !$("#settings").contains(document.activeElement)) renderSettings();
}

// ---------------------------------------------------------------- gate: connecting / boot / offline / setup
function renderGate(mode) {
  const gate = $("#gate");
  const st = S.status;
  const prevMode = S.gateMode;
  S.gateMode = mode;
  gate.className = "gate" + (mode === "offline" ? " offline" : mode === "setup" ? "" : " waiting");
  const title = $("#gate-title"), text = $("#gate-text"), setup = $("#setup");
  setup.hidden = mode !== "setup";
  if (mode === "connecting") {
    title.textContent = "Starting VocaLocate";
    text.textContent = "Connecting to the local engine…";
  } else if (mode === "offline") {
    title.textContent = "Reconnecting…";
    text.textContent = "VocaLocate's engine stopped responding. This page reconnects by itself as soon as it is back; if it doesn't, start VocaLocate again.";
  } else if (mode === "boot") {
    title.textContent = "Warming up models…";
    text.textContent = (st.boot.message ? sentence(st.boot.message) + " " : "") + "Search opens here when they're loaded.";
  } else {
    title.textContent = "Set up VocaLocate";
    text.textContent = "Two things and you can search your sounds by voice. Nothing leaves this computer.";
    renderSetup();
  }
  if (prevMode !== mode && mode === "setup") title.focus({ preventScroll: true });
}

function renderSetup() {
  const st = S.status;
  const needModels = st.needs.includes("models");
  const needLib = st.needs.includes("library");
  const keepFocus = document.activeElement && $("#setup").contains(document.activeElement) && document.activeElement.tagName === "INPUT";
  if (keepFocus) return; // don't wipe a half-typed path
  const steps = [];
  steps.push(setupStep({
    key: "models", done: !needModels, active: needModels,
    title: needModels ? "Import models" : "Models installed",
    body: needModels ? "Pick the folder that holds the VocaLocate model files (the .pt files, plus fusion.json if you have it). They are copied into VocaLocate's data folder."
      : `${plural(st.models.length, "model")} ready.`,
    action: "Import models…", run: () => pickFolder("models"),
  }));
  steps.push(setupStep({
    key: "library", done: !needLib, active: !needModels && needLib,
    title: needLib ? "Add your sound folder" : "Sound folder added",
    body: needLib ? "Choose the folder of sound effects you want to search. Files are read in place and indexed in the background; you can start searching as soon as the first pass is ready."
      : "Indexing has started.",
    action: "Add your sound folder…", run: () => pickFolder("library"),
  }));
  $("#setup").replaceChildren(...steps);
}
function setupStep({ key, done, active, title, body, action, run }) {
  const busy = S.busyFolder === key;
  const kids = [el("p", {}, body)];
  if (!done && active) {
    kids.push(el("button", {
      type: "button", class: "btn" + (active ? " primary" : "") + (busy ? " busy" : ""), disabled: !active || busy,
      onclick: run,
    }, busy ? "Waiting for the folder picker…" : action));
    if (S.fallback[key] && active) kids.push(pathForm(key));
    if (S.setupErr && S.setupErr.key === key) kids.push(el("p", { class: "error", role: "alert" }, S.setupErr.msg));
  }
  return el("li", { class: "step" + (done ? " done" : "") + (!done && !active ? " later" : "") },
    el("span", { class: "step-mark", "aria-hidden": "true" }, done ? icon("check") : null),
    el("h2", {}, title),
    el("div", { class: "step-body" }, kids));
}
function pathForm(key) {
  const id = "path-" + key;
  const input = el("input", { id, type: "text", autocomplete: "off", spellcheck: "false", placeholder: key === "models" ? "D:\\VocaLocate\\models" : "D:\\Sounds\\SFX" });
  return el("form", {
    class: "pathform",
    onsubmit: (e) => { e.preventDefault(); usePath(key, input.value.trim()); },
  }, el("label", { for: id }, key === "models" ? "Or type the models folder path" : "Or type the folder path"),
  el("div", { class: "pathrow" }, input, el("button", { type: "submit", class: "btn" }, "Use this folder")));
}

// ---------------------------------------------------------------- folder picking (native dialog, text fallback)
async function pickFolder(key) {
  if (S.busyFolder) return;
  S.setupErr = null;
  setErr("#drawer-err", "");
  if (S.missing.has("pick")) return showFallback(key);
  S.busyFolder = key;
  rerenderPickers();
  const title = key === "models" ? "Choose the folder with your VocaLocate models" : "Choose your sound folder";
  let path = null, fallback = null, offline = false;
  try {
    const r = await api("/api/pick-folder", { body: { title } });
    path = r && r.path;
    if (!path) fallback = "No folder chosen. You can also type the path.";
  } catch (e) {
    if (e.missing) { S.missing.add("pick"); fallback = ""; }
    else if (e.offline) offline = true;
    else fallback = e.message;
  }
  S.busyFolder = null;
  rerenderPickers();
  if (offline) return goOffline();
  if (fallback !== null) return showFallback(key, fallback);
  if (path) await usePath(key, path);
}
function rerenderPickers() {
  if (S.gateMode === "setup") renderSetup();
  const b = $("#add-folder");
  const busy = S.busyFolder === "drawer";
  b.disabled = busy;
  b.classList.toggle("busy", busy);
  b.lastChild.textContent = busy ? " Waiting for the folder picker…" : " Add folder";
}
function showFallback(key, msg) {
  S.fallback[key] = true;
  if (key === "drawer") {
    $("#drawer-path").hidden = false;
    if (msg) setErr("#drawer-err", msg);
    $("#drawer-path-in").focus();
  } else {
    if (msg) S.setupErr = { key, msg };
    renderSetup();
    const i = $("#path-" + key);
    if (i) i.focus();
  }
}
async function usePath(key, path) {
  const fail = (msg) => {
    if (key === "drawer") setErr("#drawer-err", msg);
    else { S.setupErr = { key, msg }; S.fallback[key] = true; renderSetup(); }
  };
  if (!path) return fail("Type or pick a folder first.");
  if (!isAbs(path)) return fail(`"${path}" isn't a full path. Use one like D:\\Sounds\\SFX.`);
  try {
    if (key === "models") {
      if (S.missing.has("import")) return fail("This version of the engine can't import models. Update VocaLocate.");
      const r = await api("/api/models/import", { body: { path } });
      announce(`Imported ${plural((r && r.models || []).length, "model")}.`);
    } else {
      await addLibrary(path);
    }
    S.fallback[key] = false;
    S.setupErr = null;
    if (key === "drawer") { $("#drawer-path").hidden = true; $("#drawer-path-in").value = ""; }
    S.statusKey = "";
    pollSoon();
  } catch (e) {
    if (e.missing && key === "models") S.missing.add("import");
    if (e.offline) return goOffline();
    fail(e.missing && key === "models" ? "This version of the engine can't import models. Update VocaLocate." : e.message);
  }
}
async function addLibrary(path) {
  const libs = S.status ? S.status.libraries : [];
  const same = libs.find((l) => l.roots.length === 1 && l.roots[0].replace(/[\\/]+$/, "").toLowerCase() === path.replace(/[\\/]+$/, "").toLowerCase());
  if (same) { S.lib = same.name; store("vl.lib", same.name); return rescan(same); }
  let name = baseName(path), n = 2;
  const taken = new Set(libs.map((l) => l.name.toLowerCase()));
  while (taken.has(name.toLowerCase())) name = `${baseName(path)} ${n++}`;
  const lib = await api("/api/libraries", { body: { name, roots: [path] } });
  S.lib = (lib && lib.name) || name;
  store("vl.lib", S.lib);
  announce(`Indexing ${S.lib}. You can search as soon as the first model is ready.`);
}
async function rescan(lib) {
  try {
    if (S.missing.has("rescan")) throw Object.assign(new ApiError(""), { missing: true });
    await api(`/api/libraries/${encodeURIComponent(lib.name)}/rescan`, { method: "POST" });
  } catch (e) {
    if (!e.missing) throw e;
    S.missing.add("rescan");
    await api("/api/libraries", { body: { name: lib.name, roots: lib.roots } });
  }
  announce(`Rescanning ${lib.name}.`);
  pollSoon();
}
async function removeLib(lib) {
  try {
    await api(`/api/libraries/${encodeURIComponent(lib.name)}`, { method: "DELETE" });
    announce(`Removed ${lib.name} from VocaLocate. Your files are untouched.`);
    if (S.lib === lib.name) { S.lib = ""; clearResults(); }
    S.statusKey = "";
    pollSoon();
  } catch (e) {
    if (e.missing) { S.missing.add("delete"); renderLibs(); return setErr("#drawer-err", "This version of the engine can't remove libraries."); }
    setErr("#drawer-err", e.message);
  }
}

// ---------------------------------------------------------------- bar + strip
function renderBar() {
  const lib = currentLib();
  const dot = $("#lib-dot");
  dot.className = "dot " + (!lib ? "" : lib.state === "error" ? "error" : libBusy(lib) ? "busy" : "ready");
  $("#lib-name").textContent = lib ? lib.name : "Add a library";
  let sub = "";
  if (lib) {
    if (lib.state === "error") sub = "Indexing failed";
    else if (libBusy(lib)) sub = lib.progress && lib.progress.total ? `Indexing ${Math.floor(100 * lib.progress.done / lib.progress.total)}%` : scanText(lib) || "Indexing";
    else if (lib.files != null) sub = plural(lib.files, "file");
  }
  $("#lib-sub").textContent = sub;
  $("#lib-btn").setAttribute("aria-label", `Libraries: ${lib ? lib.name : "none"}${sub ? ", " + sub : ""}`);
}
function renderStrip() {
  const st = S.status;
  const busy = st.libraries.filter(libBusy);
  const strip = $("#strip");
  strip.hidden = !busy.length;
  if (!busy.length) return;
  const l = busy.find((x) => x.name === S.lib) || busy[0];
  const p = l.progress;
  const bar = $("#strip-bar");
  const pct = p && p.total ? Math.min(100, (100 * p.done) / p.total) : null;
  bar.classList.toggle("indeterminate", pct == null);
  bar.firstElementChild.style.width = pct == null ? "" : pct + "%";
  if (pct == null) bar.removeAttribute("aria-valuenow"); else { bar.setAttribute("aria-valuenow", String(Math.round(pct))); bar.setAttribute("aria-valuemin", "0"); bar.setAttribute("aria-valuemax", "100"); }
  const ready = l.models_ready && l.models_ready.length;
  const what = p && p.total ? `${num(p.done)} of ${plural(p.total, "file")}` : ((scanText(l) || "").replace(/…$/, "") || (l.message ? sentence(l.message).replace(/\.$/, "") : "Scanning"));
  $("#strip-text").textContent = `Indexing ${l.name}: ${what}.` + (ready ? " Search is already on." : " Search opens when the first model is ready.") +
    (busy.length > 1 ? ` (${busy.length - 1} more queued)` : "");
}

// ---------------------------------------------------------------- console (pad, take, hint, examples)
function renderConsole() {
  const st = S.status;
  // hint availability
  const hint = $("#hint"), note = $("#hint-note");
  hint.disabled = st.text_state === "off" || st.text_state === "error";
  if (st.text_state === "off") { note.hidden = false; note.textContent = "Text hints are off in this setup."; }
  else if (st.text_state === "error") { note.hidden = false; note.textContent = "The text model failed to load, so hints are unavailable. Voice search still works."; }
  else if (st.text_state === "loading") { note.hidden = false; note.textContent = "The text model is still loading. Your hint applies as soon as it's ready."; }
  else note.hidden = true;
  renderChips();
  // examples, only rebuilt when the list changes
  const exKey = JSON.stringify(st.examples.map((e) => e.id));
  if (exKey !== S.exKey) {
    S.exKey = exKey;
    $("#examples").hidden = !st.examples.length;
    $("#example-list").replaceChildren(...st.examples.map((ex) => el("li", {}, el("button", {
      type: "button", "data-ex": ex.id, "aria-pressed": "false", onclick: () => chooseExample(ex),
    }, el("span", { class: "ex-name", title: ex.name }, ex.name.replace(/\.wav$/i, "")),
    el("span", { class: "ex-tag" }, ex.category || "")))));
    $("#examples summary").textContent = `Demo imitations (${st.examples.length})`;
  }
  syncExamples();
  renderTake();
}
function renderChips() {
  const st = S.status;
  const words = S.hint.toLowerCase().split(/[\s,]+/).filter(Boolean);
  const off = st.text_state === "off" || st.text_state === "error";
  const box = $("#chips");
  box.hidden = off || !st.hints.length;
  const key = JSON.stringify([st.hints, words, off]);
  if (key === S.chipKey) return;
  S.chipKey = key;
  const focusIdx = [...box.children].indexOf(document.activeElement);
  box.replaceChildren(...st.hints.map((h) => el("button", {
    type: "button", class: "chip", "aria-pressed": String(words.includes(h)), onclick: () => toggleHint(h),
  }, h)));
  if (focusIdx >= 0 && box.children[focusIdx]) box.children[focusIdx].focus();
}
function toggleHint(h) {
  let words = S.hint.split(/[\s,]+/).filter(Boolean);
  words = words.some((w) => w.toLowerCase() === h) ? words.filter((w) => w.toLowerCase() !== h) : words.concat(h);
  S.hint = words.join(" ");
  $("#hint").value = S.hint;
  renderChips();
  scheduleHint();
}
function scheduleHint() {
  clearTimeout(S.hintT);
  if (!S.query) return;
  S.hintT = setTimeout(() => { if (textReady() || !S.hint.trim()) requestSearch(); }, HINT_DEBOUNCE_MS);
}
function syncExamples() {
  document.querySelectorAll("[data-ex]").forEach((b) => {
    b.setAttribute("aria-pressed", String(!!(S.query && S.query.kind === "example" && String(S.query.exampleId) === b.dataset.ex)));
  });
}
function renderTake() {
  const q = S.query;
  $("#take").hidden = !q;
  $("#pad").classList.toggle("has-take", !!q && !S.rec);
  if (!S.rec) {
    $("#pad-label").textContent = q ? "Record again" : "Record";
    $("#pad-time").textContent = q && q.seconds ? fmt(q.seconds, 1) + " s" : "";
  }
  if (q) {
    $("#take-name").textContent = q.label;
    $("#take-len").textContent = q.seconds ? fmt(q.seconds, 1) + " s" : "";
  }
  drawPad();
  syncPlay();
}

// ---------------------------------------------------------------- scope / waveform drawing
function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(50, c.clientWidth), h = Math.max(20, c.clientHeight);
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}
// columns: array of [lo, hi] in -1..1, drawn right-aligned (live) or stretched (take)
function drawColumns(g, x0, w, mid, amp, cols, color) {
  g.fillStyle = color;
  const n = cols.length;
  if (!n) return;
  const bw = w / n;
  for (let i = 0; i < n; i++) {
    const [lo, hi] = cols[i];
    const y1 = mid - hi * amp, y2 = mid - lo * amp;
    g.fillRect(x0 + i * bw, y1, Math.max(1, bw - (bw > 3 ? 1 : 0)), Math.max(1.5, y2 - y1));
  }
}
function peaks(samples, n) {
  const out = [];
  let peak = 1e-4;
  for (let i = 0; i < samples.length; i++) { const a = Math.abs(samples[i]); if (a > peak) peak = a; }
  const step = samples.length / n;
  for (let x = 0; x < n; x++) {
    let lo = 0, hi = 0;
    const end = Math.min(samples.length, Math.floor((x + 1) * step));
    for (let i = Math.floor(x * step); i < end; i++) { const s = samples[i]; if (s < lo) lo = s; if (s > hi) hi = s; }
    out.push([lo / peak, hi / peak]);
  }
  return out;
}
function drawPad(sweepT) {
  const c = $("#scope");
  if (!c || $("#app").hidden) return;
  const { g, w, h } = sizeCanvas(c);
  const mid = h * 0.36, amp = h * 0.24;
  const x0 = 18, ww = w - 36;
  g.fillStyle = css("--scope-idle");
  const lineW = sweepT == null ? ww : ww * sweepT;
  g.fillRect(x0, mid - 0.5, lineW, 1);
  // tick marks, like a scope graticule
  for (let i = 0; i <= 10; i++) { const x = x0 + (ww * i) / 10; if (x - x0 <= lineW) g.fillRect(Math.round(x), mid - 3, 1, 6); }
  const rec = S.rec;
  if (rec) {
    const cols = Math.floor(ww / 3);
    const src = rec.cols.slice(-cols);
    const scale = Math.max(0.08, rec.peak);
    const norm = src.map(([lo, hi]) => [Math.max(-1, lo / scale), Math.min(1, hi / scale)]);
    drawColumns(g, x0 + (cols - norm.length) * 3, norm.length * 3, mid, amp, norm, css("--rec"));
  } else if (S.query && S.query.samples) {
    drawColumns(g, x0, ww, mid, amp, peaks(S.query.samples, Math.floor(ww / 2)), css("--wave"));
  }
}
function sweep() {
  if (S.sweepDone || reducedMotion()) { S.sweepDone = true; return drawPad(); }
  S.sweepDone = true;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 700);
    drawPad(1 - Math.pow(1 - k, 3));
    if (k < 1 && !S.rec) requestAnimationFrame(step); else drawPad();
  };
  requestAnimationFrame(step);
  setTimeout(() => { if (!S.rec) drawPad(); }, 900); // rAF can be paused in a hidden window
}

// ---------------------------------------------------------------- audio helpers
let decodeCtx = null;
function getDecodeCtx() { if (!decodeCtx) decodeCtx = new (window.AudioContext || window.webkitAudioContext)(); return decodeCtx; }
function decode(buf) {
  const ctx = getDecodeCtx();
  return new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(buf, resolve, reject);
    if (p && p.then) p.then(resolve, reject);
  });
}
function toMono(ab) {
  if (ab.numberOfChannels === 1) return ab.getChannelData(0).slice();
  const out = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const d = ab.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / ab.numberOfChannels;
  }
  return out;
}
function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}
function toBase64(buf) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
function releaseQuery() {
  const q = S.query;
  if (q && q.kind === "audio" && q.url) URL.revokeObjectURL(q.url);
}
function setAudioQuery(samples, rate, label) {
  const wav = encodeWav(samples, rate);
  releaseQuery();
  stopPlayback();
  S.query = {
    kind: "audio", b64: toBase64(wav), samples, label,
    url: URL.createObjectURL(new Blob([wav], { type: "audio/wav" })),
    seconds: samples.length / rate,
  };
  syncExamples();
  renderTake();
  requestSearch();
}

// ---------------------------------------------------------------- playback (one at a time)
function stopPlayback() {
  S.playTok++;
  if (S.player) { S.player.pause(); S.player.removeAttribute("src"); S.player.load(); }
  clearTimeout(S.playTimer);
  S.playing = null;
  syncPlay();
}
function play(key, url, { toggle = true, max = 0 } = {}) {
  if (toggle && S.playing === key) return stopPlayback();
  stopPlayback();
  if (!S.player) {
    S.player = new Audio();
    S.player.addEventListener("ended", () => stopPlayback());
    S.player.addEventListener("error", () => { if (S.playing && S.player.getAttribute("src")) { announce("Couldn't play this file."); stopPlayback(); } });
  }
  const tok = S.playTok;
  S.player.src = url;
  S.playing = key;
  S.player.play().catch(() => { if (tok === S.playTok) { announce("Playback was blocked or failed."); stopPlayback(); } });
  if (max) S.playTimer = setTimeout(() => { if (tok === S.playTok) stopPlayback(); }, max * 1000);
  syncPlay();
}
function syncPlay() {
  document.querySelectorAll("[data-play]").forEach((b) => {
    const on = b.dataset.play === S.playing;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
    setIcon(b, on ? "stop" : "play");
  });
  document.querySelectorAll(".row").forEach((r) => {
    const on = r.dataset.key === S.playing;
    if (on && !r.classList.contains("playing")) { r.classList.remove("playing"); void r.offsetWidth; }
    r.classList.toggle("playing", on);
  });
}

// ---------------------------------------------------------------- recording
const WORKLET = `class VLCapture extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor("vl-capture", VLCapture);`;

function toggleRec() { if (S.rec) stopRecording(); else startRecording(); }
async function startRecording() {
  if (S.rec || S.recStarting || !S.online) return;
  S.recStarting = true;
  try { await beginRecording(); } finally { S.recStarting = false; }
}
async function beginRecording() {
  showNotice("");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return showNotice("This window can't use the microphone. Load an audio file instead.");
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) {
    return showNotice("Microphone access was denied or there is no microphone. Allow it for this window, or load an audio file.");
  }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  sink.connect(ctx.destination);
  const rec = { ctx, stream, src, sink, chunks: [], length: 0, level: 0, peak: 0, cols: [], acc: null, t0: performance.now(), node: null };
  const COL = 256;
  const onChunk = (data) => {
    rec.chunks.push(data);
    rec.length += data.length;
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const s = data[i];
      sum += s * s;
      if (!rec.acc) rec.acc = [0, 0, 0];
      if (s < rec.acc[0]) rec.acc[0] = s;
      if (s > rec.acc[1]) rec.acc[1] = s;
      if (++rec.acc[2] >= COL) { rec.cols.push([rec.acc[0], rec.acc[1]]); rec.peak = Math.max(rec.peak * 0.9995, -rec.acc[0], rec.acc[1]); rec.acc = null; }
    }
    if (rec.cols.length > 4000) rec.cols.splice(0, rec.cols.length - 2000);
    rec.level = Math.max(Math.sqrt(sum / data.length), rec.level * 0.85);
  };
  try {
    if (!ctx.audioWorklet) throw new Error("no worklet");
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    rec.node = new AudioWorkletNode(ctx, "vl-capture");
    rec.node.port.onmessage = (e) => onChunk(e.data);
  } catch (e) {
    rec.node = ctx.createScriptProcessor(4096, 1, 1);
    rec.node.onaudioprocess = (e) => onChunk(e.inputBuffer.getChannelData(0).slice());
  }
  src.connect(rec.node);
  rec.node.connect(sink);
  if (ctx.state === "suspended") await ctx.resume();
  S.rec = rec;
  stopPlayback();
  const pad = $("#pad");
  pad.classList.add("on");
  pad.classList.remove("has-take");
  pad.setAttribute("aria-pressed", "true");
  setIcon(pad, "stop");
  $("#pad-label").textContent = "Stop";
  $("#pad-hint").innerHTML = "Press <kbd>Space</kbd> to stop and search";
  announce("Recording. Make the sound now.");
  const tick = () => {
    if (S.rec !== rec) return;
    const sec = (performance.now() - rec.t0) / 1000;
    $("#pad-time").textContent = fmt(sec, 1) + " s";
    pad.style.setProperty("--lvl", Math.min(1, rec.level * 5).toFixed(3));
    rec.level *= 0.92;
    drawPad();
    if (sec >= MAX_REC_SEC) return stopRecording();
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}
function stopRecording(discard = false) {
  const rec = S.rec;
  if (!rec) return;
  S.rec = null;
  cancelAnimationFrame(rec.raf);
  try { rec.src.disconnect(); rec.node.disconnect(); } catch (e) { /* already disconnected */ }
  rec.stream.getTracks().forEach((t) => t.stop());
  const rate = rec.ctx.sampleRate;
  rec.ctx.close();
  const pad = $("#pad");
  pad.classList.remove("on");
  pad.style.setProperty("--lvl", "0");
  pad.setAttribute("aria-pressed", "false");
  setIcon(pad, "record");
  $("#pad-hint").innerHTML = "Press <kbd>Space</kbd> anywhere to start and stop";
  if (discard) return renderTake();
  const samples = new Float32Array(rec.length);
  let o = 0;
  for (const c of rec.chunks) { samples.set(c, o); o += c.length; }
  if (samples.length < rate * 0.2) { renderTake(); return showNotice("That was too short to search with. Record for at least a quarter of a second."); }
  announce(`Recorded ${fmt(samples.length / rate, 1)} seconds. Searching.`);
  setAudioQuery(samples, rate, "Your recording");
}

async function handleFile(file) {
  if (!file) return;
  showNotice("");
  try {
    const ab = await decode(await file.arrayBuffer());
    let mono = toMono(ab);
    const cap = Math.floor(ab.sampleRate * 30);
    if (mono.length > cap) mono = mono.slice(0, cap);
    stopRecording(true);
    announce(`Loaded ${file.name}. Searching.`);
    setAudioQuery(mono, ab.sampleRate, file.name);
  } catch (e) {
    showNotice(`Couldn't read "${file.name}" as audio. Try a WAV, MP3, FLAC or OGG file.`);
  }
}
async function chooseExample(ex) {
  stopRecording(true);
  releaseQuery();
  stopPlayback();
  showNotice("");
  const q = { kind: "example", exampleId: ex.id, url: `/api/example/${ex.id}`, label: ex.name.replace(/\.wav$/i, ""), samples: null };
  S.query = q;
  if (ex.hint && S.status.text_state !== "off" && S.status.text_state !== "error") { S.hint = ex.hint; $("#hint").value = ex.hint; renderChips(); }
  syncExamples();
  renderTake();
  requestSearch();
  try {
    const res = await fetch(q.url);
    if (!res.ok) throw new Error();
    const ab = await decode(await res.arrayBuffer());
    if (S.query === q) { q.samples = toMono(ab); q.seconds = ab.duration; renderTake(); }
  } catch (e) { /* the waveform is optional */ }
}

// ---------------------------------------------------------------- search (latest wins)
function requestSearch() {
  if (!S.query) return;
  if (!canSearch()) {
    S.waiting = true;
    showNotice(currentLib() ? "Got it. This search runs as soon as the library's first model is ready." : "Add a library first: open Libraries at the top right.", "info");
    return;
  }
  if (S.waiting) showNotice("");
  S.waiting = false;
  if (!S.inflight && S.results && S.lastSig === searchSig()) return; // nothing changed (e.g. a trailing space)
  if (S.inflight) { S.pending = true; return; }
  doSearch();
}
function searchSig() {
  const q = S.query, lib = currentLib();
  return JSON.stringify([q.kind, q.exampleId, q.b64 && q.b64.length, q.url, lib.name, effModel(lib), textReady() ? S.hint.trim() : ""]);
}
async function doSearch() {
  S.inflight = true;
  S.pending = false;
  const q = S.query, lib = currentLib(), model = effModel(lib);
  const text = textReady() ? S.hint.trim() : "";
  const body = { library: lib.name, model, text, k: K };
  if (q.kind === "example") body.example = q.exampleId; else body.audio = q.b64;
  $("#r-busy").hidden = false;
  try {
    const res = await api("/api/search", { body });
    const stillCurrent = S.query === q && S.lib === lib.name && (textReady() ? S.hint.trim() : "") === text;
    if (stillCurrent || !S.pending) {
      showNotice("");
      S.results = res;
      S.resultsQuery = q;
      S.lastSig = JSON.stringify([q.kind, q.exampleId, q.b64 && q.b64.length, q.url, lib.name, model, text]);
      S.sel = -1;
      renderResults();
      announce(res.results.length ? `${res.results.length} matches. Use the arrow keys to preview.` : "No matches.");
    }
  } catch (e) {
    if (e.offline) goOffline();
    else if (e.status === 503) { S.waiting = true; showNotice(e.message, "info"); pollSoon(); } // warming up: retried by applyStatus
    else if (!S.pending) showNotice(e.message);
  } finally {
    S.inflight = false;
    if (!S.pending) $("#r-busy").hidden = true;
    if (S.pending && S.query) doSearch();
  }
}
function clearResults() {
  S.results = null; S.resultsQuery = null; S.sel = -1;
  $("#r-list").replaceChildren();
  $("#r-sum").textContent = "";
  renderEmpty();
}

// ---------------------------------------------------------------- results
function renderEmpty() {
  const box = $("#r-empty");
  box.hidden = false;
  $("#r-list").replaceChildren();
  $("#r-sum").textContent = "";
  const lib = currentLib();
  if (!lib) {
    box.replaceChildren(el("h3", {}, "No library selected"),
      el("p", {}, "Open Libraries at the top right and add the folder you keep your sound effects in."));
    return;
  }
  if (!canSearch()) {
    box.replaceChildren(el("h3", {}, lib.state === "error" ? `${lib.name} couldn't be indexed` : `Indexing ${lib.name}`),
      el("p", {}, lib.state === "error" ? sentence(lib.message || "Indexing failed") + " Rescan it from Libraries."
        : "Go ahead and record. Your search runs the moment the first model is ready."));
    return;
  }
  box.replaceChildren(
    el("h3", {}, "Make the sound. Matches land here."),
    el("p", {}, `Searching ${lib.files != null ? plural(lib.files, "file") + " in " : ""}${lib.name}.`),
    el("dl", { class: "keys" },
      el("dt", {}, el("kbd", {}, "Space")), el("dd", {}, "Record, then stop to search"),
      el("dt", {}, el("kbd", {}, "↑"), " ", el("kbd", {}, "↓")), el("dd", {}, "Move through matches and hear each one"),
      el("dt", {}, el("kbd", {}, "Enter")), el("dd", {}, "Play or stop the selected match"),
      el("dt", {}, "Drag"), el("dd", {}, "Pull a row into your editor or a folder")));
}
function renderResults() {
  const r = S.results;
  if (!r) return renderEmpty();
  $("#r-empty").hidden = true;
  const folders = r.folders ? r.folders.length : 0;
  const sum = $("#r-sum");
  sum.replaceChildren(
    el("b", { class: "num" }, String(r.results.length)), ` of ${plural(r.searched, "file")} in ${r.library}`,
    folders > 1 ? `, ${folders} folders` : "", `, ${r.ms < 1 ? "under 1" : fmt(r.ms, 0)} ms`,
    r.text && r.text_ignored ? el("span", { class: "hint-ignored", title: sentence(r.text_ignored) },
      `, hint ignored: text model ${S.status && S.status.text_state === "loading" ? "still loading" : "unavailable"}`)
      : r.text ? el("span", {}, `, hint “${r.text}”`) : "",
    `, ${shortModel(r.model_label || modelLabel(r.model))}`);
  const list = $("#r-list");
  if (!r.results.length) {
    list.replaceChildren(el("li", { class: "r-foot" }, "No matches in this library. Try a longer or more exaggerated imitation."));
    return;
  }
  const scores = r.results.map((x) => x.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const rows = r.results.map((h, i) => resultRow(h, i, r.library, hi > lo ? (h.score - lo) / (hi - lo) : 1));
  list.replaceChildren(...rows, el("li", { class: "r-foot", "aria-hidden": "true" }, "Drag a row into your editor. Preview plays the first 5 seconds."));
  syncSel(false);
  syncPlay();
}
function resultRow(h, i, library, rel) {
  const url = audioUrl(library, h.id);
  const abs = new URL(url, location.href).href;
  const key = "r" + h.id;
  const SEGS = 14;
  const lit = Math.max(1, Math.round(rel * SEGS));
  const folder = h.project || baseName(String(h.path).replace(/[\\/][^\\/]*$/, ""));
  const acts = [];
  if (!S.missing.has("reveal")) {
    acts.push(el("button", {
      type: "button", class: "btn icon ghost", title: "Show in folder", "aria-label": `Show ${h.name} in its folder`,
      onclick: (e) => { e.stopPropagation(); reveal(h); },
    }, icon("reveal")));
  }
  acts.push(el("button", {
    type: "button", class: "btn icon ghost", title: "Copy path", "aria-label": `Copy the path of ${h.name}`,
    onclick: (e) => { e.stopPropagation(); copyText(h.path); },
  }, icon("copy")));
  const row = el("li", {
    class: "row", tabindex: "-1", draggable: "true", "data-i": i, "data-key": key, style: `--dur:${PREVIEW_SEC}s`,
    "aria-label": `${h.rank}. ${h.name}, ${folder}, score ${fmt(h.score, 3)}`,
    onclick: (e) => { if (e.target.closest("button")) return; select(i, false); play(key, url, { max: PREVIEW_SEC }); },
    onkeydown: (e) => {
      if (e.key === "Enter" && e.target === row) { e.preventDefault(); play(key, url, { max: PREVIEW_SEC }); }
    },
    ondragstart: (e) => {
      const name = h.name.replace(/:/g, "_");
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData("DownloadURL", `audio/wav:${name}:${abs}`);
      e.dataTransfer.setData("text/uri-list", abs);
      e.dataTransfer.setData("text/plain", h.path);
      row.classList.add("dragging");
    },
    ondragend: () => row.classList.remove("dragging"),
  },
  el("span", { class: "rank", "aria-hidden": "true" }, h.rank),
  el("button", {
    type: "button", class: "btn icon play", "data-play": key, "aria-pressed": "false", tabindex: "-1",
    "aria-label": `Preview ${h.name}`, onclick: (e) => { e.stopPropagation(); select(i, false); play(key, url, { max: PREVIEW_SEC }); },
  }, icon("play")),
  el("div", { class: "r-main" },
    el("div", { class: "r-name", title: h.name }, h.name),
    el("div", { class: "r-where" }, el("span", { class: "proj" }, folder), el("span", { class: "path", title: h.path }, shownDir(h.path, folder)))),
  el("div", { class: "score", title: h.text_score != null ? `voice ${fmt(h.voice_score, 3)}, text ${fmt(h.text_score, 3)}` : `similarity ${fmt(h.score, 3)}` },
    el("div", { class: "meter", "aria-hidden": "true" }, el("i", { style: `width:${lit * 9}px` })),
    el("b", {}, fmt(h.score, 3))),
  el("div", { class: "r-acts" }, acts));
  return row;
}
// Parent location of the file, minus the folder name already shown beside it.
function shownDir(path, folder) {
  let dir = String(path).replace(/[\\/][^\\/]*$/, "");
  if (baseName(dir) === folder) dir = dir.replace(/[\\/][^\\/]*$/, "");
  return dir;
}
function select(i, focus = true) {
  const n = S.results ? S.results.results.length : 0;
  if (!n) return;
  S.sel = Math.max(0, Math.min(n - 1, i));
  syncSel(focus);
}
function syncSel(focus) {
  const rows = document.querySelectorAll("#r-list .row");
  rows.forEach((r, i) => {
    const on = i === S.sel;
    r.classList.toggle("sel", on);
    r.tabIndex = on || (S.sel < 0 && i === 0) ? 0 : -1;
    if (on) r.setAttribute("aria-current", "true"); else r.removeAttribute("aria-current");
  });
  const cur = rows[S.sel];
  if (cur && focus) { cur.focus({ preventScroll: true }); cur.scrollIntoView({ block: "nearest" }); }
}
function previewSel() {
  const h = S.results && S.results.results[S.sel];
  if (h) play("r" + h.id, audioUrl(S.results.library, h.id), { toggle: false, max: PREVIEW_SEC });
}
async function reveal(h) {
  try {
    await api("/api/reveal", { body: { library: S.results.library, id: h.id } });
  } catch (e) {
    if (e.missing) {
      S.missing.add("reveal");
      document.querySelectorAll('[title="Show in folder"]').forEach((b) => b.remove());
      await copyText(h.path, "This engine can't open folders, so the path was copied instead.");
      return;
    }
    showNotice(e.message);
  }
}
async function copyText(text, msg = "Path copied.") {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = el("textarea", { class: "sr-only" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e2) { /* nothing else to try */ }
    ta.remove();
  }
  announce(msg);
  showNotice(msg, "info");
  clearTimeout(S.noticeT);
  S.noticeT = setTimeout(() => { if ($("#notice").textContent === msg) showNotice(""); }, 2500);
}

// ---------------------------------------------------------------- library drawer
function openDrawer() {
  closeSettings();
  const d = $("#drawer");
  d.hidden = false;
  $("#lib-btn").setAttribute("aria-expanded", "true");
  setErr("#drawer-err", "");
  renderLibs();
  $("#drawer-close").focus();
}
function closeDrawer(restore) {
  const d = $("#drawer");
  if (d.hidden) return;
  d.hidden = true;
  S.confirmRemove = null;
  $("#lib-btn").setAttribute("aria-expanded", "false");
  if (restore) $("#lib-btn").focus();
}
function renderLibs() {
  const st = S.status;
  const box = $("#libs");
  const active = document.activeElement;
  const focusKey = active && box.contains(active) ? active.dataset.k : null;
  if (!st.libraries.length) {
    box.replaceChildren(el("li", { class: "lib" }, el("p", { class: "note" }, "No libraries yet. Add the folder you keep your sound effects in.")));
    return;
  }
  box.replaceChildren(...st.libraries.map((l) => {
    const busy = libBusy(l);
    const p = l.progress;
    const pct = p && p.total ? Math.min(100, (100 * p.done) / p.total) : null;
    let stateText;
    if (l.state === "error") stateText = sentence(l.message || "Indexing failed");
    else if (busy) stateText = (p && p.total ? `Indexing ${num(p.done)} of ${plural(p.total, "file")}` : scanText(l) || sentence(l.message || "Indexing").replace(/\.$/, "")) +
      (l.models_ready && l.models_ready.length ? ". Searchable now." : "");
    else stateText = l.files != null ? `${plural(l.files, "file")}, ready` : "Ready";
    const confirming = S.confirmRemove === l.name;
    const acts = [el("button", {
      type: "button", class: "btn small ghost", "data-k": "rescan-" + l.name, disabled: busy,
      onclick: async () => { setErr("#drawer-err", ""); try { await rescan(l); } catch (e) { setErr("#drawer-err", e.message); } },
    }, icon("rescan"), "Rescan")];
    if (!S.missing.has("delete")) {
      acts.push(el("button", {
        type: "button", class: "btn small ghost" + (confirming ? " danger" : ""), "data-k": "remove-" + l.name,
        onclick: () => {
          if (!confirming) { S.confirmRemove = l.name; renderLibs(); setTimeout(() => { if (S.confirmRemove === l.name) { S.confirmRemove = null; if (!$("#drawer").hidden) renderLibs(); } }, 4000); return; }
          S.confirmRemove = null; removeLib(l);
        },
      }, icon("remove"), confirming ? "Click again to remove" : "Remove"));
    }
    return el("li", { class: "lib" },
      el("button", {
        type: "button", class: "lib-pick", role: "radio", "aria-checked": String(l.name === S.lib), "data-k": "pick-" + l.name,
        onclick: () => { if (S.lib !== l.name) { S.lib = l.name; store("vl.lib", l.name); clearResults(); render(); if (S.query) requestSearch(); } },
      }, el("span", { class: "radio", "aria-hidden": "true" }), el("span", { class: "lib-title" }, l.name),
      el("span", { class: "lib-state" + (l.state === "error" ? " error" : "") }, stateText)),
      busy ? el("div", { class: "track" + (pct == null ? " indeterminate" : "") }, el("i", { style: pct == null ? "" : `width:${pct}%` })) : null,
      el("p", { class: "lib-roots" }, l.roots.join("\n")),
      el("div", { class: "lib-acts" }, acts));
  }));
  box.setAttribute("role", "radiogroup");
  box.setAttribute("aria-label", "Library to search");
  if (focusKey) { const n = box.querySelector(`[data-k="${CSS.escape(focusKey)}"]`); if (n) n.focus(); }
}

// ---------------------------------------------------------------- settings popover
function openSettings() {
  closeDrawer();
  $("#settings").hidden = false;
  $("#set-btn").setAttribute("aria-expanded", "true");
  setErr("#set-err", "");
  renderSettings();
  const first = $("#settings input:checked") || $("#settings input");
  if (first) first.focus();
}
function closeSettings(restore) {
  if ($("#settings").hidden) return;
  $("#settings").hidden = true;
  $("#set-btn").setAttribute("aria-expanded", "false");
  if (restore) $("#set-btn").focus();
}
function renderSettings() {
  const st = S.status;
  const lib = currentLib();
  $("#model-list").replaceChildren(...(st.models.length ? st.models.map((m) => {
    const ready = !lib || (lib.models_ready || []).includes(m.name);
    return el("label", { class: "model" + (ready ? "" : " off") },
      el("input", { type: "radio", name: "model", value: m.name, checked: m.name === S.model, onchange: () => setModel(m.name) }),
      el("b", {}, shortModel(m.label)),
      el("span", {}, (modelDesc(m.label) ? sentence(modelDesc(m.label)) : "") + (ready ? "" : " Not indexed for this library yet.")));
  }) : [el("p", { class: "note" }, "No models installed.")]));
  const row = $("#autostart-row");
  row.hidden = !st.settings.autostart_supported || S.missing.has("settings");
  $("#autostart").checked = !!st.settings.autostart;
  const about = [
    ["Data folder", st.home], ["Lake", st.lake], ["Models from", st.source],
    ["Text model", st.text_state === "ready" ? (st.clap || "ready") : st.text_state],
  ].filter(([, v]) => v);
  $("#about").replaceChildren(...about.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
}
async function setModel(name) {
  S.model = name;
  store("vl.model", name);
  setErr("#set-err", "");
  if (!S.missing.has("settings")) {
    try { await api("/api/settings", { body: { model: name } }); }
    catch (e) { if (e.missing) S.missing.add("settings"); else setErr("#set-err", e.message); }
  }
  if (S.query) requestSearch();
}
async function setAutostart(on) {
  setErr("#set-err", "");
  try { await api("/api/settings", { body: { autostart: on } }); S.statusKey = ""; pollSoon(); }
  catch (e) { $("#autostart").checked = !on; setErr("#set-err", e.missing ? "This engine can't change start-up settings." : e.message); }
}

// ---------------------------------------------------------------- keyboard + events
function typingTarget(t) { return !!(t && t.closest && t.closest("input, textarea, select, [contenteditable], summary")); }
function bind() {
  window.addEventListener("pointerdown", () => { S.kbd = false; }, true);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Tab" || e.key.startsWith("Arrow")) S.kbd = true;
  }, true);

  $("#pad").addEventListener("click", (e) => {
    if (e.detail === 0 && S.spaceHandled) return; // keyboard Space already handled
    toggleRec();
  });
  $("#take-play").addEventListener("click", () => S.query && play("query", S.query.url));
  $("#take-clear").addEventListener("click", () => {
    stopPlayback(); releaseQuery(); S.query = null; S.waiting = false;
    showNotice(""); syncExamples(); renderTake(); clearResults();
  });
  $("#file-in").addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
  $("#hint").addEventListener("input", (e) => { S.hint = e.target.value; renderChips(); scheduleHint(); });
  $("#hint").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { clearTimeout(S.hintT); requestSearch(); }
    if (e.key === "Escape") e.target.blur();
  });
  $("#lib-btn").addEventListener("click", () => ($("#drawer").hidden ? openDrawer() : closeDrawer(true)));
  $("#drawer-close").addEventListener("click", () => closeDrawer(true));
  $("#add-folder").addEventListener("click", () => pickFolder("drawer"));
  $("#drawer-path").addEventListener("submit", (e) => { e.preventDefault(); usePath("drawer", $("#drawer-path-in").value.trim()); });
  $("#set-btn").addEventListener("click", () => ($("#settings").hidden ? openSettings() : closeSettings(true)));
  $("#autostart").addEventListener("change", (e) => setAutostart(e.target.checked));
  document.addEventListener("pointerdown", (e) => {
    const s = $("#settings");
    if (!s.hidden && !s.contains(e.target) && !$("#set-btn").contains(e.target)) closeSettings();
    const d = $("#drawer");
    if (!d.hidden && !d.contains(e.target) && !$("#lib-btn").contains(e.target)) closeDrawer();
  });

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const mods = e.ctrlKey || e.metaKey || e.altKey;
    if (e.key === "Escape") {
      if (!$("#settings").hidden) return closeSettings(true);
      if (!$("#drawer").hidden) return closeDrawer(true);
      if (S.rec) return stopRecording(true);
      if (S.playing) return stopPlayback();
      return;
    }
    if ($("#app").hidden || mods) return;
    const overlay = !$("#drawer").hidden || !$("#settings").hidden;
    if (e.code === "Space") {
      if (e.repeat) { if (!typingTarget(t)) e.preventDefault(); return; }
      if (overlay || typingTarget(t)) return;
      // A keyboard user on a button/row expects Space to activate it; after a mouse click, focus
      // just lingers there, so Space records.
      const control = t.closest && t.closest("button, a, label");
      if (control && control !== $("#pad") && S.kbd) return;
      e.preventDefault();
      S.spaceHandled = true;
      toggleRec();
      return;
    }
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !overlay && !typingTarget(t) && S.results && S.results.results.length) {
      e.preventDefault();
      const n = S.results.results.length;
      const next = S.sel < 0 ? (e.key === "ArrowDown" ? 0 : n - 1) : S.sel + (e.key === "ArrowDown" ? 1 : -1);
      select(Math.max(0, Math.min(n - 1, next)));
      previewSel();
    }
    if ((e.key === "Home" || e.key === "End") && t.classList && t.classList.contains("row")) {
      e.preventDefault();
      select(e.key === "Home" ? 0 : S.results.results.length - 1);
      previewSel();
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space" && S.spaceHandled) { e.preventDefault(); setTimeout(() => { S.spaceHandled = false; }, 0); }
  });

  // drop an audio file anywhere to search with it (ignores our own result drags: no Files type)
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e) || $("#app").hidden) return; depth++; $("#drop").hidden = false; });
  window.addEventListener("dragleave", (e) => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth) $("#drop").hidden = true; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    $("#drop").hidden = true;
    if (!$("#app").hidden) handleFile(e.dataTransfer.files[0]);
  });

  let resizeT;
  window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => drawPad(), 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawPad());
}

function boot() {
  bind();
  render();
  poll();
}
boot();
