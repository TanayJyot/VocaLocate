/* VocaLocate local web UI. Plain JS, no build step, no network access beyond this origin. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const MAX_REC_SEC = 15;
const PREVIEW_SEC = 5;
const POLL_MS = 700;

const state = {
  status: null,
  library: "",        // selected library name
  model: "metadata",  // selected model name
  hint: "",
  query: null,        // {kind: "audio"|"example", b64?, exampleId?, url, label, samples?}
  last: null,         // {query, library, model, hint} that produced state.results
  results: null,
  busy: false,
  watch: null,        // library name being indexed (progress card)
  timers: new Set(),
  rec: null,          // active recording session
  recStarting: false,
  player: null,       // shared HTMLAudioElement
  playing: null,      // key of what is playing
  playTimer: null,
  view: "",
};

// ---------------------------------------------------------------- utilities
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : String(kid));
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
function setIcon(btn, name) { const u = btn.querySelector("use"); if (u) u.setAttribute("href", "#i-" + name); }
function fmt(n, d = 2) { return Number(n).toFixed(d); }
function num(n) { return Number(n).toLocaleString(); }
function announce(msg) { const l = $("#live"); l.textContent = ""; requestAnimationFrame(() => { l.textContent = msg; }); }
function showError(id, msg) { const e = $(id); e.textContent = msg || ""; e.hidden = !msg; }
function later(fn, ms) {
  const t = setTimeout(() => { state.timers.delete(t); fn(); }, ms);
  state.timers.add(t);
}
function clearTimers() { state.timers.forEach(clearTimeout); state.timers.clear(); }
function modelLabel(name) {
  const m = state.status && state.status.models.find((x) => x.name === name);
  return m ? m.label : name;
}
function currentLib() {
  return state.status ? state.status.libraries.find((l) => l.name === state.library) : null;
}
function audioUrl(lib, id) { return `/api/audio/${encodeURIComponent(lib)}/${id}`; }

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error("Can't reach the VocaLocate server. Is it still running?");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (HTTP ${res.status})`);
  return data;
}

// ---------------------------------------------------------------- audio helpers
let decodeCtx = null;
function getDecodeCtx() {
  if (!decodeCtx) decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  return decodeCtx;
}
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
  const q = state.query;
  if (q && q.kind === "audio" && !(state.last && state.last.query === q)) URL.revokeObjectURL(q.url);
}
function setAudioQuery(samples, rate, label) {
  const wav = encodeWav(samples, rate);
  releaseQuery();
  state.query = {
    kind: "audio", b64: toBase64(wav), samples, label,
    url: URL.createObjectURL(new Blob([wav], { type: "audio/wav" })),
    seconds: samples.length / rate,
  };
  renderQuery();
}

// ---------------------------------------------------------------- playback (one at a time)
function stopPlayback() {
  if (state.player) { state.player.pause(); state.player.removeAttribute("src"); state.player.load(); }
  clearTimeout(state.playTimer);
  state.playing = null;
  syncPlayButtons();
}
function play(key, url, maxSec) {
  if (state.playing === key) return stopPlayback();
  stopPlayback();
  if (!state.player) {
    state.player = new Audio();
    state.player.addEventListener("ended", stopPlayback);
    state.player.addEventListener("error", () => { if (state.playing) { announce("Could not play this file."); stopPlayback(); } });
  }
  state.player.src = url;
  state.playing = key;
  state.player.play().catch(() => { announce("Playback was blocked or failed."); stopPlayback(); });
  if (maxSec) state.playTimer = setTimeout(stopPlayback, maxSec * 1000);
  syncPlayButtons();
}
function syncPlayButtons() {
  document.querySelectorAll("[data-play]").forEach((b) => {
    const on = b.dataset.play === state.playing;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
    setIcon(b, on ? "stop" : "play");
    const t = b.querySelector(".t");
    if (t) t.textContent = on ? "Stop" : "Preview";
  });
}

// ---------------------------------------------------------------- waveform
function drawWave(canvas, samples) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(50, canvas.clientWidth), h = canvas.clientHeight || 48;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue("--border").trim();
  g.fillRect(0, h / 2 - 0.5, w, 1);
  if (!samples || !samples.length) return;
  let peak = 1e-4;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  g.fillStyle = css.getPropertyValue("--accent").trim();
  const step = samples.length / w;
  for (let x = 0; x < w; x++) {
    let lo = 0, hi = 0;
    const end = Math.min(samples.length, Math.floor((x + 1) * step));
    for (let i = Math.floor(x * step); i < end; i++) { const s = samples[i]; if (s < lo) lo = s; if (s > hi) hi = s; }
    const y1 = h / 2 - (hi / peak) * (h / 2 - 2), y2 = h / 2 - (lo / peak) * (h / 2 - 2);
    g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

// ---------------------------------------------------------------- recording
const WORKLET = `class VLCapture extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor("vl-capture", VLCapture);`;

async function startRecording() {
  if (state.rec || state.recStarting) return;
  state.recStarting = true;
  try { await beginRecording(); } finally { state.recStarting = false; }
}
async function beginRecording() {
  showError("#search-error", "");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return showError("#search-error", "This browser can't access the microphone. Try uploading a file instead.");
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) {
    return showError("#search-error", "Microphone access was denied or unavailable. Allow it in the browser, or upload a file.");
  }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  sink.connect(ctx.destination);
  const rec = { ctx, stream, src, sink, chunks: [], length: 0, level: 0, t0: performance.now(), node: null };
  const onChunk = (data) => {
    rec.chunks.push(data);
    rec.length += data.length;
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
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
  state.rec = rec;
  stopPlayback();
  const btn = $("#rec-btn");
  btn.classList.add("on"); btn.setAttribute("aria-pressed", "true"); setIcon(btn, "stop");
  $("#rec-label").textContent = "Stop";
  announce("Recording. Make the sound now.");
  const tick = () => {
    if (state.rec !== rec) return;
    const sec = (performance.now() - rec.t0) / 1000;
    $("#rec-time").textContent = fmt(sec, 1) + " s";
    $("#meter-fill").style.width = Math.min(100, rec.level * 400) + "%";
    rec.level *= 0.9;
    if (sec >= MAX_REC_SEC) return stopRecording();
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}

function stopRecording(discard = false) {
  const rec = state.rec;
  if (!rec) return;
  state.rec = null;
  cancelAnimationFrame(rec.raf);
  try { rec.src.disconnect(); rec.node.disconnect(); } catch (e) { /* already disconnected */ }
  rec.stream.getTracks().forEach((t) => t.stop());
  const rate = rec.ctx.sampleRate;
  rec.ctx.close();
  const btn = $("#rec-btn");
  btn.classList.remove("on"); btn.setAttribute("aria-pressed", "false"); setIcon(btn, "record");
  $("#rec-label").textContent = "Record";
  $("#meter-fill").style.width = "0";
  if (discard) return;
  const samples = new Float32Array(rec.length);
  let o = 0;
  for (const c of rec.chunks) { samples.set(c, o); o += c.length; }
  if (samples.length < rate * 0.2) return showError("#search-error", "That recording was too short. Hold the button a little longer.");
  setAudioQuery(samples, rate, "Your recording");
  announce(`Recorded ${fmt(samples.length / rate, 1)} seconds.`);
}

async function handleUpload(file) {
  if (!file) return;
  showError("#search-error", "");
  try {
    const ab = await decode(await file.arrayBuffer());
    let mono = toMono(ab);
    const cap = Math.floor(ab.sampleRate * 30);
    if (mono.length > cap) mono = mono.slice(0, cap);
    setAudioQuery(mono, ab.sampleRate, file.name);
    announce(`Loaded ${file.name}.`);
  } catch (e) {
    showError("#search-error", `Couldn't read "${file.name}" as audio. Try a WAV, MP3 or OGG file.`);
  }
}

async function chooseExample(ex) {
  stopRecording(true);
  releaseQuery();
  $("#rec-time").textContent = "0.0 s";  // the old recording's length no longer applies
  state.query = { kind: "example", exampleId: ex.id, url: `/api/example/${ex.id}`, label: `Example: ${ex.name}`, samples: null };
  if (ex.hint && state.status.text) { state.hint = ex.hint; $("#hint").value = ex.hint; renderChips(); }
  renderQuery();
  announce(`Selected example ${ex.name}.`);
  try {
    const res = await fetch(state.query.url);
    if (!res.ok) throw new Error();
    const ab = await decode(await res.arrayBuffer());
    if (state.query && state.query.exampleId === ex.id) {
      state.query.samples = toMono(ab);
      state.query.seconds = ab.duration;
      renderQuery();
    }
  } catch (e) { /* waveform is optional */ }
}

// ---------------------------------------------------------------- search
function queryBody(q, library, model, hint) {
  const body = { library, model, text: hint || "", k: 20 };
  if (q.kind === "example") body.example = q.exampleId; else body.audio = q.b64;
  return body;
}
async function runSearch(q, library, model, hint, errId) {
  if (state.busy) return;
  state.busy = true;
  showError(errId, "");
  setBusy(true);
  try {
    const res = await api("/api/search", queryBody(q, library, model, hint));
    state.results = res;
    state.last = { query: q, library, model, hint };
    state.library = library; state.model = model; state.hint = hint;
    announce(`Found ${res.results.length} results in ${fmt(res.ms, 0)} ms.`);
    if (state.view === "results") renderResults(); else location.hash = "#/results";
  } catch (e) {
    showError(errId, e.message);
  } finally {
    state.busy = false;
    setBusy(false);
  }
}
function setBusy(on) {
  const sb = $("#search-btn"), rg = $("#r-go");
  sb.lastChild.textContent = on ? " Searching…" : " Search";
  rg.textContent = on ? "Searching…" : "Update";
  rg.disabled = on;
  updateSearchButton();
}

// ---------------------------------------------------------------- rendering: library view
function renderLibraryList() {
  const box = $("#lib-list");
  box.replaceChildren();
  const libs = state.status ? state.status.libraries : [];
  if (!libs.length) {
    box.append(el("p", { class: "note" }, "No libraries yet. Add a folder below to index it."));
    return;
  }
  for (const lib of libs) {
    const busy = lib.state !== "ready" && lib.state !== "error";
    box.append(el("div", { class: "lib-item" },
      icon("folder"),
      el("div", { class: "grow" },
        el("div", {}, el("strong", {}, lib.name), " ",
          el("span", { class: "badge " + (busy ? "busy" : lib.state) }, lib.state),
          lib.files != null ? el("span", { class: "note" }, ` · ${num(lib.files)} files`) : null),
        el("div", { class: "roots", title: lib.roots.join("\n") }, lib.roots.join(" · "))),
      el("button", { type: "button", class: "btn small ghost", onclick: () => startIndex(lib.name, lib.roots) }, "Re-index"),
      el("button", {
        type: "button", class: "btn small primary", disabled: !lib.models_ready.length,
        onclick: () => { state.library = lib.name; location.hash = "#/search"; },
      }, "Search")));
  }
}

function addRootRow(value = "") {
  const rows = $("#root-rows");
  const n = rows.children.length + 1;
  const id = "root-" + Date.now() + "-" + n;
  const row = el("div", { class: "root-row" },
    el("label", { class: "sr-only", for: id }, `Folder ${n}`),
    el("input", { id, type: "text", value, placeholder: "/absolute/path/to/sounds", autocomplete: "off", spellcheck: "false" }),
    el("button", {
      type: "button", class: "btn ghost small", "aria-label": `Remove folder ${n}`,
      onclick: () => { if (rows.children.length > 1) row.remove(); else row.querySelector("input").value = ""; },
    }, "Remove"));
  rows.append(row);
  return row;
}

async function startIndex(name, roots) {
  showError("#lib-error", "");
  if (!roots.length) return showError("#lib-error", "Add at least one folder path.");
  const rel = roots.find((r) => !(r.startsWith("/") || r.startsWith("~") || /^[A-Za-z]:[\\/]/.test(r)));
  if (rel) return showError("#lib-error", `"${rel}" is not an absolute path. Use the full path, e.g. /Users/me/SFX.`);
  $("#index-btn").disabled = true;
  try {
    const lib = await api("/api/libraries", { name, roots });
    state.watch = lib.name;
    state.library = lib.name;
    announce(`Indexing ${lib.name}.`);
    renderProgress(lib);
    later(pollLibrary, POLL_MS);
  } catch (e) {
    showError("#lib-error", e.message);
  } finally {
    $("#index-btn").disabled = false;
  }
}

async function pollLibrary() {
  if (!state.watch || state.view !== "library") return;
  try {
    const lib = await api(`/api/libraries/${encodeURIComponent(state.watch)}`);
    renderProgress(lib);
    if (lib.state === "ready" || lib.state === "error") {
      announce(lib.state === "ready" ? `${lib.name} is ready to search.` : `Indexing failed: ${lib.message}`);
      await loadStatus();
      renderLibraryList();
      return;
    }
  } catch (e) {
    showError("#lib-error", e.message);
  }
  later(pollLibrary, POLL_MS);
}

function renderProgress(lib) {
  const card = $("#progress");
  card.hidden = false;
  const done = lib.state === "ready", failed = lib.state === "error";
  const kids = [
    el("div", { class: "progress-head" },
      el("h2", {}, `Indexing “${lib.name}”`),
      el("span", { class: "badge " + (done ? "ready" : failed ? "error" : "busy") }, lib.state)),
    el("p", { class: failed ? "error" : "note" }, lib.message || "Waiting to start…",
      ` · ${fmt(lib.seconds, 1)} s elapsed`),
  ];
  if (!done && !failed) kids.push(el("div", { class: "indeterminate", role: "progressbar", "aria-label": "Indexing in progress" }));
  kids.push(el("p", { class: "subhead" }, "Folders"));
  const folders = lib.folders.length ? lib.folders : lib.roots.map((p) => ({ path: p, files: null }));
  kids.push(el("ul", { class: "rows" }, folders.map((f) => el("li", {},
    icon("folder"), el("span", { class: "path", title: f.path }, f.path),
    el("span", { class: "num" }, f.files == null ? "counting…" : `${num(f.files)} audio files`)))));
  const models = Object.entries(lib.summary || {});
  if (models.length) {
    kids.push(el("p", { class: "subhead" }, "Models"));
    kids.push(el("ul", { class: "rows" }, models.map(([m, s]) => el("li", {},
      el("span", { class: "path" }, modelLabel(m)),
      el("span", { class: "num" }, `${num(s.embedded)} embedded · ${num(s.cached)} cached · ${fmt(s.sec, 1)} s`)))));
  }
  if (done) {
    kids.push(el("div", { class: "actions" }, el("button", {
      type: "button", class: "btn primary",
      onclick: () => { state.library = lib.name; location.hash = "#/search"; },
    }, "Search this library")));
  }
  card.replaceChildren(...kids);
}

// ---------------------------------------------------------------- rendering: search view
function fillModelSelect(sel, lib) {
  const ready = lib ? lib.models_ready : [];
  sel.replaceChildren(...state.status.models.map((m) => el("option", {
    value: m.name, selected: m.name === state.model,
  }, m.label + (ready.includes(m.name) ? "" : lib && lib.state !== "ready" ? " (indexing…)" : " (not indexed)"))));
  sel.value = state.model;
}

function renderSearch() {
  clearTimers();
  const st = state.status;
  const libSel = $("#sel-library");
  libSel.replaceChildren(...(st.libraries.length ? st.libraries.map((l) => el("option", { value: l.name }, l.name))
    : [el("option", { value: "" }, "No libraries yet")]));
  libSel.value = state.library;
  const lib = currentLib();
  fillModelSelect($("#sel-model"), lib);
  const note = $("#lib-note");
  note.replaceChildren();
  if (!lib) note.append("Index a library first. ", el("a", { href: "#/library" }, "Go to Library"));
  else if (lib.state === "error") note.append(`Indexing failed: ${lib.message} `, el("a", { href: "#/library" }, "Fix in Library"));
  else if (!lib.models_ready.includes(state.model)) note.append(`Still indexing “${lib.name}” for this model: ${lib.message}`);
  else note.append(`${num(lib.files)} files in ${lib.roots.length} folder${lib.roots.length === 1 ? "" : "s"}, ready to search.`);
  if (lib && lib.state !== "ready" && lib.state !== "error") later(refreshSearchStatus, 1500);

  const list = $("#example-list");
  $("#examples").hidden = !st.examples.length;
  list.replaceChildren(...st.examples.map((ex) => el("li", {}, el("button", {
    type: "button", "aria-pressed": String(!!(state.query && state.query.exampleId === ex.id)),
    onclick: () => chooseExample(ex),
  }, el("span", { class: "ex-name", title: ex.name }, ex.name),
  el("span", { class: "ex-meta" }, [ex.category, ex.hint && `hint: ${ex.hint}`].filter(Boolean).join(" · ") || "imitation")))));

  const hint = $("#hint");
  hint.value = state.hint;
  hint.disabled = !st.text;
  $("#hint-note").textContent = st.text ? "The hint is scored with CLAP and blended with the voice match. Click a suggestion to add or remove it."
    : "Text hints are off: this app was started without CLAP. Restart it with --text to search with hints.";
  renderChips();
  renderQuery();
}

async function refreshSearchStatus() {
  if (state.view !== "search") return;
  try { await loadStatus(); renderSearch(); } catch (e) { later(refreshSearchStatus, 3000); }
}

function renderChips() {
  const words = state.hint.toLowerCase().split(/[\s,]+/).filter(Boolean);
  $("#hint-chips").replaceChildren(...state.status.hints.map((h) => el("button", {
    type: "button", class: "chip", disabled: !state.status.text, "aria-pressed": String(words.includes(h)),
    onclick: () => toggleHint(h),
  }, h)));
}
function toggleHint(h) {
  let words = state.hint.split(/[\s,]+/).filter(Boolean);
  words = words.some((w) => w.toLowerCase() === h) ? words.filter((w) => w.toLowerCase() !== h) : words.concat(h);
  state.hint = words.join(" ");
  $("#hint").value = state.hint;
  renderChips();
}

function renderQuery() {
  const q = state.query;
  $("#query-preview").hidden = !q;
  document.querySelectorAll("#example-list button").forEach((b, i) => {
    const ex = state.status.examples[i];
    b.setAttribute("aria-pressed", String(!!(q && ex && q.exampleId === ex.id)));
  });
  if (q) {
    $("#q-desc").textContent = q.label + (q.seconds ? ` · ${fmt(q.seconds, 1)} s` : "");
    $("#q-play").dataset.play = "query";
    requestAnimationFrame(() => drawWave($("#q-wave"), q.samples));
  }
  syncPlayButtons();
  updateSearchButton();
}

function updateSearchButton() {
  const lib = currentLib();
  $("#search-btn").disabled = state.busy || !state.query || !lib || !lib.models_ready.includes(state.model);
}

// ---------------------------------------------------------------- rendering: results view
function renderResults() {
  const r = state.results;
  $("#results-empty").hidden = !!r;
  $("#results-body").hidden = !r;
  if (!r) return;
  const last = state.last;
  const qd = $("#r-query");
  qd.replaceChildren(...[last.query.label,
    r.text ? el("span", { class: "pill" }, `hint: ${r.text}`) : null,
    el("span", { class: "pill" }, r.model_label)].filter(Boolean));
  const reused = r.reused.length ? ` · reused cached ${r.reused.join(" and ")}` : "";
  const folders = r.folders ? r.folders.length : 0;
  $("#r-stats").textContent = `Searched ${num(r.searched)} files in ${folders} folder${folders === 1 ? "" : "s"} of “${r.library}” in ${fmt(r.ms, 0)} ms` +
    (r.alpha != null ? ` · voice weight ${fmt(r.alpha, 2)}` : "") + reused;
  $("#r-play").dataset.play = "query";
  fillModelSelect($("#r-model"), state.status.libraries.find((l) => l.name === r.library));
  $("#r-model").value = r.model;
  const rh = $("#r-hint");
  rh.value = r.text;
  rh.disabled = !state.status.text;
  rh.placeholder = state.status.text ? "No hint" : "Text hints unavailable (no CLAP)";

  const list = $("#result-list");
  if (!r.results.length) {
    list.replaceChildren(el("li", { class: "card" }, "No matches. Try another imitation or a different model."));
    return;
  }
  const scores = r.results.map((x) => x.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  list.replaceChildren(...r.results.map((h) => resultRow(h, r.library, hi > lo ? (h.score - lo) / (hi - lo) : 1)));
  syncPlayButtons();
}

function resultRow(h, library, rel) {
  const url = audioUrl(library, h.id);
  const abs = new URL(url, location.href).href;
  const key = "r" + h.id;
  const split = h.text_score != null
    ? el("div", { class: "sim-split" }, `voice ${fmt(h.voice_score)} · text ${fmt(h.text_score)}`) : null;
  const row = el("li", {
    class: "result", draggable: "true", title: "Drag into your editor",
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
  el("span", { class: "handle", "aria-hidden": "true" }, icon("drag")),
  el("span", { class: "rank" }, h.rank),
  el("div", { class: "r-main" },
    el("div", { class: "r-name", title: h.name }, h.name,
      h.project ? el("span", { class: "proj" }, icon("folder"), h.project) : null),
    el("div", { class: "r-path", title: h.path }, h.path)),
  el("div", { class: "sim", "aria-label": `Similarity ${fmt(h.score)}` },
    el("div", { class: "sim-top" }, el("span", {}, "similarity"), el("b", {}, fmt(h.score, 3))),
    el("div", { class: "bar" }, el("div", { style: `width:${Math.round(8 + 92 * rel)}%` })),
    split),
  el("div", { class: "r-actions" },
    el("button", {
      type: "button", class: "btn small", "data-play": key, "aria-pressed": "false",
      "aria-label": `Preview ${h.name}`, onclick: () => play(key, url, PREVIEW_SEC),
    }, icon("play"), el("span", { class: "t" }, "Preview")),
    el("button", {
      type: "button", class: "btn small icon", title: "Copy path", "aria-label": `Copy path of ${h.name}`,
      onclick: () => copyText(h.path),
    }, icon("copy"))));
  return row;
}

async function copyText(text) {
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
  announce("Path copied to the clipboard.");
}

// ---------------------------------------------------------------- routing
async function loadStatus() {
  state.status = await api("/api/status");
  const st = state.status;
  if (!st.libraries.some((l) => l.name === state.library)) {
    const pick = st.libraries.find((l) => l.models_ready.length) || st.libraries[0];
    state.library = pick ? pick.name : "";
  }
  if (!st.models.some((m) => m.name === state.model) && st.models.length) state.model = st.models[0].name;
  const f = $("#foot");
  f.textContent = `Models trained on ${st.source} · lake ${st.lake}` + (st.clap ? ` · CLAP ${st.clap}` : "");
}

function route() {
  const view = (location.hash.match(/^#\/(\w+)/) || [])[1];
  const v = ["library", "search", "results"].includes(view) ? view : "search";
  clearTimers();
  // Status messages belong to the page that produced them.
  $("#live").textContent = "";
  $("#live").classList.remove("error");
  if (v !== "search") stopRecording(true);
  stopPlayback();
  state.view = v;
  for (const name of ["library", "search", "results"]) {
    $("#view-" + name).hidden = name !== v;
    const a = document.querySelector(`[data-nav="${name}"]`);
    if (name === v) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  if (!state.status) return;
  if (v === "library") {
    renderLibraryList();
    if (state.watch) later(pollLibrary, 0);
  } else if (v === "search") renderSearch();
  else renderResults();
  const h = $("#view-" + v + " h1");
  if (h && document.activeElement && document.activeElement.closest(".tabs")) { h.tabIndex = -1; h.focus(); }
}

function bind() {
  window.addEventListener("hashchange", route);
  $("#add-root").addEventListener("click", () => addRootRow().querySelector("input").focus());
  $("#lib-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const name = $("#lib-name").value.trim() || "default";
    const roots = [...document.querySelectorAll("#root-rows input")].map((i) => i.value.trim()).filter(Boolean);
    startIndex(name, roots);
  });
  $("#sel-library").addEventListener("change", (e) => { state.library = e.target.value; renderSearch(); });
  $("#sel-model").addEventListener("change", (e) => { state.model = e.target.value; renderSearch(); });
  $("#rec-btn").addEventListener("click", () => (state.rec ? stopRecording() : startRecording()));
  $("#file-in").addEventListener("change", (e) => { handleUpload(e.target.files[0]); e.target.value = ""; });
  $("#q-play").addEventListener("click", () => state.query && play("query", state.query.url));
  $("#q-clear").addEventListener("click", () => {
    stopPlayback();
    releaseQuery();
    state.query = null;
    renderQuery();
  });
  $("#hint").addEventListener("input", (e) => { state.hint = e.target.value; renderChips(); });
  $("#search-btn").addEventListener("click", () => {
    if (state.query) runSearch(state.query, state.library, state.model, state.hint.trim(), "#search-error");
  });
  $("#refine-btn").addEventListener("click", () => {
    if (state.last) { state.query = state.last.query; state.library = state.last.library; }
    location.hash = "#/search";
  });
  $("#r-play").addEventListener("click", () => state.last && play("query", state.last.query.url));
  const rerun = () => {
    if (!state.last) return;
    const r = state.results;
    // Picking a model already reruns the search; skip a second identical run from "Update".
    if (r && r.model === $("#r-model").value && (r.text || "") === $("#r-hint").value.trim()) return;
    runSearch(state.last.query, state.last.library, $("#r-model").value, $("#r-hint").value.trim(), "#r-error");
  };
  $("#r-go").addEventListener("click", rerun);
  $("#r-model").addEventListener("change", rerun);
  $("#r-hint").addEventListener("keydown", (e) => { if (e.key === "Enter") rerun(); });
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || state.view !== "search" || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest("input, textarea, select, button, summary, a, [contenteditable]")) return;
    e.preventDefault();
    if (state.rec) stopRecording(); else startRecording();
  });
  let resizeT;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (state.query && state.view === "search") drawWave($("#q-wave"), state.query.samples); }, 150);
  });
}

async function boot() {
  bind();
  addRootRow();
  const wasEmpty = !location.hash || location.hash === "#" || location.hash === "#/";
  route();
  try {
    await loadStatus();
  } catch (e) {
    announce(e.message);
    $("#live").classList.add("error");
    return;
  }
  const st = state.status;
  const busy = st.libraries.find((l) => l.state !== "ready" && l.state !== "error");
  if (busy) state.watch = busy.name;
  if (!st.libraries.length) {
    if (wasEmpty || state.view !== "library") { location.hash = "#/library"; return; }
  } else if (wasEmpty && state.view !== "search") {
    location.hash = "#/search"; return;
  }
  route();
}

boot();
