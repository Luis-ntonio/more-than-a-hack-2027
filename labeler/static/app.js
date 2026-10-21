// Possession labeller. Labels are stored as change points [frame, label]: a label holds from its
// frame until the next change, so a 5-minute block takes a few dozen keystrokes, not 7500.
"use strict";

const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const COLORS = {};
for (const k of ["H", "A", "D", "X"]) {
  COLORS[k] = getComputedStyle(document.documentElement).getPropertyValue("--" + k).trim();
}
const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];

const S = {
  config: null, fps: 25, block: null, annotator: "",
  changes: [], done: false, notes: "",
  cur: 0, shown: -1, undo: [], redo: [],
  dirty: false, saving: false, saveTimer: null,
  playing: false, rate: 1,
  zoomFrames: 375,                       // half-width of the zoom strip, in frames
  ctx: { half: 4, speed: 1, start: 0, end: 0, center: 0, pos: -1, timer: null },
  gt: null, showGt: false,
};

const main = $("#main");
const ctxVideo = $("#ctx");

// -- label model ---------------------------------------------------------------------------------

function indexAtOrBefore(f) {
  let lo = 0, hi = S.changes.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (S.changes[mid][0] <= f) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}
function labelAt(f, changes = S.changes) {
  if (changes === S.changes) {
    const i = indexAtOrBefore(f);
    return i < 0 ? "X" : S.changes[i][1];
  }
  let label = "X";
  for (const [frame, l] of changes) { if (frame <= f) label = l; else break; }
  return label;
}
function changeAt(f) {
  const i = indexAtOrBefore(f);
  return i >= 0 && S.changes[i][0] === f ? i : -1;
}
function normalize(changes) {
  changes.sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const c of changes) {
    if (out.length && out[out.length - 1][0] === c[0]) out[out.length - 1] = c;
    else out.push(c);
  }
  return out.filter((c, i) => i === 0 || c[1] !== out[i - 1][1]);
}
function segments(changes, first, last) {
  const segs = [];
  let from = first, label = "X";
  for (const [frame, l] of changes) {
    if (frame > last) break;
    if (frame > from) segs.push({ from, to: frame - 1, label });
    from = Math.max(frame, first); label = l;
  }
  segs.push({ from, to: last, label });
  return segs;
}

function edit(mutator) {
  S.undo.push(JSON.stringify(S.changes));
  if (S.undo.length > 500) S.undo.shift();
  S.redo = [];
  mutator();
  S.changes = normalize(S.changes);
  markDirty();
  render();
}
function setLabel(label) {
  if (!S.block) return;
  const f = S.cur;
  edit(() => {
    const i = changeAt(f);
    if (i >= 0) S.changes[i][1] = label; else S.changes.push([f, label]);
  });
}
function deleteChange() {
  const i = changeAt(S.cur);
  if (i < 0) return toast("Colócate sobre un cambio ([ o ]) para borrarlo");
  edit(() => S.changes.splice(i, 1));
}
function moveChange(delta) {
  const i = changeAt(S.cur);
  if (i < 0) return toast("Colócate sobre un cambio ([ o ]) para moverlo");
  const lo = i > 0 ? S.changes[i - 1][0] + 1 : S.block.first;
  const hi = i < S.changes.length - 1 ? S.changes[i + 1][0] - 1 : S.block.last;
  const target = clamp(S.cur + delta, lo, hi);
  if (target === S.cur) return;
  edit(() => { S.changes[i][0] = target; });
  seek(target);
}
function undo() {
  if (!S.undo.length) return;
  S.redo.push(JSON.stringify(S.changes));
  S.changes = JSON.parse(S.undo.pop());
  markDirty(); render();
}
function redo() {
  if (!S.redo.length) return;
  S.undo.push(JSON.stringify(S.changes));
  S.changes = JSON.parse(S.redo.pop());
  markDirty(); render();
}

// -- navigation ----------------------------------------------------------------------------------

function seek(f) {
  if (!S.block) return;
  f = clamp(Math.round(f), S.block.first, S.block.last);
  if (S.playing) pause();
  S.cur = f;
  main.currentTime = (f + 0.5) / S.fps;  // the middle of the frame: robust to float rounding
  render();
  scheduleContext();
}
function step(n) { seek(S.cur + n); }
function jumpChange(dir) {
  if (!S.changes.length) return;
  const i = indexAtOrBefore(S.cur);
  let target;
  if (dir < 0) target = changeAt(S.cur) >= 0 ? S.changes[i - 1] : S.changes[i];
  else target = S.changes[i + 1];
  if (target) seek(target[0]);
}
function nextGap() {
  for (const seg of segments(S.changes, S.block.first, S.block.last)) {
    if (seg.label === "X" && seg.to >= S.cur) return seek(Math.max(seg.from, S.cur));
  }
  for (const seg of segments(S.changes, S.block.first, S.block.last)) {
    if (seg.label === "X") return seek(seg.from);
  }
  toast("Todo el bloque está etiquetado");
}

function play() {
  if (!S.block) return;
  if (S.cur >= S.block.last) return;
  S.playing = true;
  ctxVideo.pause();
  main.playbackRate = S.rate;
  main.play().catch(() => { S.playing = false; render(); });
  render();
}
function pause() {
  S.playing = false;
  main.pause();
  // snap to the frame on screen so the next keystroke labels exactly what you saw
  if (S.shown >= 0 && S.block) S.cur = clamp(S.shown, S.block.first, S.block.last);
  main.currentTime = (S.cur + 0.5) / S.fps;
  render();
  scheduleContext();
}
function setRate(dir) {
  const i = clamp(SPEEDS.indexOf(S.rate) + dir, 0, SPEEDS.length - 1);
  S.rate = SPEEDS[i];
  main.playbackRate = S.rate;
  $("#rate").textContent = S.rate + "×";
  toast("Velocidad " + S.rate + "×");
}

// The frame actually on screen: rVFC gives the presentation time of the composited frame.
function trackFrames(video, onFrame) {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const cb = (_, meta) => { onFrame(Math.round(meta.mediaTime * S.fps)); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    const loop = () => { onFrame(Math.floor(video.currentTime * S.fps)); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
}
trackFrames(main, (f) => {
  S.shown = f;
  if (S.playing) {
    if (f >= S.block.last) { pause(); seek(S.block.last); return; }
    S.cur = f;
  }
  renderLive();
});
main.addEventListener("seeked", renderLive);

// -- context clip --------------------------------------------------------------------------------

function scheduleContext() {
  clearTimeout(S.ctx.timer);
  S.ctx.timer = setTimeout(startContext, 180);
}
function startContext() {
  if (S.playing || !S.block) return;
  const half = Math.round(S.ctx.half * S.fps);
  S.ctx.center = S.cur;
  S.ctx.start = Math.max(0, S.cur - half);
  S.ctx.end = Math.min(S.config.n_frames - 1, S.cur + half);
  ctxVideo.playbackRate = S.ctx.speed;
  ctxVideo.currentTime = (S.ctx.start + 0.5) / S.fps;
  ctxVideo.play().catch(() => {});
}
trackFrames(ctxVideo, (f) => {
  const c = S.ctx;
  if (!ctxVideo.paused && f >= c.end) { ctxVideo.currentTime = (c.start + 0.5) / S.fps; return; }
  const crossed = c.pos >= 0 && c.pos < c.center && f >= c.center;
  c.pos = f;
  if (crossed) {
    $("#ctxbox").classList.add("flash");
    setTimeout(() => $("#ctxbox").classList.remove("flash"), 250);
  }
  const label = labelAt(f);
  const badge = $("#ctxlabel");
  badge.className = "badge " + label;
  badge.textContent = label === "X" ? "—" : label;
  const rel = (f - c.center) / S.fps;
  $("#ctxrel").textContent = (rel >= 0 ? "+" : "−") + Math.abs(rel).toFixed(1) + " s";
  drawCtxBar();
});

function segButtons(id, onPick) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    for (const x of $(id).children) x.classList.toggle("on", x === b);
    onPick(parseFloat(b.dataset.v));
    b.blur();
  });
}
segButtons("#ctxwin", (v) => { S.ctx.half = v; startContext(); });
segButtons("#ctxspeed", (v) => { S.ctx.speed = v; ctxVideo.playbackRate = v; });

// -- drawing -------------------------------------------------------------------------------------

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w: canvas.clientWidth, h: canvas.clientHeight };
}
function hatch(g) {
  if (g._hatch) return g._hatch;
  const p = document.createElement("canvas");
  p.width = p.height = 8;
  const pg = p.getContext("2d");
  pg.fillStyle = COLORS.X; pg.fillRect(0, 0, 8, 8);
  pg.strokeStyle = "#3a404c"; pg.beginPath(); pg.moveTo(0, 8); pg.lineTo(8, 0); pg.stroke();
  return (g._hatch = g.createPattern(p, "repeat"));
}
function drawTrack(g, changes, from, to, x, y, h, w) {
  for (const seg of segments(changes, from, to)) {
    const x0 = x(seg.from), x1 = x(seg.to + 1);
    g.fillStyle = seg.label === "X" ? hatch(g) : COLORS[seg.label];
    g.fillRect(x0, y, Math.max(1, x1 - x0), h);
  }
}
function drawTimeline(canvas, from, to, opts) {
  const { g, w, h } = fitCanvas(canvas);
  const span = to - from + 1;
  const x = (f) => ((f - from) / span) * w;
  g.clearRect(0, 0, w, h);
  const gtH = S.showGt && S.gt ? Math.round(h * 0.28) : 0;
  const trackH = h - gtH - (opts.ticks ? 14 : 0);
  // frames outside the block are not part of the task: draw them dark
  g.fillStyle = "#0a0b0e"; g.fillRect(0, 0, w, h);
  const bf = Math.max(from, S.block.first), bl = Math.min(to, S.block.last);
  if (bl >= bf) {
    drawTrack(g, S.changes, bf, bl, x, 0, trackH, w);
    if (gtH) {
      drawTrack(g, S.gt, bf, bl, x, trackH + 2, gtH - 2, w);
      g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(2, trackH + 3, 22, gtH - 4);
      g.fillStyle = "#fff"; g.font = "10px system-ui"; g.fillText("GT", 5, trackH + gtH - 4);
    }
  }
  // change-point diamonds
  g.fillStyle = "#fff";
  for (const [f] of S.changes) {
    if (f < from || f > to) continue;
    const cx = x(f);
    g.fillRect(cx - 0.5, 0, 1, trackH);
    if (opts.diamonds) {
      g.beginPath(); g.moveTo(cx, 1); g.lineTo(cx + 4, 5); g.lineTo(cx, 9); g.lineTo(cx - 4, 5); g.fill();
    }
  }
  if (opts.ticks) {
    g.fillStyle = "#8b93a3"; g.font = "10px system-ui";
    const pxPerFrame = w / span;
    if (pxPerFrame >= 4) {
      g.fillStyle = "#3a404c";
      for (let f = Math.ceil(from); f <= to; f++) g.fillRect(x(f), trackH, 1, 3);
    }
    g.fillStyle = "#8b93a3";
    const first = Math.ceil(from / S.fps) * S.fps;
    const every = span / S.fps > 40 ? 5 : 1;
    for (let f = first; f <= to; f += S.fps) {
      const sec = Math.round((f - S.cur) / S.fps);
      if (sec % every) continue;
      g.fillRect(x(f), trackH, 1, 6);
      g.fillText((sec > 0 ? "+" : "") + sec + "s", x(f) + 2, h - 2);
    }
  }
  // playhead
  const px = x(S.cur + 0.5);
  g.fillStyle = "#f5c518";
  g.fillRect(px - 1, 0, 2, h);
  if (opts.ticks) {
    const fw = Math.max(2, w / span);
    g.strokeStyle = "#f5c518"; g.strokeRect(x(S.cur) + 0.5, 0.5, fw - 1, trackH - 1);
  }
}
function drawCtxBar() {
  const canvas = $("#ctxbar");
  const c = S.ctx;
  if (!S.block || c.end <= c.start) return;
  const { g, w, h } = fitCanvas(canvas);
  const span = c.end - c.start + 1;
  const x = (f) => ((f - c.start) / span) * w;
  g.fillStyle = "#0a0b0e"; g.fillRect(0, 0, w, h);
  const bf = Math.max(c.start, S.block.first), bl = Math.min(c.end, S.block.last);
  if (bl >= bf) drawTrack(g, S.changes, bf, bl, x, 0, h, w);
  g.fillStyle = "#fff"; g.fillRect(x(c.center + 0.5) - 1, 0, 2, h);
  if (c.pos >= c.start) {
    g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(x(c.pos), 0, w - x(c.pos), h);
  }
}

function clock(f) {
  const half = f >= S.config.kickoffs[2] - 10 * 60 * S.fps ? 2 : 1;
  const t = (f - S.config.kickoffs[half]) / S.fps + (half === 2 ? 45 * 60 : 0);
  const sign = t < 0 ? "−" : "";
  const a = Math.abs(t), m = Math.floor(a / 60), s = Math.floor(a % 60), fr = Math.round((a % 1) * S.fps);
  return `${half}T ${sign}${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}<span class="muted">.${String(fr).padStart(2, "0")}</span>`;
}

function renderLive() {
  if (!S.block) return;
  const label = labelAt(S.cur);
  $("#clock").innerHTML = clock(S.cur);
  $("#frameinfo").textContent = `frame ${S.cur} · ${S.cur - S.block.first + 1}/${S.block.last - S.block.first + 1} del bloque`;
  const badge = $("#curlabel");
  badge.className = "badge big " + label;
  badge.textContent = label === "X" ? "sin etiqueta" : label;
  $("#onchange").classList.toggle("hidden", changeAt(S.cur) < 0);
  $("#loading").classList.toggle("hidden", S.playing || S.shown === S.cur);
  $("#playing").classList.toggle("hidden", !S.playing);
  $("#playbtn").textContent = S.playing ? "❚❚" : "▶";
  drawTimeline($("#overview"), S.block.first, S.block.last, { diamonds: true });
  const zf = S.zoomFrames;
  drawTimeline($("#zoom"), S.cur - zf, S.cur + zf, { ticks: true, diamonds: true });
  $("#zoomtxt").textContent = "±" + Math.round(zf / S.fps) + " s";
  if (S.playing) highlightChange();
}
function render() {
  if (!S.block) return;
  renderLive();
  const segs = segments(S.changes, S.block.first, S.block.last);
  const labelled = segs.filter((s) => s.label !== "X").reduce((n, s) => n + s.to - s.from + 1, 0);
  $("#progress").textContent = Math.floor((100 * labelled) / (S.block.last - S.block.first + 1)) + " % etiquetado";
  renderChangeList();
  drawCtxBar();
}
function renderChangeList() {
  const list = $("#changes");
  list.innerHTML = "";
  S.changes.forEach(([f, label], i) => {
    const next = i + 1 < S.changes.length ? S.changes[i + 1][0] : S.block.last + 1;
    const li = document.createElement("li");
    li.dataset.frame = f;
    li.innerHTML = `<span class="badge ${label}">${label}</span><span>${clock(f)}</span>` +
      `<span class="dur">${((next - f) / S.fps).toFixed(1)} s</span>`;
    li.addEventListener("click", () => seek(f));
    list.appendChild(li);
  });
  $("#nchanges").textContent = "(" + S.changes.length + ")";
  highlightChange();
}
function highlightChange() {
  const i = indexAtOrBefore(S.cur);
  const items = $("#changes").children;
  for (let k = 0; k < items.length; k++) items[k].classList.toggle("cur", k === i);
  if (i >= 0 && items[i]) items[i].scrollIntoView({ block: "nearest" });
}

// -- timeline pointer ----------------------------------------------------------------------------

// The range is frozen at pointerdown: the zoom strip is centred on the playhead, so reading it live
// while dragging would make the target run away from the pointer.
function bindScrub(canvas, range) {
  let frozen = null;
  const toFrame = (e) => {
    const r = canvas.getBoundingClientRect();
    const [from, to] = frozen;
    return Math.floor(from + ((e.clientX - r.left) / r.width) * (to - from + 1));
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!S.block) return;
    frozen = range(); canvas.setPointerCapture(e.pointerId); seek(toFrame(e));
  });
  canvas.addEventListener("pointermove", (e) => { if (frozen) seek(toFrame(e)); });
  canvas.addEventListener("pointerup", () => { frozen = null; });
}
bindScrub($("#overview"), () => [S.block.first, S.block.last]);
bindScrub($("#zoom"), () => [S.cur - S.zoomFrames, S.cur + S.zoomFrames]);
$("#zoom").addEventListener("wheel", (e) => {
  e.preventDefault();
  S.zoomFrames = clamp(Math.round(S.zoomFrames * (e.deltaY > 0 ? 1.25 : 0.8)), 25, 3000);
  renderLive();
}, { passive: false });

// -- persistence ---------------------------------------------------------------------------------

function setSaveStatus(text, cls) {
  const el = $("#save");
  el.textContent = text;
  el.className = "pill " + (cls || "muted");
}
function markDirty() {
  S.dirty = true;
  setSaveStatus("cambios sin guardar");
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 800);
}
async function save() {
  if (!S.dirty || !S.block || !S.annotator) {
    if (!S.annotator && S.dirty) setSaveStatus("escribe tu nombre para guardar", "err");
    return;
  }
  if (S.saving) { clearTimeout(S.saveTimer); S.saveTimer = setTimeout(save, 400); return; }
  S.saving = true;
  S.dirty = false;
  setSaveStatus("guardando…");
  try {
    const r = await fetch("/api/labels", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ block: S.block.id, annotator: S.annotator, changes: S.changes,
        done: S.done, notes: S.notes }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.status);
    setSaveStatus("guardado " + body.updated.slice(11), "ok");
  } catch (err) {
    S.dirty = true;
    setSaveStatus("error al guardar: " + err.message, "err");
  } finally {
    S.saving = false;
  }
}
async function flush() {
  clearTimeout(S.saveTimer);
  await save();
}
window.addEventListener("beforeunload", (e) => { if (S.dirty) { save(); e.preventDefault(); e.returnValue = ""; } });

async function loadBlock(id) {
  await flush();
  const block = S.config.blocks.find((b) => b.id === id);
  if (!block) return;
  S.block = block;
  S.undo = []; S.redo = [];
  S.gt = null; S.showGt = false;
  $("#showgt").checked = false;
  $("#compareout").textContent = "";
  $("#publicpanel").hidden = !block.public;
  S.changes = []; S.done = false; S.notes = "";
  if (S.annotator) {
    const r = await fetch(`/api/labels?block=${id}&annotator=${encodeURIComponent(S.annotator)}`);
    if (r.ok) {
      const doc = await r.json();
      S.changes = normalize(doc.changes || []);
      S.done = !!doc.done; S.notes = doc.notes || "";
    }
  }
  $("#done").checked = S.done;
  $("#notes").value = S.notes;
  S.dirty = false;
  setSaveStatus("sin cambios");
  localStore("block", id);
  // resume where the work stops: first unlabelled frame, or the block start
  const gap = segments(S.changes, block.first, block.last).find((s) => s.label === "X");
  seek(gap ? gap.from : block.first);
}

async function refreshBlockList() {
  const status = await (await fetch("/api/status")).json();
  const sel = $("#block");
  const current = sel.value;
  sel.innerHTML = "";
  for (const b of S.config.blocks) {
    const rows = status[b.id] || [];
    const mine = rows.find((r) => r.annotator === S.annotator);
    const others = rows.filter((r) => r.annotator !== S.annotator).map((r) => r.annotator + (r.done ? "✓" : ""));
    let tag = mine ? (mine.done ? " ✓" : ` · ${Math.round(mine.coverage * 100)} %`) : "";
    if (others.length) tag += "  [" + others.join(", ") + "]";
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = `${b.id}  min ${b.minute_from}–${b.minute_to}${b.public ? "  (GT pública)" : ""}${tag}`;
    sel.appendChild(opt);
  }
  if (current) sel.value = current;
}

function localStore(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("labeler." + key);
    localStorage.setItem("labeler." + key, value);
  } catch (_) { return null; }
}

// -- public-GT practice --------------------------------------------------------------------------

$("#compare").addEventListener("click", async (e) => {
  e.target.blur();
  if (!S.annotator) return toast("Escribe tu nombre primero");
  await flush();
  const r = await fetch(`/api/compare?block=${S.block.id}&annotator=${encodeURIComponent(S.annotator)}`);
  const m = await r.json();
  if (!r.ok) return toast(m.error);
  $("#compareout").innerHTML =
    `spellF1 <b>${m.spell_f1_mean.toFixed(3)}</b> · macroF1 <b>${m.macro_f1.toFixed(3)}</b><br>` +
    `vivo/muerto F1 ${m.alive_dead_f1.toFixed(3)}<br>` +
    `posesiones: tú ${m.n_pred_spells} · GT ${m.n_gt_spells}` +
    (m.n_abstain_on_graded ? `<br>${m.n_abstain_on_graded} frames calificados sin etiquetar` : "");
});
$("#showgt").addEventListener("change", async (e) => {
  e.target.blur();
  S.showGt = e.target.checked;
  if (S.showGt && !S.gt) {
    const r = await fetch(`/api/compare?block=${S.block.id}&annotator=${encodeURIComponent(S.annotator || "x")}&reveal=1`);
    S.gt = (await r.json()).gt_changes || [];
  }
  render();
});

// -- inputs --------------------------------------------------------------------------------------

$("#block").addEventListener("change", (e) => { loadBlock(e.target.value); e.target.blur(); });
$("#annotator").addEventListener("change", async (e) => {
  const name = e.target.value.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-").slice(0, 32);
  e.target.value = name;
  if (name === S.annotator) return;
  await flush();
  S.annotator = name;
  localStore("annotator", name);
  await refreshBlockList();
  if (S.block) loadBlock(S.block.id);
  e.target.blur();
});
$("#done").addEventListener("change", (e) => {
  S.done = e.target.checked; markDirty(); e.target.blur();
  if (S.done) refreshBlockList();
});
$("#notes").addEventListener("input", (e) => { S.notes = e.target.value; markDirty(); });

document.querySelector(".controls").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  b.blur();
  ({
    label: () => setLabel(b.dataset.label),
    step: () => step(parseInt(b.dataset.n, 10)),
    play: () => (S.playing ? pause() : play()),
    prevchange: () => jumpChange(-1),
    nextchange: () => jumpChange(1),
    delete: deleteChange,
    undo: undo,
    nextgap: nextGap,
  })[b.dataset.act]();
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (!S.block) return;
  const k = e.key.toLowerCase();
  const fine = e.shiftKey ? 5 : 1;
  let handled = true;
  if ((e.ctrlKey || e.metaKey) && k === "z") e.shiftKey ? redo() : undo();
  else if ((e.ctrlKey || e.metaKey) && k === "y") redo();
  else if (e.ctrlKey || e.metaKey) handled = false;
  else if (k === "h" || k === "a" || k === "d") setLabel(k.toUpperCase());
  else if (k === " ") S.playing ? pause() : play();
  else if (e.altKey && k === "arrowleft") moveChange(-fine);
  else if (e.altKey && k === "arrowright") moveChange(fine);
  else if (k === "arrowleft") step(e.shiftKey ? -S.fps : -1);
  else if (k === "arrowright") step(e.shiftKey ? S.fps : 1);
  else if (k === "arrowup") step(5 * S.fps);
  else if (k === "arrowdown") step(-5 * S.fps);
  else if (k === "[") jumpChange(-1);
  else if (k === "]") jumpChange(1);
  else if (k === "delete" || k === "backspace") deleteChange();
  else if (k === "n") nextGap();
  else if (k === "c") startContext();
  else if (k === "-") setRate(-1);
  else if (k === "+" || k === "=") setRate(1);
  else handled = false;
  if (handled) e.preventDefault();
});

let toastTimer;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 1800);
}
window.addEventListener("resize", () => render());

// -- boot ----------------------------------------------------------------------------------------

(async function boot() {
  S.config = await (await fetch("/api/config")).json();
  S.fps = S.config.fps;
  $("#game").textContent = S.config.game;
  if (!S.config.video_available) toast("No se encuentra el vídeo táctico en data/");
  main.src = ctxVideo.src = "/video";
  S.annotator = localStore("annotator") || "";
  $("#annotator").value = S.annotator;
  await refreshBlockList();
  const saved = localStore("block");
  const first = S.config.blocks.find((b) => b.id === saved) || S.config.blocks.find((b) => !b.public) || S.config.blocks[0];
  $("#block").value = first.id;
  await loadBlock(first.id);
  if (!S.annotator) { toast("Escribe tu nombre de anotador para guardar"); $("#annotator").focus(); }
})();
