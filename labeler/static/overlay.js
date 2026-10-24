"use strict";
// Tracking visualiser: the tactical video with the pitch, players and balls projected on top.
// Positions arrive on the normalised pitch (0-1) with each frame's homography; detector candidates
// arrive in video pixels. Everything is drawn in video-pixel space on a canvas over the <video>.

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const CHUNK = 250;
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const LAYERS = [
  { id: "pitch", name: "Cancha", color: "#ffffff" },
  { id: "home", name: "Jugadores Barça", color: css("--H") },
  { id: "away", name: "Jugadores Athletic", color: css("--A") },
  { id: "slots", name: "Número de slot", color: "#c9ced8" },
  { id: "ball", name: "Balón (tracking)", color: "#f5c518" },
  { id: "trail", name: "Estela del balón (1 s)", color: "#f5c518" },
  { id: "rich", name: "Balón Roboflow (relleno)", color: "#39d98a", needs: "has_enriched" },
  { id: "cands", name: "Candidatos Roboflow (top-5)", color: "#c084fc", needs: "has_detections" },
];

// /roboflow plays only the graded frames where the raw tracking has no ball, back to back
const GAPS_MODE = location.pathname === "/roboflow";

const S = {
  config: null, meta: null, fps: 25, cur: 0, shown: -1, playing: false, rate: 1,
  range: { first: 0, last: 0 }, chunks: new Map(), on: {}, pitchLines: [], pitchSpots: [],
  codes: "", gaps: [], loopGap: null, allCache: null,
  reel: [], reelCum: [], reelTotal: 0, segIdx: -1, jumpTo: null,
};
const ZOOM = 15;  // seconds each side in the zoom timeline
const SRC_ORDER = ["1", "2", "3", "0"];  // stacking order, bottom to top
const video = $("#main");
const canvas = $("#draw");

// -- data ----------------------------------------------------------------------------------------

function chunkOf(f) { return Math.floor(f / CHUNK); }

function load(idx) {
  if (idx < 0 || S.chunks.has(idx)) return;
  S.chunks.set(idx, null);
  fetch(`/api/overlay?from=${idx * CHUNK}&count=${CHUNK}`)
    .then((r) => r.json())
    .then((data) => {
      S.chunks.set(idx, data.frames);
      if (S.chunks.size > 80) {  // keep memory bounded: drop chunks far from the playhead
        const here = chunkOf(S.cur);
        for (const k of [...S.chunks.keys()]) if (Math.abs(k - here) > 20) S.chunks.delete(k);
      }
      if (chunkOf(Math.max(S.shown, 0)) === idx || chunkOf(S.cur) === idx) redraw();
    })
    .catch(() => { S.chunks.delete(idx); });
}

function frameData(f) {
  const rows = S.chunks.get(chunkOf(f));
  if (!rows) { load(chunkOf(f)); return null; }
  return rows[f - chunkOf(f) * CHUNK] || null;
}

// -- geometry ------------------------------------------------------------------------------------

function project(h, x, y) {
  const w = h[6] * x + h[7] * y + h[8];
  if (!(w > 0) && !(w < 0)) return null;
  const px = (h[0] * x + h[1] * y + h[2]) / w, py = (h[3] * x + h[4] * y + h[5]) / w;
  return Number.isFinite(px) && Number.isFinite(py) ? [px, py] : null;
}

// pixels per metre around a pitch point, averaged over both pitch axes
function scaleAt(h, x, y) {
  const [L, W] = S.meta.pitch;
  const a = project(h, x - 0.5 / L, y), b = project(h, x + 0.5 / L, y);
  const c = project(h, x, y - 0.5 / W), d = project(h, x, y + 0.5 / W);
  if (!a || !b || !c || !d) return null;
  return (Math.hypot(b[0] - a[0], b[1] - a[1]) + Math.hypot(d[0] - c[0], d[1] - c[1])) / 2;
}

function buildPitch() {
  const [L, W] = S.meta.pitch;
  const n = ([x, y]) => [x / L, y / W];
  const arc = (cx, cy, r, keep) => {
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const t = (i / 64) * 2 * Math.PI, p = [cx + r * Math.cos(t), cy + r * Math.sin(t)];
      if (!keep || keep(p)) pts.push(p); else if (pts.length) { lines.push(pts.splice(0)); }
    }
    if (pts.length) lines.push(pts);
  };
  const lines = [];
  const box = (depth, half) => {
    lines.push([[0, W / 2 - half], [depth, W / 2 - half], [depth, W / 2 + half], [0, W / 2 + half]]);
    lines.push([[L, W / 2 - half], [L - depth, W / 2 - half], [L - depth, W / 2 + half], [L, W / 2 + half]]);
  };
  lines.push([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]);
  lines.push([[L / 2, 0], [L / 2, W]]);
  arc(L / 2, W / 2, 9.15);
  box(16.5, 20.16);
  box(5.5, 9.16);
  arc(11, W / 2, 9.15, (p) => p[0] > 16.5);
  arc(L - 11, W / 2, 9.15, (p) => p[0] < L - 16.5);
  // straight lines stay straight under a homography; only the arcs need many points
  S.pitchLines = lines.map((pts) => pts.map(n));
  S.pitchSpots = [[L / 2, W / 2], [11, W / 2], [L - 11, W / 2]].map(n);
}

// -- drawing -------------------------------------------------------------------------------------

function redraw() { draw(S.shown >= 0 ? S.shown : S.cur); }

function draw(f) {
  const box = $("#viewer").getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(box.width * dpr) || canvas.height !== Math.round(box.height * dpr)) {
    canvas.width = Math.round(box.width * dpr);
    canvas.height = Math.round(box.height * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, canvas.width, canvas.height);
  const vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
  const k = Math.min(box.width / vw, box.height / vh);
  const ox = (box.width - vw * k) / 2, oy = (box.height - vh * k) / 2;
  g.setTransform(dpr * k, 0, 0, dpr * k, dpr * ox, dpr * oy);
  g.save();
  g.beginPath();
  g.rect(0, 0, vw, vh);
  g.clip();
  const px = 1 / k;  // one screen pixel in video pixels: keeps strokes and text a constant size

  const row = frameData(f);
  $("#loading").classList.toggle("hidden", !!row);
  if (!row) { g.restore(); return; }
  load(chunkOf(f) + 1);
  const h = row.h;
  const info = [];

  if (h && S.on.pitch) {
    g.strokeStyle = "rgba(255,255,255,.75)";
    g.lineWidth = 1.5 * px;
    for (const pts of S.pitchLines) {
      g.beginPath();
      let started = false;
      for (const [x, y] of pts) {
        const p = project(h, x, y);
        if (!p) { started = false; continue; }
        if (started) g.lineTo(p[0], p[1]); else { g.moveTo(p[0], p[1]); started = true; }
      }
      g.stroke();
    }
    g.fillStyle = "rgba(255,255,255,.85)";
    for (const [x, y] of S.pitchSpots) {
      const p = project(h, x, y);
      if (p) { g.beginPath(); g.arc(p[0], p[1], 2.5 * px, 0, 2 * Math.PI); g.fill(); }
    }
  }

  if (h && (S.on.home || S.on.away)) {
    g.font = `${11 * px}px system-ui, sans-serif`;
    g.textAlign = "center";
    for (const [slot, x, y] of row.p) {
      const team = S.meta.team[slot] === 0 ? "home" : "away";
      if (!S.on[team]) continue;
      const foot = project(h, x, y), s = scaleAt(h, x, y);
      if (!foot || !s) continue;
      const bw = Math.max(0.9 * s, 6 * px), bh = Math.max(2.0 * s, 12 * px);
      const color = LAYERS.find((l) => l.id === team).color;
      g.strokeStyle = color;
      g.lineWidth = 2 * px;
      g.strokeRect(foot[0] - bw / 2, foot[1] - bh, bw, bh);
      g.fillStyle = color;
      g.beginPath(); g.arc(foot[0], foot[1], 2.5 * px, 0, 2 * Math.PI); g.fill();
      if (S.on.slots) {
        g.fillStyle = "rgba(0,0,0,.65)";
        g.fillRect(foot[0] - 9 * px, foot[1] - bh - 14 * px, 18 * px, 13 * px);
        g.fillStyle = "#fff";
        g.fillText(String(slot + 1), foot[0], foot[1] - bh - 3.5 * px);
      }
    }
  }

  const ballBox = (p, s, color, dashed, tag) => {
    const size = Math.max(s ? 0.9 * s : 0, 10 * px);
    g.strokeStyle = color;
    g.lineWidth = 2 * px;
    g.setLineDash(dashed ? [4 * px, 3 * px] : []);
    g.strokeRect(p[0] - size / 2, p[1] - size / 2, size, size);
    g.setLineDash([]);
    if (tag) {
      g.font = `bold ${10 * px}px system-ui, sans-serif`;
      g.textAlign = "left";
      g.fillStyle = color;
      g.fillText(tag, p[0] + size / 2 + 3 * px, p[1] - size / 2 + 8 * px);
    }
  };

  if (h && S.on.trail) {
    g.fillStyle = LAYERS.find((l) => l.id === "trail").color;
    for (let back = S.fps; back >= 1; back--) {
      const past = frameData(f - back);
      const xy = past && (past.b || (S.on.rich && past.e));
      if (!xy) continue;
      const p = project(h, xy[0], xy[1]);  // past pitch position seen through today's camera
      if (!p) continue;
      g.globalAlpha = 0.15 + 0.6 * (1 - back / S.fps);
      g.beginPath(); g.arc(p[0], p[1], 2.5 * px, 0, 2 * Math.PI); g.fill();
    }
    g.globalAlpha = 1;
  }

  if (h && S.on.ball && row.b) {
    const p = project(h, row.b[0], row.b[1]);
    if (p) ballBox(p, scaleAt(h, row.b[0], row.b[1]), LAYERS.find((l) => l.id === "ball").color, false, "");
  }
  if (h && S.on.rich && row.e) {
    const p = project(h, row.e[0], row.e[1]);
    if (p) ballBox(p, scaleAt(h, row.e[0], row.e[1]), LAYERS.find((l) => l.id === "rich").color, true,
                   row.e[2] === 2 ? "RF" : "interp");
  }
  if (S.on.cands && row.d) {
    const color = LAYERS.find((l) => l.id === "cands").color;
    g.font = `${10 * px}px ui-monospace, monospace`;
    g.textAlign = "left";
    row.d.forEach(([cx, cy, conf, size], i) => {
      const half = Math.max(size, 6 * px) / 2;
      g.strokeStyle = color;
      g.lineWidth = (i === 0 ? 2 : 1) * px;
      g.globalAlpha = i === 0 ? 1 : 0.75;
      g.strokeRect(cx - half, cy - half, 2 * half, 2 * half);
      g.fillStyle = color;
      g.fillText(conf.toFixed(2), cx + half + 2 * px, cy + half);
    });
    g.globalAlpha = 1;
  }
  g.restore();

  if (!h) info.push("sin homografía");
  info.push(row.b ? "balón: tracking" : row.e ? (row.e[2] === 2 ? "balón: Roboflow" : "balón: interpolado")
                  : "sin balón");
  if (row.d) info.push(row.d.length + " candidatos");
  $("#ballinfo").textContent = info.join(" · ");
}

// -- ball-source timelines -----------------------------------------------------------------------

// runs of frames where the raw tracking has no ball: [first, last] inclusive
function findGaps(codes) {
  const gaps = [];
  let start = -1;
  for (let f = 0; f <= codes.length; f++) {
    const missing = f < codes.length && codes[f] !== "1";
    if (missing && start < 0) start = f;
    else if (!missing && start >= 0) { gaps.push([start, f - 1]); start = -1; }
  }
  return gaps;
}

function gapAt(f) {
  let lo = 0, hi = S.gaps.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1, g = S.gaps[mid];
    if (f < g[0]) hi = mid - 1; else if (f > g[1]) lo = mid + 1; else return g;
  }
  return null;
}

function counts(from, to) {
  const c = { 0: 0, 1: 0, 2: 0, 3: 0 };
  for (let f = Math.max(0, from); f <= Math.min(S.codes.length - 1, to); f++) c[S.codes[f]]++;
  return c;
}

// stacked bars: each pixel column shows the mix of ball sources over the frames it covers
function paintSources(c, from, to, toReal) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d");
  g.fillStyle = css("--X");
  g.fillRect(0, 0, w, h);
  const span = to - from + 1;
  const colors = { 0: css("--src0"), 1: css("--src1"), 2: css("--src2"), 3: css("--src3") };
  for (let x = 0; x < w; x++) {
    const a = from + Math.floor((x / w) * span), b = Math.max(a + 1, from + Math.floor(((x + 1) / w) * span));
    if (a < 0 || (!toReal && a >= S.codes.length) || (toReal && a >= S.reelTotal)) continue;
    const n = toReal ? reelCounts(a, Math.min(b, S.reelTotal) - 1) : counts(a, b - 1);
    const total = n[0] + n[1] + n[2] + n[3];
    if (!total) continue;
    let y = h;
    for (const k of SRC_ORDER) {
      if (!n[k]) continue;
      const bh = (n[k] / total) * h;
      g.fillStyle = colors[k];
      g.fillRect(x, y - bh, 1, bh);
      y -= bh;
    }
  }
  return { g, w, h, x: (f) => ((f - from + 0.5) / span) * w };
}

// -- reel of gaps (/roboflow) --------------------------------------------------------------------

function buildReel() {
  const { first, last } = S.range;
  const minLen = Math.max(1, parseFloat($("#minlen").value) * S.fps);
  S.reel = [];
  for (const g of S.gaps) {
    if (g[1] - g[0] + 1 < minLen) continue;
    for (const b of S.config.blocks) {  // graded frames only: halftime and the like are not gaps
      const a = Math.max(g[0], b.first, first), z = Math.min(g[1], b.last, last);
      if (z >= a) S.reel.push([a, z]);
    }
  }
  S.reelCum = [];
  let total = 0;
  for (const [a, z] of S.reel) { S.reelCum.push(total); total += z - a + 1; }
  S.reelTotal = total;
  S.allCache = null;
}

// index of the segment containing f, or of the first one after it (-1 past the end)
function segIndexOf(f) {
  let lo = 0, hi = S.reel.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (S.reel[mid][1] >= f) { ans = mid; hi = mid - 1; } else lo = mid + 1;
  }
  return ans;
}

function virtualOf(f) {
  const i = segIndexOf(f);
  if (i < 0) return Math.max(0, S.reelTotal - 1);
  return S.reelCum[i] + Math.max(0, f - S.reel[i][0]);
}

function realOf(v) {
  v = clamp(v, 0, Math.max(0, S.reelTotal - 1));
  let lo = 0, hi = S.reel.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (S.reelCum[mid] <= v) lo = mid; else hi = mid - 1;
  }
  return S.reel.length ? S.reel[lo][0] + (v - S.reelCum[lo]) : S.range.first;
}

function reelCounts(from, to) {
  const c = { 0: 0, 1: 0, 2: 0, 3: 0 };
  for (let v = from; v <= to; v++) c[S.codes[realOf(v)]]++;
  return c;
}

function mmss(frames) {
  const s = Math.floor(frames / S.fps);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function drawTimelines(f) {
  if (!S.codes) return;
  const { first, last } = S.range;
  const all = $("#srcall");
  const key = [first, last, all.clientWidth, window.devicePixelRatio, S.reelTotal].join();
  if (!S.allCache || S.allCache.key !== key) {
    if (GAPS_MODE) paintSources(all, 0, Math.max(0, S.reelTotal - 1), realOf);
    else paintSources(all, first, last);
    S.allCache = { key, image: all.getContext("2d").getImageData(0, 0, all.width, all.height) };
  }
  const g = all.getContext("2d");
  g.putImageData(S.allCache.image, 0, 0);
  const z = ZOOM * S.fps;
  if (GAPS_MODE) {
    const span = Math.max(1, S.reelTotal), x = (v) => ((v + 0.5) / span) * all.width;
    g.fillStyle = "#fff";
    g.fillRect(x(virtualOf(f)) - 1, 0, 2, all.height);
  } else {
    const span = last - first + 1, x = (fr) => ((fr - first + 0.5) / span) * all.width;
    g.fillStyle = "rgba(255,255,255,.18)";
    g.fillRect(x(f - z), 0, Math.max(2, x(f + z) - x(f - z)), all.height);
    g.fillStyle = "#fff";
    g.fillRect(x(f) - 1, 0, 2, all.height);
  }

  const zoom = paintSources($("#srczoom"), f - z, f + z);
  zoom.g.fillStyle = "rgba(255,255,255,.55)";
  for (let t = Math.ceil((f - z) / S.fps) * S.fps; t <= f + z; t += S.fps) zoom.g.fillRect(zoom.x(t), 0, 1, 5);
  zoom.g.fillStyle = "#fff";
  zoom.g.fillRect(zoom.x(f) - 1, 0, 2, zoom.h);

  const gap = gapAt(f);
  if (GAPS_MODE) {
    const i = segIndexOf(f), seg = S.reel[i];
    if (!seg) { $("#gapinfo").textContent = "no hay huecos en este rango"; return; }
    const n = counts(seg[0], seg[1]), len = seg[1] - seg[0] + 1;
    $("#gapinfo").textContent = `hueco ${i + 1} de ${S.reel.length} · ${(len / S.fps).toFixed(1)} s · ` +
      `Roboflow cubre ${Math.round((100 * (n[2] + n[3])) / len)} % · ${mmss(virtualOf(f))} de ${mmss(S.reelTotal)}`;
  } else if (gap) {
    const n = counts(gap[0], gap[1]);
    const len = gap[1] - gap[0] + 1;
    $("#gapinfo").textContent = `en hueco del tracking: ${(len / S.fps).toFixed(1)} s · Roboflow cubre ` +
      `${Math.round((100 * (n[2] + n[3])) / len)} %`;
  } else {
    const inRange = S.gaps.filter((g2) => g2[1] >= first && g2[0] <= last).length;
    $("#gapinfo").textContent = `${inRange} huecos del tracking en este rango`;
  }
}

function updateLegend() {
  const { first, last } = S.range;
  let n = counts(first, last), total = last - first + 1;
  if (GAPS_MODE) {
    n = { 0: 0, 1: 0, 2: 0, 3: 0 };
    for (const [a, z] of S.reel) { const c = counts(a, z); for (const k in c) n[k] += c[k]; }
    total = Math.max(1, S.reelTotal);
  }
  for (const k of ["0", "1", "2", "3"]) $("#pct" + k).textContent = Math.round((100 * n[k]) / total) + " %";
}

function frameAtClick(e, from, to) {
  const r = e.currentTarget.getBoundingClientRect();
  return from + Math.floor(((e.clientX - r.left) / r.width) * (to - from + 1));
}

function jumpGap(dir) {
  if (GAPS_MODE) {
    const i = segIndexOf(S.cur), here = S.reel[i];
    const target = dir > 0 ? (here && S.cur >= here[0] ? i + 1 : i) : (here && S.cur > here[0] ? i : i - 1);
    if (i < 0 && dir < 0) return seek(S.reel[S.reel.length - 1][0]);
    if (target < 0 || target >= S.reel.length) return toast(dir > 0 ? "Último hueco" : "Primer hueco");
    return seek(S.reel[target][0]);
  }
  const { first, last } = S.range;
  const here = S.cur + S.fps;  // seeks land 1 s before a gap, so measure from where it starts
  const inRange = S.gaps.filter((g) => g[1] >= first && g[0] <= last);
  const target = dir > 0 ? inRange.find((g) => g[0] > here) : [...inRange].reverse().find((g) => g[0] < here);
  if (!target) return toast(dir > 0 ? "No hay más huecos en este rango" : "No hay huecos antes");
  seek(target[0] - S.fps);
  S.loopGap = target;
}

// -- navigation ----------------------------------------------------------------------------------

function clock(f) {
  const half = f >= S.config.kickoffs[2] - 10 * 60 * S.fps ? 2 : 1;
  const t = (f - S.config.kickoffs[half]) / S.fps + (half === 2 ? 45 * 60 : 0);
  const sign = t < 0 ? "−" : "";
  const a = Math.abs(t), m = Math.floor(a / 60), s = Math.floor(a % 60), fr = Math.round((a % 1) * S.fps);
  return `${half}T ${sign}${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}<span class="muted">.${String(fr).padStart(2, "0")}</span>`;
}

// "1T 12:30", "2T 50:05", "12:30" (first half) or a bare frame number
function parseTime(text) {
  const m = text.trim().toUpperCase().match(/^(?:([12])T\s*)?(\d+):(\d{1,2})$/);
  if (!m) return /^\d+$/.test(text.trim()) ? parseInt(text, 10) : null;
  const half = m[1] ? parseInt(m[1], 10) : 1;
  const t = parseInt(m[2], 10) * 60 + parseInt(m[3], 10) - (half === 2 ? 45 * 60 : 0);
  return S.config.kickoffs[half] + t * S.fps;
}

function updateHud(f) {
  $("#clock").innerHTML = clock(f);
  $("#frameinfo").textContent = "frame " + f;
  drawTimelines(f);
}

function seek(f) {
  f = clamp(Math.round(f), S.range.first, S.range.last);
  if (GAPS_MODE && S.reel.length) { f = realOf(virtualOf(f)); S.segIdx = segIndexOf(f); }
  if (S.playing) pause();
  S.cur = f;
  S.loopGap = null;
  load(chunkOf(f));
  video.currentTime = (f + 0.5) / S.fps;
  updateHud(f);
}

function play() {
  S.playing = true;
  video.playbackRate = S.rate;
  video.play().catch(() => { S.playing = false; });
  $("#playbtn").textContent = "❚❚";
}

function pause() {
  video.pause();
  S.playing = false;
  if (S.shown >= 0) S.cur = S.shown;
  $("#playbtn").textContent = "▶";
}

function setRate(dir) {
  S.rate = SPEEDS[clamp(SPEEDS.indexOf(S.rate) + dir, 0, SPEEDS.length - 1)];
  video.playbackRate = S.rate;
  $("#rate").textContent = S.rate + "×";
}

function setRange(first, last, label) {
  S.range = { first, last };
  $("#rangetxt").textContent = label;
  if (GAPS_MODE) buildReel();
  updateLegend();
  seek(S.cur >= first && S.cur <= last ? S.cur : first);
}

// frame steps: in /roboflow they walk the reel, skipping the frames that have a tracking ball
function stepBy(n) {
  const base = S.playing ? S.shown : S.cur;
  if (GAPS_MODE && S.reel.length) seek(realOf(virtualOf(base) + n));
  else seek(base + n);
}

function trackFrames(onFrame) {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const cb = (_, meta) => { onFrame(Math.round(meta.mediaTime * S.fps)); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else {
    const loop = () => { onFrame(Math.floor(video.currentTime * S.fps)); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
}

// -- layers --------------------------------------------------------------------------------------

const LAYER_KEY = GAPS_MODE ? "overlay.layers.gaps" : "overlay.layers";

function saveLayers() {
  try { localStorage.setItem(LAYER_KEY, JSON.stringify(S.on)); } catch (e) { /* private mode */ }
}

function buildLayers() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(LAYER_KEY) || "null"); } catch (e) { saved = null; }
  const box = $("#layers");
  box.innerHTML = "";
  LAYERS.forEach((layer, i) => {
    const available = !layer.needs || S.meta[layer.needs];
    S.on[layer.id] = available && (saved && layer.id in saved ? saved[layer.id] : layer.id !== "slots");
    const row = document.createElement("label");
    row.className = "toggle layer" + (available ? "" : " off");
    row.innerHTML = `<input type="checkbox" ${S.on[layer.id] ? "checked" : ""} ${available ? "" : "disabled"}>` +
      `<span class="swatch" style="background:${layer.color}"></span>${layer.name}` +
      `<kbd>${i + 1}</kbd>` + (available ? "" : `<span class="muted small">(sin datos: ejecuta enrich)</span>`);
    row.querySelector("input").addEventListener("change", (e) => {
      S.on[layer.id] = e.target.checked;
      saveLayers();
      redraw();
    });
    box.appendChild(row);
  });
}

function toggleLayer(i) {
  const input = $("#layers").children[i] && $("#layers").children[i].querySelector("input");
  if (!input || input.disabled) return;
  input.checked = !input.checked;
  input.dispatchEvent(new Event("change"));
}

function setAll(value) {
  [...$("#layers").querySelectorAll("input")].forEach((input) => {
    if (!input.disabled && input.checked !== value) { input.checked = value; input.dispatchEvent(new Event("change")); }
  });
}

// -- wiring --------------------------------------------------------------------------------------

function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 1800);
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const act = b.dataset.act;
  if (act === "play") (S.playing ? pause : play)();
  else if (act === "step") stepBy(parseInt(b.dataset.n, 10));
  else if (act === "rate") setRate(parseInt(b.dataset.n, 10));
  else if (act === "gap") jumpGap(parseInt(b.dataset.n, 10));
  b.blur();
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input[type=text], input:not([type]), select")) return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); (S.playing ? pause : play)(); }
  else if (k === "arrowleft") { e.preventDefault(); stepBy(-(e.shiftKey ? S.fps : 1)); }
  else if (k === "arrowright") { e.preventDefault(); stepBy(e.shiftKey ? S.fps : 1); }
  else if (k === "arrowup") { e.preventDefault(); stepBy(5 * S.fps); }
  else if (k === "arrowdown") { e.preventDefault(); stepBy(-5 * S.fps); }
  else if (k === "-") setRate(-1);
  else if (k === "+" || k === "=") setRate(1);
  else if (/^[1-9]$/.test(k)) toggleLayer(parseInt(k, 10) - 1);
  else if (k === "[") jumpGap(-1);
  else if (k === "]") jumpGap(1);
});

$("#srcall").addEventListener("click", (e) => {
  if (GAPS_MODE) seek(realOf(frameAtClick(e, 0, Math.max(0, S.reelTotal - 1))));
  else seek(frameAtClick(e, S.range.first, S.range.last));
});
$("#minlen").addEventListener("change", (e) => {
  setRange(S.range.first, S.range.last, $("#rangetxt").textContent);
  e.target.blur();
});
$("#srczoom").addEventListener("click", (e) => {
  const f = S.playing ? S.shown : S.cur;
  seek(frameAtClick(e, f - ZOOM * S.fps, f + ZOOM * S.fps));
});
$("#goto").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const f = parseTime(e.target.value);
  if (f === null) return toast("Formato: 1T 12:30, 2T 50:05 o un número de frame");
  $("#block").value = "";
  setRange(0, S.config.n_frames - 1, "partido");
  seek(f);
  e.target.blur();
});
$("#block").addEventListener("change", (e) => {
  const b = S.config.blocks.find((x) => x.id === e.target.value);
  if (b) setRange(b.first, b.last, b.id);
  else setRange(0, S.config.n_frames - 1, "partido");
  e.target.blur();
});
$("#dim").addEventListener("input", (e) => { video.style.opacity = e.target.value / 100; });
$("#allon").addEventListener("click", () => setAll(true));
$("#alloff").addEventListener("click", () => setAll(false));
window.addEventListener("resize", () => { redraw(); drawTimelines(S.playing ? S.shown : S.cur); });
video.addEventListener("loadedmetadata", redraw);
video.addEventListener("ended", pause);

trackFrames((f) => {
  S.shown = f;
  if (GAPS_MODE && S.playing && S.reel.length) {
    if (S.jumpTo !== null) {  // frames decoded before the jump landed: ignore them
      const seg = S.reel[S.jumpTo];
      if (f < seg[0] || f > seg[1]) return;
      S.segIdx = S.jumpTo;
      S.jumpTo = null;
    }
    const seg = S.reel[S.segIdx];
    if (seg && f > seg[1]) {
      const next = $("#gaploop").checked ? S.segIdx : S.segIdx + 1;
      if (next >= S.reel.length) { pause(); return; }
      S.jumpTo = next;
      video.currentTime = (S.reel[next][0] + 0.5) / S.fps;
      return;
    }
  }
  const g = S.loopGap;
  if (S.playing && g && $("#gaploop").checked && f > g[1] + S.fps / 2) {
    video.currentTime = (g[0] - S.fps + 0.5) / S.fps;  // replay the gap from 1 s before it
    return;
  }
  if (S.playing) { S.cur = f; updateHud(f); }
  draw(f);
});

(async function boot() {
  if (GAPS_MODE) {
    document.body.classList.add("gaps");
    document.title = "Solo huecos del tracking";
    $("#tab-roboflow").classList.add("on");
    $("#gaploop").parentElement.lastChild.textContent = " repetir este hueco";
  } else {
    $("#tab-overlay").classList.add("on");
  }
  S.config = await (await fetch("/api/config")).json();
  S.fps = S.config.fps;
  $("#game").textContent = S.config.game;
  for (const b of S.config.blocks) {
    const o = document.createElement("option");
    o.value = b.id;
    o.textContent = `${b.id} · min ${b.minute_from}–${b.minute_to}`;
    $("#block").appendChild(o);
  }
  $("#status").textContent = "cargando tracking…";
  S.meta = await (await fetch("/api/overlay/meta")).json();
  $("#status").textContent = (S.meta.has_detections ? "con" : "sin") + " detecciones Roboflow";
  buildPitch();
  buildLayers();
  S.codes = (await (await fetch("/api/overlay/sources")).json()).codes;
  S.gaps = findGaps(S.codes);
  if (!S.config.video_available) toast("No se encuentra el vídeo táctico en data/");
  video.src = "/video";
  if (GAPS_MODE) {  // the reel spans the whole match by default
    setRange(0, S.config.n_frames - 1, "partido");
  } else {
    const first = S.config.blocks[0];
    $("#block").value = first.id;
    setRange(first.first, first.last, first.id);
  }
})();
