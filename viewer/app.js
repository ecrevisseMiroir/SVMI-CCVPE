/* SVMI+ viewer: camera frames + GT / predicted poses on the IGN orthophoto.
 * Plain JS, no build step. Serve the repo root with `python3 -m http.server`
 * and open /viewer/. All data paths are relative to viewer/. */
'use strict';

const CFG = {
  fps: 10.14,                      // 2799 frames over 4'36"
  paths: {
    gt: '../data/gt.json',
    mapMeta: '../data/map/ign_z19.json',
    mapViewer: '../data/map/ign_z19_viewer.json',
    mapDir: '../data/map/',
    frames: '../data/viewer_frames/',
    pano: '../data/viewer_pano/',
    results: '../results/',
  },
  speeds: [0.25, 0.5, 1, 2, 4, 8],
  preload: {
    seconds: 4,        // look this far ahead in playback time
    minAhead: 40,
    maxAhead: 300,
    around: 12,        // neighbours on both sides (for stepping / scrubbing)
    inflight: 8,       // concurrent requests
    cache: 200,        // max decoded frames kept (3 images each)
  },
  gapBreak: 50,        // break prediction polylines across index gaps larger than this
  gtColor: '#4cc2ff',
  fallbackColors: ['#e4572e', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'],
  minScale: 0.04,      // CSS px per mosaic px
  maxScale: 16,
};

/* ------------------------------------------------------------------ geo -- */
// Same constants as svmi/geo.py.
const TILE = 256;
const M_PER_DEG_LAT = 111229.5357;
const M_PER_DEG_LON = 71745.380;

function latLonToWorldPx(lat, lon, zoom) {
  const n = TILE * 2 ** zoom;
  const s = Math.sin(lat * Math.PI / 180);
  return [(lon + 180) / 360 * n, (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * n];
}

/* ---------------------------------------------------------------- state -- */
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

const S = {
  N: 0,
  frame: 0,
  playhead: 0,          // fractional frame position while playing
  playing: false,
  speed: 1,
  follow: false,
  chartMode: 'alt',
  mapHover: -1,
  chartHover: -1,
  view: { cx: 0, cy: 0, scale: 0.25 },
  buffering: false,
  bufferSince: 0,
};

let ORIGIN = null;      // {lat, lon}
let MAP = null;         // {zoom, ox, oy, w, h, mPerPx, image, source}
const GT = {};          // typed arrays, see buildGT()
let methods = [];
const levels = [];      // basemap pyramid: [{src, k}] with k = level px per mosaic px
let fullResState = 'none';

const dirty = { cam: true, map: true, chart: true, chartBg: true, hud: true };
let rafId = 0;
let lastT = 0;

/* ------------------------------------------------------------ utilities -- */
function fmtTime(sec) {
  const t = Math.round(sec * 10);
  const m = Math.floor(t / 600);
  const r = (t - m * 600) / 10;
  return `${String(m).padStart(2, '0')}:${r.toFixed(1).padStart(4, '0')}`;
}
const signed = (v, d = 1) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const compass = (h) => COMPASS[Math.round(((h % 360) + 360) % 360 / 22.5) % 16];
const fmtM = (v) => (isNum(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + ' m' : '–');
const fmtDeg = (v) => (isNum(v) ? v.toFixed(1) + '°' : '–');
const angDiff = (a, b) => { const d = Math.abs(((a - b) % 360 + 360) % 360); return d > 180 ? 360 - d : d; };
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}
function fmtNum(v) {
  if (!isNum(v)) return String(v);
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
}

async function fetchJSON(url, optional = false) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) {
    if (optional && r.status === 404) return null;
    throw new Error(`${url}: HTTP ${r.status}`);
  }
  return r.json();
}

function setStatus(text, warn = false) {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('warn', warn);
}

/* ---------------------------------------------------------- data setup -- */
function toMosaic(lat, lon) {
  const [x, y] = latLonToWorldPx(lat, lon, MAP.zoom);
  return [x - MAP.ox, y - MAP.oy];
}

function buildGT(gt) {
  const fr = gt.frames;
  const N = fr.length;
  S.N = N;
  ORIGIN = gt.origin;
  Object.assign(GT, {
    x: new Float64Array(N), y: new Float64Array(N),
    north: new Float64Array(N), east: new Float64Array(N),
    alt: new Float64Array(N), hdg: new Float64Array(N),
    files: new Array(N),
  });
  for (let k = 0; k < N; k++) {
    const f = fr[k];
    const i = isNum(f.i) ? f.i : k;
    if (i !== k) console.warn('gt.json frames are not dense/sorted at', k);
    const [x, y] = toMosaic(f.lat, f.lon);
    GT.x[k] = x; GT.y[k] = y;
    GT.north[k] = f.north; GT.east[k] = f.east;
    GT.alt[k] = f.alt;
    GT.hdg[k] = isNum(f.heading_deg) ? f.heading_deg : NaN;
    GT.files[k] = f.image ? f.image.replace(/\.png$/i, '.jpg') : String(k).padStart(6, '0') + '.jpg';
  }
}

function buildMethod(spec, data, k) {
  const N = S.N;
  const nan = () => new Float64Array(N).fill(NaN);
  const m = {
    name: spec.name || data.method || spec.file,
    color: spec.color || data.color || CFG.fallbackColors[k % CFG.fallbackColors.length],
    file: spec.file,
    enabled: spec.visible !== false,   // manifest may hide a method by default
    x: nan(), y: nan(), hdg: nan(), err: nan(), herr: nan(),
    idx: [],
    summary: data.summary || {},
    params: data.params || {},
  };
  for (const f of data.frames || []) {
    const i = f.i;
    if (!Number.isInteger(i) || i < 0 || i >= N) continue;
    let lat = f.lat, lon = f.lon, north = f.north, east = f.east;
    if (!(isNum(lat) && isNum(lon))) {
      if (!(isNum(north) && isNum(east))) continue;
      lat = ORIGIN.lat + north / M_PER_DEG_LAT;
      lon = ORIGIN.lon + east / M_PER_DEG_LON;
    }
    if (!(isNum(north) && isNum(east))) {
      north = (lat - ORIGIN.lat) * M_PER_DEG_LAT;
      east = (lon - ORIGIN.lon) * M_PER_DEG_LON;
    }
    const [x, y] = toMosaic(lat, lon);
    m.x[i] = x; m.y[i] = y;
    m.hdg[i] = isNum(f.heading_deg) ? f.heading_deg : NaN;
    m.err[i] = isNum(f.err_m) ? f.err_m : Math.hypot(north - GT.north[i], east - GT.east[i]);
    m.herr[i] = isNum(f.heading_err_deg) ? f.heading_err_deg
      : (isNum(m.hdg[i]) && isNum(GT.hdg[i]) ? angDiff(m.hdg[i], GT.hdg[i]) : NaN);
    m.idx.push(i);
  }
  m.idx = [...new Set(m.idx)].sort((a, b) => a - b);
  const errs = m.idx.map((i) => m.err[i]).filter(isNum).sort((a, b) => a - b);
  const herrs = m.idx.map((i) => m.herr[i]).filter(isNum).sort((a, b) => a - b);
  m.stats = {
    n: m.idx.length,
    median: quantile(errs, 0.5),
    mean: errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : NaN,
    hMedian: quantile(herrs, 0.5),
    hasHeading: herrs.length > 0,
  };
  return m;
}

async function loadMethods() {
  let manifest;
  try {
    manifest = await fetchJSON(CFG.paths.results + 'manifest.json', true);
  } catch (e) {
    console.warn(e);
    $('methodsNote').textContent = 'results/manifest.json could not be read';
    return;
  }
  if (!manifest || !Array.isArray(manifest.methods) || !manifest.methods.length) {
    $('methodsNote').textContent = 'none yet — GT only (add results/manifest.json)';
    return;
  }
  const loaded = await Promise.all(manifest.methods.map(async (spec, k) => {
    try {
      const data = await fetchJSON(CFG.paths.results + spec.file);
      return buildMethod(spec, data, k);
    } catch (e) {
      console.warn('prediction load failed', spec, e);
      return { name: spec.name || spec.file, color: spec.color || '#888', failed: String(e.message || e), enabled: false, idx: [] };
    }
  }));
  methods = loaded;
  $('methodsNote').textContent = '';
  if (methods.some((m) => !m.failed)) S.chartMode = 'err';
}

/* ------------------------------------------------------------- basemap -- */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => (img.decode ? img.decode().catch(() => {}) : Promise.resolve()).then(() => resolve(img));
    img.onerror = () => reject(new Error('failed to load ' + url));
    img.src = url;
  });
}

function addBaseLevel(img) {
  // Register the image and a runtime pyramid of halvings for zoomed-out views,
  // so drawImage never has to minify a huge texture on every frame.
  let src = img, w = img.naturalWidth, h = img.naturalHeight;
  levels.push({ src: img, k: w / MAP.w });
  while (w > 600) {
    w = Math.round(w / 2); h = Math.round(h / 2);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    cx.imageSmoothingEnabled = true;
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(src, 0, 0, w, h);
    levels.push({ src: c, k: w / MAP.w });
    src = c;
  }
  levels.sort((a, b) => a.k - b.k);
}

async function loadBasemap(viewerMeta) {
  const small = viewerMeta && viewerMeta.image ? CFG.paths.mapDir + viewerMeta.image : null;
  try {
    if (small) {
      addBaseLevel(await loadImage(small));
    } else {
      addBaseLevel(await loadImage(CFG.paths.mapDir + MAP.image));
      fullResState = 'done';
    }
  } catch (e) {
    console.warn(e);
    if (small) {  // fall back to the full-resolution mosaic
      try { addBaseLevel(await loadImage(CFG.paths.mapDir + MAP.image)); fullResState = 'done'; } catch (e2) { console.warn(e2); }
    }
  }
  if (!levels.length) setStatus('basemap image failed to load', true);
  dirty.map = true; kick();
}

function requestFullRes() {
  if (fullResState !== 'none') return;
  fullResState = 'loading';
  loadImage(CFG.paths.mapDir + MAP.image).then((img) => {
    levels.push({ src: img, k: img.naturalWidth / MAP.w });
    levels.sort((a, b) => a.k - b.k);
    fullResState = 'done';
    dirty.map = true; kick();
  }, () => { fullResState = 'failed'; });
}

function pickLevel(devScale) {
  if (!levels.length) return null;
  for (const l of levels) if (l.k >= devScale * 0.85) return l;
  return levels[levels.length - 1];
}

/* ------------------------------------------------------- frame preload -- */
// Views shown side by side; each frame loads one image per view.
const VIEWS = [
  { key: 'fisheye', dir: () => CFG.paths.frames, canvas: 'cam' },
  { key: 'pano', dir: () => CFG.paths.pano, canvas: 'camPano' },
];
const cache = new Map();   // i -> {imgs: {view: Image}, state: 'loading' | 'ready' | 'error', ok: {view: bool}}
let queue = [];
let inflight = 0;
const lastCamImg = {};

function startLoad(i) {
  const e = { imgs: {}, ok: {}, state: 'loading' };
  cache.set(i, e);
  inflight++;
  const loads = VIEWS.map((v) => {
    const img = new Image();
    img.decoding = 'async';
    img.src = v.dir() + GT.files[i];
    e.imgs[v.key] = img;
    return img.decode().then(() => { e.ok[v.key] = true; }, () => { e.ok[v.key] = false; });
  });
  Promise.all(loads).then(() => { e.state = e.ok.fisheye ? 'ready' : 'error'; }).finally(() => {
    inflight--;
    if (i === S.frame) { dirty.cam = true; kick(); }
    pump();
  });
}

function pump() {
  while (inflight < CFG.preload.inflight && queue.length) {
    const i = queue.shift();
    if (!cache.has(i)) startLoad(i);
  }
}

function planPreload() {
  const P = CFG.preload, i0 = S.frame, N = S.N;
  const ahead = clamp(Math.ceil(S.speed * CFG.fps * P.seconds), P.minAhead, P.maxAhead);
  const q = [];
  const push = (i) => { if (i >= 0 && i < N && !cache.has(i)) q.push(i); };
  push(i0);
  for (let k = 1; k <= P.around; k++) { push(i0 + k); push(i0 - k); }
  for (let k = P.around + 1; k <= ahead; k++) push(i0 + k);
  queue = q;
  pump();
  if (cache.size > P.cache) evict(i0, ahead);
}

function evict(i0, ahead) {
  const cost = (i) => (i >= i0 ? (i - i0 <= ahead ? 0 : i - i0) : 2 * (i0 - i));
  const keys = [];
  for (const [i, e] of cache) if (e.state !== 'loading') keys.push(i);
  keys.sort((a, b) => cost(b) - cost(a));
  const target = Math.floor(CFG.preload.cache * 0.85);
  for (const i of keys) {
    if (cache.size <= target || cost(i) === 0) break;
    cache.delete(i);
  }
}

const frameState = (i) => { const e = cache.get(i); return e ? e.state : 'none'; };
const playable = (i) => { const s = frameState(i); return s === 'ready' || s === 'error'; };

/* ------------------------------------------------------------ playback -- */
function setFrame(i, fromPlay = false) {
  i = clamp(Math.round(i), 0, S.N - 1);
  if (!fromPlay) S.playhead = i;
  S.frame = i;
  planPreload();
  dirty.cam = dirty.map = dirty.chart = dirty.hud = true;
  if (!S.playing) scheduleHash();
  kick();
}

function step(d) {
  if (S.playing) pause();
  setFrame(S.frame + d);
}

function play() {
  if (S.frame >= S.N - 1) setFrame(0);
  S.playing = true;
  S.playhead = S.frame;
  document.body.classList.add('playing');
  $('btnPlay').setAttribute('aria-label', 'Pause');
  lastT = 0;
  planPreload();
  kick();
}

function pause() {
  S.playing = false;
  setBuffering(false);
  document.body.classList.remove('playing');
  $('btnPlay').setAttribute('aria-label', 'Play');
  scheduleHash();
  dirty.hud = true; kick();
}

const togglePlay = () => (S.playing ? pause() : play());

function setSpeed(v) {
  S.speed = v;
  $('speed').value = String(v);
  planPreload();
}
function bumpSpeed(d) {
  const k = CFG.speeds.indexOf(S.speed);
  setSpeed(CFG.speeds[clamp((k < 0 ? 2 : k) + d, 0, CFG.speeds.length - 1)]);
}

function setBuffering(b) {
  if (b && !S.buffering) S.bufferSince = performance.now();
  if (b !== S.buffering) { S.buffering = b; dirty.hud = true; }
}

function advance(dt) {
  const N = S.N;
  if (S.frame >= N - 1) { pause(); return; }
  const next = S.playhead + dt * CFG.fps * S.speed;
  const target = Math.min(N - 1, Math.floor(next));
  if (target <= S.frame) { S.playhead = next; setBuffering(false); return; }
  if (playable(target)) {
    S.playhead = next;
    setFrame(target, true);
    setBuffering(false);
  } else {
    // Show the furthest decoded frame on the way, then hold until the next one arrives.
    let j = target - 1;
    while (j > S.frame && !playable(j)) j--;
    if (j > S.frame) setFrame(j, true);
    S.playhead = S.frame + 0.999;
    setBuffering(true);
  }
  if (S.frame >= N - 1) pause();
}

/* URL hash (#f=123) so a frame can be linked / survives reload. */
let hashTimer = 0;
function scheduleHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    try { history.replaceState(null, '', `#f=${S.frame}`); } catch (e) { /* ignore */ }
  }, 300);
}
function frameFromHash() {
  const m = /f=(\d+)/.exec(location.hash);
  return m ? +m[1] : 0;
}

/* -------------------------------------------------------------- canvas -- */
function makeCanvas(id) {
  const canvas = $(id);
  const o = { canvas, ctx: canvas.getContext('2d'), w: 1, h: 1, dpr: 1 };
  new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    o.dpr = window.devicePixelRatio || 1;
    o.w = Math.max(1, r.width); o.h = Math.max(1, r.height);
    canvas.width = Math.round(o.w * o.dpr);
    canvas.height = Math.round(o.h * o.dpr);
    dirty.cam = dirty.map = dirty.chart = dirty.chartBg = true;
    if (o.onResize) o.onResize();
    kick();
  }).observe(canvas);
  return o;
}

let cam, map, chart;
const camViews = {};

/* -------------------------------------------------------------- camera -- */
function drawCam() {
  const e = cache.get(S.frame);
  const msg = $('camMsg');
  if (e && e.state === 'ready') for (const v of VIEWS) if (e.ok[v.key]) lastCamImg[v.key] = e.imgs[v.key];
  if (e && e.state === 'error') {
    msg.hidden = false;
    msg.innerHTML = `Frame <code>${escapeHtml(GT.files[S.frame])}</code> not found.<br>` +
      'Run <code>.venv/bin/python scripts/make_viewer_frames.py</code>';
  } else {
    msg.hidden = true;
  }
  for (const v of VIEWS) drawView(camViews[v.key], lastCamImg[v.key]);
}

function drawView(view, img) {
  const { ctx, canvas } = view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!img) return;
  const s = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const w = img.naturalWidth * s, h = img.naturalHeight * s;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
}

/* ----------------------------------------------------------------- map -- */
const toScreen = (mx, my) => [
  (mx - S.view.cx) * S.view.scale + map.w / 2,
  (my - S.view.cy) * S.view.scale + map.h / 2,
];
const toMosaicPx = (sx, sy) => [
  (sx - map.w / 2) / S.view.scale + S.view.cx,
  (sy - map.h / 2) / S.view.scale + S.view.cy,
];

function tracePath(ctx, xs, ys, indices, from, to) {
  // indices: null for a dense 0..N-1 range, or a sorted list of frame indices.
  const s = S.view.scale, cx = S.view.cx, cy = S.view.cy, hw = map.w / 2, hh = map.h / 2;
  ctx.beginPath();
  let prev = -Infinity;
  const visit = (i) => {
    const x = xs[i], y = ys[i];
    if (!isNum(x)) { prev = -Infinity; return; }
    const sx = (x - cx) * s + hw, sy = (y - cy) * s + hh;
    if (i - prev > CFG.gapBreak) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    prev = i;
  };
  if (!indices) for (let i = from; i <= to; i++) visit(i);
  else for (const i of indices) { if (i < from) continue; if (i > to) break; visit(i); }
}

function strokeHalo(ctx, color, width, alpha = 1) {
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.lineWidth = width + 2.5;
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawPose(ctx, x, y, hdg, color, r, len, ring) {
  if (isNum(hdg)) {
    const a = hdg * Math.PI / 180, dx = Math.sin(a), dy = -Math.cos(a);
    const tx = x + dx * len, ty = y + dy * len;
    const hl = 9, hw = 5, bx = tx - dx * hl, by = ty - dy * hl, px = -dy, py = dx;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(bx, by);
    ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 5; ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tx, ty); ctx.lineTo(bx + px * hw, by + py * hw); ctx.lineTo(bx - px * hw, by - py * hw);
    ctx.closePath();
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = color; ctx.fill();
  }
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke();
}

function drawMap() {
  const { ctx, w: W, h: H, dpr } = map;
  const v = S.view, s = v.scale;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0c0d';
  ctx.fillRect(0, 0, W, H);

  // Basemap: draw only the visible part of the best pyramid level.
  const lvl = pickLevel(s * dpr);
  if (lvl) {
    if (s * dpr > lvl.k * 1.15) requestFullRes();
    const mx0 = Math.max(0, v.cx - W / 2 / s), mx1 = Math.min(MAP.w, v.cx + W / 2 / s);
    const my0 = Math.max(0, v.cy - H / 2 / s), my1 = Math.min(MAP.h, v.cy + H / 2 / s);
    if (mx1 > mx0 && my1 > my0) {
      const [dx, dy] = toScreen(mx0, my0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium';
      ctx.drawImage(lvl.src, mx0 * lvl.k, my0 * lvl.k, (mx1 - mx0) * lvl.k, (my1 - my0) * lvl.k,
        dx, dy, (mx1 - mx0) * s, (my1 - my0) * s);
      ctx.fillStyle = 'rgba(0,0,0,0.12)';  // slight dim so overlays pop
      ctx.fillRect(dx, dy, (mx1 - mx0) * s, (my1 - my0) * s);
    }
  }

  const i = S.frame, N = S.N;
  const live = methods.filter((m) => m.enabled && !m.failed);

  // Predicted trajectories (dim), under the GT.
  for (const m of live) {
    tracePath(ctx, m.x, m.y, m.idx, 0, N - 1);
    strokeHalo(ctx, m.color, 1.5, 0.5);
  }
  // GT: full path dim, traversed part highlighted.
  tracePath(ctx, GT.x, GT.y, null, 0, N - 1);
  strokeHalo(ctx, 'rgba(255,255,255,0.7)', 1.5, 0.8);
  if (i > 0) {
    tracePath(ctx, GT.x, GT.y, null, 0, i);
    strokeHalo(ctx, CFG.gtColor, 3);
  }

  const [gx, gy] = toScreen(GT.x[i], GT.y[i]);
  // Current predictions: GT->prediction error line + pose.
  for (const m of live) {
    if (!isNum(m.x[i])) continue;
    const [px, py] = toScreen(m.x[i], m.y[i]);
    ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(px, py);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = m.color; ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const m of live) {
    if (!isNum(m.x[i])) continue;
    const [px, py] = toScreen(m.x[i], m.y[i]);
    drawPose(ctx, px, py, m.hdg[i], m.color, 5.5, 26, '#111');
  }
  drawPose(ctx, gx, gy, GT.hdg[i], CFG.gtColor, 7, 34, '#fff');

  // Hovered frame on the GT path.
  const hf = S.mapHover;
  if (hf >= 0 && hf !== i) {
    const [hx, hy] = toScreen(GT.x[hf], GT.y[hf]);
    ctx.beginPath(); ctx.arc(hx, hy, 6, 0, Math.PI * 2);
    ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    const label = `#${hf} · ${fmtTime(hf / CFG.fps)} · ${GT.alt[hf].toFixed(0)} m`;
    ctx.font = '12px system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    let lx = hx + 10, ly = hy - 26;
    if (lx + tw + 12 > W) lx = hx - tw - 22;
    if (ly < 4) ly = hy + 10;
    ctx.fillStyle = 'rgba(15,16,18,0.9)';
    ctx.fillRect(lx, ly, tw + 12, 20);
    ctx.fillStyle = '#e8e8eb';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 6, ly + 10);
  }

  updateScalebar();
}

function updateScalebar() {
  const mPerCss = MAP.mPerPx / S.view.scale;
  let len = niceStep(110 * mPerCss);
  if (len / mPerCss > 140) len /= 2;
  const px = len / mPerCss;
  const el = $('scalebar');
  el.querySelector('.bar').style.width = px.toFixed(1) + 'px';
  el.querySelector('span').textContent = (len >= 1000 ? len / 1000 + ' km' : len + ' m') + ' · north up';
}

function fitView(pad = 36) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < S.N; i++) {
    x0 = Math.min(x0, GT.x[i]); x1 = Math.max(x1, GT.x[i]);
    y0 = Math.min(y0, GT.y[i]); y1 = Math.max(y1, GT.y[i]);
  }
  const topPad = pad + 36;  // room for the key / tool overlays
  const sw = (map.w - 2 * pad) / Math.max(1, x1 - x0);
  const sh = (map.h - pad - topPad) / Math.max(1, y1 - y0);
  S.view.scale = clamp(Math.min(sw, sh), CFG.minScale, CFG.maxScale);
  S.view.cx = (x0 + x1) / 2;
  S.view.cy = (y0 + y1) / 2 - (topPad - pad) / 2 / S.view.scale;
  setFollow(false);
  dirty.map = true; kick();
}

function zoomAt(factor, sx, sy) {
  const v = S.view;
  const ns = clamp(v.scale * factor, CFG.minScale, CFG.maxScale);
  if (S.follow || sx == null) { v.scale = ns; dirty.map = true; kick(); return; }
  const [mx, my] = toMosaicPx(sx, sy);
  v.scale = ns;
  v.cx = mx - (sx - map.w / 2) / ns;
  v.cy = my - (sy - map.h / 2) / ns;
  dirty.map = true; kick();
}

function setFollow(on) {
  S.follow = on;
  const b = $('btnFollow');
  b.setAttribute('aria-pressed', String(on));
  if (on) kick();
}

// Glide the view towards the drone while following (GT positions are quantised, so snapping jumps).
function followStep(dt) {
  if (!S.follow || !S.N) return false;
  const v = S.view, tx = GT.x[S.frame], ty = GT.y[S.frame];
  const dx = tx - v.cx, dy = ty - v.cy;
  if (Math.hypot(dx, dy) * v.scale < 0.3) {
    if (dx || dy) { v.cx = tx; v.cy = ty; dirty.map = true; }
    return false;
  }
  const a = dt > 0 ? 1 - Math.exp(-dt / 0.12) : 0.2;
  v.cx += dx * a; v.cy += dy * a;
  dirty.map = true;
  return true;
}

function nearestFrame(sx, sy, maxPx) {
  const [mx, my] = toMosaicPx(sx, sy);
  let best = -1, bd = Infinity;
  for (let i = 0; i < S.N; i++) {
    const dx = GT.x[i] - mx, dy = GT.y[i] - my, d = dx * dx + dy * dy;
    // Exact ties (the drone hovering / quantised GT) resolve to the frame closest in time.
    if (d < bd || (d === bd && Math.abs(i - S.frame) < Math.abs(best - S.frame))) { bd = d; best = i; }
  }
  return Math.sqrt(bd) * S.view.scale <= maxPx ? best : -1;
}

function setupMapInput() {
  const c = map.canvas;
  let drag = null;
  const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  c.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    c.setPointerCapture(e.pointerId);
    const [x, y] = pos(e);
    drag = { x, y, cx: S.view.cx, cy: S.view.cy, moved: false };
  });
  c.addEventListener('pointermove', (e) => {
    const [x, y] = pos(e);
    if (drag) {
      const dx = x - drag.x, dy = y - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4) {
        drag.moved = true;
        setFollow(false);
        c.classList.add('dragging');
        S.mapHover = -1;
      }
      if (drag.moved) {
        S.view.cx = drag.cx - dx / S.view.scale;
        S.view.cy = drag.cy - dy / S.view.scale;
        dirty.map = true; kick();
      }
      return;
    }
    const f = nearestFrame(x, y, 12);
    if (f !== S.mapHover) { S.mapHover = f; c.classList.toggle('hover', f >= 0); dirty.map = true; kick(); }
  });
  const end = (e) => {
    if (!drag) return;
    const wasClick = !drag.moved;
    drag = null;
    c.classList.remove('dragging');
    if (wasClick && e.type === 'pointerup') {
      const [x, y] = pos(e);
      const f = nearestFrame(x, y, 14);
      if (f >= 0) setFrame(f);
    }
  };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
  c.addEventListener('pointerleave', () => {
    if (!drag && S.mapHover >= 0) { S.mapHover = -1; c.classList.remove('hover'); dirty.map = true; kick(); }
  });
  c.addEventListener('wheel', (e) => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const [x, y] = pos(e);
    zoomAt(Math.exp(-e.deltaY * unit * 0.0015), x, y);
  }, { passive: false });
  c.addEventListener('dblclick', (e) => { const [x, y] = pos(e); zoomAt(2, x, y); });
}

/* --------------------------------------------------------------- chart -- */
const CH = { pad: { l: 44, r: 12, t: 8, b: 20 }, y0: 0, y1: 1, bg: null, series: [] };

function chartSeries() {
  const live = methods.filter((m) => m.enabled && !m.failed);
  if (S.chartMode === 'err') return live.map((m) => ({ color: m.color, v: m.err, idx: m.idx, name: m.name, unit: 'm' }));
  if (S.chartMode === 'herr') return live.filter((m) => m.stats.hasHeading)
    .map((m) => ({ color: m.color, v: m.herr, idx: m.idx, name: m.name, unit: '°' }));
  return [{ color: CFG.gtColor, v: GT.alt, idx: null, name: 'GT altitude', unit: 'm' }];
}

const chX = (i) => CH.pad.l + (i / Math.max(1, S.N - 1)) * (chart.w - CH.pad.l - CH.pad.r);
const chY = (v) => CH.pad.t + (1 - (v - CH.y0) / (CH.y1 - CH.y0)) * (chart.h - CH.pad.t - CH.pad.b);
const chFrame = (sx) => clamp(Math.round((sx - CH.pad.l) / (chart.w - CH.pad.l - CH.pad.r) * (S.N - 1)), 0, S.N - 1);

function buildChartBg() {
  const { w: W, h: H, dpr } = chart;
  if (!CH.bg) CH.bg = document.createElement('canvas');
  CH.bg.width = chart.canvas.width; CH.bg.height = chart.canvas.height;
  const ctx = CH.bg.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const series = CH.series = chartSeries();
  const P = CH.pad, x0 = P.l, x1 = W - P.r, yTop = P.t, yBot = H - P.b;

  // y range
  const vals = [];
  for (const s of series) {
    if (s.idx) for (const i of s.idx) { if (isNum(s.v[i])) vals.push(s.v[i]); }
    else for (let i = 0; i < S.N; i++) if (isNum(s.v[i])) vals.push(s.v[i]);
  }
  vals.sort((a, b) => a - b);
  if (S.chartMode === 'herr') { CH.y0 = 0; CH.y1 = 180; }
  else if (S.chartMode === 'err') { CH.y0 = 0; CH.y1 = Math.max(1, quantile(vals, 0.98) * 1.1 || 1); }
  else { CH.y0 = vals.length ? vals[0] : 0; CH.y1 = vals.length ? vals[vals.length - 1] : 1; }
  const yStep = S.chartMode === 'herr' ? 45 : niceStep((CH.y1 - CH.y0) / 4 || 1);
  CH.y0 = Math.floor(CH.y0 / yStep) * yStep;
  CH.y1 = Math.max(CH.y0 + yStep, Math.ceil(CH.y1 / yStep) * yStep);

  // grid + y labels
  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  const unit = S.chartMode === 'herr' ? '°' : '';
  const dec = Number.isInteger(yStep) ? 0 : 1;
  for (let v = CH.y0; v <= CH.y1 + 1e-9; v += yStep) {
    const y = Math.round(chY(v)) + 0.5;
    ctx.strokeStyle = v === CH.y0 ? '#3a3c43' : '#26282d';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillStyle = '#8b8e97';
    ctx.fillText(v.toFixed(dec) + unit, x0 - 6, y);
  }

  // x ticks (frames)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  const pxPerFrame = (x1 - x0) / Math.max(1, S.N - 1);
  const xStep = [100, 250, 500, 1000, 2000].find((st) => st * pxPerFrame >= 70) || 5000;
  for (let f = 0; f < S.N; f += xStep) {
    const x = Math.round(chX(f)) + 0.5;
    ctx.strokeStyle = '#3a3c43';
    ctx.beginPath(); ctx.moveTo(x, yBot); ctx.lineTo(x, yBot + 4); ctx.stroke();
    ctx.fillStyle = '#8b8e97';
    ctx.fillText(String(f), x, H - 5);
  }

  if (!series.length) {
    ctx.fillStyle = '#8b8e97';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('no enabled method with data for this view', (x0 + x1) / 2, (yTop + yBot) / 2);
    return;
  }

  // series
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, yTop - 2, x1 - x0, yBot - yTop + 4); ctx.clip();
  for (const s of series) {
    ctx.beginPath();
    let prev = -Infinity;
    const visit = (i) => {
      const v = s.v[i];
      if (!isNum(v)) { prev = -Infinity; return; }
      const x = chX(i), y = chY(Math.min(v, CH.y1));
      if (i - prev > CFG.gapBreak) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      prev = i;
    };
    if (s.idx) s.idx.forEach(visit); else for (let i = 0; i < S.N; i++) visit(i);
    ctx.lineJoin = 'round';
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.stroke();
    if (s.idx && s.idx.length < (x1 - x0) / 5) {  // sparse predictions: show the samples too
      ctx.fillStyle = s.color;
      for (const i of s.idx) {
        if (!isNum(s.v[i])) continue;
        ctx.beginPath(); ctx.arc(chX(i), chY(Math.min(s.v[i], CH.y1)), 2, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  ctx.restore();
}

function drawChart() {
  const { ctx, canvas, h: H, dpr } = chart;
  if (dirty.chartBg || !CH.bg) { buildChartBg(); dirty.chartBg = false; }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(CH.bg, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const yTop = CH.pad.t, yBot = H - CH.pad.b;

  if (S.chartHover >= 0) {
    const x = Math.round(chX(S.chartHover)) + 0.5;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#6b6e77'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
    ctx.setLineDash([]);
  }
  // playhead
  const i = S.frame, x = Math.round(chX(i)) + 0.5;
  ctx.strokeStyle = '#e8e8eb'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
  ctx.fillStyle = '#e8e8eb';
  ctx.beginPath(); ctx.moveTo(x - 4, yTop - 1); ctx.lineTo(x + 4, yTop - 1); ctx.lineTo(x, yTop + 5); ctx.fill();
  for (const s of CH.series) {
    const v = s.v[i];
    if (!isNum(v)) continue;
    ctx.beginPath(); ctx.arc(x, chY(Math.min(v, CH.y1)), 4, 0, Math.PI * 2);
    ctx.fillStyle = s.color; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#16171a'; ctx.stroke();
  }
}

function updateChartReadout() {
  const f = S.chartHover >= 0 ? S.chartHover : S.frame;
  const parts = [`<b>#${f}</b> ${fmtTime(f / CFG.fps)}`];
  for (const s of CH.series) {
    const v = s.v[f];
    const val = isNum(v) ? (s.unit === '°' ? fmtDeg(v) : fmtM(v)) : '–';
    parts.push(`<span class="sw" style="background:${escapeHtml(s.color)}"></span>${escapeHtml(s.name)} <b>${val}</b>`);
  }
  $('chartReadout').innerHTML = parts.join('');
}

function buildChartTabs() {
  const ok = methods.filter((m) => !m.failed);
  const tabs = [];
  if (ok.length) tabs.push(['err', 'Position error (m)']);
  if (ok.some((m) => m.stats.hasHeading)) tabs.push(['herr', 'Heading error (°)']);
  tabs.push(['alt', 'Altitude (m)']);
  if (!tabs.some(([k]) => k === S.chartMode)) S.chartMode = tabs[0][0];
  const el = $('chartTabs');
  el.innerHTML = '';
  for (const [k, label] of tabs) {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(k === S.chartMode));
    b.addEventListener('click', () => {
      S.chartMode = k;
      for (const o of el.children) o.setAttribute('aria-pressed', String(o === b));
      b.blur();
      dirty.chartBg = dirty.chart = dirty.hud = true; kick();
    });
    el.appendChild(b);
  }
}

function setupChartInput() {
  const c = chart.canvas;
  let scrubbing = false, resume = false;
  const fx = (e) => chFrame(e.clientX - c.getBoundingClientRect().left);
  c.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    c.setPointerCapture(e.pointerId);
    scrubbing = true; resume = S.playing;
    if (S.playing) pause();
    setFrame(fx(e));
  });
  c.addEventListener('pointermove', (e) => {
    const f = fx(e);
    if (scrubbing) setFrame(f);
    if (f !== S.chartHover) { S.chartHover = f; dirty.chart = dirty.hud = true; kick(); }
  });
  const end = () => { if (scrubbing) { scrubbing = false; if (resume) play(); } };
  c.addEventListener('pointerup', end);
  c.addEventListener('pointercancel', end);
  c.addEventListener('pointerleave', () => { S.chartHover = -1; dirty.chart = dirty.hud = true; kick(); });
}

/* ----------------------------------------------------------------- HUD -- */
function buildMethodsTable() {
  const body = $('methodsBody');
  body.innerHTML = '';
  $('methodsTable').hidden = !methods.length;
  methods.forEach((m) => {
    const tr = document.createElement('tr');
    if (m.failed) tr.className = 'err';
    const extra = m.failed ? m.failed : summaryLine(m);
    const tip = m.failed ? m.failed : JSON.stringify({ file: m.file, params: m.params, summary: m.summary }, null, 1);
    tr.innerHTML = `
      <td class="name"><label title="${escapeHtml(tip)}">
        <input type="checkbox" ${m.enabled ? 'checked' : ''} ${m.failed ? 'disabled' : ''} style="accent-color:${escapeHtml(m.color)}">
        <span class="sw" style="background:${escapeHtml(m.color)}"></span>
        <span><span class="mname">${escapeHtml(m.name)}</span><span class="msub">${escapeHtml(extra)}</span></span>
      </label></td>
      <td class="now"></td><td class="now"></td>
      <td class="stat">${m.failed ? '' : fmtM(m.stats.median)}</td>
      <td class="stat">${m.failed ? '' : fmtM(m.stats.mean)}</td>`;
    const cb = tr.querySelector('input');
    cb.addEventListener('change', () => {
      m.enabled = cb.checked;
      cb.blur();
      tr.classList.toggle('off', !m.enabled);
      buildMapKey();
      dirty.map = dirty.chart = dirty.chartBg = dirty.hud = true; kick();
    });
    m.row = tr;
    m.cells = tr.querySelectorAll('td.now');
    body.appendChild(tr);
  });
}

function summaryLine(m) {
  const parts = [`${m.stats.n} frames`];
  if (isNum(m.stats.hMedian)) parts.push(`median Δψ ${m.stats.hMedian.toFixed(1)}°`);
  // Show a few scalar summary fields as-is (keys are method-defined).
  let shown = 0;
  for (const [k, v] of Object.entries(m.summary || {})) {
    if (shown >= 3) break;
    if (/median_err|mean_err|^n$|n_frames|count/i.test(k)) continue;
    if (isNum(v)) { parts.push(`${k} ${fmtNum(v)}`); shown++; }
    else if (typeof v === 'string' && v.length < 24) { parts.push(`${k} ${v}`); shown++; }
  }
  return parts.join(' · ');
}

function buildMapKey() {
  const items = [
    `<span><i style="background:rgba(255,255,255,.75)"></i>GT path</span>`,
    `<span><i style="background:${CFG.gtColor}"></i>traversed</span>`,
    `<span><i class="dot" style="background:${CFG.gtColor}"></i>GT pose</span>`,
  ];
  for (const m of methods) {
    if (m.failed || !m.enabled) continue;
    items.push(`<span><i class="dot" style="background:${escapeHtml(m.color)};box-shadow:0 0 0 1.5px #111"></i>${escapeHtml(m.name)}</span>`);
  }
  $('mapKey').innerHTML = items.join('');
}

function updateHud() {
  const i = S.frame;
  $('hudFrame').innerHTML = `${i}<small> / ${S.N - 1}</small>`;
  $('hudTime').textContent = `${(i / CFG.fps).toFixed(1)} s`;
  $('hudAlt').textContent = GT.alt[i].toFixed(1) + ' m';
  $('hudN').textContent = signed(GT.north[i]) + ' m';
  $('hudE').textContent = signed(GT.east[i]) + ' m';
  const h = GT.hdg[i];
  $('hudHdg').innerHTML = isNum(h) ? `${h.toFixed(1)}°<small> ${compass(h)}</small>` : '–';

  const scrub = $('scrub');
  if (+scrub.value !== i) scrub.value = String(i);
  scrub.style.setProperty('--pct', (i / Math.max(1, S.N - 1) * 100).toFixed(3) + '%');
  $('timeReadout').textContent = `${fmtTime(i / CFG.fps)} / ${fmtTime((S.N - 1) / CFG.fps)}`;

  const buf = S.buffering && performance.now() - S.bufferSince > 250;
  $('camBadge').innerHTML = `#${String(i).padStart(6, '0')} · ${fmtTime(i / CFG.fps)}` +
    (S.playing ? ` · ${S.speed}×` : '') + (buf ? '<span class="buf">buffering…</span>' : '');

  for (const m of methods) {
    if (!m.cells) continue;
    const e = m.failed ? NaN : m.err[i], he = m.failed ? NaN : m.herr[i];
    m.cells[0].textContent = fmtM(e);
    m.cells[1].textContent = fmtDeg(he);
  }
  updateChartReadout();
}

/* ------------------------------------------------------------ controls -- */
function setupControls() {
  const btn = (id, fn) => $(id).addEventListener('click', (e) => { fn(e); e.currentTarget.blur(); });
  btn('btnPlay', togglePlay);
  btn('btnFirst', () => step(-S.N));
  btn('btnLast', () => step(S.N));
  btn('btnBack1', () => step(-1));
  btn('btnFwd1', () => step(1));
  btn('btnBack10', () => step(-10));
  btn('btnFwd10', () => step(10));
  btn('btnFollow', () => setFollow(!S.follow));
  btn('btnFit', () => fitView());
  btn('btnZoomIn', () => zoomAt(1.5));
  btn('btnZoomOut', () => zoomAt(1 / 1.5));

  const sel = $('speed');
  for (const v of CFG.speeds) {
    const o = document.createElement('option');
    o.value = String(v); o.textContent = `${v}×`;
    sel.appendChild(o);
  }
  sel.value = String(S.speed);
  sel.addEventListener('change', () => { setSpeed(+sel.value); sel.blur(); });

  const scrub = $('scrub');
  // Pause while the slider is held, resume on release.
  scrub.addEventListener('pointerdown', () => {
    if (!S.playing) return;
    pause();
    window.addEventListener('pointerup', () => play(), { once: true });
  });
  scrub.addEventListener('input', () => setFrame(+scrub.value));
  scrub.addEventListener('change', () => scrub.blur());

  window.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || t.isContentEditable ||
      (t.tagName === 'INPUT' && !['range', 'checkbox', 'button'].includes(t.type)))) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    switch (e.key) {
      case ' ': case 'k': togglePlay(); break;
      case 'ArrowRight': step(e.shiftKey ? 10 : 1); break;
      case 'ArrowLeft': step(e.shiftKey ? -10 : -1); break;
      case 'Home': step(-S.N); break;
      case 'End': step(S.N); break;
      case 'f': case 'F': setFollow(!S.follow); break;
      case 'r': case 'R': fitView(); break;
      case '[': bumpSpeed(-1); break;
      case ']': bumpSpeed(1); break;
      case '+': case '=': zoomAt(1.5); break;
      case '-': case '_': zoomAt(1 / 1.5); break;
      default: return;
    }
    e.preventDefault();
    // Keep Space/arrows from also activating whatever control still has focus.
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
  }, true);

  window.addEventListener('hashchange', () => setFrame(frameFromHash()));
}

/* ---------------------------------------------------------------- loop -- */
function kick() { if (!rafId) rafId = requestAnimationFrame(tick); }

function tick(now) {
  rafId = 0;
  const dt = lastT ? Math.min(0.1, (now - lastT) / 1000) : 0;
  lastT = now;
  if (!S.N || !MAP) return;
  if (S.playing) advance(dt);
  const gliding = followStep(dt);
  if (S.buffering) dirty.hud = true;
  if (dirty.cam) { drawCam(); dirty.cam = false; }
  if (dirty.map) { drawMap(); dirty.map = false; }
  if (dirty.chart || dirty.chartBg) { drawChart(); dirty.chart = false; }
  if (dirty.hud) { updateHud(); dirty.hud = false; }
  if (S.playing || gliding) kick(); else lastT = 0;
}

/* ---------------------------------------------------------------- init -- */
async function init() {
  cam = makeCanvas('cam');
  for (const v of VIEWS) camViews[v.key] = v.key === 'fisheye' ? cam : makeCanvas(v.canvas);
  map = makeCanvas('map');
  chart = makeCanvas('chart');
  let fitted = false;
  map.onResize = () => { if (!fitted && S.N && MAP) { fitted = true; fitView(); } };

  let gt, meta, viewerMeta;
  try {
    [gt, meta, viewerMeta] = await Promise.all([
      fetchJSON(CFG.paths.gt),
      fetchJSON(CFG.paths.mapMeta),
      fetchJSON(CFG.paths.mapViewer, true).catch(() => null),
    ]);
  } catch (e) {
    setStatus(`Failed to load data: ${e.message}. Serve the repo root (python3 -m http.server) and open /viewer/.`, true);
    throw e;
  }
  MAP = {
    zoom: meta.zoom,
    ox: meta.origin_world_px[0], oy: meta.origin_world_px[1],
    w: meta.size_px[0], h: meta.size_px[1],
    mPerPx: meta.m_per_px,
    image: meta.image || 'ign_z19.jpg',
  };
  $('attrib').textContent = meta.source ? `Orthophoto: ${meta.source}` : 'Orthophoto: IGN';
  $('attrib').title = $('attrib').textContent;

  buildGT(gt);
  // Conventions come from gt.json so the HUD stays truthful if they change.
  if (gt.alt_convention) $('hudAlt').parentElement.title = `Altitude: ${gt.alt_convention}`;
  if (gt.heading_convention) $('hudHdg').parentElement.title = `Heading: ${gt.heading_convention}`;
  $('scrub').max = String(S.N - 1);
  setupControls();
  setupMapInput();
  setupChartInput();
  loadBasemap(viewerMeta);

  await loadMethods();
  buildMethodsTable();
  buildMapKey();
  buildChartTabs();

  const ok = methods.filter((m) => !m.failed).length, bad = methods.length - ok;
  setStatus(`${S.N} frames · ${CFG.fps} fps · ${ok} method${ok === 1 ? '' : 's'}` + (bad ? ` · ${bad} failed to load` : ''), bad > 0);

  if (!fitted && map.w > 1) { fitted = true; fitView(); }
  setFrame(frameFromHash());
  dirty.chartBg = true;
  kick();
}

init();
