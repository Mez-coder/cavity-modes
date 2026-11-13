"use strict";
/* Cavity modes UI. Lengths in mm on screen; the server works in metres. */

const $ = (id) => document.getElementById(id);
const COL = {
  ink: "#1c2b39", muted: "#5d6c7a", rule: "#c3ccd3", copper: "#b1622b", steel: "#2e5c88",
  grid: "#dde3e8",
};
const REST_MEV = { electron: 0.51099895, positron: 0.51099895, proton: 938.27208816 };

// ------------------------------------------------------------------ presets
const PRESETS = {
  smooth: {
    points: [[-25, 9, 0], [0, 9, 0], [6, 14, 1], [14, 36, 1], [25, 41.5, 1], [36, 36, 1], [44, 14, 1], [50, 9, 0], [75, 9, 0]],
    cell: 50, fmax: 8, h: 0.8,
  },
  pillbox: { points: [[0, 0, 0], [0, 38.3, 0], [49.9, 38.3, 0], [49.9, 0, 0]], cell: 49.9, fmax: 8, h: 0.8 },
  pillbox_pipes: {
    points: [[-30, 8, 0], [0, 8, 0], [0, 38.3, 0], [49.9, 38.3, 0], [49.9, 8, 0], [79.9, 8, 0]],
    cell: 49.9, fmax: 8, h: 1.0, haxis: 0.4,
  },
  nose: {
    points: [[-20, 6, 0], [0, 6, 0], [4, 9, 1], [10, 10, 1], [14, 30, 1], [24.9, 35, 1], [35.8, 30, 1], [39.8, 10, 1],
             [45.8, 9, 1], [49.8, 6, 0], [69.8, 6, 0]],
    cell: 49.8, fmax: 8, h: 1.0, haxis: 0.3,
  },
};

// ------------------------------------------------------------------ state
const S = {
  points: [], wall: [], geomError: null, solve: null, stale: false, selMode: null, fields: null,
  quantity: null, phase: 0, playing: false, drives: [], selPoint: null, V: null, track: null,
};

// ------------------------------------------------------------------ utilities
function setStatus(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + kind;
}
async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
  if (!res.ok || data.error) throw new Error(data.error || `Server returned ${res.status}`);
  return data;
}
function fmt(v, p = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(p - 1).replace("e+", "e");
  return Number(v.toPrecision(p)).toLocaleString("en-GB", { maximumFractionDigits: 6 });
}
function niceTicks(a, b, n = 5) {
  const span = b - a;
  if (!(span > 0)) return [a];
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function tickLabel(v) { return Math.abs(v) < 1e-12 ? "0" : +v.toPrecision(6) + ""; }
function fitCanvas(cv, w, h) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
function beta() {
  const m0 = REST_MEV[$("species").value], g = 1 + (+$("ek").value) / m0;
  return Math.sqrt(1 - 1 / (g * g));
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ------------------------------------------------------------------ colour maps
function lerpStops(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
const DIVERGING = [[24, 66, 110], [86, 132, 176], [190, 207, 222], [244, 244, 240], [229, 196, 168], [196, 123, 70], [124, 52, 18]];
const SEQUENTIAL = [[244, 244, 240], [233, 205, 178], [207, 146, 96], [168, 88, 40], [96, 42, 14]];
function makeLUT(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) { const c = lerpStops(stops, i / 255); lut.set([c[0], c[1], c[2]], i * 3); }
  return lut;
}
const LUT_DIV = makeLUT(DIVERGING), LUT_SEQ = makeLUT(SEQUENTIAL);

// ------------------------------------------------------------------ view transform
function computeView() {
  const ed = $("editor");
  const width = Math.max(320, ed.clientWidth || ed.parentElement.clientWidth);
  const zs = S.points.map((p) => p.z), rs = S.points.map((p) => p.r);
  let z0 = Math.min(...zs), z1 = Math.max(...zs), rmax = Math.max(...rs);
  const padz = 0.04 * (z1 - z0);
  z0 -= padz; z1 += padz;
  const rtop = rmax * 1.18 + 2;
  const left = 46, right = 12, top = 24, bottom = 30;
  let s = (width - left - right) / (z1 - z0);
  const maxH = Math.min(440, window.innerHeight * 0.5);
  if (rtop * s > maxH) s = maxH / rtop;
  S.V = { z0, z1, rtop, s, left, top, bottom, right, W: left + (z1 - z0) * s + right, H: top + rtop * s + bottom };
}
const X = (z) => S.V.left + (z - S.V.z0) * S.V.s;
const Y = (r) => S.V.top + (S.V.rtop - r) * S.V.s;
const Z_ = (x) => S.V.z0 + (x - S.V.left) / S.V.s;
const R_ = (y) => S.V.rtop - (y - S.V.top) / S.V.s;

// ------------------------------------------------------------------ geometry drawing
function drawSVG() {
  const svg = $("svg"), V = S.V;
  svg.setAttribute("width", V.W); svg.setAttribute("height", V.H);
  svg.innerHTML = "";
  const defs = svgEl("defs", {}, svg);
  const pat = svgEl("pattern", { id: "hatch", patternUnits: "userSpaceOnUse", width: 7, height: 7, patternTransform: "rotate(45)" }, defs);
  svgEl("rect", { width: 7, height: 7, fill: "#efe2d6" }, pat);
  svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: COL.copper, "stroke-width": 1, "stroke-opacity": 0.55 }, pat);

  // ticks
  const gt = svgEl("g", { class: "tick" }, svg);
  for (const z of niceTicks(V.z0, V.z1, 8)) {
    svgEl("line", { x1: X(z), x2: X(z), y1: Y(0), y2: Y(0) + 5 }, gt);
    const t = svgEl("text", { x: X(z), y: Y(0) + 17, "text-anchor": "middle" }, gt); t.textContent = tickLabel(z);
  }
  for (const r of niceTicks(0, V.rtop, 4)) {
    svgEl("line", { x1: V.left - 5, x2: V.left, y1: Y(r), y2: Y(r) }, gt);
    const t = svgEl("text", { x: V.left - 8, y: Y(r) + 4, "text-anchor": "end" }, gt); t.textContent = tickLabel(r);
  }
  const zt = svgEl("text", { x: V.W - V.right, y: V.H - 2, "text-anchor": "end", class: "axis-title" }, svg); zt.textContent = "z / mm";
  const rt = svgEl("text", { x: 4, y: 12, class: "axis-title" }, svg); rt.textContent = "r / mm";

  const wall = S.wall.length ? S.wall : S.points.map((p) => [p.z, p.r]);
  if (wall.length >= 2) {
    // metal: from the wall up to the top of the view, continued to the view edges
    const first = wall[0], last = wall[wall.length - 1];
    let d = `M${X(V.z0)},${Y(V.rtop)} L${X(V.z0)},${Y(first[1])}`;
    for (const [z, r] of wall) d += ` L${X(z)},${Y(r)}`;
    d += ` L${X(V.z1)},${Y(last[1])} L${X(V.z1)},${Y(V.rtop)} Z`;
    svgEl("path", { d, class: "metal" }, svg);
    let w = `M${X(V.z0)},${Y(first[1])}`;
    for (const [z, r] of wall) w += ` L${X(z)},${Y(r)}`;
    w += ` L${X(V.z1)},${Y(last[1])}`;
    svgEl("path", { d: w, class: "wall" + (S.geomError ? " invalid" : "") }, svg);
    // port planes
    for (const [pt, side] of [[first, "in"], [last, "out"]]) {
      if (pt[1] > 0) {
        svgEl("line", { x1: X(pt[0]), x2: X(pt[0]), y1: Y(0), y2: Y(pt[1]), class: "port" }, svg);
        const bc = $(side === "in" ? "bcin" : "bcout").value;
        const t = svgEl("text", { x: X(pt[0]) + (side === "in" ? 4 : -4), y: Y(pt[1] * 0.5),
          "text-anchor": side === "in" ? "start" : "end", class: "port-label" }, svg);
        t.textContent = bc === "metal" ? "metal" : bc[0].toUpperCase();
        t.appendChild(svgEl("title", {})).textContent = `${bc} boundary`;
      }
    }
  }
  if (S.geomError && S.cross) {
    const cx = X(S.cross[0]), cy = Y(S.cross[1]);
    svgEl("circle", { cx, cy, r: 9, fill: "none", stroke: "#a3271b", "stroke-width": 2 }, svg);
  }
  // centre line (axis)
  svgEl("line", { x1: X(V.z0), x2: X(V.z1), y1: Y(0), y2: Y(0), class: "centre" }, svg);
  // control polygon + points
  svgEl("polyline", { points: S.points.map((p) => `${X(p.z)},${Y(p.r)}`).join(" "), class: "ctrl-poly" }, svg);
  S.points.forEach((p, i) => {
    const attrs = { class: "pt" + (i === S.selPoint ? " sel" : ""), tabindex: 0, "data-i": i,
      role: "slider", "aria-label": `Wall point ${i + 1}, ${p.smooth ? "smooth" : "corner"}, z ${p.z.toFixed(1)} mm, r ${p.r.toFixed(1)} mm` };
    const el = p.smooth
      ? svgEl("circle", { ...attrs, cx: X(p.z), cy: Y(p.r), r: 5 }, svg)
      : svgEl("rect", { ...attrs, x: X(p.z) - 4.5, y: Y(p.r) - 4.5, width: 9, height: 9 }, svg);
    el.addEventListener("pointerdown", (e) => startDrag(e, i));
    el.addEventListener("contextmenu", (e) => { e.preventDefault(); toggleSmooth(i); });
    el.addEventListener("focus", () => selectPoint(i, false));
  });
  if (S.geomError || (S.stale && S.fields)) {
    const t = svgEl("text", { x: V.left + 8, y: V.top + 16, class: "stale-note" }, svg);
    t.textContent = S.geomError ? `${S.geomError} Move a point to untangle it.` : "Shape changed. The field shown is from the last solve.";
  }
}

// ------------------------------------------------------------------ point editing
let drag = null;
function startDrag(e, i) {
  e.preventDefault();
  selectPoint(i, true);
  drag = { i, id: e.pointerId };
  $("editor").classList.add("dragging");
  $("svg").setPointerCapture(e.pointerId);
}
$("svg").addEventListener("pointermove", (e) => {
  if (!drag) return;
  const rect = $("svg").getBoundingClientRect();
  let z = Z_(e.clientX - rect.left), r = Math.max(0, R_(e.clientY - rect.top));
  const snap = e.altKey ? 0 : 0.1;
  if (snap) { z = Math.round(z / snap) * snap; r = Math.round(r / snap) * snap; }
  if (r < 0.15) r = 0;
  S.points[drag.i].z = +z.toFixed(3); S.points[drag.i].r = +r.toFixed(3);
  geometryChanged(false);
});
$("svg").addEventListener("pointerup", () => {
  if (!drag) return;
  drag = null;
  $("editor").classList.remove("dragging");
  geometryChanged(true);
});
$("svg").addEventListener("dblclick", (e) => {
  const rect = $("svg").getBoundingClientRect();
  const z = Z_(e.clientX - rect.left), r = Math.max(0, R_(e.clientY - rect.top));
  // insert into the nearest control-polygon segment
  let best = 0, bestD = Infinity;
  for (let i = 0; i < S.points.length - 1; i++) {
    const a = S.points[i], b = S.points[i + 1];
    const dz = b.z - a.z, dr = b.r - a.r, L2 = dz * dz + dr * dr || 1e-12;
    const t = Math.max(0, Math.min(1, ((z - a.z) * dz + (r - a.r) * dr) / L2));
    const d = Math.hypot(a.z + t * dz - z, a.r + t * dr - r);
    if (d < bestD) { bestD = d; best = i; }
  }
  S.points.splice(best + 1, 0, { z: +z.toFixed(2), r: +r.toFixed(2), smooth: true });
  selectPoint(best + 1, true);
  geometryChanged(true);
});
document.addEventListener("keydown", (e) => {
  if (S.selPoint === null || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const p = S.points[S.selPoint];
  const step = e.shiftKey ? 1 : 0.1;
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
  if (moves[e.key]) {
    e.preventDefault();
    p.z = +(p.z + moves[e.key][0]).toFixed(3); p.r = Math.max(0, +(p.r + moves[e.key][1]).toFixed(3));
    geometryChanged(false); debouncedCommit();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault(); deletePoint(S.selPoint);
  }
});
const debouncedCommit = debounce(() => geometryChanged(true), 600);

function selectPoint(i, focus) {
  S.selPoint = i;
  const p = S.points[i];
  $("inspector").hidden = false;
  $("insp-name").textContent = `Point ${i + 1} of ${S.points.length}`;
  $("insp-z").value = p.z; $("insp-r").value = p.r; $("insp-smooth").checked = !!p.smooth;
  $("insp-smooth").disabled = i === 0 || i === S.points.length - 1;
  drawSVG();
  if (focus) { const el = $("svg").querySelector(`[data-i="${i}"]`); if (el) el.focus({ preventScroll: true }); }
}
function toggleSmooth(i) {
  if (i === 0 || i === S.points.length - 1) return;
  S.points[i].smooth = !S.points[i].smooth;
  selectPoint(i, false);
  geometryChanged(true);
}
function deletePoint(i) {
  if (S.points.length <= 2) return;
  S.points.splice(i, 1);
  S.selPoint = null; $("inspector").hidden = true;
  geometryChanged(true);
}
$("insp-z").addEventListener("change", () => { S.points[S.selPoint].z = +$("insp-z").value; geometryChanged(true); });
$("insp-r").addEventListener("change", () => { S.points[S.selPoint].r = Math.max(0, +$("insp-r").value); geometryChanged(true); });
$("insp-smooth").addEventListener("change", () => toggleSmooth(S.selPoint));
$("insp-del").addEventListener("click", () => deletePoint(S.selPoint));

// ------------------------------------------------------------------ profile preview + solve
function geomSpec() {
  return {
    points: S.points.map((p) => ({ z: p.z, r: p.r, smooth: !!p.smooth })),
    bc_in: $("bcin").value, bc_out: $("bcout").value,
    cell_length: +$("cell").value || null,
  };
}
// Previews are serialised; rapid requests (dragging) coalesce into one queued
// call, and every caller gets a promise that settles after a preview that
// includes its change.
let previewChain = Promise.resolve(), previewQueued = false;
function refreshPreview() {
  if (!previewQueued) {
    previewQueued = true;
    previewChain = previewChain.then(() => { previewQueued = false; return doPreview(); });
  }
  return previewChain;
}
async function doPreview() {
  try {
    const d = await post("/api/profile", geomSpec());
    S.wall = d.wall; S.geomError = d.problem; S.cross = d.cross;
    if (d.problem) setStatus(d.problem, "error");
  } catch (err) {
    S.geomError = err.message; S.cross = null;
    setStatus(err.message, "error");
  }
  $("solve").disabled = !!S.geomError;
  updateEndControls();
  drawSVG();
}
function updateEndControls() {
  const p = S.points;
  $("bcin-wrap").hidden = !(p.length && p[0].r > 0);
  $("bcout-wrap").hidden = !(p.length && p[p.length - 1].r > 0);
}
function syncInspector() {
  if (S.selPoint === null || !S.points[S.selPoint]) return;
  const p = S.points[S.selPoint];
  if (document.activeElement !== $("insp-z")) $("insp-z").value = p.z;
  if (document.activeElement !== $("insp-r")) $("insp-r").value = p.r;
}
function geometryChanged(commit) {
  if (S.solve) S.stale = true;
  syncInspector();
  drawSVG(); drawHeat();
  refreshPreview().then(() => {
    if (commit && !S.geomError && $("auto").checked) solve();
    else if (commit && !S.geomError) setStatus("Shape changed. Solve to update the modes.");
  });
}

let solveSeq = 0;
async function solve() {
  if (S.geomError) { setStatus(S.geomError, "error"); return; }
  const seq = ++solveSeq;
  setStatus("Solving…", "busy");
  $("solve").disabled = true;
  try {
    const spec = {
      ...geomSpec(), f_max_GHz: +$("fmax").value, h_mm: +$("h").value,
      h_axis_mm: +$("haxis").value || null, beta: beta(),
      kinds: { TM: $("tm").checked, TE: $("te").checked },
    };
    const d = await post("/api/solve", spec);
    if (seq !== solveSeq) return; // a newer solve superseded this one
    S.solve = d; S.wall = d.wall; S.stale = false;
    S.drives = S.drives.filter((dr) => dr.mode < d.modes.length);
    if (!S.drives.length) {
      const fund = d.modes.find((m) => m.kind === "TM");
      if (fund) S.drives.push({ mode: fund.index, set: "voltage", value: 1, phase: 0 });
    }
    const nm = d.modes.length;
    setStatus(`${nm} mode${nm === 1 ? "" : "s"} below ${$("fmax").value} GHz, solved in ${d.time_s.toFixed(2)} s ` +
              `on a ${d.n_nodes.toLocaleString("en-GB")}-node ${d.mesher} mesh.`);
    renderModes(); renderDrives();
    const keep = S.selMode !== null && S.selMode < nm ? S.selMode : (nm ? 0 : null);
    if (keep !== null) await selectMode(keep); else { S.fields = null; drawHeat(); drawAxisPlot(); }
    drawSVG();
  } catch (err) {
    if (seq === solveSeq) setStatus(err.message, "error");
  } finally {
    if (seq === solveSeq) $("solve").disabled = !!S.geomError;
  }
}

// ------------------------------------------------------------------ mode table
function renderModes() {
  const tb = $("modes").querySelector("tbody");
  $("beta-note").textContent = S.solve ? `T, R/Q and ZT² are for β = ${beta().toFixed(4)} (from the beam energy).` : "";
  if (!S.solve || !S.solve.modes.length) {
    tb.innerHTML = `<tr><td colspan="8" class="empty">${S.solve ? "No modes below this frequency. Raise the limit." : "Solve to list the modes of this cell."}</td></tr>`;
    return;
  }
  tb.innerHTML = "";
  for (const m of S.solve.modes) {
    const tr = document.createElement("tr");
    tr.className = (m.index === S.selMode ? "sel " : "") + m.kind.toLowerCase();
    tr.tabIndex = 0;
    const driven = S.drives.some((d) => d.mode === m.index);
    const tm = m.kind === "TM";
    tr.innerHTML = `<td>${m.label}</td><td>${m.f_GHz.toFixed(4)}</td><td>${fmt(m.Q0, 5)}</td>` +
      `<td>${tm ? fmt(m.T, 3) : "–"}</td><td>${tm ? fmt(m.R_over_Q, 4) : "–"}</td><td>${tm ? fmt(m.ZTT_MOhm_m, 4) : "–"}</td>` +
      `<td>${tm ? fmt(m.Epk_Eacc, 3) : "–"}</td>` +
      `<td><input type="checkbox" aria-label="Drive ${m.label}" ${driven ? "checked" : ""}></td>`;
    tr.addEventListener("click", (e) => { if (e.target.tagName !== "INPUT") selectMode(m.index); });
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") selectMode(m.index); });
    tr.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) S.drives.push({ mode: m.index, set: tm ? "voltage" : "energy", value: tm ? 1 : 0.1, phase: 0 });
      else S.drives = S.drives.filter((d) => d.mode !== m.index);
      renderDrives();
    });
    tb.appendChild(tr);
  }
}
const refreshFigures = debounce(async () => {
  if (!S.solve) return;
  try {
    const d = await post("/api/figures", { solve_id: S.solve.solve_id, beta: beta() });
    S.solve.modes = d.modes; renderModes();
  } catch (err) { setStatus(err.message, "error"); }
}, 300);

// ------------------------------------------------------------------ field display
const QUANTITIES = {
  TM: [["Emag", "|E|"], ["Ez", "Ez"], ["Er", "Er"], ["Bphi", "Bφ"]],
  TE: [["Ephi", "Eφ"], ["Bmag", "|B|"], ["Bz", "Bz"], ["Br", "Br"]],
};
async function selectMode(i) {
  S.selMode = i;
  renderModes();
  try {
    const d = await post("/api/fields", { solve_id: S.solve.solve_id, index: i });
    if (S.selMode !== i) return;
    prepareFields(d);
  } catch (err) { setStatus(err.message, "error"); }
}
function prepareFields(d) {
  const F = {};
  for (const [k, v] of Object.entries(d.fields)) F[k] = Float32Array.from(v);
  const n = d.nr * d.nz;
  if (d.kind === "TM") { F.Emag = new Float32Array(n); for (let i = 0; i < n; i++) F.Emag[i] = Math.hypot(F.Ez[i], F.Er[i]); }
  else { F.Bmag = new Float32Array(n); for (let i = 0; i < n; i++) F.Bmag[i] = Math.hypot(F.Bz[i], F.Br[i]); }
  const max = {};
  for (const [k, v] of Object.entries(F)) { let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); max[k] = m || 1; }
  S.fields = { ...d, F, max, inside: Uint8Array.from(d.inside) };
  const sel = $("quantity"), opts = QUANTITIES[d.kind];
  if (!opts.some(([k]) => k === S.quantity)) S.quantity = opts[0][0];
  sel.innerHTML = opts.map(([k, l]) => `<option value="${k}" ${k === S.quantity ? "selected" : ""}>${l}</option>`).join("");
  drawHeat(); drawAxisPlot();
}
$("quantity").addEventListener("change", (e) => { S.quantity = e.target.value; drawHeat(); });

const off = document.createElement("canvas");
function drawHeat() {
  const V = S.V, cv = $("heat");
  const ctx = fitCanvas(cv, V.W, V.H);
  $("editor").style.height = V.H + "px";
  const f = S.fields;
  if (!f) return;
  const q = S.quantity, vals = f.F[q], mx = f.max[q];
  const isE = q.startsWith("E"), tf = isE ? Math.cos(S.phase) : Math.sin(S.phase);
  const signed = !q.endsWith("mag");
  const lut = signed ? LUT_DIV : LUT_SEQ;
  off.width = f.nz; off.height = f.nr;
  const octx = off.getContext("2d"), img = octx.createImageData(f.nz, f.nr);
  for (let j = 0; j < f.nr; j++) {
    const row = f.nr - 1 - j; // image row 0 is the top (largest r)
    for (let i = 0; i < f.nz; i++) {
      const k = j * f.nz + i, o = (row * f.nz + i) * 4;
      if (!f.inside[k]) { img.data[o + 3] = 0; continue; }
      const v = vals[k] * (signed ? tf : Math.abs(tf)) / mx;
      const t = signed ? 0.5 + 0.5 * v : v;
      const li = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
      img.data[o] = lut[li]; img.data[o + 1] = lut[li + 1]; img.data[o + 2] = lut[li + 2];
      img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.globalAlpha = S.stale ? 0.3 : 1;
  ctx.imageSmoothingEnabled = true;
  const x0 = X(f.z_mm[0]), x1 = X(f.z_mm[1]), y0 = Y(f.r_mm[1]), y1 = Y(0);
  ctx.drawImage(off, x0, y0, x1 - x0, y1 - y0);
  // field arrows: E for TM, B for TE
  const [az, ar] = f.kind === "TM" ? ["Ez", "Er"] : ["Bz", "Br"];
  const vmax = f.max[f.kind === "TM" ? "Emag" : "Bmag"];
  const atf = f.kind === "TM" ? Math.cos(S.phase) : Math.sin(S.phase);
  const spacing = 22, px = (x1 - x0) / (f.nz - 1), py = (y1 - y0) / (f.nr - 1);
  ctx.strokeStyle = "rgba(28,43,57,0.6)"; ctx.fillStyle = "rgba(28,43,57,0.6)"; ctx.lineWidth = 1;
  for (let y = y1 - spacing / 2; y > y0; y -= spacing) {
    for (let x = x0 + spacing / 2; x < x1; x += spacing) {
      const i = Math.round((x - x0) / px), j = Math.round((y1 - y) / py);
      if (i < 0 || j < 0 || i >= f.nz || j >= f.nr) continue;
      const k = j * f.nz + i;
      if (!f.inside[k]) continue;
      const vz = f.F[az][k] * atf / vmax, vr = f.F[ar][k] * atf / vmax;
      const m = Math.hypot(vz, vr);
      if (m < 0.04) continue;
      const L = 0.85 * spacing * Math.sqrt(m), ux = vz / m, uy = -vr / m;
      const xa = x - ux * L / 2, ya = y - uy * L / 2, xb = x + ux * L / 2, yb = y + uy * L / 2;
      ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb, yb); ctx.stroke();
      const hs = Math.min(4, L * 0.35);
      ctx.beginPath(); ctx.moveTo(xb, yb);
      ctx.lineTo(xb - ux * hs - uy * hs * 0.6, yb - uy * hs + ux * hs * 0.6);
      ctx.lineTo(xb - ux * hs + uy * hs * 0.6, yb - uy * hs - ux * hs * 0.6);
      ctx.closePath(); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function drawAxisPlot() {
  const V = S.V, h = 130;
  const ctx = fitCanvas($("axisplot"), V.W, h);
  const top = 12, bottom = 22, ph = h - top - bottom;
  ctx.font = "11px system-ui, sans-serif";
  const f = S.fields;
  ctx.fillStyle = COL.muted;
  if (!f) return;
  if (f.kind !== "TM") {
    ctx.fillText(`${f.label} has no on-axis Ez, so it cannot exchange energy with a beam on axis.`, V.left, top + 20);
    return;
  }
  const z = f.axis.z_mm, ez = f.axis.Ez.map((v) => v / 1e6);
  let m = Math.max(...ez.map(Math.abs)) || 1;
  const yv = (v) => top + ph / 2 - (v / m) * (ph / 2);
  ctx.strokeStyle = COL.rule; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(V.left, yv(0)); ctx.lineTo(V.W - V.right, yv(0)); ctx.stroke();
  for (const t of niceTicks(-m, m, 3)) {
    ctx.fillStyle = COL.muted; ctx.textAlign = "right"; ctx.fillText(tickLabel(t), V.left - 8, yv(t) + 4);
  }
  ctx.textAlign = "left";
  ctx.fillText(`Ez on axis, MV/m at 1 J (${f.label}, ${f.f_GHz.toFixed(4)} GHz)`, V.left + 6, top - 1);
  ctx.strokeStyle = COL.copper; ctx.lineWidth = 1.6;
  ctx.beginPath();
  const tf = Math.cos(S.phase);
  z.forEach((zz, i) => { const x = X(zz), y = yv(ez[i] * (S.playing || S.phase ? tf : 1)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = COL.muted; ctx.textAlign = "right";
  ctx.fillText("z / mm", V.W - V.right, h - 4);
  ctx.textAlign = "left";
}

// animation (user-started only)
function frame() {
  if (!S.playing) return;
  S.phase = (S.phase + (2 * Math.PI) / 120) % (2 * Math.PI);
  $("rfphase").value = Math.round((S.phase * 180) / Math.PI);
  drawHeat(); drawAxisPlot();
  requestAnimationFrame(frame);
}
$("play").addEventListener("click", () => {
  S.playing = !S.playing;
  $("play").setAttribute("aria-pressed", S.playing);
  $("play").textContent = S.playing ? "Pause" : "Animate";
  if (S.playing) requestAnimationFrame(frame);
});
$("rfphase").addEventListener("input", (e) => { S.phase = (+e.target.value * Math.PI) / 180; drawHeat(); drawAxisPlot(); });

// ------------------------------------------------------------------ drives
function renderDrives() {
  const tb = $("drives").querySelector("tbody");
  $("track").disabled = !S.solve || !S.drives.length;
  if (!S.drives.length) { tb.innerHTML = `<tr><td colspan="5" class="empty">Tick “Drive” on a mode to add it.</td></tr>`; return; }
  tb.innerHTML = "";
  S.drives.forEach((d, k) => {
    const m = S.solve.modes[d.mode];
    const tr = document.createElement("tr");
    const tm = m.kind === "TM";
    tr.innerHTML = `<td>${m.label} <span class="note">${m.f_GHz.toFixed(3)} GHz</span></td>
      <td><select aria-label="Set ${m.label} by">${tm ? `<option value="voltage" ${d.set === "voltage" ? "selected" : ""}>V eff (MV)</option>` : ""}
        <option value="energy" ${d.set === "energy" ? "selected" : ""}>U (J)</option></select></td>
      <td><input type="number" step="0.1" value="${d.value}" aria-label="${m.label} amplitude"></td>
      <td><input type="number" step="5" value="${d.phase}" aria-label="${m.label} phase"></td>
      <td><button type="button" aria-label="Stop driving ${m.label}">×</button></td>`;
    const [sel, val, ph] = [tr.querySelector("select"), ...tr.querySelectorAll("input")];
    sel.addEventListener("change", () => { d.set = sel.value; });
    val.addEventListener("change", () => { d.value = +val.value; });
    ph.addEventListener("change", () => { d.phase = +ph.value; });
    tr.querySelector("button").addEventListener("click", () => { S.drives.splice(k, 1); renderDrives(); renderModes(); });
    tb.appendChild(tr);
  });
}

// ------------------------------------------------------------------ tracking
async function runTrack() {
  if (!S.solve) return;
  if (S.stale) { setStatus("The shape changed since the last solve. Solve first, then track.", "error"); return; }
  const n = +$("npart").value;
  setStatus(`Tracking ${n.toLocaleString("en-GB")} particles…`, "busy");
  $("track").disabled = true;
  try {
    const d = await post("/api/track", {
      solve_id: S.solve.solve_id, crest: $("crest").value,
      beam: {
        species: $("species").value, Ek_MeV: +$("ek").value, n, seed: +$("seed").value,
        emit_nx_um: +$("enx").value, emit_ny_um: +$("eny").value, beta_x: +$("bx").value, beta_y: +$("by").value,
        alpha_x: +$("ax").value, alpha_y: +$("ay").value, sigma_t_ps: +$("st").value, sigma_delta: +$("sd").value / 100,
      },
      components: S.drives.map((dr) => ({ mode: dr.mode, set: dr.set, value: dr.value, phase_deg: dr.phase })),
    });
    S.track = d;
    const gain = d.stats.out.Ek_ref_MeV - d.stats.in.Ek_ref_MeV;
    setStatus(`Tracked ${d.n.toLocaleString("en-GB")} particles in ${d.time_s.toFixed(2)} s. ` +
              `The reference particle ${gain >= 0 ? "gained" : "lost"} ${fmt(Math.abs(gain) * 1e3, 4)} keV.`);
    renderTrack();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    $("track").disabled = !S.drives.length;
  }
}
$("track").addEventListener("click", runTrack);

function axesBox(ctx, box, xr, yr, xlabel, ylabel) {
  const { x, y, w, h } = box;
  ctx.font = "11px system-ui, sans-serif";
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1; ctx.fillStyle = COL.muted;
  const xs = (v) => x + ((v - xr[0]) / (xr[1] - xr[0])) * w;
  const ys = (v) => y + h - ((v - yr[0]) / (yr[1] - yr[0])) * h;
  ctx.textAlign = "center";
  for (const t of niceTicks(xr[0], xr[1], 4)) {
    ctx.beginPath(); ctx.moveTo(xs(t), y); ctx.lineTo(xs(t), y + h); ctx.stroke();
    ctx.fillText(tickLabel(t), xs(t), y + h + 13);
  }
  ctx.textAlign = "right";
  for (const t of niceTicks(yr[0], yr[1], 4)) {
    ctx.beginPath(); ctx.moveTo(x, ys(t)); ctx.lineTo(x + w, ys(t)); ctx.stroke();
    ctx.fillText(tickLabel(t), x - 5, ys(t) + 4);
  }
  ctx.strokeStyle = COL.rule; ctx.strokeRect(x, y, w, h);
  ctx.textAlign = "center"; ctx.fillText(xlabel, x + w / 2, y + h + 27);
  ctx.save(); ctx.translate(x - 40, y + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore();
  ctx.textAlign = "left";
  return { xs, ys };
}
function range(arrs, pad = 0.06) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!(hi > lo)) { hi = lo + 1; lo -= 1; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}
function scatter(cv, kx, ky, xlabel, ylabel) {
  const w = cv.clientWidth || 300, h = Math.round(Math.min(300, Math.max(200, w * 0.72)));
  const ctx = fitCanvas(cv, w, h);
  const box = { x: 50, y: 8, w: w - 60, h: h - 44 };
  const ps = S.track && S.track.phase_space;
  if (!ps) { axesBox(ctx, box, [-1, 1], [-1, 1], xlabel, ylabel); return; }
  const xr = range([ps.in[kx], ps.out[kx]]), yr = range([ps.in[ky], ps.out[ky]]);
  const { xs, ys } = axesBox(ctx, box, xr, yr, xlabel, ylabel);
  ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
  for (const [tag, col] of [["in", "rgba(46,92,136,0.35)"], ["out", "rgba(177,98,43,0.35)"]]) {
    ctx.fillStyle = col;
    const a = ps[tag][kx], b = ps[tag][ky];
    for (let i = 0; i < a.length; i++) ctx.fillRect(xs(a[i]) - 0.8, ys(b[i]) - 0.8, 1.6, 1.6);
  }
  ctx.restore();
}
function drawEnvelope() {
  const cv = $("envelope"), w = cv.clientWidth || 600, h = 190;
  const ctx = fitCanvas(cv, w, h);
  const box = { x: 50, y: 22, w: w - 60, h: h - 58 };
  const e = S.track && S.track.envelope;
  if (!e) { axesBox(ctx, box, [0, 1], [0, 1], "z / mm", "rms size / mm"); return; }
  const xr = [e.z_mm[0], e.z_mm[e.z_mm.length - 1]], yr = range([e.sx_mm, e.sy_mm, [0]], 0.08);
  const { xs, ys } = axesBox(ctx, box, xr, yr, "z / mm", "rms size / mm");
  for (const [k, col, lab] of [["sx_mm", COL.copper, "σx"], ["sy_mm", COL.steel, "σy"]]) {
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.beginPath();
    e.z_mm.forEach((z, i) => (i ? ctx.lineTo(xs(z), ys(e[k][i])) : ctx.moveTo(xs(z), ys(e[k][i]))));
    ctx.stroke();
  }
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = COL.copper; ctx.fillText("σx", box.x + 6, 14);
  ctx.fillStyle = COL.steel; ctx.fillText("σy", box.x + 30, 14);
  ctx.fillStyle = COL.muted; ctx.fillText("rms beam size along the cell", box.x + 56, 14);
}
const STAT_ROWS = [
  ["Reference energy / MeV", "Ek_ref_MeV", 1, "diff"],
  ["rms energy spread / keV", "sigma_E_keV", 1, "diff"],
  ["Bunch length σt / ps", "sigma_t_ps", 1, "diff"],
  ["εn,x / µm", "emit_nx_m", 1e6, "pct"],
  ["εn,y / µm", "emit_ny_m", 1e6, "pct"],
  ["βx / m", "beta_x_m", 1, "diff"],
  ["αx", "alpha_x", 1, "diff"],
  ["βy / m", "beta_y_m", 1, "diff"],
  ["αy", "alpha_y", 1, "diff"],
  ["Longitudinal ε / keV ps", "emit_long_keV_ps", 1, "pct"],
  ["x–t tilt / mm ps⁻¹", "tilt_x_mm_per_ps", 1, "diff"],
  ["Mean x′ / mrad", "xp_mean_mrad", 1, "diff"],
];
function renderTrack() {
  scatter($("ps-x"), "x", "xp", "x / mm", "x′ / mrad");
  scatter($("ps-y"), "y", "yp", "y / mm", "y′ / mrad");
  const fr = S.track ? `Δφ / deg at ${S.track.f_ref_GHz.toFixed(3)} GHz` : "Δφ / deg";
  scatter($("ps-l"), "dphi", "dE", fr, "ΔE / keV");
  scatter($("ps-t"), "dphi", "x", fr, "x / mm");
  drawEnvelope();
  const tb = $("stats").querySelector("tbody");
  if (!S.track) return;
  tb.innerHTML = "";
  for (const [label, key, k, mode] of STAT_ROWS) {
    const a = S.track.stats.in[key], b = S.track.stats.out[key];
    if (a === null && b === null) continue;
    let ch = "–";
    if (a !== null && b !== null) {
      if (mode === "pct" && a !== 0) ch = `${((b / a - 1) * 100 >= 0 ? "+" : "")}${fmt((b / a - 1) * 100, 3)} %`;
      else ch = `${b - a >= 0 ? "+" : ""}${fmt((b - a) * k, 4)}`;
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${fmt(a * k, 5)}</td><td>${fmt(b * k, 5)}</td><td>${ch}</td>`;
    tb.appendChild(tr);
  }
}

// ------------------------------------------------------------------ wiring
function loadPreset(name) {
  const p = PRESETS[name];
  S.points = p.points.map(([z, r, sm]) => ({ z, r, smooth: !!sm }));
  $("cell").value = p.cell; $("fmax").value = p.fmax; $("h").value = p.h; $("haxis").value = p.haxis || "";
  S.selPoint = null; $("inspector").hidden = true;
  S.fields = null; S.solve = null; S.stale = false; S.track = null; S.drives = [];
  renderModes(); renderDrives(); renderTrack();
  $("stats").querySelector("tbody").innerHTML = `<tr><td colspan="4" class="empty">Track a bunch to compare.</td></tr>`;
  computeView(); drawSVG(); drawHeat(); drawAxisPlot();
  refreshPreview().then(() => { if (!S.geomError) solve(); });
}
$("preset").addEventListener("change", (e) => loadPreset(e.target.value));
$("solve").addEventListener("click", solve);
$("fit").addEventListener("click", () => { computeView(); drawSVG(); drawHeat(); drawAxisPlot(); });
for (const id of ["bcin", "bcout"]) $(id).addEventListener("change", () => geometryChanged(true));
for (const id of ["ek", "species"]) $(id).addEventListener("change", () => { refreshFigures(); renderModes(); });
window.addEventListener("resize", debounce(() => { computeView(); drawSVG(); drawHeat(); drawAxisPlot(); renderTrack(); }, 150));

loadPreset("smooth");
