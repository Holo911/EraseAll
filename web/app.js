"use strict";

/* EraseAll - canvas UI.
 *
 * Coordinate model
 * ----------------
 * The two on-screen canvases (image + overlay) fill the whole stage viewport.
 * The image is drawn inside them through a single `view` transform:
 *     canvasX = imgX * view.scale + view.offsetX
 * The authoritative selection lives on an offscreen `maskCanvas` at FULL image
 * resolution (white = selected). All painting happens there in image coordinates;
 * the overlay is re-rendered from it each frame. Because every pointer event is
 * mapped through the same `view`, strokes land exactly under the cursor at any
 * window size (and, later, any zoom level).
 */

const el = (id) => document.getElementById(id);

const imageCanvas = el("imageCanvas");
const overlayCanvas = el("overlayCanvas");
const imageCtx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");
const container = el("canvasContainer");
const stage = el("stage");

// Offscreen full-resolution layers (white on transparent), all at image res.
//   rawCanvas   - hand-painted brush strokes (raw mode + eraser)
//   smartCanvas - the latest MobileSAM mask (smart mode)
//   maskCanvas  - the authoritative selection = union of the two (exported/rendered)
const rawCanvas = document.createElement("canvas");
const rawCtx = rawCanvas.getContext("2d");
const smartCanvas = document.createElement("canvas");
const smartCtx = smartCanvas.getContext("2d");
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

// Viewport-sized scratch buffer used to tint a mask layer before compositing it
// onto the overlay (lets us stack the red selection and the cyan hover).
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d");

const state = {
  img: null,          // HTMLImageElement of the current photo
  imageId: null,      // current image_id on the server
  natW: 0,            // natural (original) image width
  natH: 0,
  view: { scale: 1, offsetX: 0, offsetY: 0 },
  mode: "raw",        // "raw" | "smart"
  method: "structure",// remove fill method: "structure" (LaMa) | "pattern" (shift-map)
  brush: 40,          // brush diameter in IMAGE pixels
  eraser: false,
  painting: false,
  lastPt: null,       // last image-space point during a stroke
  currentStroke: [],  // image-space points of the in-progress smart stroke
  strokeNeg: false,   // is the current smart stroke a negative (carve-out) stroke?
  selectionUndo: [],  // stack of {raw,smart} layer snapshots (cap 10)
  segmenting: false,  // a /api/segment call is in flight
  cursor: null,       // last cursor position in canvas space (for the brush ring)
  history: [],        // list of image_ids
  cursorIdx: -1,      // index into history of the displayed image
  busy: false,
  // view interaction
  spaceDown: false,   // space held → pan mode
  panning: false,
  panStart: null,
  showBefore: false,  // hold-B compare
  beforeImg: null,    // cached Image of history[0] (the original upload)
  // people mode
  peopleActive: false,
  people: [],         // [{id, box, conf, layer:canvas, checked}]
  smartBase: null,    // snapshot of the smart layer before people were overlaid
  // object mode
  objectsStatus: "unavailable", // pending | ready | unavailable
  hoverObjectId: null,
  hoverLayer: null,   // canvas of the object under the cursor (cyan highlight)
  objectCache: new Map(), // object_id -> layer canvas
  objDragStart: null, // image-space point where an object-mode drag began
  objDragCur: null,
  // variant strip (M9): non-null while the user is picking a fill
  variants: null,     // {ids, editBox, imgs, chosen, baseImg}
  // clone stamp (M10)
  stamp: null,        // {canvas, ctx, source, offset, dabs, undo:[]}
  edge: 0,            // selection edge adjustment in px (-20..20)
  crop: null,         // {srcW, srcH, quarter, straighten, canvas, rotW, rotH, rect, drag}
};

const STAMP_HARDNESS = 0.7;   // solid core out to 0.7r, then fades to 0 at r
const STAMP_UNDO_MAX = 10;

const MIN_SCALE = 0.1;
const MAX_SCALE = 8;

/* ---------------- View / sizing ---------------- */

function resizeCanvases() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  for (const c of [imageCanvas, overlayCanvas, scratchCanvas]) {
    c.width = w;
    c.height = h;
  }
  if (state.img) fitView();
  render();
}

// Fit the whole image into the viewport (letterboxed), centered.
function fitView() {
  const vw = imageCanvas.width;
  const vh = imageCanvas.height;
  const scale = Math.min(vw / state.natW, vh / state.natH);
  state.view.scale = scale;
  state.view.offsetX = (vw - state.natW * scale) / 2;
  state.view.offsetY = (vh - state.natH * scale) / 2;
}

// canvas(screen) space -> image space
function toImage(cx, cy) {
  return {
    x: (cx - state.view.offsetX) / state.view.scale,
    y: (cy - state.view.offsetY) / state.view.scale,
  };
}

function eventToCanvas(e) {
  const r = overlayCanvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

/* ---------------- Rendering ---------------- */

function render() {
  const { scale, offsetX, offsetY } = state.view;
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!state.img) return;
  if (state.peopleActive) positionChips();

  const dw = state.natW * scale;
  const dh = state.natH * scale;
  imageCtx.imageSmoothingEnabled = true;

  // Hold-B compare: draw the original and skip the selection overlay.
  if (state.showBefore && state.beforeImg) {
    imageCtx.drawImage(state.beforeImg, 0, 0, state.natW, state.natH, offsetX, offsetY, dw, dh);
    return;
  }

  // Stamping shows the working copy; cropping shows the straightened copy.
  let shown = state.img;
  if (state.mode === "stamp" && state.stamp) shown = state.stamp.canvas;
  else if (state.mode === "crop" && state.crop) shown = state.crop.canvas;
  imageCtx.drawImage(shown, 0, 0, state.natW, state.natH, offsetX, offsetY, dw, dh);
  // While picking a fill, hide the red selection so the result can be judged.
  if (!state.variants) renderOverlay();
}

function renderOverlay() {
  const { scale, offsetX, offsetY } = state.view;
  const dw = state.natW * scale;
  const dh = state.natH * scale;

  // Paint the mask shape, then tint it red via source-in. Skipped while cropping:
  // there natW/natH describe the ROTATED image, so the mask layers don't align.
  if (state.mode !== "crop") {
    overlayCtx.save();
    overlayCtx.imageSmoothingEnabled = true;
    overlayCtx.drawImage(maskCanvas, 0, 0, state.natW, state.natH, offsetX, offsetY, dw, dh);
    overlayCtx.globalCompositeOperation = "source-in";
    overlayCtx.fillStyle = "rgba(255,60,60,0.45)";
    overlayCtx.fillRect(offsetX, offsetY, dw, dh);
    overlayCtx.restore();
  }

  drawHover();
  drawObjectBox();
  drawStampSource();
  drawCropOverlay();
  drawScribble();
  drawBrushCursor();
}

// Crosshair marking the clone source (Stamp mode).
function drawStampSource() {
  if (state.mode !== "stamp" || !state.stamp || !state.stamp.source) return;
  const { scale, offsetX, offsetY } = state.view;
  const s = state.stamp.source;
  const x = s.x * scale + offsetX, y = s.y * scale + offsetY;
  overlayCtx.save();
  overlayCtx.strokeStyle = "rgba(60,220,255,0.95)";
  overlayCtx.lineWidth = 1.5;
  overlayCtx.beginPath();
  overlayCtx.moveTo(x - 8, y); overlayCtx.lineTo(x + 8, y);
  overlayCtx.moveTo(x, y - 8); overlayCtx.lineTo(x, y + 8);
  overlayCtx.stroke();
  overlayCtx.beginPath();
  overlayCtx.arc(x, y, 10, 0, Math.PI * 2);
  overlayCtx.stroke();
  overlayCtx.restore();
}

// Cyan highlight of the object under the cursor (Object mode).
function drawHover() {
  if (state.mode !== "object" || !state.hoverLayer || state.objDragStart) return;
  const { scale, offsetX, offsetY } = state.view;
  const dw = state.natW * scale, dh = state.natH * scale;
  scratchCtx.globalCompositeOperation = "source-over";
  scratchCtx.clearRect(0, 0, scratchCanvas.width, scratchCanvas.height);
  scratchCtx.drawImage(state.hoverLayer, 0, 0, state.natW, state.natH, offsetX, offsetY, dw, dh);
  scratchCtx.globalCompositeOperation = "source-in";
  scratchCtx.fillStyle = "rgba(60,220,255,0.45)";
  scratchCtx.fillRect(offsetX, offsetY, dw, dh);
  scratchCtx.globalCompositeOperation = "source-over";
  overlayCtx.drawImage(scratchCanvas, 0, 0);
}

// Live rectangle while box-dragging in Object mode.
function drawObjectBox() {
  if (state.mode !== "object" || !state.objDragStart || !state.objDragCur) return;
  const { scale, offsetX, offsetY } = state.view;
  const a = state.objDragStart, b = state.objDragCur;
  const x = Math.min(a.x, b.x) * scale + offsetX;
  const y = Math.min(a.y, b.y) * scale + offsetY;
  const w = Math.abs(b.x - a.x) * scale;
  const h = Math.abs(b.y - a.y) * scale;
  overlayCtx.save();
  overlayCtx.setLineDash([6, 4]);
  overlayCtx.lineWidth = 1.5;
  overlayCtx.strokeStyle = "rgba(60,220,255,0.95)";
  overlayCtx.strokeRect(x, y, w, h);
  overlayCtx.restore();
}

// Live preview of the in-progress smart scribble (green = add, blue = carve).
function drawScribble() {
  if (state.mode !== "smart" || state.currentStroke.length === 0) return;
  const { scale, offsetX, offsetY } = state.view;
  overlayCtx.save();
  overlayCtx.beginPath();
  state.currentStroke.forEach((p, i) => {
    const sx = p.x * scale + offsetX, sy = p.y * scale + offsetY;
    if (i === 0) overlayCtx.moveTo(sx, sy); else overlayCtx.lineTo(sx, sy);
  });
  overlayCtx.lineWidth = 3;
  overlayCtx.lineCap = "round";
  overlayCtx.strokeStyle = state.strokeNeg ? "rgba(90,180,255,0.95)" : "rgba(90,230,120,0.95)";
  overlayCtx.stroke();
  overlayCtx.restore();
}

function drawBrushCursor() {
  if (!state.cursor || state.busy) return;
  overlayCtx.save();
  if (state.mode === "smart") {
    // Small dot - smart mode uses point prompts, not a brush footprint.
    overlayCtx.beginPath();
    overlayCtx.arc(state.cursor.x, state.cursor.y, 3, 0, Math.PI * 2);
    overlayCtx.fillStyle = "rgba(255,255,255,0.9)";
    overlayCtx.fill();
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeStyle = "rgba(0,0,0,0.6)";
    overlayCtx.stroke();
  } else {
    const r = (state.brush / 2) * state.view.scale;
    overlayCtx.beginPath();
    overlayCtx.arc(state.cursor.x, state.cursor.y, r, 0, Math.PI * 2);
    overlayCtx.lineWidth = 1.5;
    overlayCtx.strokeStyle = state.eraser ? "rgba(120,200,255,0.95)" : "rgba(255,255,255,0.95)";
    overlayCtx.stroke();
    overlayCtx.strokeStyle = "rgba(0,0,0,0.5)";
    overlayCtx.lineWidth = 0.75;
    overlayCtx.stroke();
  }
  overlayCtx.restore();
}

/* ---------------- Painting ---------------- */

// Union the two layers into the authoritative mask.
function recomposeMask() {
  maskCtx.globalCompositeOperation = "source-over";
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.drawImage(rawCanvas, 0, 0);
  maskCtx.drawImage(smartCanvas, 0, 0);
}

// Paint a round stroke segment on the RAW layer (image coordinates).
function paintRawDab(a, b) {
  rawCtx.lineCap = "round";
  rawCtx.lineJoin = "round";
  rawCtx.lineWidth = state.brush;
  rawCtx.strokeStyle = "#ffffff";
  rawCtx.fillStyle = "#ffffff";
  rawCtx.globalCompositeOperation = state.eraser ? "destination-out" : "source-over";
  rawCtx.beginPath();
  rawCtx.moveTo(a.x, a.y);
  rawCtx.lineTo(b.x, b.y);
  rawCtx.stroke();
  rawCtx.beginPath();
  rawCtx.arc(b.x, b.y, state.brush / 2, 0, Math.PI * 2);
  rawCtx.fill();
}

// Evenly-spaced subsample of a stroke to at most n points.
function samplePoints(points, n) {
  if (points.length <= n) return points.slice();
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(points[Math.round((i * (points.length - 1)) / (n - 1))]);
  }
  return out;
}

function onPointerDown(e) {
  if (!state.img || state.busy || state.segmenting) return;
  try { overlayCanvas.setPointerCapture(e.pointerId); } catch {}
  const c = eventToCanvas(e);
  state.cursor = c;

  // Pan: space-drag or middle-button drag.
  if (state.spaceDown || e.button === 1) {
    state.panning = true;
    state.panStart = { x: e.clientX, y: e.clientY, ox: state.view.offsetX, oy: state.view.offsetY };
    overlayCanvas.style.cursor = "grabbing";
    return;
  }

  // A hand stroke finalizes any pending people selection first, so the stroke
  // isn't wiped by the next people rebuild.
  if (state.peopleActive) finalizePeople();

  const p = toImage(c.x, c.y);

  // Crop mode: drag a new crop rectangle (in rotated-image coords).
  if (state.mode === "crop") {
    if (!state.crop) initCrop();
    state.crop.drag = { x: p.x, y: p.y };
    state.painting = true;
    render();
    return;
  }

  // Stamp mode: Alt-click sets the clone source; otherwise paint dabs.
  if (state.mode === "stamp") {
    initStamp();
    if (e.altKey || e.button === 2) {
      state.stamp.source = p;
      state.stamp.offset = null;     // re-lock the offset on the next stroke
      render();
      return;
    }
    if (!state.stamp.source) { toast("Alt-click to set a source point first."); return; }
    stampSnapshot();                 // one undo entry per stroke
    // Classic aligned clone: the first dab locks the source->brush offset.
    if (!state.stamp.offset) {
      state.stamp.offset = { dx: state.stamp.source.x - p.x, dy: state.stamp.source.y - p.y };
    }
    state.painting = true;
    state.lastPt = p;
    stampDab(p);
    updateApplyBar();
    render();
    return;
  }

  // Object mode: a press starts either a click-select or a box-drag; decided on release.
  if (state.mode === "object") {
    state.strokeNeg = e.altKey || e.button === 2;
    state.objDragStart = p;
    state.objDragCur = p;
    state.painting = true;
    render();
    return;
  }

  state.painting = true;
  if (state.mode === "smart") {
    state.strokeNeg = e.altKey || e.button === 2;   // Alt or right-button = negative
    state.currentStroke = [p];
  } else {
    pushSelectionSnapshot();   // snapshot before a raw stroke so Ctrl+Z undoes it
    state.lastPt = p;
    paintRawDab(p, p);
    recomposeMask();
  }
  render();
}

function onPointerMove(e) {
  const c = eventToCanvas(e);
  state.cursor = c;
  if (state.panning) {
    state.view.offsetX = state.panStart.ox + (e.clientX - state.panStart.x);
    state.view.offsetY = state.panStart.oy + (e.clientY - state.panStart.y);
    render();
    return;
  }
  if (state.mode === "object") {
    const p = toImage(c.x, c.y);
    if (state.objDragStart) { state.objDragCur = p; render(); }
    else { hoverProbe(p); }          // throttled: one request in flight at a time
    render();
    return;
  }
  if (state.mode === "crop") {
    if (state.painting && state.crop && state.crop.drag) {
      const p = toImage(c.x, c.y);
      const d = state.crop.drag;
      const x0 = Math.max(0, Math.round(Math.min(d.x, p.x)));
      const y0 = Math.max(0, Math.round(Math.min(d.y, p.y)));
      const x1 = Math.min(state.crop.rotW, Math.round(Math.max(d.x, p.x)));
      const y1 = Math.min(state.crop.rotH, Math.round(Math.max(d.y, p.y)));
      if (x1 - x0 >= 8 && y1 - y0 >= 8) state.crop.rect = [x0, y0, x1, y1];
    }
    render();
    return;
  }
  if (state.mode === "stamp") {
    if (state.painting) {
      const p = toImage(c.x, c.y);
      // Interpolate so fast strokes don't leave gaps between dabs.
      const step = Math.max(2, state.brush * 0.25);
      const dist = Math.hypot(p.x - state.lastPt.x, p.y - state.lastPt.y);
      const n = Math.max(1, Math.floor(dist / step));
      for (let i = 1; i <= n; i++) {
        stampDab({
          x: state.lastPt.x + (p.x - state.lastPt.x) * (i / n),
          y: state.lastPt.y + (p.y - state.lastPt.y) * (i / n),
        });
      }
      state.lastPt = p;
      updateApplyBar();
    }
    render();
    return;
  }
  if (state.painting) {
    const p = toImage(c.x, c.y);
    if (state.mode === "smart") {
      state.currentStroke.push(p);
    } else {
      paintRawDab(state.lastPt, p);
      state.lastPt = p;
      recomposeMask();
    }
  }
  render();
}

async function onPointerUp(e) {
  if (state.panning) {
    state.panning = false;
    overlayCanvas.style.cursor = state.spaceDown ? "grab" : "none";
    try { overlayCanvas.releasePointerCapture(e.pointerId); } catch {}
    return;
  }
  if (!state.painting) return;
  state.painting = false;
  try { overlayCanvas.releasePointerCapture(e.pointerId); } catch {}

  if (state.mode === "crop") {
    if (state.crop) state.crop.drag = null;
    render();
    return;
  }

  // Object mode: short press = click-select, longer drag = box-select.
  if (state.mode === "object") {
    const a = state.objDragStart, b = state.objDragCur;
    const negative = state.strokeNeg;
    state.objDragStart = null; state.objDragCur = null;
    if (!a) { render(); return; }
    const moved = Math.hypot((b.x - a.x) * state.view.scale, (b.y - a.y) * state.view.scale);
    if (moved < CLICK_SLOP_PX) await objectClick(a, negative);
    else await objectBox(a, b, negative);
    render();
    return;
  }

  if (state.mode === "smart") {
    const stroke = state.currentStroke;
    const negative = state.strokeNeg;
    state.currentStroke = [];
    render();
    await segmentStroke(stroke, negative);  // segment & compose just this stroke
  } else {
    state.lastPt = null;
    render();
    updateActionButtons();
  }
}

function onPointerLeave() {
  state.cursor = null;
  render();
}

// Cursor-anchored wheel zoom: the image point under the cursor stays put.
function onWheel(e) {
  if (!state.img) return;
  e.preventDefault();
  const c = eventToCanvas(e);
  const imgPt = toImage(c.x, c.y);
  const factor = Math.exp(-e.deltaY * 0.0015);
  const ns = Math.max(MIN_SCALE, Math.min(MAX_SCALE, state.view.scale * factor));
  state.view.scale = ns;
  state.view.offsetX = c.x - imgPt.x * ns;
  state.view.offsetY = c.y - imgPt.y * ns;
  render();
}

function resetView() {
  if (!state.img) return;
  fitView();
  render();
}

// 100% zoom, anchored at the cursor (or the viewport centre if it's outside).
function zoomTo100() {
  if (!state.img) return;
  const c = state.cursor || { x: overlayCanvas.width / 2, y: overlayCanvas.height / 2 };
  const imgPt = toImage(c.x, c.y);
  state.view.scale = 1;
  state.view.offsetX = c.x - imgPt.x;
  state.view.offsetY = c.y - imgPt.y;
  render();
}

/* ---------------- Smart select (SAM) ---------------- */

// Decode the returned single-channel mask into a full-res white-on-transparent
// offscreen canvas we can union/subtract into the selection layers.
function maskToLayer(b64) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      const w = smartCanvas.width, h = smartCanvas.height;
      const layer = document.createElement("canvas");
      layer.width = w; layer.height = h;
      const lctx = layer.getContext("2d");
      lctx.drawImage(im, 0, 0, w, h);
      const id = lctx.getImageData(0, 0, w, h);
      const d = id.data;
      for (let i = 0; i < d.length; i += 4) {
        const a = d[i];             // R channel holds the mask value (0/255)
        d[i] = d[i + 1] = d[i + 2] = 255;
        d[i + 3] = a;               // use it as alpha → white where selected
      }
      lctx.putImageData(id, 0, 0);
      resolve(layer);
    };
    im.onerror = () => reject(new Error("Bad mask from server"));
    im.src = "data:image/png;base64," + b64;
  });
}

// Bounding box of a stroke, expanded 2.5x (>=96px each side), clamped to image.
function strokeBox(points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of points) {
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
  }
  const padX = Math.max(96, (x1 - x0) * 0.75);
  const padY = Math.max(96, (y1 - y0) * 0.75);
  return [
    Math.max(0, Math.floor(x0 - padX)),
    Math.max(0, Math.floor(y0 - padY)),
    Math.min(state.natW, Math.ceil(x1 + padX)),
    Math.min(state.natH, Math.ceil(y1 + padY)),
  ];
}

// Segment ONE stroke locally and compose it into the selection incrementally.
async function segmentStroke(strokePoints, negative) {
  if (!strokePoints || strokePoints.length === 0) return;
  const pts = samplePoints(strokePoints, 10).map((p) => ({ x: p.x, y: p.y, label: 1 }));
  const box = strokeBox(strokePoints);
  state.segmenting = true;
  showCursorSpinner(true);
  try {
    const res = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId, points: pts, box }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Segment failed");
    const data = await res.json();
    const layer = await maskToLayer(data.mask);

    pushSelectionSnapshot();  // so Ctrl+Z can undo just this stroke
    if (negative) {
      // Carve this whole sub-object out of BOTH selection layers, locally.
      for (const ctx of [smartCtx, rawCtx]) {
        ctx.globalCompositeOperation = "destination-out";
        ctx.drawImage(layer, 0, 0);
        ctx.globalCompositeOperation = "source-over";
      }
    } else {
      // Union into the smart layer - never clears what's already there.
      smartCtx.globalCompositeOperation = "source-over";
      smartCtx.drawImage(layer, 0, 0);
    }
    recomposeMask();
    render();
    updateActionButtons();
  } catch (err) {
    toast(err.message, true);
  } finally {
    state.segmenting = false;
    showCursorSpinner(false);
  }
}

/* ---------------- Selection undo (stroke-level) ---------------- */

const SELECTION_UNDO_MAX = 10;

function snapshotLayer(srcCanvas) {
  const c = document.createElement("canvas");
  c.width = srcCanvas.width; c.height = srcCanvas.height;
  c.getContext("2d").drawImage(srcCanvas, 0, 0);
  return c;
}

// Capture the current raw+smart layers so the next stroke can be undone.
function pushSelectionSnapshot() {
  state.selectionUndo.push({ raw: snapshotLayer(rawCanvas), smart: snapshotLayer(smartCanvas) });
  if (state.selectionUndo.length > SELECTION_UNDO_MAX) state.selectionUndo.shift();
}

// Restore the most recent snapshot. Returns false if the stack was empty.
function popSelectionSnapshot() {
  const snap = state.selectionUndo.pop();
  if (!snap) return false;
  for (const [ctx, snapCanvas] of [[rawCtx, snap.raw], [smartCtx, snap.smart]]) {
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    ctx.drawImage(snapCanvas, 0, 0);
  }
  recomposeMask();
  render();
  updateActionButtons();
  return true;
}

function showCursorSpinner(on) {
  const s = el("cursorSpinner");
  if (on && state.cursor) {
    s.style.left = state.cursor.x + "px";
    s.style.top = state.cursor.y + "px";
    s.classList.remove("hidden");
  } else {
    s.classList.add("hidden");
  }
}

/* ---------------- Crop / straighten (M11) ---------------- */

// Mirrors PIL's Image.rotate(expand=True) geometry EXACTLY (ceil(max)-floor(min)
// of the transformed corners, not the naive formula, which is 1-2px off). Keeping
// this identical to the server is what makes the crop land on precisely the
// region the user previewed. `angleParam` is what the server passes to rotate().
function pilRotatedSize(w, h, angleParam) {
  const rad = -angleParam * Math.PI / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const cx = w / 2, cy = h / 2;
  const m2 = cos * (-cx) + sin * (-cy) + cx;
  const m5 = -sin * (-cx) + cos * (-cy) + cy;
  const xs = [], ys = [];
  for (const [x, y] of [[0, 0], [w, 0], [w, h], [0, h]]) {
    xs.push(cos * x + sin * y + m2);
    ys.push(-sin * x + cos * y + m5);
  }
  return [
    Math.ceil(Math.max(...xs)) - Math.floor(Math.min(...xs)),
    Math.ceil(Math.max(...ys)) - Math.floor(Math.min(...ys)),
  ];
}

// Largest axis-aligned rectangle that fits inside the rotated image (the
// suggestion shown on straighten; the user's own rect overrides it).
function inscribedRect(w, h, angleDeg, rw, rh) {
  let a = Math.abs(angleDeg % 180) * Math.PI / 180;
  a = Math.min(a, Math.PI - a);
  const sin = Math.abs(Math.sin(a)), cos = Math.abs(Math.cos(a));
  const longer = Math.max(w, h), shorter = Math.min(w, h);
  let W, H;
  if (shorter <= 2 * sin * cos * longer || Math.abs(sin - cos) < 1e-10) {
    const x = 0.5 * shorter;
    if (w >= h) { W = x / sin; H = x / cos; } else { W = x / cos; H = x / sin; }
  } else {
    const c2 = cos * cos - sin * sin;
    W = (w * cos - h * sin) / c2;
    H = (h * cos - w * sin) / c2;
  }
  if (!isFinite(W) || !isFinite(H) || W <= 0 || H <= 0) { W = rw; H = rh; }
  W = Math.max(8, Math.min(W, rw)); H = Math.max(8, Math.min(H, rh));
  const x0 = Math.round((rw - W) / 2), y0 = Math.round((rh - H) / 2);
  return [x0, y0, Math.round(x0 + W), Math.round(y0 + H)];
}

function cropTotalAngle() {
  return state.crop ? state.crop.quarter * 90 + state.crop.straighten : 0;
}

function initCrop() {
  if (!state.img || state.crop) return;
  const srcW = state.natW, srcH = state.natH;
  state.crop = {
    srcW, srcH, quarter: 0, straighten: 0,
    canvas: document.createElement("canvas"),
    rotW: srcW, rotH: srcH, rect: null, drag: null,
  };
  updateCropCanvas(true);
}

// Re-render the straightened image and (re)fit the view to its new size. While
// cropping, natW/natH describe the ROTATED image - that's the space the crop rect
// and all pointer coords live in, and exactly what the server crops.
function updateCropCanvas(resetRect) {
  const c = state.crop;
  const a = cropTotalAngle();
  const [rw, rh] = pilRotatedSize(c.srcW, c.srcH, -a);
  c.rotW = rw; c.rotH = rh;
  c.canvas.width = rw; c.canvas.height = rh;
  const ctx = c.canvas.getContext("2d");
  ctx.clearRect(0, 0, rw, rh);
  ctx.save();
  ctx.translate(rw / 2, rh / 2);
  ctx.rotate(a * Math.PI / 180);
  ctx.drawImage(state.img, -c.srcW / 2, -c.srcH / 2, c.srcW, c.srcH);
  ctx.restore();
  state.natW = rw; state.natH = rh;
  if (resetRect || !c.rect) c.rect = inscribedRect(c.srcW, c.srcH, a, rw, rh);
  fitView();
  render();
}

function setStraighten(v) {
  if (!state.crop) return;
  state.crop.straighten = Math.max(-45, Math.min(45, +v));
  el("straightenSlider").value = state.crop.straighten;
  el("straightenReadout").textContent = `${state.crop.straighten}°`;
  updateCropCanvas(true);   // new angle → fresh inscribed suggestion
}

function rotateQuarter(dir) {
  if (!state.crop) return;
  state.crop.quarter = (state.crop.quarter + dir + 4) % 4;
  updateCropCanvas(true);
}

async function applyCrop() {
  const c = state.crop;
  if (!c) return;
  showProgress("Applying crop…");
  try {
    const r = await fetch("/api/transform", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId, angle: cropTotalAngle(), crop: c.rect }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Crop failed");
    const { image_id } = await r.json();
    state.crop = null;
    clearMaskSilently();
    await loadImageId(image_id);
    pushHistory(image_id);
    updateActionButtons();
    updateApplyBar();
    toast("Cropped");
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

function cancelCrop() {
  if (!state.crop) return;
  state.natW = state.crop.srcW;
  state.natH = state.crop.srcH;
  state.crop = null;
  updateApplyBar();
  fitView();
  render();
}

// Dim outside the crop, plus the rule-of-thirds grid.
function drawCropOverlay() {
  if (state.mode !== "crop" || !state.crop || !state.crop.rect) return;
  const { scale, offsetX, offsetY } = state.view;
  const [x0, y0, x1, y1] = state.crop.rect;
  const sx = x0 * scale + offsetX, sy = y0 * scale + offsetY;
  const sw = (x1 - x0) * scale, sh = (y1 - y0) * scale;

  overlayCtx.save();
  overlayCtx.fillStyle = "rgba(0,0,0,0.5)";
  overlayCtx.beginPath();
  overlayCtx.rect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayCtx.rect(sx, sy, sw, sh);
  overlayCtx.fill("evenodd");

  overlayCtx.strokeStyle = "rgba(255,255,255,0.95)";
  overlayCtx.lineWidth = 1.5;
  overlayCtx.strokeRect(sx, sy, sw, sh);
  overlayCtx.strokeStyle = "rgba(255,255,255,0.35)";
  overlayCtx.lineWidth = 1;
  for (let i = 1; i <= 2; i++) {
    const gx = sx + (sw * i) / 3, gy = sy + (sh * i) / 3;
    overlayCtx.beginPath(); overlayCtx.moveTo(gx, sy); overlayCtx.lineTo(gx, sy + sh); overlayCtx.stroke();
    overlayCtx.beginPath(); overlayCtx.moveTo(sx, gy); overlayCtx.lineTo(sx + sw, gy); overlayCtx.stroke();
  }
  overlayCtx.restore();
}

/* ---------------- Export (M11) ---------------- */

function exportUrl() {
  const fmt = document.querySelector('input[name="fmt"]:checked').value;
  const q = +el("qualitySlider").value;
  const exif = el("exifToggle").checked ? 1 : 0;
  return `/api/export?image_id=${state.imageId}&fmt=${fmt}&q=${q}&exif=${exif}`;
}

function doExport() {
  if (!state.imageId) return;
  const a = document.createElement("a");
  a.href = exportUrl();
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
  el("exportPopover").classList.add("hidden");
}

/* ---------------- Clone stamp (M10) ---------------- */

// The stamp works on a full-res offscreen copy of the current image. Nothing is
// sent to the server until Apply, and only painted dabs ever change pixels.
function initStamp() {
  if (state.stamp || !state.img) return;
  const c = document.createElement("canvas");
  c.width = state.natW; c.height = state.natH;
  const ctx = c.getContext("2d");
  ctx.drawImage(state.img, 0, 0, state.natW, state.natH);
  state.stamp = { canvas: c, ctx, source: null, offset: null, dabs: 0, undo: [] };
  updateApplyBar();
}

function stampSnapshot() {
  const s = state.stamp;
  const c = document.createElement("canvas");
  c.width = s.canvas.width; c.height = s.canvas.height;
  c.getContext("2d").drawImage(s.canvas, 0, 0);
  s.undo.push(c);
  if (s.undo.length > STAMP_UNDO_MAX) s.undo.shift();
}

function stampUndo() {
  const s = state.stamp;
  if (!s || !s.undo.length) return false;
  const prev = s.undo.pop();
  s.ctx.globalCompositeOperation = "source-over";
  s.ctx.clearRect(0, 0, s.canvas.width, s.canvas.height);
  s.ctx.drawImage(prev, 0, 0);
  s.dabs = Math.max(0, s.dabs - 1);
  updateApplyBar();
  render();
  return true;
}

// One soft-edged dab: copy a brush-sized disc from (dst + offset) to dst.
function stampDab(dst) {
  const s = state.stamp;
  if (!s || !s.offset) return;
  const r = Math.max(1, state.brush / 2);
  const d = Math.round(r * 2);
  const sx = dst.x + s.offset.dx, sy = dst.y + s.offset.dy;

  const tmp = document.createElement("canvas");
  tmp.width = d; tmp.height = d;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(s.canvas, sx - r, sy - r, d, d, 0, 0, d, d);  // source patch
  // Feather the patch so dabs blend instead of stamping hard discs.
  const g = tctx.createRadialGradient(d / 2, d / 2, 0, d / 2, d / 2, r);
  g.addColorStop(0, "rgba(0,0,0,1)");
  g.addColorStop(STAMP_HARDNESS, "rgba(0,0,0,1)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  tctx.globalCompositeOperation = "destination-in";
  tctx.fillStyle = g;
  tctx.fillRect(0, 0, d, d);

  s.ctx.globalCompositeOperation = "source-over";
  s.ctx.drawImage(tmp, dst.x - r, dst.y - r);
  s.dabs++;
}

// One bar serves both Stamp and Crop; the buttons route by current mode.
function updateApplyBar() {
  const stampPending = !!(state.stamp && state.stamp.dabs > 0 && state.mode === "stamp");
  const cropping = !!(state.crop && state.mode === "crop");
  el("applyBar").classList.toggle("hidden", !(stampPending || cropping));
  el("applyBarLabel").textContent = cropping ? "Crop" : `Stamp edits (${state.stamp ? state.stamp.dabs : 0})`;
}

function applyBarApply() { return state.mode === "crop" ? applyCrop() : applyStamp(); }
function applyBarCancel() { return state.mode === "crop" ? cancelCrop() : cancelStamp(); }

async function applyStamp() {
  const s = state.stamp;
  if (!s || s.dabs === 0) return;
  showProgress("Applying stamp…");
  try {
    const blob = await new Promise((res) => s.canvas.toBlob(res, "image/png"));
    const fd = new FormData();
    fd.append("base_image_id", state.imageId);
    fd.append("file", blob, "edit.png");
    const r = await fetch("/api/upload_edit", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Apply failed");
    const { image_id } = await r.json();
    state.stamp = null;
    updateApplyBar();
    clearMaskSilently();
    await loadImageId(image_id);
    pushHistory(image_id);
    updateActionButtons();
    toast("Stamp applied");
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

function cancelStamp() {
  state.stamp = null;      // discard the edit canvas; base image untouched
  updateApplyBar();
  if (state.mode === "stamp") initStamp();
  render();
}

/* ---------------- Object mode ---------------- */

const CLICK_SLOP_PX = 5;      // drag shorter than this = a click, not a box
const FALLBACK_BOX_PX = 128;  // click box before the object index is ready

// Apply an object's mask layer to the selection (union, or subtract with Alt).
function applyObjectLayer(layer, negative) {
  pushSelectionSnapshot();
  if (negative) {
    for (const ctx of [smartCtx, rawCtx]) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.drawImage(layer, 0, 0);
      ctx.globalCompositeOperation = "source-over";
    }
  } else {
    smartCtx.globalCompositeOperation = "source-over";
    smartCtx.drawImage(layer, 0, 0);
  }
  recomposeMask();
  render();
  updateActionButtons();
}

let hoverBusy = false;   // one in-flight object_at at a time (natural throttle)

async function hoverProbe(p) {
  if (state.mode !== "object" || state.objectsStatus !== "ready") return;
  if (hoverBusy || state.busy || state.objDragStart) return;
  hoverBusy = true;
  try {
    const res = await fetch("/api/object_at", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId, x: p.x, y: p.y }),
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.object_id === null || d.object_id === undefined) {
      if (state.hoverObjectId !== null) { state.hoverObjectId = null; state.hoverLayer = null; render(); }
      return;
    }
    if (d.object_id !== state.hoverObjectId) {
      let layer = state.objectCache.get(d.object_id);
      if (!layer) { layer = await maskToLayer(d.mask); state.objectCache.set(d.object_id, layer); }
      state.hoverObjectId = d.object_id;
      state.hoverLayer = layer;
      render();
    }
  } catch { /* hover is best-effort */ } finally {
    hoverBusy = false;
  }
}

// Ask the server for a mask via /api/segment and apply it.
async function segmentAndApply(body, negative, busyText) {
  showProgress(busyText || "Selecting…");
  try {
    const res = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Select failed");
    const d = await res.json();
    applyObjectLayer(await maskToLayer(d.mask), negative);
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

async function objectClick(p, negative) {
  // Index ready → use the highlighted object directly (instant, no model call).
  if (state.objectsStatus === "ready" && state.hoverLayer) {
    applyObjectLayer(state.hoverLayer, negative);
    return;
  }
  // Not ready yet → fall back to a 1-point smart segment around the click.
  const box = [
    Math.max(0, p.x - FALLBACK_BOX_PX), Math.max(0, p.y - FALLBACK_BOX_PX),
    Math.min(state.natW, p.x + FALLBACK_BOX_PX), Math.min(state.natH, p.y + FALLBACK_BOX_PX),
  ];
  await segmentAndApply(
    { image_id: state.imageId, points: [{ x: p.x, y: p.y, label: 1 }], box },
    negative, "Selecting…");
}

async function objectBox(a, b, negative) {
  const box = [
    Math.max(0, Math.min(a.x, b.x)), Math.max(0, Math.min(a.y, b.y)),
    Math.min(state.natW, Math.max(a.x, b.x)), Math.min(state.natH, Math.max(a.y, b.y)),
  ];
  await segmentAndApply({ image_id: state.imageId, points: [], box }, negative, "Selecting…");
}

// Ask the server to build this image's object index if it doesn't have one yet
// (uploads index eagerly; edit results index on demand), then poll to ready.
async function ensureObjectIndex() {
  if (!state.imageId) return;
  try {
    const r = await fetch("/api/objects_index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId }),
    });
    if (r.ok) state.objectsStatus = (await r.json()).status;
  } catch { /* best-effort */ }
  updateObjectsBadge();
  if (state.objectsStatus === "pending") pollObjectsStatus();
}

// Poll the object index until it's ready (drives the "Analyzing objects…" badge).
async function pollObjectsStatus() {
  if (!state.imageId) return;
  const id = state.imageId;
  for (let i = 0; i < 120; i++) {
    if (state.imageId !== id) return;              // image changed; stop polling
    try {
      const r = await fetch(`/api/objects_status/${id}`);
      if (!r.ok) return;
      state.objectsStatus = (await r.json()).status;
    } catch { return; }
    updateObjectsBadge();
    if (state.objectsStatus !== "pending") return;
    await new Promise((r) => setTimeout(r, 1500));
  }
}

function updateObjectsBadge() {
  const show = state.mode === "object" && state.objectsStatus === "pending" && !!state.img;
  el("objectsBadge").classList.toggle("hidden", !show);
}

/* ---------------- People mode ---------------- */

// Toggle button: first click detects & shows chips; a second click finalizes.
async function peopleButton() {
  if (!state.img || state.busy) return;
  if (state.peopleActive) { finalizePeople(); return; }
  await detectPeople();
}

async function detectPeople() {
  showProgress("Finding people…");
  try {
    const res = await fetch("/api/detect_people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Detection failed");
    const data = await res.json();
    if (!data.people.length) { toast("No people detected - select manually."); return; }

    // Snapshot the current smart layer so toggling never disturbs prior picks,
    // and push one selection-undo entry for the whole people operation.
    pushSelectionSnapshot();
    state.smartBase = snapshotLayer(smartCanvas);
    state.people = [];
    for (const p of data.people) {
      const layer = await maskToLayer(p.mask);
      state.people.push({ id: p.id, box: p.box, conf: p.confidence, layer, checked: true });
    }
    state.peopleActive = true;
    el("peopleBtn").textContent = "Done";
    el("peopleBtn").classList.add("active");
    rebuildPeopleSelection();
    buildChips();
    toast(`${data.people.length} ${data.people.length === 1 ? "person" : "people"} found - click a chip to toggle`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

// smart layer = base selection UNION the checked people masks.
function rebuildPeopleSelection() {
  smartCtx.globalCompositeOperation = "source-over";
  smartCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  if (state.smartBase) smartCtx.drawImage(state.smartBase, 0, 0);
  for (const person of state.people) {
    if (person.checked) smartCtx.drawImage(person.layer, 0, 0);
  }
  recomposeMask();
  render();
  updateActionButtons();
}

function finalizePeople() {
  // Checked people are already baked into the smart layer; just drop the chips.
  state.peopleActive = false;
  state.people = [];
  state.smartBase = null;
  el("peopleBtn").textContent = "Select people";
  el("peopleBtn").classList.remove("active");
  buildChips();
  render();
}

// Called by clearAllLayers so any selection reset also exits people mode.
function clearPeopleState() {
  state.peopleActive = false;
  state.people = [];
  state.smartBase = null;
  const btn = document.getElementById("peopleBtn");
  if (btn) { btn.textContent = "Select people"; btn.classList.remove("active"); }
  buildChips();
}

function buildChips() {
  const box = el("peopleChips");
  box.innerHTML = "";
  if (!state.peopleActive) return;
  for (const person of state.people) {
    const chip = document.createElement("div");
    chip.className = "chip" + (person.checked ? "" : " off");
    chip.textContent = `Person ${person.id + 1}`;
    chip.addEventListener("click", () => {
      person.checked = !person.checked;
      chip.classList.toggle("off", !person.checked);
      rebuildPeopleSelection();
    });
    box.appendChild(chip);
    person._chip = chip;
  }
  positionChips();
}

// Anchor each chip at the top-center of its person's box (image → screen).
function positionChips() {
  if (!state.peopleActive) return;
  const { scale, offsetX, offsetY } = state.view;
  for (const person of state.people) {
    if (!person._chip) continue;
    const [x0, y0, x1] = person.box;
    person._chip.style.left = ((x0 + x1) / 2) * scale + offsetX + "px";
    person._chip.style.top = y0 * scale + offsetY + "px";
  }
}

/* ---------------- Selection helpers ---------------- */

function clearAllLayers() {
  for (const c of [rawCtx, smartCtx, maskCtx]) {
    c.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }
  state.currentStroke = [];
  state.selectionUndo = [];
  clearPeopleState();
}

function clearSelection() {
  clearAllLayers();
  render();
  updateActionButtons();
}

// True if any pixel is selected. Used to enable/disable Remove.
function maskHasContent() {
  if (!state.natW) return false;
  const { width, height } = maskCanvas;
  // Sample a downscaled copy for speed.
  const s = 200;
  const sw = Math.max(1, Math.min(s, width));
  const sh = Math.max(1, Math.min(s, height));
  const tmp = document.createElement("canvas");
  tmp.width = sw; tmp.height = sh;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(maskCanvas, 0, 0, sw, sh);
  const data = tctx.getImageData(0, 0, sw, sh).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 10) return true;
  }
  return false;
}

// Export the mask as a black/white PNG (white = selected) base64 (no data: prefix).
function exportMaskBase64() {
  const out = document.createElement("canvas");
  out.width = state.natW;
  out.height = state.natH;
  const octx = out.getContext("2d");
  octx.fillStyle = "#000000";
  octx.fillRect(0, 0, out.width, out.height);
  octx.drawImage(maskCanvas, 0, 0); // white strokes over black
  return out.toDataURL("image/png").split(",")[1];
}

/* ---------------- Loading images ---------------- */

function loadImageId(imageId) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      state.img = img;
      state.imageId = imageId;
      state.natW = img.naturalWidth;
      state.natH = img.naturalHeight;
      // (Re)size the full-res layers to match; resizing also clears them.
      if (maskCanvas.width !== state.natW || maskCanvas.height !== state.natH) {
        for (const cv of [rawCanvas, smartCanvas, maskCanvas]) {
          cv.width = state.natW;
          cv.height = state.natH;
        }
      }
      container.classList.remove("empty");
      // The object index is per-image: drop the old image's hover cache. Only
      // pull an index if the user is actually in Object mode.
      state.hoverObjectId = null;
      state.hoverLayer = null;
      state.objectCache = new Map();
      state.objectsStatus = "unavailable";
      updateObjectsBadge();
      if (state.mode === "object") ensureObjectIndex();
      fitView();
      render();
      resolve();
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = `/api/image/${imageId}?t=${Date.now()}`;
  });
}

async function uploadFile(file) {
  if (!file) return;
  // HEIC often arrives with an empty or nonstandard MIME type, so accept by
  // extension too; the server's PIL open is the real validator either way.
  const okType = /^image\/(png|jpeg|webp|heic|heif)$/.test(file.type);
  const okExt = /\.(png|jpe?g|webp|heic|heif)$/i.test(file.name || "");
  if (!okType && !okExt) {
    toast("Unsupported file type. Use PNG, JPEG, WebP, or HEIC.", true);
    return;
  }
  showProgress("Uploading…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    const data = await res.json();
    clearMaskSilently();
    await loadImageId(data.image_id);
    state.history = [data.image_id];
    state.cursorIdx = 0;
    // Cache the original image for the hold-B before/after compare.
    const bimg = new Image();
    bimg.src = `/api/image/${data.image_id}?t=${Date.now()}`;
    state.beforeImg = bimg;
    updateActionButtons();
    toast(`Loaded ${data.width}×${data.height}px`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

function clearMaskSilently() {
  clearAllLayers();
}

/* ---------------- History (undo/redo) ---------------- */

async function gotoHistory(idx) {
  if (idx < 0 || idx >= state.history.length) return;
  state.cursorIdx = idx;
  clearMaskSilently();
  await loadImageId(state.history[idx]);
  updateActionButtons();
}

function pushHistory(imageId) {
  // Truncate any redo tail, then append.
  state.history = state.history.slice(0, state.cursorIdx + 1);
  state.history.push(imageId);
  state.cursorIdx = state.history.length - 1;
}

// Undo: fixing a selection is cheap - first pop the stroke-level selection stack;
// only when it's empty does undo step the image (removal) history.
function undo() {
  if (state.busy || state.segmenting) return;
  if (state.mode === "stamp" && stampUndo()) return;  // undo dabs while stamping
  if (popSelectionSnapshot()) return;
  if (state.cursorIdx > 0) gotoHistory(state.cursorIdx - 1);
}

/* ---------------- Remove object ---------------- */

function loadImageElement(imageId) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Failed to load result"));
    im.src = `/api/image/${imageId}?t=${Date.now()}`;
  });
}

// Remove now offers three fills and lets the user pick (M9).
async function removeObject() {
  if (!state.img || state.busy || state.variants || !maskHasContent()) return;
  const mask = exportMaskBase64();
  showProgress("Trying 3 fills… (~30s)");
  try {
    const res = await fetch("/api/remove_variants", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.imageId, mask, edge: state.edge }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Remove failed");
    const data = await res.json();
    const imgs = await Promise.all(data.variants.map(loadImageElement));
    state.variants = {
      ids: data.variants,
      editBox: data.edit_box,
      imgs,
      chosen: 0,                 // V1 is the default highlighted choice
      baseImg: state.img,        // so Cancel can restore the pre-remove image
    };
    state.img = imgs[0];
    buildVariantStrip();
    render();
    updateActionButtons();
  } catch (err) {
    toast(err.message, true);
  } finally {
    hideProgress();
  }
}

function chooseVariant(k) {
  const v = state.variants;
  if (!v || k < 0 || k >= v.imgs.length) return;
  v.chosen = k;
  state.img = v.imgs[k];        // instant full-size preview; strip stays up
  document.querySelectorAll("#variantThumbs .thumb").forEach((t, i) =>
    t.classList.toggle("selected", i === k));
  render();
}

async function keepVariant() {
  const v = state.variants;
  if (!v) return;
  const id = v.ids[v.chosen];
  state.variants = null;
  el("variantStrip").classList.add("hidden");
  clearMaskSilently();
  await loadImageId(id);
  pushHistory(id);              // only the chosen id enters history
  updateActionButtons();
  toast(`Kept fill ${v.chosen + 1}`);
}

function cancelVariants() {
  const v = state.variants;
  if (!v) return;
  state.variants = null;
  el("variantStrip").classList.add("hidden");
  state.img = v.baseImg;        // back to the pre-remove image; nothing pushed
  render();
  updateActionButtons();
}

// Thumbnails are cropped to edit_box so the user judges the edited area, not the
// whole photo (a person-sized fix in a 12MP frame would be invisible otherwise).
function buildVariantStrip() {
  const v = state.variants;
  const wrap = el("variantThumbs");
  wrap.innerHTML = "";
  const [bx0, by0, bx1, by1] = v.editBox;
  const bw = Math.max(1, bx1 - bx0), bh = Math.max(1, by1 - by0);
  const TW = 150, TH = 115;
  const s = Math.min(TW / bw, TH / bh);
  v.imgs.forEach((im, i) => {
    const d = document.createElement("div");
    d.className = "thumb" + (i === v.chosen ? " selected" : "");
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(bw * s));
    c.height = Math.max(1, Math.round(bh * s));
    c.getContext("2d").drawImage(im, bx0, by0, bw, bh, 0, 0, c.width, c.height);
    const n = document.createElement("span");
    n.className = "num";
    n.textContent = String(i + 1);
    d.appendChild(c); d.appendChild(n);
    d.addEventListener("click", () => chooseVariant(i));
    wrap.appendChild(d);
  });
  el("variantStrip").classList.remove("hidden");
}

/* ---------------- UI wiring ---------------- */

function updateActionButtons() {
  const has = maskHasContent();
  el("removeBtn").disabled = state.busy || !state.img || !has;
  el("undoBtn").disabled = state.busy || (state.selectionUndo.length === 0 && state.cursorIdx <= 0);
  el("redoBtn").disabled = state.busy || state.cursorIdx >= state.history.length - 1;
  el("downloadBtn").disabled = !state.img;
  el("beforeBtn").disabled = state.history.length < 2;
  el("peopleBtn").disabled = state.busy || !state.img;
}

function setMode(mode) {
  // Never silently drop stamp work: make the user Apply or Cancel first.
  if (state.mode === "stamp" && mode !== "stamp" && state.stamp && state.stamp.dabs > 0) {
    toast("Apply or Cancel your stamp edits first.");
    updateApplyBar();
    document.querySelectorAll('input[name="mode"]').forEach((r) => { r.checked = r.value === "stamp"; });
    return;
  }
  state.mode = mode;
  document.querySelectorAll("#modeGroup .pill").forEach((p) => {
    p.classList.toggle("selected", p.querySelector("input").value === mode);
  });
  if (mode === "stamp") initStamp();
  else if (state.stamp) { state.stamp = null; updateApplyBar(); }
  if (mode === "crop") initCrop();
  else if (state.crop) cancelCrop();          // crop is non-destructive; just drop it
  el("cropGroup").classList.toggle("hidden", mode !== "crop");
  updateApplyBar();
  // Eraser only meaningful in raw mode.
  el("eraserBtn").style.display = mode === "raw" ? "" : "none";
  // Leaving object mode drops the cyan highlight.
  if (mode !== "object") { state.hoverObjectId = null; state.hoverLayer = null; }
  updateObjectsBadge();
  if (mode === "object" && state.img) ensureObjectIndex();
  render();
}

function setEdge(v) {
  state.edge = Math.max(-20, Math.min(20, v | 0));
  el("edgeSlider").value = state.edge;
  el("edgeReadout").textContent = (state.edge > 0 ? "+" : "") + state.edge;
}

function setMethod(method) {
  state.method = method;
  document.querySelectorAll("#methodGroup .pill").forEach((p) => {
    p.classList.toggle("selected", p.querySelector("input").value === method);
  });
}

function setEraser(on) {
  state.eraser = on;
  el("eraserBtn").classList.toggle("active", on);
  render();
}

function setBrush(v) {
  state.brush = Math.max(5, Math.min(200, v | 0));
  el("brushSize").value = state.brush;
  el("brushReadout").textContent = state.brush;
  render();
}

/* ---------------- Progress / toasts ---------------- */

function showProgress(text) {
  state.busy = true;
  el("progressText").textContent = text || "Working…";
  el("progress").classList.remove("hidden");
  updateActionButtons();
}
function hideProgress() {
  state.busy = false;
  el("progress").classList.add("hidden");
  updateActionButtons();
}

let toastTimer = null;
function toast(msg, isError) {
  const box = el("toasts");
  const t = document.createElement("div");
  t.className = "toast" + (isError ? " error" : "");
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => t.remove(), isError ? 5000 : 2600);
}

/* ---------------- Event listeners ---------------- */

function bindEvents() {
  el("fileInput").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
    e.target.value = "";
  });

  // Drag & drop onto the stage.
  ["dragenter", "dragover"].forEach((ev) =>
    stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("dragover"); })
  );
  stage.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  // Mode radios
  document.querySelectorAll('input[name="mode"]').forEach((r) =>
    r.addEventListener("change", () => setMode(r.value))
  );
  document.querySelectorAll('input[name="method"]').forEach((r) =>
    r.addEventListener("change", () => setMethod(r.value))
  );

  el("brushSize").addEventListener("input", (e) => setBrush(+e.target.value));
  el("eraserBtn").addEventListener("click", () => setEraser(!state.eraser));
  el("peopleBtn").addEventListener("click", peopleButton);
  el("clearBtn").addEventListener("click", clearSelection);
  el("undoBtn").addEventListener("click", undo);
  el("redoBtn").addEventListener("click", () => gotoHistory(state.cursorIdx + 1));
  el("removeBtn").addEventListener("click", removeObject);
  el("downloadBtn").addEventListener("click", downloadCurrent);
  el("helpBtn").addEventListener("click", () => el("shortcuts").classList.toggle("hidden"));
  el("variantKeep").addEventListener("click", keepVariant);
  el("variantCancel").addEventListener("click", cancelVariants);
  el("applyBtn").addEventListener("click", applyBarApply);
  el("applyCancelBtn").addEventListener("click", applyBarCancel);
  el("edgeSlider").addEventListener("input", (e) => setEdge(+e.target.value));
  el("straightenSlider").addEventListener("input", (e) => setStraighten(+e.target.value));
  el("rotCwBtn").addEventListener("click", () => rotateQuarter(1));
  el("rotCcwBtn").addEventListener("click", () => rotateQuarter(-1));
  // Export popover
  el("exportGo").addEventListener("click", doExport);
  el("exportClose").addEventListener("click", () => el("exportPopover").classList.add("hidden"));
  el("qualitySlider").addEventListener("input", (e) => { el("qualityReadout").textContent = e.target.value; });
  document.querySelectorAll('input[name="fmt"]').forEach((r) =>
    r.addEventListener("change", () => {
      document.querySelectorAll("#exportPopover .pill").forEach((p) =>
        p.classList.toggle("selected", p.querySelector("input").value === r.value));
      el("qualityRow").classList.toggle("hidden", r.value !== "jpeg");
    })
  );

  // Pointer painting on the overlay canvas.
  overlayCanvas.addEventListener("pointerdown", onPointerDown);
  overlayCanvas.addEventListener("pointermove", onPointerMove);
  overlayCanvas.addEventListener("pointerup", onPointerUp);
  overlayCanvas.addEventListener("pointercancel", onPointerUp);
  overlayCanvas.addEventListener("pointerleave", onPointerLeave);
  overlayCanvas.addEventListener("wheel", onWheel, { passive: false });
  overlayCanvas.addEventListener("dblclick", resetView);
  // Right-drag = negative stroke in smart mode; suppress the context menu.
  overlayCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // Before/After: press-and-hold the Before button (alternative to holding B).
  const beforeBtn = el("beforeBtn");
  const beforeOn = () => { if (state.beforeImg) { state.showBefore = true; render(); } };
  const beforeOff = () => { if (state.showBefore) { state.showBefore = false; render(); } };
  beforeBtn.addEventListener("pointerdown", beforeOn);
  beforeBtn.addEventListener("pointerup", beforeOff);
  beforeBtn.addEventListener("pointerleave", beforeOff);

  window.addEventListener("resize", resizeCanvases);

  // Keyboard shortcuts
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
}

function onKeyDown(e) {
  const typing = /input|textarea/i.test(document.activeElement.tagName);
  if (typing) return;

  // The variant strip owns 1/2/3 + Enter/Esc while it's open.
  if (state.variants) {
    if (e.key === "1" || e.key === "2" || e.key === "3") { e.preventDefault(); chooseVariant(+e.key - 1); return; }
    if (e.key === "Enter") { e.preventDefault(); keepVariant(); return; }
    if (e.key === "Escape") { e.preventDefault(); cancelVariants(); return; }
  }

  if (e.key === "[") { setBrush(state.brush - 5); }
  else if (e.key === "]") { setBrush(state.brush + 5); }
  else if (e.key === "e" || e.key === "E") { if (state.mode === "raw") setEraser(!state.eraser); }
  else if (e.key === " ") {
    e.preventDefault();
    if (!state.spaceDown) { state.spaceDown = true; overlayCanvas.style.cursor = "grab"; }
  }
  else if ((e.key === "b" || e.key === "B") && !e.ctrlKey && !e.metaKey) {
    if (!state.showBefore && state.beforeImg) { state.showBefore = true; render(); }
  }
  else if (e.key === "0" || e.key === "f" || e.key === "F") { resetView(); }
  else if (e.key === "1") { zoomTo100(); }   // variant strip claims 1/2/3 above
  else if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) { e.preventDefault(); undo(); }
  else if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) { e.preventDefault(); if (state.cursorIdx < state.history.length - 1) gotoHistory(state.cursorIdx + 1); }
}

function onKeyUp(e) {
  if (e.key === " ") {
    state.spaceDown = false;
    if (!state.panning) overlayCanvas.style.cursor = "none";
  } else if (e.key === "b" || e.key === "B") {
    if (state.showBefore) { state.showBefore = false; render(); }
  }
}

function downloadCurrent() {
  if (!state.imageId) return;
  el("exportPopover").classList.toggle("hidden");
}

/* ---------------- Init ---------------- */

setMode("raw");
setBrush(40);
bindEvents();
resizeCanvases();
