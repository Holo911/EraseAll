"""MobileSAM smart-select engine (local, CPU, offline).

The expensive step is computing the image embedding (`set_image`, ~1-2s CPU).
We do that once per image in a background thread at upload / after each edit, cache
the embedding per image_id (small LRU), and signal readiness with an Event. Per-
stroke `predict` then restores the cached embedding into the shared predictor and
runs only the light mask decoder (~100ms), so smart-select feels instant.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from server import store

_CKPT = os.environ.get(
    "MOBILE_SAM_MODEL",
    str(Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt"),
)
_CACHE_MAX = 4

_predictor = None
_load_lock = threading.Lock()   # guards model load
_run_lock = threading.Lock()    # serializes predictor access (embed + decode)

_cache: "OrderedDict[str, dict]" = OrderedDict()  # image_id -> embedding state
_events: "dict[str, threading.Event]" = {}
_events_lock = threading.Lock()


def is_loaded() -> bool:
    return _predictor is not None


def _get_predictor():
    global _predictor
    if _predictor is None:
        with _load_lock:
            if _predictor is None:
                if not Path(_CKPT).exists():
                    raise FileNotFoundError(
                        f"MobileSAM checkpoint not found at {_CKPT} (see plan/02)."
                    )
                from mobile_sam import SamPredictor, sam_model_registry
                sam = sam_model_registry["vit_t"](checkpoint=_CKPT)
                sam.to("cpu")
                sam.eval()
                _predictor = SamPredictor(sam)
    return _predictor


def _event_for(image_id: str) -> threading.Event:
    with _events_lock:
        ev = _events.get(image_id)
        if ev is None:
            ev = threading.Event()
            _events[image_id] = ev
        return ev


# NOTE: capture exactly what SamPredictor.predict()/predict_torch() read back.
# This version sets `original_size`/`input_size` in set_torch_image(), while
# reset_image() nulls a *different* set of names (orig_h/orig_w/input_h/input_w)
# that nothing else uses. Caching those instead left `original_size` pointing at
# whichever image was prepared LAST, so segmenting any other image silently used
# the wrong scale - masks came back at the wrong resolution. Keep these in sync
# with the predictor's real attributes.
def _capture_state(pred) -> dict:
    return {
        "features": pred.features,
        "original_size": pred.original_size,
        "input_size": pred.input_size,
    }


def _restore_state(pred, st: dict) -> None:
    pred.is_image_set = True
    pred.features = st["features"]
    pred.original_size = st["original_size"]
    pred.input_size = st["input_size"]


def prepare(image_id: str) -> None:
    """Compute and cache the embedding for image_id. Safe to call in a thread."""
    ev = _event_for(image_id)
    try:
        if image_id in _cache:
            return
        img = np.asarray(store.load_image(image_id))  # HxWx3 RGB uint8
        pred = _get_predictor()
        with _run_lock:
            pred.set_image(img)
            state = _capture_state(pred)
        _cache[image_id] = state
        _cache.move_to_end(image_id)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    finally:
        ev.set()  # readiness signaled even on failure (predict re-checks the cache)


def prepare_async(image_id: str) -> None:
    threading.Thread(target=prepare, args=(image_id,), daemon=True).start()


def _ensure_ready(image_id: str, timeout: float = 60.0) -> None:
    if image_id in _cache:
        return
    ev = _event_for(image_id)
    ev.wait(timeout)
    if image_id not in _cache:
        # Never prepared (or evicted) - compute synchronously now.
        prepare(image_id)
    if image_id not in _cache:
        raise TimeoutError(f"embedding not ready for {image_id}")


def _expand_box(box, factor: float, W: int, H: int):
    """Scale a XYXY box about its center by `factor`, clamped to the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) / 2.0 * factor, (y1 - y0) / 2.0 * factor
    return (
        max(0, int(round(cx - hw))),
        max(0, int(round(cy - hh))),
        min(W, int(round(cx + hw))),
        min(H, int(round(cy + hh))),
    )


def predict(image_id: str, points: list[dict], box: list[float] | None = None) -> np.ndarray:
    """Segment the object under a single stroke's points (all treated as fg).

    points: [{'x','y','label'} ...] in original-image pixels. `box` (XYXY) scopes
    the prompt and hard-clips the result so a stroke can never affect the selection
    far from itself. `points` may be empty when `box` is given (box-only select, as
    used by Object mode's drag gesture) - then the box alone drives the prompt and
    the clip is tighter (1.5x). Returns a binary uint8 HxW mask (0/255).
    """
    if not points and box is None:
        raise ValueError("need points or a box")
    _ensure_ready(image_id)
    pred = _get_predictor()

    if points:
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([int(p["label"]) for p in points], dtype=np.int32)
        clip_factor = 2.0
    else:
        coords = labels = None          # box-only prompt
        clip_factor = 1.5
    box_arr = np.array(box, dtype=np.float32) if box else None

    # Always multimask + pick the best candidate: no behaviour flip at 4 points.
    # The box disambiguates scale far better than point count ever did.
    with _run_lock:
        _restore_state(pred, _cache[image_id])
        masks, scores, _ = pred.predict(
            point_coords=coords,
            point_labels=labels,
            box=box_arr,
            multimask_output=True,
        )

    best = int(np.argmax(scores))
    mask = (masks[best] > 0).astype(np.uint8) * 255
    pp_coords = coords if coords is not None else np.zeros((0, 2), np.float32)
    pp_labels = labels if labels is not None else np.zeros((0,), np.int32)
    mask = _postprocess(mask, pp_coords, pp_labels)

    # Locality guarantee: even if SAM reaches far away, clip around the request box.
    if box is not None:
        H, W = mask.shape
        ex0, ey0, ex1, ey1 = _expand_box(box, clip_factor, W, H)
        clip = np.zeros_like(mask)
        clip[ey0:ey1, ex0:ex1] = mask[ey0:ey1, ex0:ex1]
        mask = clip
    return mask


def predict_box(image_id: str, box: list[float]) -> np.ndarray:
    """Segment the object inside an XYXY box (no point prompts). Returns a binary
    uint8 HxW mask (0/255). Used by people-mode to refine YOLO's coarse masks."""
    _ensure_ready(image_id)
    pred = _get_predictor()
    box_arr = np.array(box, dtype=np.float32)
    with _run_lock:
        _restore_state(pred, _cache[image_id])
        masks, scores, _ = pred.predict(
            point_coords=None, point_labels=None, box=box_arr, multimask_output=True,
        )
    best = int(np.argmax(scores))
    mask = (masks[best] > 0).astype(np.uint8) * 255
    # No points → _postprocess keeps the largest component and smooths it.
    return _postprocess(mask, np.zeros((0, 2), np.float32), np.zeros((0,), np.int32))


def _postprocess(mask: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Keep only components touched by a positive point; smooth ragged edges."""
    H, W = mask.shape
    n, comp = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if n <= 1:
        return mask

    keep_labels = set()
    for (x, y), lab in zip(coords, labels):
        if lab != 1:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W and comp[yi, xi] > 0:
            keep_labels.add(int(comp[yi, xi]))

    if keep_labels:
        out = np.isin(comp, list(keep_labels)).astype(np.uint8) * 255
    else:
        # No positive point landed on the mask - keep the largest component.
        counts = np.bincount(comp.ravel())
        counts[0] = 0
        out = (comp == int(counts.argmax())).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out
