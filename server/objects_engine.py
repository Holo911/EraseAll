"""Object index for hover-highlight (MobileSAM automatic mask generator).

After upload we run SamAutomaticMaskGenerator once per image on a 1024-long-side
copy and keep the resulting masks as packed bits. Hovering then costs a bit-test
per candidate instead of a model call, so `object_at` answers in milliseconds.

The AMG runs on its OWN model instance so it never takes sam_engine's predictor
lock - a removal or a smart-select stroke stays responsive while indexing runs.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from server import store

_CKPT = os.environ.get(
    "MOBILE_SAM_MODEL",
    str(Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt"),
)

LONG_SIDE = 1024        # AMG runs at this scale; cost is ~constant vs megapixels
POINTS_PER_SIDE = 16
# AMG's default quality gates (iou .88 / stability .95) reject uncertain masks and
# left ~24% of bus.jpg uncovered - hovering a whole person lit up nothing. Relaxing
# them takes coverage 76%->97.5% at the SAME 23s; raising points_per_side to 24
# instead cost 2.3x the time for +0.4% coverage, so we tune the gates, not the grid.
PRED_IOU_THRESH = 0.80
STABILITY_THRESH = 0.85
MIN_AREA_FRAC = 0.0005  # 0.05% of the (downscaled) image
MAX_MASKS = 200
CACHE_MAX = 2           # ~20MB per image worst case

_amg = None
_amg_load_lock = threading.Lock()
_amg_run_lock = threading.Lock()   # separate from sam_engine's predictor lock

_index: "OrderedDict[str, dict]" = OrderedDict()   # image_id -> {shape, items}
_status: "dict[str, str]" = {}                     # image_id -> pending|ready|unavailable
_status_lock = threading.Lock()


def _get_amg():
    global _amg
    if _amg is None:
        with _amg_load_lock:
            if _amg is None:
                if not Path(_CKPT).exists():
                    raise FileNotFoundError(f"MobileSAM checkpoint not found at {_CKPT}")
                from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
                sam = sam_model_registry["vit_t"](checkpoint=_CKPT)
                sam.to("cpu")
                sam.eval()
                _amg = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=POINTS_PER_SIDE,
                    pred_iou_thresh=PRED_IOU_THRESH,
                    stability_score_thresh=STABILITY_THRESH,
                )
    return _amg


def mark_pending(image_id: str) -> None:
    with _status_lock:
        _status[image_id] = "pending"


def status(image_id: str) -> str:
    if image_id in _index:
        return "ready"
    with _status_lock:
        return _status.get(image_id, "unavailable")


def ensure_index_async(image_id: str) -> str:
    """Start building the index if it isn't ready/building. Returns the status.
    Lets Object mode pull an index on demand for edited images, instead of every
    edit eagerly burning ~20s of CPU on an index the user may never hover."""
    if image_id in _index:
        return "ready"
    with _status_lock:
        if _status.get(image_id) == "pending":
            return "pending"
        _status[image_id] = "pending"
    threading.Thread(target=build, args=(image_id,), daemon=True).start()
    return "pending"


def build(image_id: str) -> None:
    """Compute and cache the object index. Safe to call from a background thread."""
    mark_pending(image_id)
    try:
        pil = store.load_image(image_id)
        W, H = pil.size
        scale = LONG_SIDE / max(W, H)
        if scale < 1.0:
            small = pil.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.LANCZOS)
        else:
            small = pil
        arr = np.asarray(small)
        h, w = arr.shape[:2]

        amg = _get_amg()
        with _amg_run_lock:
            raw = amg.generate(arr)

        min_area = MIN_AREA_FRAC * (h * w)
        keep = [m for m in raw if m["area"] >= min_area]
        keep.sort(key=lambda m: m["area"])          # ascending → smallest match first
        keep = keep[:MAX_MASKS]

        items = [(int(m["area"]), np.packbits(m["segmentation"].astype(bool).ravel())) for m in keep]
        _index[image_id] = {"shape": (h, w), "orig": (H, W), "items": items}
        _index.move_to_end(image_id)
        while len(_index) > CACHE_MAX:
            _index.popitem(last=False)
        with _status_lock:
            _status[image_id] = "ready"
    except Exception:
        with _status_lock:
            _status[image_id] = "unavailable"


def _bit_at(packed: np.ndarray, idx: int) -> bool:
    return bool((packed[idx >> 3] >> (7 - (idx & 7))) & 1)


def object_at(image_id: str, x: float, y: float) -> tuple[int | None, np.ndarray | None]:
    """Smallest indexed object containing (x, y) in ORIGINAL image coords.
    Returns (object_id, uint8 HxW mask 0/255) or (None, None)."""
    entry = _index.get(image_id)
    if entry is None:
        return None, None
    _index.move_to_end(image_id)
    h, w = entry["shape"]
    H, W = entry["orig"]
    sx = int(round(x * w / W))
    sy = int(round(y * h / H))
    if not (0 <= sx < w and 0 <= sy < h):
        return None, None

    idx = sy * w + sx
    for oid, (_area, packed) in enumerate(entry["items"]):   # ascending area
        if _bit_at(packed, idx):
            small = np.unpackbits(packed, count=h * w).reshape(h, w).astype(np.uint8) * 255
            full = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 3px close
            full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel)
            return oid, full
    return None, None
