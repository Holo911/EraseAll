"""People detection engine (YOLO11n-seg, local, CPU, offline).

Detects every person in an image and returns a refined mask per person. YOLO's
segmentation masks come from low-res prototypes and are coarse, so each person's
box is fed to MobileSAM (whose embedding is already cached per image_id) for a
tight mask; if SAM wanders (IoU < 0.5 vs YOLO) we fall back to YOLO's mask.

Fully offline: the checkpoint is loaded by explicit local path (no auto-download)
and telemetry sync is disabled. YOLO_CONFIG_DIR is pointed inside models/ so
ultralytics never writes settings into the repo root.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CFG_DIR = _ROOT / "models" / ".ultralytics"
_CFG_DIR.mkdir(parents=True, exist_ok=True)
# Must be set before ultralytics is imported (done lazily below).
os.environ.setdefault("YOLO_CONFIG_DIR", str(_CFG_DIR))

from server import sam_engine, store  # noqa: E402

_MODEL_PATH = os.environ.get("YOLO_MODEL", str(_ROOT / "models" / "yolo11n-seg.pt"))
_PERSON_CLASS = 0
_CONF = 0.35
_REFINE_IOU = 0.5
_DILATE_PX = 3

_model = None
_lock = threading.Lock()


def is_loaded() -> bool:
    return _model is not None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if not Path(_MODEL_PATH).exists():
                    raise FileNotFoundError(
                        f"YOLO checkpoint not found at {_MODEL_PATH}. "
                        "Run download_models.py to enable People mode."
                    )
                from ultralytics import YOLO
                try:
                    from ultralytics import settings
                    settings.update({"sync": False})  # no telemetry / online sync
                except Exception:
                    pass
                _model = YOLO(_MODEL_PATH)
    return _model


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def detect_people(image_id: str) -> list[dict]:
    """Return [{id, box:[x0,y0,x1,y1], confidence, mask:uint8 HxW 0/255}, ...]."""
    pil = store.load_image(image_id)  # RGB
    W, H = pil.size
    model = _get_model()

    with _lock:
        results = model.predict(
            pil, classes=[_PERSON_CLASS], conf=_CONF,
            device="cpu", retina_masks=True, verbose=False,
        )
    r = results[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    ymasks = r.masks.data.cpu().numpy()  # N x h x w, float {0,1}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1))

    people = []
    for i in range(len(boxes)):
        ym = ymasks[i]
        if ym.shape != (H, W):
            ym = cv2.resize(ym.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        y_bool = ym > 0.5
        box = [float(v) for v in boxes[i]]

        # Refine with SAM using the person box; fall back if SAM grabbed the wrong thing.
        try:
            s_bool = sam_engine.predict_box(image_id, box) > 0
            final = s_bool if _iou(s_bool, y_bool) >= _REFINE_IOU else y_bool
        except Exception:
            final = y_bool

        mask_u8 = cv2.dilate((final.astype(np.uint8) * 255), kernel)  # hide hair/edges
        people.append({
            "id": i,
            "box": box,
            "confidence": float(confs[i]),
            "mask": mask_u8,
        })
    return people
