"""Mask utilities - pure functions, unit-testable.

Convention: a mask is a single-channel uint8 array, HxW, values in {0, 255},
where 255 (white) means "selected". Masks travel over the wire as base64 PNG.
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def decode_mask(b64: str, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """base64 PNG -> uint8 HxW {0,255}. Resized (NEAREST) to target_hw if given."""
    if "," in b64:  # tolerate a data: URL prefix
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    mask = np.array(img, dtype=np.uint8)
    mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    if target_hw is not None and mask.shape != target_hw:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    return mask


def encode_mask(mask: np.ndarray) -> str:
    """uint8 HxW -> base64 PNG (no data: prefix)."""
    m = np.where(mask >= 128, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow the mask by `px` using an elliptical kernel."""
    if px <= 0:
        return mask.copy()
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink the mask by `px` using an elliptical kernel."""
    if px <= 0:
        return mask.copy()
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    """Gaussian-blur a binary mask into a float32 alpha in [0,1]."""
    if px <= 0:
        return (mask.astype(np.float32) / 255.0)
    k = 2 * px + 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    return (blurred.astype(np.float32) / 255.0)


def bbox_with_context(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    pad_frac: float = 0.25,
    min_size: int = 512,
    multiple: int = 8,
) -> tuple[int, int, int, int] | None:
    """Tight bbox of the mask, expanded by pad_frac each side, grown to at least
    min_size, snapped up to `multiple`, clamped to image bounds.
    Returns (x0, y0, x1, y1) half-open, or None if the mask is empty.
    """
    H, W = image_shape
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0

    # Expand by a fraction of the bbox size on each side.
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    x0 -= px; x1 += px
    y0 -= py; y1 += py

    # Grow to at least min_size (centered), capped at image size.
    def grow(a0, a1, limit):
        size = a1 - a0
        if size < min_size:
            extra = min_size - size
            a0 -= extra // 2
            a1 += extra - extra // 2
        return a0, a1
    x0, x1 = grow(x0, x1, W)
    y0, y1 = grow(y0, y1, H)

    # Snap the *size* up to a multiple, then clamp to the image.
    def snap(a0, a1, limit):
        size = a1 - a0
        size = ((size + multiple - 1) // multiple) * multiple
        size = min(size, limit)
        a1 = a0 + size
        if a1 > limit:
            a1 = limit
            a0 = max(0, limit - size)
        if a0 < 0:
            a0 = 0
            a1 = min(limit, size)
        return a0, a1
    x0, x1 = snap(x0, x1, W)
    y0, y1 = snap(y0, y1, H)
    return x0, y0, x1, y1


def composite(original: np.ndarray, edited: np.ndarray, feathered: np.ndarray) -> np.ndarray:
    """orig*(1-m) + edited*m per channel. `feathered` is HxW float [0,1].

    Pixels where feathered == 0 are returned byte-identical to `original`.
    """
    m = feathered[..., None]  # HxWx1
    orig = original.astype(np.float32)
    edit = edited.astype(np.float32)
    out = orig * (1.0 - m) + edit * m
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Guarantee exact identity where alpha is truly zero (avoid float drift).
    zero = feathered <= 0.0
    out[zero] = original[zero]
    return out
