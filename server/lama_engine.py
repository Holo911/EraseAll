"""LaMa inpainting engine (local, CPU, offline).

Lazy singleton around the big-lama TorchScript checkpoint. We load the model
ourselves with map_location="cpu" - the released checkpoint carries CUDA device
tags, so simple_lama_inpainting's own `torch.jit.load` (no map_location) fails on
a CPU-only machine. We reuse the package's `prepare_img_and_mask` preprocessing so
inputs match what the model was traced for.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting.utils import prepare_img_and_mask

_MODEL_PATH = os.environ.get(
    "LAMA_MODEL",
    str(Path(__file__).resolve().parent.parent / "models" / "big-lama.pt"),
)

_model = None
_lock = threading.Lock()
_device = torch.device("cpu")


def is_loaded() -> bool:
    return _model is not None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if not Path(_MODEL_PATH).exists():
                    raise FileNotFoundError(
                        f"LaMa model not found at {_MODEL_PATH}. "
                        "Download big-lama.pt into models/ (see plan/02)."
                    )
                m = torch.jit.load(_MODEL_PATH, map_location=_device)
                m.eval()
                _model = m
    return _model


def inpaint(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Inpaint `image` (RGB PIL) under `mask` (uint8 HxW, 255=fill).

    Returns an RGB PIL image the same size as `image`. The model pads internally
    to a multiple of 8, so we crop the result back to the input dimensions.
    """
    model = _get_model()
    W, H = image.size
    mask_pil = Image.fromarray(mask, mode="L")

    # Serialize model calls: one worker, but /api/remove runs in a threadpool.
    with _lock:
        img_t, mask_t = prepare_img_and_mask(image, mask_pil, _device)
        with torch.inference_mode():
            out = model(img_t, mask_t)
    res = out[0].permute(1, 2, 0).cpu().numpy()
    res = np.clip(res * 255, 0, 255).astype(np.uint8)
    res = res[:H, :W]  # drop modulo-8 padding
    return Image.fromarray(res, mode="RGB")
