"""Unit tests for mask utilities. Run: venv\\Scripts\\python -m pytest tests -q
(or plain: venv\\Scripts\\python tests/test_masks.py)"""
import base64
import io

import numpy as np
from PIL import Image

from server import masks


def _mask_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_roundtrip_and_threshold():
    a = np.zeros((20, 30), np.uint8)
    a[5:15, 10:20] = 255
    b64 = _mask_b64(a)
    got = masks.decode_mask(b64)
    assert got.shape == (20, 30)
    assert set(np.unique(got)).issubset({0, 255})
    assert got[10, 15] == 255 and got[0, 0] == 0


def test_decode_resizes_to_target():
    a = np.zeros((10, 10), np.uint8)
    a[2:8, 2:8] = 255
    got = masks.decode_mask(_mask_b64(a), target_hw=(40, 50))
    assert got.shape == (40, 50)


def test_dilate_grows_and_erode_shrinks():
    a = np.zeros((60, 60), np.uint8)
    a[28:32, 28:32] = 255
    assert masks.dilate(a, 5).sum() > a.sum()
    assert masks.erode(a, 1).sum() < a.sum()


def test_feather_is_float_0_1():
    a = np.zeros((40, 40), np.uint8)
    a[10:30, 10:30] = 255
    f = masks.feather(a, 5)
    assert f.dtype == np.float32
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert f[20, 20] > 0.9  # solidly inside stays ~1


def test_bbox_with_context_snapped_and_clamped():
    H, W = 400, 500
    m = np.zeros((H, W), np.uint8)
    m[190:210, 240:260] = 255
    box = masks.bbox_with_context(m, (H, W), pad_frac=0.3, min_size=128, multiple=8)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
    assert (x1 - x0) % 8 == 0 and (y1 - y0) % 8 == 0
    assert (x1 - x0) >= 128 and (y1 - y0) >= 128
    # contains the original mask bbox
    assert x0 <= 240 and x1 >= 260 and y0 <= 190 and y1 >= 210


def test_bbox_empty_mask_is_none():
    assert masks.bbox_with_context(np.zeros((10, 10), np.uint8), (10, 10)) is None


def test_composite_identity_where_alpha_zero():
    orig = np.random.randint(0, 256, (30, 30, 3), np.uint8)
    edited = np.random.randint(0, 256, (30, 30, 3), np.uint8)
    alpha = np.zeros((30, 30), np.float32)
    alpha[10:20, 10:20] = 1.0
    out = masks.composite(orig, edited, alpha)
    # outside the alpha region: byte-identical to original
    assert np.array_equal(out[0:10, :], orig[0:10, :])
    # fully inside: equals edited
    assert np.array_equal(out[12:18, 12:18], edited[12:18, 12:18])


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
