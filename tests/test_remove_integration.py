"""Integration test for the removal pipeline (loads the real LaMa model).
Run: PYTHONPATH=. venv\\Scripts\\python tests/test_remove_integration.py
Proves the M2 acceptance criteria:
  - object is plausibly filled (removed region no longer matches the object color)
  - pixels OUTSIDE the dilated mask are byte-identical to the original
  - output is the same resolution as the input
"""
import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from server import masks
from server.main import _decode_mask_for, _run_removal


def make_scene():
    W, H = 900, 600
    img = Image.new("RGB", (W, H), (60, 110, 70))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(0)
    # light texture so LaMa has something to continue
    noise = rng.integers(-18, 18, (H, W, 1), dtype=np.int16)
    arr = np.clip(np.asarray(img).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse([360, 200, 560, 400], fill=(230, 120, 40), outline=(120, 50, 8), width=5)
    return img


def circle_mask():
    W, H = 900, 600
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse([360, 200, 560, 400], fill=255)
    buf = io.BytesIO(); m.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    img = make_scene()
    orig = np.asarray(img)
    H, W = orig.shape[:2]
    mask_b64 = circle_mask()

    edited, _box = _run_removal(img, _decode_mask_for(img, mask_b64))
    out = np.asarray(edited)

    # Reproduce the pipeline's dilated mask to define "outside the dilated mask".
    # Must match _run_removal: base dilation + FEATHER_PX.
    from server.main import FEATHER_PX
    mask = masks.decode_mask(mask_b64, target_hw=(H, W))
    dpx = max(5, round(0.01 * max(H, W))) + FEATHER_PX
    dmask = masks.dilate(mask, dpx)
    outside = dmask == 0

    checks = []

    # 1) same resolution
    checks.append(("same resolution", out.shape == orig.shape))

    # 2) pixels outside the dilated mask are byte-identical
    identical = np.array_equal(out[outside], orig[outside])
    n_diff = int(np.count_nonzero(np.any(out != orig, axis=2) & outside))
    checks.append((f"outside dilated mask byte-identical (diff px={n_diff})", identical))

    # 3) object removed: mean color inside the object is now far from orange and
    #    close to the green background (plausible fill).
    obj = mask == 255
    orange = np.array([230, 120, 40])
    bg = np.array([60, 110, 70])
    mean_after = out[obj].mean(axis=0)
    d_orange = np.linalg.norm(mean_after - orange)
    d_bg = np.linalg.norm(mean_after - bg)
    checks.append(
        (f"object filled (dist->orange {d_orange:.0f} > dist->bg {d_bg:.0f})",
         d_orange > d_bg and d_orange > 60),
    )

    # 4) some pixels inside the mask actually changed (an edit happened)
    changed_inside = int(np.count_nonzero(np.any(out != orig, axis=2) & obj))
    checks.append((f"edit occurred inside mask (changed px={changed_inside})", changed_inside > 1000))

    ok = all(c[1] for c in checks)
    for name, passed in checks:
        print(("PASS " if passed else "FAIL ") + name)
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
    # Save an artifact for visual inspection.
    edited.save("remove_result.out.png")
    print("wrote remove_result.out.png")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
