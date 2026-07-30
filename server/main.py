"""FastAPI backend for EraseAll.

Serves the vanilla-JS frontend from web/ and exposes a small JSON API. Everything
runs locally, CPU-only, offline - no network calls, no API keys.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps

from server import lama_engine, masks, objects_engine, people_engine, sam_engine, store

try:  # iPhone .heic/.heif import (pillow-heif, BSD)
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB

app = FastAPI(title="EraseAll")


def _warm_async(image_id: str, build_index: bool = False) -> None:
    """Warm an image in the background: SAM embedding first (smart-select becomes
    usable in ~1-2s), then optionally the object index for hover (~20s). One
    thread, so indexing never starves the embedding.

    build_index is True for uploads (so Object mode is ready when first opened)
    and False for edit results - those index on demand via /api/objects_index,
    otherwise every removal would burn ~20s of CPU on an index nobody hovers."""
    if build_index:
        objects_engine.mark_pending(image_id)

    def work():
        try:
            sam_engine.prepare(image_id)
        except Exception:
            pass
        if build_index:
            try:
                objects_engine.build(image_id)
            except Exception:
                pass

    threading.Thread(target=work, daemon=True).start()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a JPEG/PNG/WebP, normalize to RGB PNG, store it under a new id."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 30 MB).")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # honor camera orientation
        img = img.convert("RGB")            # strip alpha / palette
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image_id = store.save_image(img)
    # Keep the untouched upload so Export can copy its EXIF onto an edited result.
    store.save_original_bytes(image_id, raw)
    store.record_lineage(image_id)          # a fresh upload is its own root
    _warm_async(image_id, build_index=True)  # embedding, then the hover index
    return {"image_id": image_id, "width": img.width, "height": img.height}


@app.post("/api/upload_edit")
async def upload_edit(base_image_id: str = Form(...), file: UploadFile = File(...)):
    """Accept a client-rendered full-res edit (clone stamp) as a new immutable
    image. Dimensions must match the base image it was derived from."""
    if not store.exists(base_image_id):
        raise HTTPException(status_code=404, detail="Unknown base_image_id.")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Edit too large (max 30 MB).")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read edited image.")

    base = store.load_image(base_image_id)
    if img.size != base.size:
        raise HTTPException(
            status_code=400,
            detail=f"Edit must match the base image size {base.size}, got {img.size}.",
        )
    new_id = store.save_image(img)
    store.record_lineage(new_id, base_image_id)
    _warm_async(new_id)
    return {"image_id": new_id}


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    if not store.exists(image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    return FileResponse(
        store.image_path(image_id),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


class RemoveRequest(BaseModel):
    image_id: str
    mask: str  # base64 PNG, white = selected
    method: str = "structure"  # "structure" (LaMa) | "pattern" (shift-map)
    edge: int = 0              # grow(+)/shrink(-) the selection, -20..20 px


# Cap the crop long-side fed to LaMa; larger crops are downscaled for inference.
MAX_CROP_SIDE = 1600
FEATHER_PX = 8


def _fill_crop(method: str, crop_pil: Image.Image, crop_mask_u8: "np.ndarray") -> Image.Image:
    """Fill the masked region of a crop. `structure` = LaMa (blends structure);
    `pattern` = OpenCV shift-map, which copies real patches from the surroundings
    so strong repeating textures (carpets, fabrics, bricks) stay crisp."""
    if method == "pattern":
        src = np.ascontiguousarray(np.asarray(crop_pil.convert("RGB")))
        # xphoto mask semantics are inverted: non-zero = KNOWN, zero = hole.
        known = (255 - crop_mask_u8).astype(np.uint8)
        dst = np.zeros_like(src)
        cv2.xphoto.inpaint(src, known, dst, cv2.xphoto.INPAINT_SHIFTMAP)
        return Image.fromarray(dst, "RGB")
    return lama_engine.inpaint(crop_pil, crop_mask_u8)


MAX_EDGE_PX = 20


def _decode_mask_for(image: Image.Image, mask_b64: str, edge: int = 0) -> np.ndarray:
    """Decode the client's mask and apply the selection-edge adjustment (M10):
    grow (+) or shrink (-) it before the normal pipeline, so a chronic slight
    under/over-selection can be fixed without re-brushing."""
    W, H = image.size
    mask = masks.decode_mask(mask_b64, target_hw=(H, W))
    if not mask.any():
        raise HTTPException(status_code=400, detail="Selection is empty.")
    edge = max(-MAX_EDGE_PX, min(MAX_EDGE_PX, int(edge)))
    if edge > 0:
        mask = masks.dilate(mask, edge)
    elif edge < 0:
        mask = masks.erode(mask, -edge)
        if not mask.any():
            raise HTTPException(status_code=400, detail="Selection vanished at that edge setting.")
    return mask


def _run_removal(
    image: Image.Image,
    mask: np.ndarray,
    method: str = "structure",
    mirror: bool = False,
    pad_frac: float = 0.30,
    min_size: int = 512,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Full removal pipeline (05 §Removal). Returns (edited full-size image, crop box).

    `mirror` horizontally flips the crop+mask before inpainting and flips the result
    back: LaMa is deterministic, so a flipped context is what makes a genuinely
    different-but-plausible variant (M9 V2) rather than a reseed.
    """
    W, H = image.size
    orig = np.asarray(image)  # HxWx3 uint8

    # 1) Dilate so LaMa repaints past the object edge (no ghost halo). We add
    #    FEATHER_PX to the base dilation: the composite feathers an *eroded* copy
    #    (by FEATHER_PX), so this keeps the object's true edge inside the fully-
    #    opaque core while the ramp still reaches 0 at the dilated boundary.
    dpx = max(5, round(0.01 * max(H, W))) + FEATHER_PX
    dmask = masks.dilate(mask, dpx)

    # 2) Crop around the mask with context (big CPU win; LaMa likes ~512 contexts).
    box = masks.bbox_with_context(dmask, (H, W), pad_frac=pad_frac, min_size=min_size)
    x0, y0, x1, y1 = box
    crop_img = image.crop((x0, y0, x1, y1))
    crop_dmask = dmask[y0:y1, x0:x1]
    cw, ch = crop_img.size

    # Optionally downscale the crop for inference, then upscale the result back.
    long_side = max(cw, ch)
    scale = MAX_CROP_SIDE / long_side if long_side > MAX_CROP_SIDE else 1.0
    if scale < 1.0:
        iw, ih = max(8, round(cw * scale)), max(8, round(ch * scale))
        infer_img = crop_img.resize((iw, ih), Image.LANCZOS)
        infer_mask = cv2.resize(crop_dmask, (iw, ih), interpolation=cv2.INTER_NEAREST)
    else:
        infer_img, infer_mask = crop_img, crop_dmask

    # 3) Fill (LaMa for structure, shift-map for pattern). A mirrored context
    #    yields a different plausible fill from the same deterministic model.
    if mirror:
        infer_img = ImageOps.mirror(infer_img)
        infer_mask = np.ascontiguousarray(infer_mask[:, ::-1])
    result = _fill_crop(method, infer_img, infer_mask)
    if mirror:
        result = ImageOps.mirror(result)

    if scale < 1.0:
        result = result.resize((cw, ch), Image.LANCZOS)
    result_np = np.asarray(result.convert("RGB"))

    # 4) Feathered composite. Feather an *eroded* dilated mask so the alpha ramp
    #    stays inside the dilated region → pixels outside it are byte-identical.
    alpha = masks.feather(masks.erode(crop_dmask, FEATHER_PX), FEATHER_PX)
    crop_orig = orig[y0:y1, x0:x1]
    composited = masks.composite(crop_orig, result_np, alpha)

    out = orig.copy()
    out[y0:y1, x0:x1] = composited
    return Image.fromarray(out, mode="RGB"), box


@app.post("/api/remove")
def remove(req: RemoveRequest):
    """Remove the selected object and return a new immutable image_id."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    method = req.method if req.method in ("structure", "pattern") else "structure"
    try:
        image = store.load_image(req.image_id)
        edited, _box = _run_removal(image, _decode_mask_for(image, req.mask, req.edge), method)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Removal failed: {e}")

    new_id = store.save_image(edited)
    store.record_lineage(new_id, req.image_id)
    _warm_async(new_id)  # keep chained smart-select / object hover snappy
    return {"image_id": new_id}


class Point(BaseModel):
    x: float
    y: float
    label: int  # 1 = foreground, 0 = negative


class SegmentRequest(BaseModel):
    image_id: str
    points: list[Point]
    box: list[float] | None = None  # optional XYXY box scoping this stroke


@app.post("/api/segment")
def segment(req: SegmentRequest):
    """Segment the object under one stroke's points, scoped to an optional box.
    `points` may be empty when `box` is given (Object mode's box-drag select)."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    if not req.points and not req.box:
        raise HTTPException(status_code=400, detail="Provide points or a box.")
    pts = [{"x": p.x, "y": p.y, "label": p.label} for p in req.points]
    try:
        mask = sam_engine.predict(req.image_id, pts, box=req.box)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="SAM embedding not ready; retry.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segment failed: {e}")
    return {"mask": masks.encode_mask(mask)}


# Above this Laplacian variance in the ring around the hole, the surroundings are
# a strong repeating texture and shift-map beats LaMa. Measured: carpet 7256,
# natural photo 3295, smooth gradient 488 - so 5000 sends carpets to `pattern`
# and leaves natural scenes (where shift-map looks worse) on wide-context LaMa.
PATTERN_ENERGY_THRESH = 5000.0
BAND_PX = 32


def _band_texture_energy(image: Image.Image, mask: np.ndarray) -> float:
    """Laplacian variance of the ring just outside the hole - how patterned is
    the context we'd have to continue?"""
    gray = np.asarray(image.convert("L")).astype(np.float64)
    band = (masks.dilate(mask, BAND_PX) > 0) & (mask == 0)
    if int(band.sum()) < 100:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[band].var())


class RemoveVariantsRequest(BaseModel):
    image_id: str
    mask: str
    edge: int = 0


@app.post("/api/remove_variants")
def remove_variants(req: RemoveVariantsRequest):
    """Three plausible fills for one selection; the user picks. LaMa is
    deterministic, so the variants come from varying its *input*, not a seed:
      V1 baseline · V2 mirrored context · V3 adaptive (pattern vs wide-context)."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    try:
        image = store.load_image(req.image_id)
        mask = _decode_mask_for(image, req.mask, req.edge)

        v1, box = _run_removal(image, mask)                       # baseline
        v2, _ = _run_removal(image, mask, mirror=True)            # mirrored context

        energy = _band_texture_energy(image, mask)
        if energy >= PATTERN_ENERGY_THRESH:
            v3, _ = _run_removal(image, mask, method="pattern")
            engine = "pattern"
        else:
            v3, _ = _run_removal(image, mask, pad_frac=0.6, min_size=768)
            engine = "lama_wide"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Removal failed: {e}")

    ids = [store.save_image(v) for v in (v1, v2, v3)]
    for i in ids:
        store.record_lineage(i, req.image_id)
        _warm_async(i)  # embedding only; the object index waits for Object mode
    return {
        "variants": ids,
        "edit_box": [int(v) for v in box],
        "v3_engine": engine,
        "band_energy": round(energy, 1),
    }


class ObjectAtRequest(BaseModel):
    image_id: str
    x: float
    y: float


@app.post("/api/object_at")
def object_at(req: ObjectAtRequest):
    """Smallest indexed object containing (x, y) - powers Object-mode hover."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    oid, mask = objects_engine.object_at(req.image_id, req.x, req.y)
    if oid is None:
        return {"object_id": None}
    return {"object_id": oid, "mask": masks.encode_mask(mask)}


@app.get("/api/objects_status/{image_id}")
def objects_status(image_id: str):
    if not store.exists(image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    return {"status": objects_engine.status(image_id)}


class ObjectsIndexRequest(BaseModel):
    image_id: str


@app.post("/api/objects_index")
def objects_index(req: ObjectsIndexRequest):
    """Ensure an object index exists for this image (Object mode opens on an
    edited image). No-op if it's already ready or building."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    return {"status": objects_engine.ensure_index_async(req.image_id)}


class DetectPeopleRequest(BaseModel):
    image_id: str


@app.post("/api/detect_people")
def detect_people(req: DetectPeopleRequest):
    """Detect all people and return a refined mask + box + confidence for each."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    try:
        found = people_engine.detect_people(req.image_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"People detection failed: {e}")
    people = [
        {
            "id": p["id"],
            "box": p["box"],
            "confidence": p["confidence"],
            "mask": masks.encode_mask(p["mask"]),
        }
        for p in found
    ]
    return {"people": people}


@app.get("/api/export")
def export(image_id: str, fmt: str = "png", q: int = 90, exif: int = 0):
    """Re-encode the stored result for download. `exif=1` copies the EXIF block
    from the ORIGINAL upload this image descends from (camera, date, lens)."""
    if not store.exists(image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    fmt = fmt.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise HTTPException(status_code=400, detail="fmt must be png or jpeg.")
    img = store.load_image(image_id)
    buf = io.BytesIO()

    exif_bytes = store.original_exif(store.get_root(image_id)) if exif else None
    if fmt in ("jpeg", "jpg"):
        kwargs = {"quality": max(60, min(100, int(q))), "optimize": True, "subsampling": 1}
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        img.save(buf, format="JPEG", **kwargs)
        media, ext = "image/jpeg", "jpg"
    else:
        kwargs = {}
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        img.save(buf, format="PNG", **kwargs)
        media, ext = "image/png", "png"

    return Response(
        content=buf.getvalue(),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="edited-{image_id}.{ext}"'},
    )


class TransformRequest(BaseModel):
    image_id: str
    angle: float = 0.0                  # degrees, positive = clockwise
    crop: list[int] | None = None       # XYXY in ROTATED image coords


@app.post("/api/transform")
def transform(req: TransformRequest):
    """Straighten then crop. The client mirrors PIL's expand geometry exactly, so
    the crop lands on precisely the region it previewed."""
    if not store.exists(req.image_id):
        raise HTTPException(status_code=404, detail="Unknown image_id.")
    img = store.load_image(req.image_id)
    # Any angle is valid here: the slider sends -45..45, the rotate buttons ±90/±180,
    # and the client previews with the same PIL-exact geometry either way.
    angle = float(req.angle)
    if angle:
        # PIL rotates counter-clockwise; negate so positive angle = clockwise.
        img = img.rotate(-angle, resample=Image.BICUBIC, expand=True)
    if req.crop:
        x0, y0, x1, y1 = [int(v) for v in req.crop]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, x1), min(img.height, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise HTTPException(status_code=400, detail="Crop is too small.")
        img = img.crop((x0, y0, x1, y1))

    new_id = store.save_image(img)
    store.record_lineage(new_id, req.image_id)
    _warm_async(new_id)
    return {"image_id": new_id, "width": img.width, "height": img.height}


@app.get("/api/health")
def health():
    return {
        "sam": "ready" if sam_engine.is_loaded() else "loading",
        "lama": "ready" if lama_engine.is_loaded() else "not_loaded",
        "people": "ready" if people_engine.is_loaded() else "not_loaded",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Browsers that ignore the <link rel=icon> still probe here; serve the real
    icon so the console/log isn't full of 404s."""
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


# Static frontend mounted last so /api/* routes above take precedence.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
