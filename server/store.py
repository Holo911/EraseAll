"""Immutable image store.

Every image (upload or edit result) lives as data/{image_id}.png. image_id is a
12-char hex string. Images are never mutated in place - an edit writes a new id, so
undo/redo is just the client pointing at an earlier id.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# image_id -> the id of the ORIGINAL upload it descends from. Export uses this to
# copy the untouched camera EXIF onto an edited result.
_LINEAGE_PATH = DATA_DIR / "lineage.json"
_lineage_lock = threading.Lock()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def valid_id(image_id: str) -> bool:
    return bool(_ID_RE.match(image_id))


def image_path(image_id: str) -> Path:
    """Path for an image_id. Raises on a malformed id (blocks path traversal)."""
    if not valid_id(image_id):
        raise ValueError(f"invalid image_id: {image_id!r}")
    return DATA_DIR / f"{image_id}.png"


def save_image(img: Image.Image) -> str:
    """Persist a PIL image under a fresh id and return that id."""
    image_id = new_id()
    img.save(image_path(image_id), format="PNG")
    return image_id


def load_image(image_id: str) -> Image.Image:
    """Load an image as RGB. Raises FileNotFoundError if the id is unknown."""
    path = image_path(image_id)
    if not path.exists():
        raise FileNotFoundError(image_id)
    return Image.open(path).convert("RGB")


def exists(image_id: str) -> bool:
    return valid_id(image_id) and image_path(image_id).exists()


# ---------------------------------------------------------------- originals

def original_path(image_id: str) -> Path:
    """Where the untouched uploaded bytes live (whatever format they arrived in)."""
    if not valid_id(image_id):
        raise ValueError(f"invalid image_id: {image_id!r}")
    return DATA_DIR / f"{image_id}.orig"


def save_original_bytes(image_id: str, raw: bytes) -> None:
    original_path(image_id).write_bytes(raw)


def original_exif(image_id: str) -> bytes | None:
    """EXIF block from the original upload, or None if it had none."""
    p = original_path(image_id)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            exif = im.getexif()
            if not exif:
                return None
            data = exif.tobytes()
            return data or None
    except Exception:
        return None


# ---------------------------------------------------------------- lineage

def _load_lineage() -> dict:
    if not _LINEAGE_PATH.exists():
        return {}
    try:
        return json.loads(_LINEAGE_PATH.read_text())
    except Exception:
        return {}


def record_lineage(new_id: str, source_id: str | None = None) -> None:
    """Point new_id at its root original. A fresh upload is its own root."""
    with _lineage_lock:
        data = _load_lineage()
        root = new_id if source_id is None else data.get(source_id, source_id)
        data[new_id] = root
        try:
            _LINEAGE_PATH.write_text(json.dumps(data))
        except Exception:
            pass


def get_root(image_id: str) -> str:
    with _lineage_lock:
        return _load_lineage().get(image_id, image_id)
