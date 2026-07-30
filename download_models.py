"""Download the local model checkpoints into models/.

Run once after installing dependencies:

    venv\\Scripts\\python download_models.py

Idempotent: skips a file that is already present with a plausible size. After this
completes the whole app runs fully offline - no further network access is needed.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# (filename, url, minimum plausible size in bytes)
FILES = [
    (
        "mobile_sam.pt",
        "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
        30 * 1024 * 1024,   # ~40 MB
    ),
    (
        "big-lama.pt",
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        150 * 1024 * 1024,  # ~200 MB
    ),
    (
        "yolo11n-seg.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        4 * 1024 * 1024,    # ~6 MB (People mode)
    ),
]


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done * 100 // total
        sys.stdout.write(f"\r    {pct:3d}%  ({done // (1024*1024)} / {total // (1024*1024)} MB)")
        sys.stdout.flush()


def download(name: str, url: str, min_size: int) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"[skip] {name} already present ({dest.stat().st_size // (1024*1024)} MB)")
        return
    print(f"[get ] {name}\n       {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    print()
    tmp.replace(dest)
    if dest.stat().st_size < min_size:
        raise SystemExit(f"Downloaded {name} looks too small - check your connection.")
    print(f"[done] {name} ({dest.stat().st_size // (1024*1024)} MB)")


def main() -> None:
    print("Downloading model checkpoints into models/ (one-time, needs internet)...")
    for name, url, min_size in FILES:
        download(name, url, min_size)
    print("\nAll models ready. The app now runs fully offline.")


if __name__ == "__main__":
    main()
