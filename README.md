# EraseAll

Remove objects from photos on your own PC. Brush over something, click Remove, and
it's gone - the gap is filled in to match the surrounding image.

Everything runs locally on the CPU. Your photos never leave your computer, and
after setup the app works with no internet connection at all.

## What it does

- **Remove objects** - three different fills are generated per removal, you pick
- **Smart select** - scribble across an object and the selection snaps to its edges
- **Object mode** - hover to highlight whatever is under the cursor, click to select
- **People mode** - finds everyone in the photo; tick who to remove
- **Clone stamp** - copy texture from elsewhere in the photo to fix details by hand
- **Crop, straighten, rotate**
- Opens HEIC (iPhone) photos; exports PNG or JPEG, optionally keeping the original EXIF

## Requirements

- Windows 10 or 11
- [Python 3.11 or newer](https://www.python.org/downloads/) - tick **Add Python to PATH** during install
- About 2 GB of free disk space
- No graphics card needed

## Setup (once)

Open PowerShell in this folder and run:

```powershell
py -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python download_models.py
```

The last command downloads about 240 MB of AI models. It needs internet - nothing
after this does.

## Run

Double-click **`run.bat`**. Your browser opens by itself once the app is ready.

It normally runs at <http://127.0.0.1:8000>. If another program is already using
that port, the launcher moves up (8001, 8002, …) and prints the address it used.

## Stop

Close the `run.bat` window, or press `Ctrl+C` inside it.

## Using it

1. **Upload** a photo, or drag one onto the window.
2. Choose how to select what you want gone:
   - **Raw** - a plain brush; exactly what you paint
   - **Smart** - scribble across an object and it snaps to the whole object
   - **Object** - hover to highlight, click to select (give it ~20 s after upload
     while it scans the photo)
   - **People** - click *Select people*, then untick anyone you want to keep
3. Click **Remove object**. Three versions are produced - press `1`, `2`, `3` to
   compare them, `Enter` to keep one, `Esc` to discard them all.
4. **Download** to export as PNG or JPEG.

Removal takes roughly 10-30 seconds on a typical laptop CPU.

**Fill quality:** *Auto* suits most photos. Switch to *Pattern* for repeating
backgrounds - carpets, brickwork, tiles - where it rebuilds the pattern instead of
blurring it.

**Other tools:** *Stamp* (Alt-click a source spot, then paint to clone texture) and
*Crop*. `Ctrl+Z` undoes, hold `B` to compare against the original, and the `?`
button lists every shortcut.

## Troubleshooting

The `run.bat` window stays open and explains what went wrong. The usual causes:

- **"Setup needed"** - the setup steps above didn't complete
- **"Missing AI model(s)"** - run `venv\Scripts\python download_models.py`
- **Browser didn't open** - the window prints the address; open it manually

Your uploads and edits are kept in `data\`. Delete that folder any time to clear them.

## License

[AGPL-3.0](LICENSE).

People mode uses [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO,
which is AGPL-3.0 - so this project is licensed the same way. The other components
are permissively licensed: MobileSAM and LaMa (Apache-2.0), OpenCV (Apache-2.0),
FastAPI (MIT).
