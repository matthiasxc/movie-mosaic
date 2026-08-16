# movie-mosaic

Turn an MP4 into a **1120 × 1120** contact sheet: **70 × 70** tiles, each **16 × 16**, sampled evenly through the film.

Two render modes:

- **resize** — drop letterbox bars, center-crop the remaining picture to a square, scale to 16 × 16
- **color** — drop letterbox bars, average every remaining pixel, fill a 16 × 16 block with that color

## Install

Python 3.10+ on Windows, macOS, or Linux. PyAV’s wheels include FFmpeg, so you do not need a separate FFmpeg install.

```powershell
cd movie-mosaic
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

If `py -3.11` is not available, `python -m venv .venv` is fine as long as that interpreter is 3.10 or newer.

## Usage

Single file:

```powershell
movie-mosaic --mode resize --input "D:\Exports\Inception (2010).mp4" --output "D:\Mosaics"
```

Every `.mp4` or `.m4v` sitting directly in a folder (not recursive):

```powershell
movie-mosaic --mode color --input "D:\Exports" --output "D:\Mosaics"
```

Folder runs keep a resume file in the output directory: `render-queue-color.txt` or `render-queue-resize.txt`. Each line is `{filename} - {status} - {render time}`. Status starts as `Pending`, becomes `Rendering` while a movie is in progress, then `Done` (with elapsed time) or `Failed`. Re-run the same command after a stop or crash; `Done` and `Failed` are skipped, and a leftover `Rendering` row is retried first. Use `--overwrite` to retry `Done` or `Failed` rows. Existing mosaics from before this feature are adopted as `Done` and not rebuilt.

Output files look like `Inception (2010).mosaic-resize.png` and `Inception (2010).mosaic-color.png`.

| Flag | Meaning |
| --- | --- |
| `--mode resize\|color` | Required. Tile renderer. |
| `--input PATH` | One `.mp4`/`.m4v` or a flat folder of those files. |
| `--output DIR` | Where PNGs are written. Created if missing. |
| `--overwrite` | Replace existing mosaics. In a folder run, also retries queue rows marked Done or Failed. |
| `--no-letterbox` | Disable bar detection if a dark film is cropped too aggressively. |
| `--hwaccel auto\|cuda\|off` | Video decoder. **auto** (default) tries NVIDIA NVDEC, then software. **cuda** requires the GPU. **off** forces software. |

Each movie prints `decode: cuda` or `decode: software (...)` so you can see which path ran. If a render is still ~20 minutes on 1080p H.264, the files are likely disk-bound (NAS / Plex share); the GPU cannot fix that.

You can also run `python -m movie_mosaic` with the same flags.

## How frames are chosen

If the movie has `N` frames, tile `i` (0…4899) comes from frame

```text
round(i × (N − 1) / 4899)
```

That always includes the first and last frame. Movies shorter than 4,900 frames reuse some frames.

Letterbox / pillarbox bars are estimated from about 40 samples in the middle 90% of the file, then applied to every tile.

A feature-length 1080p encode is decoded at ~360p for speed, and only the 4,900 sampled frames are converted to RGB. With NVIDIA NVDEC (`--hwaccel auto` or `cuda`) a local 1080p H.264 file should take a few minutes; software-only is slower. Disk-bound network copies stay slow regardless of decoder.

## Tests

```powershell
pytest
```
