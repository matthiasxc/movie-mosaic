# Speed up mosaic renders (NVIDIA, 1080p H.264)

Parked plan. Implement after the folder render-queue feature.

## What the 20 minutes actually is

Tile rendering is not the bottleneck. A mosaic is 4,900 tiny 16×16 operations. That is seconds.

The pipeline walks **every frame** of the movie so it can hit exact sample indices. For a 2-hour 24 fps file that is ~173,000 decodes. Today `iter_rgb_frames()` also **scales and copies every one of those frames to a 360p RGB NumPy array**, even though only 4,900 are kept:

```python
        for frame in container.decode(video=0):
            yield _to_rgb(frame, width, height)
```

On 1080p H.264 that extra convert-every-frame work is a large share of the 20 minutes. Putting the 16×16 tiles on a GPU would not move the needle.

Two levers matter, in this order:

1. **Stop converting frames we throw away** (CPU, low risk, likely the bigger win on 1080p H.264).
2. **Decode H.264 with NVDEC** on the NVIDIA GPU (real hardware path, medium risk). The installed PyAV 18.1.0 wheel already lists `cuda` and `h264_cuvid`.

Expected outcome if both land, for a typical 1080p H.264 feature:

| Change | Rough time per movie |
|---|---|
| Today (software decode + RGB every frame) | ~20 min |
| After skip-unused-RGB | ~5–8 min |
| After that + NVDEC | ~2–5 min |

If the files live on a slow NAS/Plex share, disk will cap both. GPU cannot fix that.

## Phase 1 — only RGB-convert the 4,900 keepers

**Effort: half a day. Risk: low.**

Change `iter_rgb_frames` / `extract_tiles` so the hot loop is:

- `decode()` each packet (unavoidable for exact frame numbers)
- if this presentation index is not a target, discard the `VideoFrame` and continue
- only then `reformat` → RGB NumPy → tile

Also set the software decoder to auto threads (`codec_context.thread_count = 0`) so libavcodec can use more than one core on H.264.

Keep the letterbox pass as-is (40 seeks). It is not the 20-minute cost.

**Tests:** existing synthetic MP4s must still produce the same 1120×1120 output. Add a unit-level assertion that a 30-frame clip still yields 4,900 tiles and the same first/last colors.

This phase is worth doing even if we never touch the GPU.

## Phase 2 — optional NVDEC via PyAV `HWAccel`

**Effort: 1–1.5 days on top of Phase 1. Risk: medium.**

PyAV 18 API (already in the venv):

```python
from av.codec.hwaccel import HWAccel

container = av.open(
    str(path),
    hwaccel=HWAccel(device_type="cuda", allow_software_fallback=True),
)
# stream.codec_context.is_hwaccel  → log whether GPU actually engaged
```

### Design

- New `--hwaccel {auto,cuda,off}`. Default **`auto`**: try CUDA, fall back to today’s software path if device create / open / first decode fails.
- `--hwaccel cuda` is strict: fail the file if the GPU is not used (`is_hwaccel` is false).
- `--hwaccel off` is the current decoder (plus Phase 1).
- Print one line per movie: `decode: cuda` or `decode: software (cuda unavailable: …)` so a 20-minute run is diagnosable.
- Still only RGB-convert target frames. Hardware frames are downloaded only for those 4,900 hits (`is_hw_owned=False`, the default).
- Letterbox samples stay on the same decoder as the main pass so crop math matches the frames we will actually use.

No new Python dependencies. No custom FFmpeg build. The wheel already exposes the device.

### What we will not do

- CUDA kernels for 16×16 tiles or color averages.
- `scale_cuda` / keeping the whole stream in VRAM (`is_hw_owned=True`). Extra complexity, tiny gain once we only download 4,900 frames.
- Seeking to 4,900 timestamps instead of a sequential pass. On H.264 that re-decodes from each keyframe and is usually slower and less accurate.
- Parallel movies on one GPU in v1. One NVDEC stream at a time is the stable setting.

## Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Wrong bottleneck** | 1080p H.264 software decode is already fast; NAS I/O or the RGB-every-frame loop may dominate. NVDEC then saves little. | Ship Phase 1 first (or in the same PR but land it independently). Log frames/sec and `decode: cuda\|software`. |
| **CUDA init fails on this PC** | Wheel lists `cuda`, but the driver, Optimus laptop, or a missing nvcuda can still fail at `av.open`. | `auto` fallback. Never make GPU required. |
| **Frame-index drift** | NVDEC + B-frames can emit a different count than `stream.frames`. Last tiles would pad or shift. | Compare decoded count vs probe; pad as today; add a fixture test on software and a GPU test that skips if `cuda` is missing. |
| **Slightly different pixels** | NVDEC vs software is usually identical for 8-bit H.264; not a guarantee. | Keep software as `--hwaccel off` so you can A/B a title. Do not change sampling math. |
| **10-bit / unusual profiles** | Rare in “mostly 1080p H.264,” but a stray HEVC 10-bit rip may fall back or fail. | `allow_software_fallback=True` in `auto`; surface the reason. |
| **Tests on machines without NVIDIA** | Existing pytest must stay green. | Hardware tests are skip-if-no-cuda. Synthetic MPEG-4 clips keep using software. |
| **Windows Group Policy** | Already blocks `Activate.ps1`. Unrelated to CUDA, but we keep invoking `python.exe -m movie_mosaic`. | No new scripts to execute. |

## How long

| Scope | Calendar time | What you get |
|---|---|---|
| Phase 1 only | **~0.5 day** | Likely the majority of the 20 → ~5–8 min drop on 1080p H.264 |
| Phase 2 | **~1–1.5 days** after Phase 1 | NVDEC flag, fallback, logging, skip-if-no-GPU tests |
| Both in one pass | **~2 days** | One CLI change, one README note, both speedups |

“2 days” assumes we can run at least one real 1080p H.264 file on this NVIDIA box to confirm `is_hwaccel` and time it. Without a real file, GPU work can be coded but not honestly benchmarked.

## Implementation sketch

1. Refactor `video.py` so decode and RGB convert are separate. `extract_tiles` asks for RGB only on hits.
2. Enable software decoder auto-threads.
3. Add `open_video(path, hwaccel=...)` that tries `HWAccel(device_type="cuda")` and returns `(container, "cuda"|"software", warning)`.
4. Wire `probe` / `iter` / letterbox sample through that opener (probe can stay software; it only reads metadata).
5. CLI `--hwaccel` + per-file decode line.
6. Tests: Phase 1 behavior unchanged; new tests for CLI flag parsing; optional live CUDA test marked `pytest.mark.skipif`.
7. README: flags, and a short “if it is still slow, check disk / look at the decode: line.”

## Recommendation

Do **both**, but treat Phase 1 as the real fix for 1080p H.264 and Phase 2 as the NVIDIA bonus. If you only want one change, do Phase 1. GPU tile rendering is not on this list on purpose.
