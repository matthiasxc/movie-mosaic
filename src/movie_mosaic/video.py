"""Probe MP4s and decode frames, converting to RGB only when needed."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np

from movie_mosaic.config import DECODE_MAX_HEIGHT
from movie_mosaic.letterbox import (
    EdgeCrop,
    combine_crops,
    detect_frame_bars,
    letterbox_sample_indices,
)

HwaccelMode = Literal["auto", "cuda", "off"]
HWACCEL_MODES: tuple[HwaccelMode, ...] = ("auto", "cuda", "off")


class VideoError(RuntimeError):
    """Raised when a file cannot be probed or decoded as video."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    width: int
    height: int
    rate: float


@dataclass(frozen=True)
class DecodeBackend:
    """Resolved decoder: CUDA NVDEC or software libavcodec."""

    name: Literal["cuda", "software"]
    warning: str | None = None

    def describe(self) -> str:
        if self.name == "cuda":
            return "decode: cuda"
        if self.warning:
            return f"decode: software ({self.warning})"
        return "decode: software"


def scaled_size(
    width: int,
    height: int,
    max_height: int = DECODE_MAX_HEIGHT,
) -> tuple[int, int]:
    """Preserve aspect ratio, capping height at ``max_height``."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if height <= max_height:
        return width, height
    new_height = max_height
    new_width = max(1, round(width * max_height / height))
    return new_width, new_height


def _stream_rate(stream: av.video.stream.VideoStream) -> float | None:
    rate = stream.average_rate or stream.guessed_rate
    if rate is None:
        return None
    value = float(rate)
    return value if value > 0 else None


def _frame_count(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> int:
    if stream.frames and stream.frames > 0:
        return int(stream.frames)

    rate = _stream_rate(stream)
    if rate is None:
        raise VideoError("could not determine frame rate")

    if stream.duration is not None and stream.time_base is not None:
        return max(1, int(stream.duration * stream.time_base * rate))

    if container.duration is not None:
        return max(1, int(container.duration / av.time_base * rate))

    raise VideoError("could not determine frame count")


def probe(path: Path) -> VideoInfo:
    """Read frame count and size from the first video stream (software open)."""
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"cannot open {path}: {exc}") from exc

    try:
        if not container.streams.video:
            raise VideoError(f"no video stream in {path}")
        stream = container.streams.video[0]
        if stream.width is None or stream.height is None:
            raise VideoError(f"video stream in {path} has no size")
        rate = _stream_rate(stream)
        if rate is None:
            raise VideoError(f"could not determine frame rate for {path}")
        return VideoInfo(
            path=path,
            frame_count=_frame_count(container, stream),
            width=int(stream.width),
            height=int(stream.height),
            rate=rate,
        )
    finally:
        container.close()


def _cuda_probe_error(path: Path) -> str | None:
    """Return None if CUDA decode works on this file, else a short reason."""
    try:
        from av.codec.hwaccel import HWAccel

        hwaccel = HWAccel(device_type="cuda", allow_software_fallback=False)
        container = av.open(str(path), hwaccel=hwaccel)
    except Exception as exc:
        return str(exc)

    try:
        if not container.streams.video:
            return "no video stream"
        if not container.streams.video[0].codec_context.is_hwaccel:
            return "decoder did not enable hardware acceleration"
        return None
    finally:
        container.close()


def resolve_backend(path: Path, hwaccel: HwaccelMode = "off") -> DecodeBackend:
    """Pick cuda or software. ``auto`` tries CUDA and falls back."""
    if hwaccel not in HWACCEL_MODES:
        raise ValueError(f"unknown hwaccel mode {hwaccel!r}")
    if hwaccel == "off":
        return DecodeBackend("software")

    error = _cuda_probe_error(path)
    if error is None:
        return DecodeBackend("cuda")
    if hwaccel == "cuda":
        raise VideoError(f"CUDA decode required but unavailable: {error}")
    return DecodeBackend("software", warning=f"cuda unavailable: {error}")


def open_container(path: Path, backend: DecodeBackend) -> av.container.InputContainer:
    """Open a video for decode using a previously resolved backend."""
    try:
        if backend.name == "cuda":
            from av.codec.hwaccel import HWAccel

            hwaccel = HWAccel(device_type="cuda", allow_software_fallback=False)
            container = av.open(str(path), hwaccel=hwaccel)
        else:
            container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"cannot open {path}: {exc}") from exc

    if not container.streams.video:
        container.close()
        raise VideoError(f"no video stream in {path}")

    if backend.name == "software":
        # 0 = libavcodec auto thread count.
        container.streams.video[0].codec_context.thread_count = 0
    return container


def frame_to_rgb(frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    """Scale a decoded frame to RGB. Call only for frames that will become tiles."""
    return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()


def iter_decoded_frames(path: Path, backend: DecodeBackend) -> Iterator[av.VideoFrame]:
    """Yield sequential decoded frames without converting to RGB."""
    container = open_container(path, backend)
    try:
        for frame in container.decode(video=0):
            yield frame
    except av.FFmpegError as exc:
        raise VideoError(f"decode failed for {path}: {exc}") from exc
    finally:
        container.close()


def _seek_to_index(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    frame_index: int,
    rate: float,
) -> None:
    if stream.time_base is None:
        return
    seconds = frame_index / rate
    offset = int(seconds / float(stream.time_base))
    container.seek(offset, stream=stream)


def sample_rgb_frames(
    path: Path,
    indices: list[int],
    width: int,
    height: int,
    rate: float,
    backend: DecodeBackend | None = None,
) -> list[np.ndarray]:
    """Decode one frame near each index via approximate seeks."""
    if not indices:
        return []

    backend = backend or DecodeBackend("software")
    container = open_container(path, backend)
    frames: list[np.ndarray] = []
    try:
        stream = container.streams.video[0]
        for index in indices:
            try:
                _seek_to_index(container, stream, index, rate)
                frame = next(container.decode(video=0))
            except (av.FFmpegError, StopIteration, OSError):
                continue
            frames.append(frame_to_rgb(frame, width, height))
    finally:
        container.close()
    return frames


def detect_video_letterbox(
    path: Path,
    info: VideoInfo,
    width: int,
    height: int,
    backend: DecodeBackend | None = None,
) -> EdgeCrop:
    """Estimate a single crop box from samples across the movie."""
    indices = letterbox_sample_indices(info.frame_count)
    samples = sample_rgb_frames(path, indices, width, height, info.rate, backend=backend)
    crops = [crop for frame in samples if (crop := detect_frame_bars(frame)) is not None]
    return combine_crops(crops, height, width)
