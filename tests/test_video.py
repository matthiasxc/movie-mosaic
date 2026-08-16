from pathlib import Path

import pytest

from movie_mosaic.video import DecodeBackend, VideoError, resolve_backend
from tests.conftest import write_solid_mp4


def test_resolve_backend_off_is_software(tmp_path: Path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=8)
    assert resolve_backend(video, "off") == DecodeBackend("software")


def test_resolve_backend_auto_does_not_raise(tmp_path: Path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=8)
    backend = resolve_backend(video, "auto")
    assert backend.name in {"cuda", "software"}
    assert backend.describe().startswith("decode: ")


def test_resolve_backend_rejects_unknown_mode(tmp_path: Path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=8)
    with pytest.raises(ValueError, match="unknown hwaccel"):
        resolve_backend(video, "opencl")  # type: ignore[arg-type]


def test_resolve_backend_cuda_strict_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "gone.mp4"
    with pytest.raises(VideoError, match="CUDA decode required"):
        resolve_backend(missing, "cuda")


def _h264_cuda_works(path: Path) -> bool:
    try:
        return resolve_backend(path, "cuda").name == "cuda"
    except (VideoError, Exception):
        return False


def test_cuda_decode_h264_when_available(tmp_path: Path) -> None:
    try:
        video = write_solid_mp4(
            tmp_path / "clip.mp4",
            frames=12,
            codec="libx264",
            color=(30, 30, 220),
        )
    except Exception as exc:
        pytest.skip(f"cannot encode libx264: {exc}")

    if not _h264_cuda_works(video):
        pytest.skip("CUDA NVDEC is not usable on this machine")

    from movie_mosaic.config import MOSAIC_SIZE
    from movie_mosaic.mosaic import build_mosaic

    mosaic = build_mosaic(
        video,
        "color",
        detect_letterbox=False,
        progress=False,
        hwaccel="cuda",
    )
    assert mosaic.size == (MOSAIC_SIZE, MOSAIC_SIZE)
    pixel = mosaic.getpixel((0, 0))
    assert pixel[2] > 150
