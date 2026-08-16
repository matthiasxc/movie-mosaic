from __future__ import annotations

from pathlib import Path

import numpy as np


def write_solid_mp4(
    path: Path,
    *,
    frames: int = 30,
    width: int = 80,
    height: int = 60,
    fps: int = 10,
    color: tuple[int, int, int] = (220, 30, 30),
    bar: int = 0,
    codec: str = "mpeg4",
) -> Path:
    """Write a tiny video of solid (optionally letterboxed) frames."""
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    if codec == "libx264":
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

    image = np.zeros((height, width, 3), dtype=np.uint8)
    if bar > 0:
        image[bar : height - bar, :] = color
    else:
        image[:, :] = color

    try:
        for _ in range(frames):
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path
