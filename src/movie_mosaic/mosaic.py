"""Render 16×16 tiles and assemble the 70×70 mosaic."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from movie_mosaic.config import GRID, TILE_COUNT, TILE_SIZE
from movie_mosaic.letterbox import EdgeCrop
from movie_mosaic.sampling import sample_indices
from movie_mosaic.video import (
    DecodeBackend,
    HwaccelMode,
    VideoError,
    detect_video_letterbox,
    frame_to_rgb,
    iter_decoded_frames,
    probe,
    resolve_backend,
    scaled_size,
)

RenderMode = str


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Crop the middle square, dropping left/right (or top/bottom if portrait)."""
    height, width = frame.shape[:2]
    if width > height:
        x0 = (width - height) // 2
        return frame[:, x0 : x0 + height]
    if height > width:
        y0 = (height - width) // 2
        return frame[y0 : y0 + width, :]
    return frame


def render_resize_tile(frame: np.ndarray, crop: EdgeCrop, tile_size: int = TILE_SIZE) -> Image.Image:
    picture = center_square_crop(crop.apply(frame))
    return Image.fromarray(picture, mode="RGB").resize(
        (tile_size, tile_size),
        Image.Resampling.LANCZOS,
    )


def render_color_tile(frame: np.ndarray, crop: EdgeCrop, tile_size: int = TILE_SIZE) -> Image.Image:
    picture = crop.apply(frame)
    mean = picture.reshape(-1, picture.shape[-1])[:, :3].mean(axis=0)
    color = tuple(int(round(channel)) for channel in mean)
    return Image.new("RGB", (tile_size, tile_size), color)


_RENDERERS: dict[str, Callable[[np.ndarray, EdgeCrop, int], Image.Image]] = {
    "resize": render_resize_tile,
    "color": render_color_tile,
}


def render_tile(
    frame: np.ndarray,
    mode: RenderMode,
    crop: EdgeCrop,
    tile_size: int = TILE_SIZE,
) -> Image.Image:
    try:
        renderer = _RENDERERS[mode]
    except KeyError as exc:
        raise ValueError(f"unknown mosaic mode {mode!r}") from exc
    return renderer(frame, crop, tile_size)


def assemble_mosaic(
    tiles: list[Image.Image],
    grid: int = GRID,
    tile_size: int = TILE_SIZE,
) -> Image.Image:
    expected = grid * grid
    if len(tiles) != expected:
        raise ValueError(f"expected {expected} tiles, got {len(tiles)}")

    mosaic = Image.new("RGB", (grid * tile_size, grid * tile_size))
    for index, tile in enumerate(tiles):
        if tile.size != (tile_size, tile_size):
            tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        row, column = divmod(index, grid)
        mosaic.paste(tile.convert("RGB"), (column * tile_size, row * tile_size))
    return mosaic


def _pad_tiles(tiles: list[Image.Image], count: int = TILE_COUNT) -> list[Image.Image]:
    if not tiles:
        raise VideoError("decoded zero frames")
    if len(tiles) >= count:
        return tiles[:count]
    last = tiles[-1]
    return tiles + [last.copy() for _ in range(count - len(tiles))]


def extract_tiles(
    path: Path,
    indices: list[int],
    mode: RenderMode,
    crop: EdgeCrop,
    width: int,
    height: int,
    *,
    backend: DecodeBackend | None = None,
    progress: bool = True,
    label: str | None = None,
) -> list[Image.Image]:
    """Decode sequentially and RGB-convert only the target frames."""
    backend = backend or DecodeBackend("software")
    tiles: list[Image.Image] = []
    next_slot = 0
    target_count = len(indices)
    bar = tqdm(
        total=target_count,
        desc=label or path.name,
        unit="tile",
        disable=not progress,
    )
    try:
        for frame_index, frame in enumerate(iter_decoded_frames(path, backend)):
            rgb = None
            while next_slot < target_count and indices[next_slot] == frame_index:
                if rgb is None:
                    rgb = frame_to_rgb(frame, width, height)
                tiles.append(render_tile(rgb, mode, crop))
                next_slot += 1
                bar.update(1)
            if next_slot >= target_count:
                break
    finally:
        bar.close()
    return _pad_tiles(tiles, target_count)


def build_mosaic(
    path: Path,
    mode: RenderMode,
    *,
    detect_letterbox: bool = True,
    progress: bool = True,
    hwaccel: HwaccelMode = "off",
    backend: DecodeBackend | None = None,
) -> Image.Image:
    """Probe, crop-detect, extract tiles, and assemble one mosaic image."""
    if mode not in _RENDERERS:
        raise ValueError(f"unknown mosaic mode {mode!r}")

    backend = backend or resolve_backend(path, hwaccel)
    info = probe(path)
    width, height = scaled_size(info.width, info.height)
    crop = (
        detect_video_letterbox(path, info, width, height, backend=backend)
        if detect_letterbox
        else EdgeCrop()
    )
    indices = sample_indices(info.frame_count)
    tiles = extract_tiles(
        path,
        indices,
        mode,
        crop,
        width,
        height,
        backend=backend,
        progress=progress,
        label=path.name,
    )
    return assemble_mosaic(tiles)
