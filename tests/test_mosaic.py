import numpy as np
from PIL import Image

from movie_mosaic.config import GRID, MOSAIC_SIZE, TILE_COUNT, TILE_SIZE
from movie_mosaic.letterbox import EdgeCrop
from movie_mosaic.mosaic import (
    assemble_mosaic,
    build_mosaic,
    center_square_crop,
    render_color_tile,
    render_resize_tile,
)
from tests.conftest import write_solid_mp4


def test_center_square_crop_drops_left_and_right() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, 20:60] = (0, 255, 0)
    cropped = center_square_crop(frame)
    assert cropped.shape == (40, 40, 3)
    assert (cropped == (0, 255, 0)).all()


def test_center_square_crop_drops_top_and_bottom_when_portrait() -> None:
    frame = np.zeros((80, 40, 3), dtype=np.uint8)
    frame[20:60, :] = (0, 0, 255)
    cropped = center_square_crop(frame)
    assert cropped.shape == (40, 40, 3)
    assert (cropped == (0, 0, 255)).all()


def test_color_tile_is_solid_average() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:10, :] = (100, 0, 0)
    frame[10:, :] = (0, 0, 50)
    tile = render_color_tile(frame, EdgeCrop())
    assert tile.size == (TILE_SIZE, TILE_SIZE)
    assert tile.getpixel((0, 0)) == (50, 0, 25)
    assert tile.getpixel((TILE_SIZE - 1, TILE_SIZE - 1)) == (50, 0, 25)


def test_color_tile_ignores_letterbox_pixels() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[8:32, :] = (200, 10, 10)
    tile = render_color_tile(frame, EdgeCrop(top=8, bottom=8))
    assert tile.getpixel((0, 0)) == (200, 10, 10)


def test_resize_tile_from_solid_wide_frame() -> None:
    frame = np.full((40, 80, 3), (30, 180, 30), dtype=np.uint8)
    tile = render_resize_tile(frame, EdgeCrop())
    assert tile.size == (TILE_SIZE, TILE_SIZE)
    assert tile.getpixel((8, 8)) == (30, 180, 30)


def test_assemble_mosaic_dimensions() -> None:
    red = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 0, 0))
    tiles = [red] * TILE_COUNT
    mosaic = assemble_mosaic(tiles)
    assert mosaic.size == (MOSAIC_SIZE, MOSAIC_SIZE)
    assert mosaic.getpixel((0, 0)) == (255, 0, 0)
    assert mosaic.getpixel((MOSAIC_SIZE - 1, MOSAIC_SIZE - 1)) == (255, 0, 0)
    assert mosaic.getpixel((TILE_SIZE, 0)) == (255, 0, 0)


def test_build_mosaic_from_generated_mp4(tmp_path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=24, color=(220, 30, 30))
    mosaic = build_mosaic(video, "color", detect_letterbox=False, progress=False)
    assert mosaic.size == (MOSAIC_SIZE, MOSAIC_SIZE)
    # mpeg4 is lossy; stay in the red family rather than requiring an exact RGB.
    pixel = mosaic.getpixel((GRID // 2 * TILE_SIZE, GRID // 2 * TILE_SIZE))
    assert pixel[0] > 150
    assert pixel[1] < 80
    assert pixel[2] < 80


def test_short_clip_fills_mosaic_with_matching_end_tiles(tmp_path) -> None:
    video = write_solid_mp4(tmp_path / "short.mp4", frames=30, color=(220, 30, 30))
    mosaic = build_mosaic(
        video,
        "color",
        detect_letterbox=False,
        progress=False,
        hwaccel="off",
    )
    assert mosaic.size == (MOSAIC_SIZE, MOSAIC_SIZE)
    first = mosaic.getpixel((0, 0))
    last = mosaic.getpixel((MOSAIC_SIZE - 1, MOSAIC_SIZE - 1))
    assert first == last
    assert first[0] > 150
    assert first[1] < 80
    assert first[2] < 80


def test_build_mosaic_color_ignores_letterbox(tmp_path) -> None:
    video = write_solid_mp4(
        tmp_path / "boxed.mp4",
        frames=24,
        width=80,
        height=60,
        color=(220, 30, 30),
        bar=10,
    )
    mosaic = build_mosaic(video, "color", detect_letterbox=True, progress=False)
    pixel = mosaic.getpixel((0, 0))
    assert pixel[0] > 150
    assert pixel[1] < 80
    assert pixel[2] < 80
