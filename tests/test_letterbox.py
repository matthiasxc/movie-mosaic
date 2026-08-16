import numpy as np

from movie_mosaic.letterbox import (
    EdgeCrop,
    combine_crops,
    detect_frame_bars,
    letterbox_sample_indices,
)


def _frame(height: int, width: int, fill: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = fill
    return image


def test_detects_letterbox_bars() -> None:
    frame = _frame(200, 320, (180, 40, 40))
    frame[:40, :] = 0
    frame[-40:, :] = 0
    crop = detect_frame_bars(frame)
    assert crop is not None
    assert crop.top == 40
    assert crop.bottom == 40
    assert crop.left == 0
    assert crop.right == 0


def test_detects_pillarbox_bars() -> None:
    frame = _frame(180, 320, (40, 180, 40))
    frame[:, :50] = 0
    frame[:, -50:] = 0
    crop = detect_frame_bars(frame)
    assert crop is not None
    assert crop.left == 50
    assert crop.right == 50
    assert crop.top == 0
    assert crop.bottom == 0


def test_no_bars_on_full_frame() -> None:
    frame = _frame(180, 320, (200, 200, 40))
    assert detect_frame_bars(frame) == EdgeCrop()


def test_all_black_frame_is_ignored() -> None:
    assert detect_frame_bars(_frame(180, 320, (0, 0, 0))) is None


def test_apply_crop_keeps_picture() -> None:
    frame = _frame(100, 200, (10, 20, 30))
    frame[10:90, 20:180] = (255, 128, 0)
    cropped = EdgeCrop(top=10, bottom=10, left=20, right=20).apply(frame)
    assert cropped.shape == (80, 160, 3)
    assert (cropped == (255, 128, 0)).all()


def test_combine_crops_uses_median_and_drops_thin_bars() -> None:
    crops = [
        EdgeCrop(top=40, bottom=40),
        EdgeCrop(top=42, bottom=38),
        EdgeCrop(top=41, bottom=39),
        EdgeCrop(top=40, bottom=40),
        EdgeCrop(top=39, bottom=41),
        EdgeCrop(left=1, right=1),
    ]
    combined = combine_crops(crops, height=200, width=320)
    assert combined.top == 40
    assert combined.bottom == 40
    assert combined.left == 0
    assert combined.right == 0


def test_combine_crops_needs_enough_samples() -> None:
    assert combine_crops([EdgeCrop(top=40, bottom=40)], 200, 320) == EdgeCrop()


def test_letterbox_sample_indices_skip_edges() -> None:
    indices = letterbox_sample_indices(1000, samples=10, edge_skip=0.05)
    assert indices[0] >= 50
    assert indices[-1] <= 949
    assert indices == sorted(indices)
    assert len(indices) == 10
