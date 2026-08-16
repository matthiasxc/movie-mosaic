"""Detect constant letterbox / pillarbox bars from sample frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from movie_mosaic.config import (
    LETTERBOX_ALL_BLACK_MEAN,
    LETTERBOX_EDGE_SKIP,
    LETTERBOX_LUMINANCE_THRESHOLD,
    LETTERBOX_MIN_BAR_FRACTION,
    LETTERBOX_MIN_SAMPLES,
    LETTERBOX_SAMPLE_COUNT,
)


@dataclass(frozen=True)
class EdgeCrop:
    """Pixels to drop from each edge of a frame."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        y0 = min(max(self.top, 0), height - 1)
        x0 = min(max(self.left, 0), width - 1)
        y1 = max(y0 + 1, height - max(self.bottom, 0))
        x1 = max(x0 + 1, width - max(self.right, 0))
        return frame[y0:y1, x0:x1]


def _luminance(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32, copy=False)
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _edge_run(means: np.ndarray, threshold: float) -> int:
    """Count how far a near-black run extends from the start of ``means``."""
    limit = len(means) // 2
    count = 0
    while count < limit and means[count] < threshold:
        count += 1
    return count


def detect_frame_bars(
    frame: np.ndarray,
    luminance_threshold: float = LETTERBOX_LUMINANCE_THRESHOLD,
    all_black_mean: float = LETTERBOX_ALL_BLACK_MEAN,
) -> EdgeCrop | None:
    """Return bars for one RGB frame, or ``None`` if the frame is too dark."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame must be an HxWx3 RGB array")

    gray = _luminance(frame)
    if float(gray.mean()) < all_black_mean:
        return None

    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    return EdgeCrop(
        top=_edge_run(row_means, luminance_threshold),
        bottom=_edge_run(row_means[::-1], luminance_threshold),
        left=_edge_run(col_means, luminance_threshold),
        right=_edge_run(col_means[::-1], luminance_threshold),
    )


def combine_crops(
    crops: list[EdgeCrop],
    height: int,
    width: int,
    min_samples: int = LETTERBOX_MIN_SAMPLES,
    min_bar_fraction: float = LETTERBOX_MIN_BAR_FRACTION,
) -> EdgeCrop:
    """Median-combine per-frame crops and drop bars thinner than ``min_bar_fraction``."""
    if len(crops) < min_samples:
        return EdgeCrop()

    def _median(values: list[int]) -> int:
        return int(round(float(np.median(np.asarray(values, dtype=np.float64)))))

    crop = EdgeCrop(
        top=_median([c.top for c in crops]),
        bottom=_median([c.bottom for c in crops]),
        left=_median([c.left for c in crops]),
        right=_median([c.right for c in crops]),
    )
    min_h = max(1, int(round(height * min_bar_fraction)))
    min_w = max(1, int(round(width * min_bar_fraction)))
    return EdgeCrop(
        top=crop.top if crop.top >= min_h else 0,
        bottom=crop.bottom if crop.bottom >= min_h else 0,
        left=crop.left if crop.left >= min_w else 0,
        right=crop.right if crop.right >= min_w else 0,
    )


def letterbox_sample_indices(
    frame_count: int,
    samples: int = LETTERBOX_SAMPLE_COUNT,
    edge_skip: float = LETTERBOX_EDGE_SKIP,
) -> list[int]:
    """Evenly spaced indices in the middle ``1 - 2*edge_skip`` of the movie."""
    if frame_count <= 0:
        return []
    start = int(frame_count * edge_skip)
    end = int(frame_count * (1.0 - edge_skip)) - 1
    if end < start:
        start, end = 0, frame_count - 1
    span = end - start
    count = min(samples, span + 1)
    if count <= 1:
        return [start]
    return [start + round(i * span / (count - 1)) for i in range(count)]
