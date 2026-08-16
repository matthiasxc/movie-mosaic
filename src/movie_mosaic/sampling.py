"""Evenly spaced frame-index selection."""

from movie_mosaic.config import TILE_COUNT


def sample_indices(frame_count: int, sample_count: int = TILE_COUNT) -> list[int]:
    """Return ``sample_count`` frame indices spaced through ``[0, frame_count)``.

    Uses ``round(i * (N - 1) / (sample_count - 1))`` so the first and last
    frames are always included. Short videos reuse frames when ``N`` is
    smaller than ``sample_count``.
    """
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if sample_count == 1 or frame_count == 1:
        return [0] * sample_count
    last = frame_count - 1
    denom = sample_count - 1
    return [round(i * last / denom) for i in range(sample_count)]
