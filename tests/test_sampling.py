import pytest

from movie_mosaic.config import TILE_COUNT
from movie_mosaic.sampling import sample_indices


def test_sample_indices_include_first_and_last() -> None:
    indices = sample_indices(100_000)
    assert len(indices) == TILE_COUNT
    assert indices[0] == 0
    assert indices[-1] == 99_999
    assert indices == sorted(indices)


def test_sample_indices_are_unique_when_film_is_long() -> None:
    indices = sample_indices(20_000)
    assert len(set(indices)) == TILE_COUNT


def test_short_video_reuses_frames() -> None:
    indices = sample_indices(30)
    assert len(indices) == TILE_COUNT
    assert indices[0] == 0
    assert indices[-1] == 29
    assert len(set(indices)) == 30


def test_single_frame_video() -> None:
    assert sample_indices(1) == [0] * TILE_COUNT


def test_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError):
        sample_indices(0)
    with pytest.raises(ValueError):
        sample_indices(10, sample_count=0)
