from pathlib import Path

import pytest

from movie_mosaic.queue import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RENDERING,
    QueueItem,
    count_statuses,
    format_duration,
    load_queue,
    parse_line,
    queue_path,
    save_queue,
    sync_queue,
    work_items,
)


def test_queue_path_is_per_mode(tmp_path: Path) -> None:
    assert queue_path(tmp_path, "color").name == "render-queue-color.txt"
    assert queue_path(tmp_path, "resize").name == "render-queue-resize.txt"


def test_parse_line_keeps_dash_in_filename() -> None:
    item = parse_line("The Fast - Furious.m4v - Rendering -")
    assert item == QueueItem("The Fast - Furious.m4v", STATUS_RENDERING, "")


def test_parse_line_accepts_missing_trailing_space() -> None:
    assert parse_line("clip.mp4 - Failed -") == QueueItem("clip.mp4", STATUS_FAILED, "")


def test_parse_done_line_with_time() -> None:
    item = parse_line("Dune (2021).mp4 - Done - 12m 04s")
    assert item == QueueItem("Dune (2021).mp4", STATUS_DONE, "12m 04s")


def test_round_trip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "render-queue-color.txt"
    items = [
        QueueItem("Inception (2010).mp4", STATUS_DONE, "12m 04s"),
        QueueItem("The Fast - Furious.m4v", STATUS_RENDERING),
        QueueItem("Dune (2021).mp4", STATUS_PENDING),
        QueueItem("Broken File.mp4", STATUS_FAILED),
    ]
    save_queue(path, items)
    loaded = load_queue(path)
    assert loaded == items
    text = path.read_text(encoding="utf-8")
    assert "The Fast - Furious.m4v - Rendering - \n" in text
    assert "Dune (2021).mp4 - Pending - \n" in text


def test_sync_queue_appends_new_names_only() -> None:
    items = [QueueItem("old.mp4", STATUS_DONE, "4s")]
    synced = sync_queue(items, ["old.mp4", "new.m4v"])
    assert [item.filename for item in synced] == ["old.mp4", "new.m4v"]
    assert synced[0].status == STATUS_DONE
    assert synced[1].status == STATUS_PENDING


def test_work_items_retries_rendering_before_pending() -> None:
    items = [
        QueueItem("a.mp4", STATUS_PENDING),
        QueueItem("b.mp4", STATUS_RENDERING),
        QueueItem("c.mp4", STATUS_DONE, "1s"),
        QueueItem("d.mp4", STATUS_FAILED),
        QueueItem("gone.mp4", STATUS_PENDING),
    ]
    present = {"a.mp4", "b.mp4", "c.mp4", "d.mp4"}
    names = [item.filename for item in work_items(items, present, overwrite=False)]
    assert names == ["b.mp4", "a.mp4"]

    overwritten = [item.filename for item in work_items(items, present, overwrite=True)]
    assert overwritten == ["b.mp4", "a.mp4", "d.mp4", "c.mp4"]


def test_count_statuses() -> None:
    items = [
        QueueItem("a.mp4", STATUS_DONE, "1s"),
        QueueItem("b.mp4", STATUS_DONE, "2s"),
        QueueItem("c.mp4", STATUS_PENDING),
    ]
    assert count_statuses(items) == {
        STATUS_PENDING: 1,
        STATUS_RENDERING: 0,
        STATUS_DONE: 2,
        STATUS_FAILED: 0,
    }


def test_format_duration() -> None:
    assert format_duration(4.2) == "4s"
    assert format_duration(64) == "1m 04s"
    assert format_duration(3723) == "1h 02m 03s"


def test_parse_line_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown queue status"):
        parse_line("clip.mp4 - Working - ")
