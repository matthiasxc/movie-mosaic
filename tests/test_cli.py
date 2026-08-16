from pathlib import Path

from movie_mosaic.cli import build_parser, collect_movies, main, output_path_for
from movie_mosaic.config import MOSAIC_SIZE
from movie_mosaic.queue import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RENDERING,
    QueueItem,
    load_queue,
    save_queue,
)
from tests.conftest import write_solid_mp4


def test_collect_movies_file_and_flat_folder(tmp_path: Path) -> None:
    movie = write_solid_mp4(tmp_path / "A Movie (1999).mp4", frames=8)
    write_solid_mp4(tmp_path / "other.mp4", frames=8)
    m4v = write_solid_mp4(tmp_path / "Show.m4v", frames=8)
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    write_solid_mp4(nested / "hidden.mp4", frames=8)

    assert collect_movies(movie) == [movie]
    assert collect_movies(m4v) == [m4v]
    names = [path.name for path in collect_movies(tmp_path)]
    assert names == ["A Movie (1999).mp4", "other.mp4", "Show.m4v"]


def test_output_path_includes_mode(tmp_path: Path) -> None:
    movie = tmp_path / "Title (2010).mp4"
    assert output_path_for(movie, tmp_path / "out", "resize").name == "Title (2010).mosaic-resize.png"


def test_cli_writes_png_and_skips_existing(tmp_path: Path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=12, color=(20, 20, 200))
    out = tmp_path / "mosaics"

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(video), "--output", str(out)]) == 0
    png = out / "clip.mosaic-color.png"
    assert png.is_file()

    from PIL import Image

    image = Image.open(png)
    assert image.size == (MOSAIC_SIZE, MOSAIC_SIZE)

    first_mtime = png.stat().st_mtime_ns
    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(video), "--output", str(out)]) == 0
    assert png.stat().st_mtime_ns == first_mtime

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(video), "--output", str(out), "--overwrite"]) == 0
    assert png.stat().st_mtime_ns >= first_mtime


def test_cli_rejects_missing_input(tmp_path: Path) -> None:
    assert main(["--mode", "resize", "--input", str(tmp_path / "missing.mp4"), "--output", str(tmp_path)]) == 2


def test_single_file_run_does_not_create_queue(tmp_path: Path) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=8)
    out = tmp_path / "mosaics"
    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(video), "--output", str(out)]) == 0
    assert not (out / "render-queue-color.txt").exists()
    assert not (out / "render-queue-resize.txt").exists()


def test_folder_run_creates_queue_and_marks_done(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    write_solid_mp4(inbox / "one.mp4", frames=8)
    write_solid_mp4(inbox / "two.m4v", frames=8)
    out = tmp_path / "mosaics"

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out)]) == 0
    queue = out / "render-queue-color.txt"
    assert queue.is_file()
    assert not (out / "render-queue-resize.txt").exists()
    items = {item.filename: item for item in load_queue(queue)}
    assert set(items) == {"one.mp4", "two.m4v"}
    assert items["one.mp4"].status == STATUS_DONE
    assert items["two.m4v"].status == STATUS_DONE
    assert items["one.mp4"].render_time
    assert (out / "one.mosaic-color.png").is_file()
    assert (out / "two.mosaic-color.png").is_file()


def test_folder_run_skips_done_and_failed_retries_rendering(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    write_solid_mp4(inbox / "done.mp4", frames=8, color=(200, 20, 20))
    write_solid_mp4(inbox / "next.mp4", frames=8, color=(20, 200, 20))
    write_solid_mp4(inbox / "later.mp4", frames=8, color=(20, 20, 200))
    out = tmp_path / "mosaics"
    out.mkdir()
    save_queue(
        out / "render-queue-color.txt",
        [
            QueueItem("done.mp4", STATUS_DONE, "4s"),
            QueueItem("next.mp4", STATUS_RENDERING),
            QueueItem("later.mp4", STATUS_PENDING),
            QueueItem("bad.mp4", STATUS_FAILED),
        ],
    )
    # Pretend done.mp4 was already mosaiced so a resume must not rebuild it.
    (out / "done.mosaic-color.png").write_bytes(b"old")

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out)]) == 0
    assert (out / "done.mosaic-color.png").read_bytes() == b"old"
    assert (out / "next.mosaic-color.png").is_file()
    assert (out / "later.mosaic-color.png").is_file()
    items = {item.filename: item for item in load_queue(out / "render-queue-color.txt")}
    assert items["done.mp4"].status == STATUS_DONE
    assert items["next.mp4"].status == STATUS_DONE
    assert items["later.mp4"].status == STATUS_DONE
    assert items["bad.mp4"].status == STATUS_FAILED


def test_folder_run_adopts_existing_png_as_done(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    write_solid_mp4(inbox / "clip.mp4", frames=8)
    out = tmp_path / "mosaics"
    out.mkdir()
    (out / "clip.mosaic-color.png").write_bytes(b"already")

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out)]) == 0
    assert (out / "clip.mosaic-color.png").read_bytes() == b"already"
    items = load_queue(out / "render-queue-color.txt")
    assert items == [QueueItem("clip.mp4", STATUS_DONE, "")]


def test_folder_run_marks_unreadable_file_failed_and_continues(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    (inbox / "bad.mp4").write_text("not a video", encoding="utf-8")
    write_solid_mp4(inbox / "good.mp4", frames=8)
    out = tmp_path / "mosaics"

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out)]) == 1
    items = {item.filename: item for item in load_queue(out / "render-queue-color.txt")}
    assert items["bad.mp4"].status == STATUS_FAILED
    assert items["good.mp4"].status == STATUS_DONE
    assert (out / "good.mosaic-color.png").is_file()


def test_folder_overwrite_retries_done_and_failed(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    write_solid_mp4(inbox / "ok.mp4", frames=8)
    write_solid_mp4(inbox / "retry.mp4", frames=8)
    out = tmp_path / "mosaics"
    out.mkdir()
    (out / "ok.mosaic-color.png").write_bytes(b"stale")
    save_queue(
        out / "render-queue-color.txt",
        [
            QueueItem("ok.mp4", STATUS_DONE, "4s"),
            QueueItem("retry.mp4", STATUS_FAILED),
        ],
    )

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out), "--overwrite"]) == 0
    assert (out / "ok.mosaic-color.png").stat().st_size > 10
    assert (out / "retry.mosaic-color.png").is_file()
    items = {item.filename: item for item in load_queue(out / "render-queue-color.txt")}
    assert items["ok.mp4"].status == STATUS_DONE
    assert items["retry.mp4"].status == STATUS_DONE


def test_folder_appends_new_movie_to_existing_queue(tmp_path: Path) -> None:
    inbox = tmp_path / "movies"
    inbox.mkdir()
    write_solid_mp4(inbox / "old.mp4", frames=8)
    write_solid_mp4(inbox / "new.mp4", frames=8)
    out = tmp_path / "mosaics"
    out.mkdir()
    save_queue(out / "render-queue-color.txt", [QueueItem("old.mp4", STATUS_DONE, "3s")])
    (out / "old.mosaic-color.png").write_bytes(b"old")

    assert main(["--mode", "color", "--hwaccel", "off", "--input", str(inbox), "--output", str(out)]) == 0
    items = load_queue(out / "render-queue-color.txt")
    assert [item.filename for item in items] == ["old.mp4", "new.mp4"]
    assert items[0].status == STATUS_DONE
    assert items[1].status == STATUS_DONE
    assert (out / "old.mosaic-color.png").read_bytes() == b"old"
    assert (out / "new.mosaic-color.png").is_file()


def test_hwaccel_defaults_to_auto() -> None:
    args = build_parser().parse_args(
        ["--mode", "color", "--input", "in", "--output", "out"]
    )
    assert args.hwaccel == "auto"


def test_cli_hwaccel_off_prints_software_decode(tmp_path: Path, capsys) -> None:
    video = write_solid_mp4(tmp_path / "clip.mp4", frames=8)
    out = tmp_path / "mosaics"
    assert main(
        ["--mode", "color", "--hwaccel", "off", "--input", str(video), "--output", str(out)]
    ) == 0
    captured = capsys.readouterr()
    assert "decode: software" in captured.out
