"""Command-line interface for movie-mosaic."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from movie_mosaic.mosaic import build_mosaic
from movie_mosaic.queue import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RENDERING,
    count_statuses,
    format_duration,
    load_queue,
    queue_path,
    save_queue,
    sync_queue,
    work_items,
)
from movie_mosaic.video import HWACCEL_MODES, HwaccelMode, VideoError, resolve_backend

_MODES = ("resize", "color")
_MOVIE_SUFFIXES = {".mp4", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-mosaic",
        description=(
            "Sample 4,900 frames from an MP4/M4V and assemble a 1120×1120 mosaic. "
            "Point --input at one .mp4/.m4v file or a flat folder of those files."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        required=True,
        help="resize: center-crop each frame to 16×16. color: solid average-color tiles.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A single .mp4/.m4v file, or a folder containing those files (not recursive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where mosaic PNGs are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace mosaics that already exist. In folder mode this also retries "
            "queue rows marked Done or Failed."
        ),
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Do not detect or remove letterbox / pillarbox bars.",
    )
    parser.add_argument(
        "--hwaccel",
        choices=HWACCEL_MODES,
        default="auto",
        help=(
            "Video decoder. auto (default) tries NVIDIA NVDEC and falls back to "
            "software. cuda requires the GPU. off forces software."
        ),
    )
    return parser


def collect_movies(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in _MOVIE_SUFFIXES:
            raise ValueError(f"input file must be an .mp4 or .m4v, got {input_path.name}")
        return [input_path]

    movies = sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in _MOVIE_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )
    if not movies:
        raise ValueError(f"no .mp4 or .m4v files in {input_path}")
    return movies


def output_path_for(movie: Path, output_dir: Path, mode: str) -> Path:
    return output_dir / f"{movie.stem}.mosaic-{mode}.png"


def _render_one(
    movie: Path,
    destination: Path,
    mode: str,
    *,
    detect_letterbox: bool,
    hwaccel: HwaccelMode,
) -> None:
    backend = resolve_backend(movie, hwaccel)
    print(backend.describe())
    mosaic = build_mosaic(
        movie,
        mode,
        detect_letterbox=detect_letterbox,
        progress=True,
        backend=backend,
    )
    mosaic.save(destination, format="PNG")


def _run_single_file(
    movie: Path,
    output_dir: Path,
    mode: str,
    *,
    overwrite: bool,
    detect_letterbox: bool,
    hwaccel: HwaccelMode,
) -> int:
    destination = output_path_for(movie, output_dir, mode)
    if destination.exists() and not overwrite:
        print(f"[1/1] skip {movie.name} (exists: {destination.name})")
        print("1 skipped")
        return 0

    print(f"[1/1] {mode} {movie.name} -> {destination.name}")
    try:
        _render_one(
            movie,
            destination,
            mode,
            detect_letterbox=detect_letterbox,
            hwaccel=hwaccel,
        )
    except (VideoError, ValueError, OSError) as exc:
        print(f"[1/1] failed {movie.name}: {exc}", file=sys.stderr)
        print("0 ok, 0 skipped, 1 failed")
        return 1
    print("1 ok, 0 skipped, 0 failed")
    return 0


def _run_folder(
    movies: list[Path],
    output_dir: Path,
    mode: str,
    *,
    overwrite: bool,
    detect_letterbox: bool,
    hwaccel: HwaccelMode,
) -> int:
    queue_file = queue_path(output_dir, mode)
    try:
        items = sync_queue(load_queue(queue_file), [movie.name for movie in movies])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    save_queue(queue_file, items)
    counts = count_statuses(items)
    print(
        f"queue: {counts[STATUS_DONE]} done, {counts[STATUS_RENDERING]} rendering, "
        f"{counts[STATUS_PENDING]} pending, {counts[STATUS_FAILED]} failed"
    )

    present = {movie.name: movie for movie in movies}
    todo = work_items(items, set(present), overwrite=overwrite)

    ok = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(todo, start=1):
        movie = present[item.filename]
        destination = output_path_for(movie, output_dir, mode)
        prefix = f"[{index}/{len(todo)}]"

        if item.status == STATUS_PENDING and destination.exists() and not overwrite:
            item.status = STATUS_DONE
            item.render_time = ""
            save_queue(queue_file, items)
            print(f"{prefix} skip {movie.name} (exists: {destination.name})")
            skipped += 1
            continue

        item.status = STATUS_RENDERING
        item.render_time = ""
        save_queue(queue_file, items)

        print(f"{prefix} {mode} {movie.name} -> {destination.name}")
        started = time.perf_counter()
        try:
            _render_one(
                movie,
                destination,
                mode,
                detect_letterbox=detect_letterbox,
                hwaccel=hwaccel,
            )
        except (VideoError, ValueError, OSError) as exc:
            item.status = STATUS_FAILED
            item.render_time = ""
            save_queue(queue_file, items)
            print(f"{prefix} failed {movie.name}: {exc}", file=sys.stderr)
            failed += 1
            continue

        item.status = STATUS_DONE
        item.render_time = format_duration(time.perf_counter() - started)
        save_queue(queue_file, items)
        ok += 1

    print(f"{ok} ok, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        movies = collect_movies(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input.is_file():
        return _run_single_file(
            movies[0],
            args.output,
            args.mode,
            overwrite=args.overwrite,
            detect_letterbox=not args.no_letterbox,
            hwaccel=args.hwaccel,
        )

    return _run_folder(
        movies,
        args.output,
        args.mode,
        overwrite=args.overwrite,
        detect_letterbox=not args.no_letterbox,
        hwaccel=args.hwaccel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
