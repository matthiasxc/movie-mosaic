"""Durable folder-run queue so a long batch can stop and resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STATUS_PENDING = "Pending"
STATUS_RENDERING = "Rendering"
STATUS_DONE = "Done"
STATUS_FAILED = "Failed"

STATUSES = frozenset(
    {STATUS_PENDING, STATUS_RENDERING, STATUS_DONE, STATUS_FAILED}
)

_SEPARATOR = " - "


@dataclass
class QueueItem:
    filename: str
    status: str
    render_time: str = ""

    def line(self) -> str:
        return f"{self.filename}{_SEPARATOR}{self.status}{_SEPARATOR}{self.render_time}"


def queue_path(output_dir: Path, mode: str) -> Path:
    return output_dir / f"render-queue-{mode}.txt"


def parse_line(line: str) -> QueueItem | None:
    raw = line.rstrip("\r\n")
    if not raw.strip() or raw.lstrip().startswith("#"):
        return None
    # Empty time is written as a trailing " - ". If that last space was
    # stripped, rsplit would swallow the status into the filename split.
    if raw.endswith(" -"):
        raw += " "
    parts = raw.rsplit(_SEPARATOR, 2)
    if len(parts) != 3:
        raise ValueError(f"invalid queue line: {line!r}")
    filename, status, render_time = (part.strip() for part in parts)
    if not filename:
        raise ValueError(f"queue line is missing a file name: {line!r}")
    if status not in STATUSES:
        raise ValueError(f"unknown queue status {status!r} in line: {line!r}")
    return QueueItem(filename=filename, status=status, render_time=render_time)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def load_queue(path: Path) -> list[QueueItem]:
    if not path.exists():
        return []
    items: list[QueueItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = parse_line(line)
        if item is not None:
            items.append(item)
    return items


def save_queue(path: Path, items: list[QueueItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{item.line()}\n" for item in items)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sync_queue(items: list[QueueItem], movie_names: list[str]) -> list[QueueItem]:
    """Keep existing rows and append any new movie names as Pending."""
    known = {item.filename for item in items}
    extras = [
        QueueItem(name, STATUS_PENDING) for name in movie_names if name not in known
    ]
    return [*items, *extras]


def count_statuses(items: list[QueueItem]) -> dict[str, int]:
    counts = {
        STATUS_PENDING: 0,
        STATUS_RENDERING: 0,
        STATUS_DONE: 0,
        STATUS_FAILED: 0,
    }
    for item in items:
        if item.status in counts:
            counts[item.status] += 1
    return counts


def work_items(
    items: list[QueueItem],
    present_names: set[str],
    *,
    overwrite: bool,
) -> list[QueueItem]:
    """Rendering first, then Pending. With overwrite, also Failed then Done."""
    if overwrite:
        buckets = (
            STATUS_RENDERING,
            STATUS_PENDING,
            STATUS_FAILED,
            STATUS_DONE,
        )
    else:
        buckets = (STATUS_RENDERING, STATUS_PENDING)

    grouped = {status: [] for status in buckets}
    for item in items:
        if item.filename not in present_names:
            continue
        if item.status in grouped:
            grouped[item.status].append(item)
    ordered: list[QueueItem] = []
    for status in buckets:
        ordered.extend(grouped[status])
    return ordered
