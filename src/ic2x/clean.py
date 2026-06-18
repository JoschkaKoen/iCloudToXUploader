"""Discard non-posted image records and their queue/approved files (posted
history is kept). For a full reset, stop the bot and `rm state.db`."""

from __future__ import annotations

from ic2x.config import ensure_dirs, load_config
from ic2x.db import DB
from ic2x.status import Status
from ic2x.utils import ui


def clean() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    db = DB(cfg.db_path)
    counts = db.get_cleanable_counts()
    total_rows = sum(counts.values())

    queue_files = list(cfg.queue_dir.glob("*.jpg"))
    approved_files = list(cfg.approved_dir.glob("*.jpg"))
    total_files = len(queue_files) + len(approved_files)

    if total_rows == 0 and total_files == 0:
        ui.info("Nothing to clean.")
        db.close()
        return

    ui.warn(
        f"This will delete:\n"
        f"  {len(queue_files)} file(s) from queue/\n"
        f"  {len(approved_files)} file(s) from approved/\n"
        f"  {total_rows} non-posted DB record(s) "
        f"(queued: {counts[Status.QUEUED.value]}, approved: {counts[Status.APPROVED.value]}, "
        f"seen: {counts[Status.SEEN.value]})\n"
        f"Posted images, the asset index, and posted DB records are NOT touched."
    )
    if input("Proceed? [y/N] ").strip().lower() != "y":
        ui.info("Aborted — no changes made.")
        db.close()
        return

    deleted_files = 0
    for path in queue_files + approved_files:
        try:
            path.unlink()
            deleted_files += 1
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"Could not delete {path.name}: {exc}")

    deleted_rows = db.clean_pipeline()
    db.close()
    ui.ok(f"Cleaned {deleted_files} file(s) and {deleted_rows} DB record(s).")
