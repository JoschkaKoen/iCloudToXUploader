"""Discard queued and approved images — files and DB records — keeping posted history."""

from __future__ import annotations

from ic2x.config import load_config, ensure_dirs
from ic2x.db import DB
from ic2x.utils import ui


def clean() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    db = DB(cfg.db_path)

    counts        = db.get_cleanable_counts()
    queued_rows   = counts["queued"]
    approved_rows = counts["approved"]
    seen_rows     = counts["seen"]
    total_rows    = queued_rows + approved_rows + seen_rows
    inbox_names   = set(db.get_cleanable_filenames())

    queue_files    = list(cfg.queue_dir.glob("*.jpg")) + list(cfg.queue_dir.glob("*.json"))
    approved_files = list(cfg.approved_dir.glob("*.jpg")) + list(cfg.approved_dir.glob("*.json"))
    inbox_files    = [p for p in cfg.inbox_dir.rglob("*") if p.is_file() and p.name in inbox_names]

    total_files = len(queue_files) + len(approved_files) + len(inbox_files)

    if total_rows == 0 and total_files == 0:
        ui.info("Nothing to clean — queue and approved directories are already empty.")
        db.close()
        return

    ui.warn(
        f"This will delete:\n"
        f"  {len(queue_files)} file(s) from queue/\n"
        f"  {len(approved_files)} file(s) from approved/\n"
        f"  {len(inbox_files)} source file(s) from inbox/ (so icloudpd re-downloads them)\n"
        f"  {total_rows} DB record(s)  "
        f"(queued: {queued_rows}, approved: {approved_rows}, seen: {seen_rows})\n"
        f"Posted images and their DB records will NOT be touched."
    )

    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        ui.info("Aborted — no changes made.")
        db.close()
        return

    deleted_files = 0
    for path in queue_files + approved_files + inbox_files:
        try:
            path.unlink()
            deleted_files += 1
        except Exception as exc:
            ui.warn(f"Could not delete {path.name}: {exc}")

    deleted_rows = db.clean_pipeline()
    db.close()

    ui.ok(
        f"Cleaned {deleted_files} file(s) and {deleted_rows} DB record(s). "
        f"Run `ic2x run` to re-download and re-evaluate from iCloud."
    )
