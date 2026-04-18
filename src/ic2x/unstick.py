"""Reset rows stuck in 'posting' status back to 'approved' for retry."""

from __future__ import annotations

from ic2x.config import load_config, ensure_dirs
from ic2x.db import DB
from ic2x.utils import ui


def unstick() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    db = DB(cfg.db_path)

    stuck = db.get_stuck_posting()
    if not stuck:
        ui.info("No rows stuck in 'posting' — nothing to do.")
        db.close()
        return

    ui.warn(f"Found {len(stuck)} row(s) stuck in 'posting':")
    for r in stuck:
        ui.info(f"  sha256={r['sha256']}  file={r['source_filename']}")

    ui.warn("Only proceed if you have verified these tweets were NOT posted.")
    confirm = input("Reset to 'approved' for retry? [y/N] ").strip().lower()
    if confirm != "y":
        ui.info("Aborted — no changes made.")
        db.close()
        return

    n = db.reset_stuck_posting()
    ui.ok(f"Reset {n} row(s) to 'approved'. Run `ic2x post` to retry.")
    db.close()
