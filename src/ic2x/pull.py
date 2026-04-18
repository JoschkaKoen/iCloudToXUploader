"""
Pull new photos from iCloud using icloudpd as a subprocess.

Key behaviour:
- Downloads the current (edited/adjusted) version by default — no --version original flag.
- Passes --skip-videos to avoid .MOV Live Photo sidecars entirely.
- Tracks the newest photo date seen across runs via the DB (run_state.last_asset_date).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pillow_heif  # noqa: F401 — registers HEIC opener with PIL on import
pillow_heif.register_heif_opener()

from ic2x.config import Config
from ic2x.db import DB

logger = logging.getLogger("ic2x.pull")


def pull(cfg: Config, db: DB) -> list[Path]:
    """
    Run icloudpd, scan inbox/ for new files, update last_asset_date in DB.
    Returns list of new file paths found in inbox/ after the run.
    """
    inbox = cfg.inbox_dir
    inbox.mkdir(parents=True, exist_ok=True)

    _run_icloudpd(cfg)

    new_files = _scan_inbox(inbox)
    if new_files:
        newest_mtime = max(p.stat().st_mtime for p in new_files)
        db.set_last_asset_date(datetime.utcfromtimestamp(newest_mtime))
        db.increment_images_pulled(len(new_files))
        logger.info("pull: found %d new file(s) in inbox/", len(new_files))
    else:
        logger.info("pull: no new files in inbox/")

    return new_files


def _run_icloudpd(cfg: Config) -> None:
    cmd = [
        sys.executable, "-m", "icloudpd",
        "--username", cfg.icloud_username,
        "--password-provider", "parameter",
        "-p", cfg.icloud_password,
        "--mfa-provider", "console",
        "--directory", str(cfg.inbox_dir),
        "--cookie-directory", str(cfg.icloud_cookie_dir),
        "--recent", str(cfg.icloud_recent_count),
        "--skip-videos",
        "--no-progress-bar",
        "--auto-delete",
    ]

    logger.info("pull: running icloudpd %s", " ".join(cmd[3:]))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode not in (0, 2):
        # Exit 2 = "nothing new to download" — that's fine
        logger.warning("icloudpd exited with code %d", result.returncode)


def _scan_inbox(inbox: Path) -> list[Path]:
    """Return all image files in inbox/ (recursively), sorted by mtime ascending."""
    extensions = {".heic", ".jpg", ".jpeg", ".png"}
    files = [
        p for p in inbox.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)
