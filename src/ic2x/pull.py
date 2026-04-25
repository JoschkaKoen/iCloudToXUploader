"""
Pull new photos from iCloud using icloudpd as a subprocess.

Key behaviour:
- Downloads the current (edited/adjusted) version by default — no --version original flag.
- Passes --skip-videos to avoid .MOV Live Photo sidecars entirely.
- Tracks the newest photo date seen across runs via the DB (run_state.last_asset_date).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pillow_heif  # noqa: F401 — registers HEIC opener with PIL on import
pillow_heif.register_heif_opener()

from ic2x.config import Config
from ic2x.db import DB

logger = logging.getLogger("ic2x.pull")


def pull(cfg: Config, db: DB, recent_override: int = 0) -> list[Path]:
    """
    Run icloudpd, scan inbox/ for new files, update last_asset_date in DB.
    Returns only files not yet processed — already-processed originals are
    left in inbox/ so icloudpd skips re-downloading them next run.

    recent_override: if > 0, use this instead of cfg.icloud_recent_count.
    When recent_override > cfg.icloud_recent_count (lookback expansion mode),
    --until-found is disabled so icloudpd walks back the full window.
    """
    inbox = cfg.inbox_dir
    inbox.mkdir(parents=True, exist_ok=True)

    _run_icloudpd(cfg, recent_override=recent_override)

    all_files = _scan_inbox(inbox)

    # Skip files already in the DB — leave them in inbox/ for icloudpd's dedup.
    all_names = {p.name for p in all_files}
    already_done = db.filenames_processed(all_names)
    new_files = [p for p in all_files if p.name not in already_done]

    if already_done:
        logger.info("pull: skipped %d already-processed file(s) in inbox/", len(already_done))

    if new_files:
        newest_mtime = max(p.stat().st_mtime for p in new_files)
        db.set_last_asset_date(datetime.fromtimestamp(newest_mtime, tz=timezone.utc))
        db.increment_images_pulled(len(new_files))
        logger.info("pull: found %d new file(s) in inbox/", len(new_files))
    else:
        logger.info("pull: no new files in inbox/")

    return new_files


def _run_icloudpd(cfg: Config, recent_override: int = 0) -> None:
    recent = recent_override if recent_override > 0 else cfg.icloud_recent_count
    # Disable --until-found when expanding the lookback window: icloudpd would otherwise
    # stop after the first N consecutive already-downloaded photos, defeating the expansion.
    disable_until_found = recent_override > cfg.icloud_recent_count
    cmd = [
        sys.executable, "-m", "icloudpd",
        "--username", cfg.icloud_username,
        "--password-provider", "parameter",
        "-p", cfg.icloud_password,
        "--mfa-provider", "console",
        "--directory", str(cfg.inbox_dir),
        "--cookie-directory", str(cfg.icloud_cookie_dir),
        "--recent", str(recent),
        *(
            []
            if disable_until_found
            else (["--until-found", str(cfg.icloud_until_found)] if cfg.icloud_until_found > 0 else [])
        ),
        "--skip-videos",
        "--no-progress-bar",
        "--auto-delete",
    ]

    _PROXY_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"}
    env = {k: v for k, v in os.environ.items() if k not in _PROXY_VARS}

    logger.info(
        "pull: running icloudpd --username %s --recent %d%s",
        cfg.icloud_username,
        recent,
        " (--until-found disabled)" if disable_until_found else "",
    )
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode not in (0, 2):
        # Exit 2 = "nothing new to download" — that's fine.
        # Anything else (auth expired, MFA needed, rate limit) means the
        # inbox may be empty for the wrong reason; tell the user where to look.
        logger.warning(
            "icloudpd exited with code %d — re-run interactively to refresh "
            "MFA or check %s for stale cookies",
            result.returncode, cfg.icloud_cookie_dir,
        )


def _scan_inbox(inbox: Path) -> list[Path]:
    """Return all image files in inbox/ (recursively), sorted by mtime ascending."""
    extensions = {".heic", ".jpg", ".jpeg", ".png"}
    files = [
        p for p in inbox.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)
