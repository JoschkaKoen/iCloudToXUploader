"""
Fully automatic daemon: pull → filter → judge → (auto-approve) → post.

Runs in a continuous loop, firing the pipeline whenever a post is due
(controlled by POST_INTERVAL_HOURS). When no new images are found, the
iCloud lookback window doubles each tick up to ICLOUD_LOOKBACK_MAX.

Requires AUTO_APPROVE=true for fully unattended operation.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timedelta, timezone

from ic2x.config import Config, load_config, ensure_dirs
from ic2x.db import DB
from ic2x.utils import ui

logger = logging.getLogger("ic2x.daemon")


def daemon() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    from ic2x.run import setup_logging
    setup_logging(cfg.logs_dir)

    _stop = False

    def _handle_signal(sig, frame):  # noqa: ANN001
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ui.info(
        f"Daemon started — post interval {cfg.post_interval_hours}h, "
        f"check every {cfg.daemon_check_interval}s"
    )
    if not cfg.auto_approve:
        ui.warn(
            "AUTO_APPROVE=false: daemon will queue images but won't post automatically. "
            "Set AUTO_APPROVE=true in .env for fully unattended operation."
        )

    try:
        while not _stop:
            _tick(cfg)
            if not _stop:
                next_check = datetime.now(timezone.utc) + timedelta(seconds=cfg.daemon_check_interval)
                ui.info(
                    f"Sleeping {cfg.daemon_check_interval // 60}m — "
                    f"next check at {next_check.strftime('%H:%M')} UTC"
                )
            # Sleep in 1-second increments so SIGTERM is handled promptly
            for _ in range(cfg.daemon_check_interval):
                if _stop:
                    break
                time.sleep(1)
    finally:
        ui.info("Daemon stopped.")


def _tick(cfg: Config) -> None:
    db = DB(cfg.db_path)
    last_posted = db.get_last_posted_at()
    depth = db.get_lookback_depth() or cfg.icloud_recent_count
    approved_waiting = len(db.get_approved())
    db.close()

    now = datetime.utcnow()
    if last_posted is None:
        hours_since = float("inf")
    else:
        hours_since = (now - last_posted).total_seconds() / 3600

    post_due = hours_since >= cfg.post_interval_hours

    if not post_due:
        logger.info("daemon: next post in %.1fh", cfg.post_interval_hours - hours_since)
        return

    # Post is due
    if approved_waiting > 0:
        logger.info("daemon: %d approved image(s) ready, posting now", approved_waiting)
        from ic2x.post import post
        post()
        return

    # No ready images — pull and process with current lookback depth
    logger.info("daemon: running pipeline with --recent %d", depth)
    from ic2x.run import run
    new_pulled, queued, approved = run(recent_override=depth, auto_unstick=True, show_banner=False)

    # Images were found but all rejected (or daily AI limit hit) — don't expand lookback.
    # The AI limit resets at midnight; rejections are normal — retry same depth next tick.
    if new_pulled > 0 and queued == 0 and approved == 0:
        logger.info(
            "daemon: %d image(s) pulled but none passed filters (or AI limit reached)",
            new_pulled,
        )
        return

    if new_pulled == 0:
        new_depth = min(depth * 2, cfg.icloud_lookback_max)
        db2 = DB(cfg.db_path)
        db2.set_lookback_depth(new_depth)
        db2.close()
        if new_depth == depth:
            logger.info("daemon: lookback already at max (%d); waiting for new iCloud photos", depth)
        else:
            logger.info("daemon: nothing new at depth %d; expanding to %d", depth, new_depth)
        return

    # Found new images — reset lookback depth to default
    db2 = DB(cfg.db_path)
    db2.set_lookback_depth(cfg.icloud_recent_count)
    db2.close()

    if cfg.auto_approve and approved > 0:
        logger.info("daemon: %d auto-approved image(s), posting now", approved)
        from ic2x.post import post
        post()
    elif not cfg.auto_approve and queued > 0:
        logger.info(
            "daemon: %d image(s) queued — run `ic2x review` then `ic2x post`", queued
        )
