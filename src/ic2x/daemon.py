"""
Fully automatic daemon: pull → filter → judge → (auto-approve) → post.

Runs in a continuous loop, firing the pipeline whenever a post is due
(controlled by POST_INTERVAL_HOURS). When no new images are found the daemon
searches backwards through iCloud history one image at a time, downloading
only the next older unprocessed photo on each step.

Requires AUTO_APPROVE=true for fully unattended operation.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

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
            _tick(cfg, lambda: _stop)
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


def _tick(cfg: Config, should_stop: Callable[[], bool]) -> None:
    # ── Read state ────────────────────────────────────────────────────────────
    db = DB(cfg.db_path)

    # Auto-unstick once per tick (daemon is unattended — can't ask for confirmation)
    stuck = db.get_stuck_posting()
    if stuck:
        logger.warning("daemon: auto-resetting %d stuck 'posting' row(s)", len(stuck))
        db.reset_stuck_posting()

    last_posted      = db.get_last_posted_at()
    saved_depth      = db.get_lookback_depth() or cfg.icloud_recent_count
    approved_waiting = len(db.get_approved())
    db.close()

    # ── Is a post due? ────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    hours_since = (now - last_posted).total_seconds() / 3600 if last_posted else float("inf")
    post_due = hours_since >= cfg.post_interval_hours

    if not post_due:
        logger.info("daemon: next post in %.1fh", cfg.post_interval_hours - hours_since)
        return

    # ── Approved images already waiting → post immediately ───────────────────
    if approved_waiting > 0:
        logger.info("daemon: %d approved image(s) ready, posting now", approved_waiting)
        from ic2x.post import post
        post()
        return

    # ── Search loop: one image at a time ─────────────────────────────────────
    # Every tick starts with a fresh check of the latest ICLOUD_RECENT_COUNT
    # images (--until-found enabled, normal mode) to catch newly uploaded photos.
    # After the initial batch, the loop jumps to saved_depth and continues
    # one image at a time through older history (--until-found disabled).
    from ic2x.run import run

    depth = cfg.icloud_recent_count  # always start fresh each tick

    while not should_stop() and depth <= cfg.icloud_lookback_max:
        # Check AI budget before each pipeline run (avoids a wasted icloudpd call)
        db_chk = DB(cfg.db_path)
        limit_hit = db_chk.check_daily_ai_limit(cfg.daily_ai_calls)
        db_chk.close()
        if limit_hit:
            logger.info("daemon: daily AI limit reached — will retry tomorrow")
            break

        if depth == cfg.icloud_recent_count:
            logger.info("daemon: checking latest %d images", depth)
        else:
            logger.info("daemon: searching at lookback depth %d / %d", depth, cfg.icloud_lookback_max)

        new_pulled, queued, approved = run(
            recent_override=depth,
            auto_unstick=False,  # already handled above
            show_banner=False,
        )

        if approved > 0:
            logger.info("daemon: postable image found at depth %d, posting now", depth)
            from ic2x.post import post
            post()
            # Reset so the next post cycle restarts from the newest photos
            db_rst = DB(cfg.db_path)
            db_rst.set_lookback_depth(cfg.icloud_recent_count)
            db_rst.close()
            return

        if queued > 0 and not cfg.auto_approve:
            # Manual mode: something is interesting but needs human review
            db_sv = DB(cfg.db_path)
            db_sv.set_lookback_depth(depth)
            db_sv.close()
            logger.info(
                "daemon: %d image(s) queued for manual review — "
                "run `ic2x review` then `ic2x post`",
                queued,
            )
            return

        # Nothing postable at this depth — step one image deeper.
        db_sv = DB(cfg.db_path)
        if depth >= cfg.icloud_lookback_max:
            # Exhausted the full lookback — reset so next tick starts fresh
            # rather than re-entering expansion mode immediately.
            db_sv.set_lookback_depth(cfg.icloud_recent_count)
            db_sv.close()
            break
        # After the initial batch, jump directly to the saved search position
        # to skip re-processing the already-examined history range.
        if depth == cfg.icloud_recent_count and saved_depth > cfg.icloud_recent_count:
            depth = saved_depth
        else:
            depth += 1
        db_sv.set_lookback_depth(depth)
        db_sv.close()

    # ── Post-loop: log why we stopped ────────────────────────────────────────
    if should_stop():
        logger.info("daemon: search interrupted by stop signal at depth %d", depth)
    elif depth >= cfg.icloud_lookback_max:
        logger.info(
            "daemon: exhausted %d-image lookback — will check for new photos next tick",
            cfg.icloud_lookback_max,
        )
