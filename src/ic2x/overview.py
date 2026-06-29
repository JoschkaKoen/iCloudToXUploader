"""
ic2x status — a quick, offline operational snapshot of the bot, read straight from
state.db. No iCloud / X / AI calls. Shows the dry-run/live mode + judge model, the
last post time and next-post countdown, the rolling-24h post count vs the cap,
lifetime posted/pending/rejected/seen counts, and today's AI calls + spend. Handy
for checking a long-running server bot's health at a glance.

`--json` prints the same snapshot as one JSON object — for a monitoring/healthcheck
cron to parse.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ic2x.config import load_config
from ic2x.db import DB
from ic2x.utils import ui
from ic2x.utils.cost_report import pricing_currency


def _next_post(last: datetime | None, cfg, posts_24h: int, now: datetime) -> str:
    if posts_24h >= cfg.max_posts_per_day:
        return f"daily cap reached ({posts_24h}/{cfg.max_posts_per_day}) — waiting for a 24h slot"
    if last is None:
        return "due now (never posted yet)"
    nxt = last + timedelta(hours=cfg.post_interval_hours)
    if nxt <= now:
        return "due now"
    secs = (nxt - now).total_seconds()
    return f"in ~{int(secs // 3600)}h {int(secs % 3600 // 60):02d}m"


def _seconds_until_next(last: datetime | None, cfg, posts_24h: int, now: datetime) -> int:
    """Seconds until the next post is allowed (0 = due now); for the JSON snapshot."""
    if last is None:
        return 0
    return max(0, int((last + timedelta(hours=cfg.post_interval_hours) - now).total_seconds()))


def snapshot(cfg, db: DB) -> dict:
    """Build the status snapshot dict from the DB (pure read; no side effects)."""
    now = datetime.now(timezone.utc)
    ov = db.overview()
    today = db.get_today_stats()
    posts_24h = db.count_posts_rolling_24h()
    last = db.get_last_posted_at()
    return {
        "mode": "dry_run" if cfg.x_dry_run else "live",
        "judge_model": cfg.judge_model,
        "last_posted_at": last.isoformat() if last else None,
        "next_post_human": _next_post(last, cfg, posts_24h, now),
        "seconds_until_next_post": _seconds_until_next(last, cfg, posts_24h, now),
        "posts_24h": posts_24h,
        "max_posts_per_day": cfg.max_posts_per_day,
        "posted_total": ov["posted"],
        "approved_pending": ov["approved_pending"],
        "rejected_total": ov["rejected"],
        "assets_seen": ov["seen"],
        "assets_indexed": ov["indexed"],
        "today_ai_calls": today["ai_calls"],
        "today_cost": round(today["cost_rmb"], 4),
        "currency": pricing_currency(),
        "color_calls_this_month": db.month_color_calls(),
        "color_free_quota": cfg.color_enhance_free_quota,
    }


def status(as_json: bool = False) -> None:
    cfg = load_config()
    db = DB(cfg.db_path)
    try:
        snap = snapshot(cfg, db)
    finally:
        db.close()

    if as_json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return

    sym = "¥" if snap["currency"] == "RMB" else ""
    mode = "DRY-RUN (no real tweets)" if snap["mode"] == "dry_run" else "LIVE (posting to X)"
    last_str = (datetime.fromisoformat(snap["last_posted_at"]).strftime("%Y-%m-%d %H:%M UTC")
                if snap["last_posted_at"] else "—")
    free, used = snap["color_free_quota"], snap["color_calls_this_month"]

    print("\n=== ic2x status ===")
    print(f"  mode            : {mode}")
    print(f"  judge model     : {snap['judge_model']}")
    print(f"  last post       : {last_str}")
    print(f"  next post       : {snap['next_post_human']}")
    print(f"  posts (24h)     : {snap['posts_24h']}/{snap['max_posts_per_day']}")
    print(f"  posted (all)    : {snap['posted_total']}")
    print(f"  pending approved: {snap['approved_pending']}")
    print(f"  rejected (all)  : {snap['rejected_total']}")
    print(f"  decided / index : {snap['assets_seen']} / {snap['assets_indexed']} iCloud assets")
    print(f"  today           : {snap['today_ai_calls']} AI calls · {sym}{snap['today_cost']:.4f}")
    print(f"  color-enhance   : {used} this month ({max(0, free - used)}/{free} free remaining)")
    ui.info("read-only snapshot of state.db — no iCloud / X / AI calls made.")
