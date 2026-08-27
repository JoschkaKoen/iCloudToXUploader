"""
ic2x stats — what the posts actually did on X (read-only, no API calls).

Reads the snapshots the startup reconcile stores in `tweet_metrics` (captured inside
the get_tweets call it already makes, so collecting them costs nothing). Answers the
questions the account's growth actually turns on:

  • is reach moving at all, and which posts carried it;
  • WHEN should the bot post — median impressions by Beijing hour, the readout that
    judges POST_WINDOW;
  • does the one hashtag help — tagged vs untagged medians, the A/B readout.

Medians, not means: with a handful of tweets one outlier (a 95-impression post among
a field of 5s) makes a mean say the opposite of what happened.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from ic2x.config import load_config
from ic2x.db import DB
from ic2x.utils import ui

_BEIJING = timezone(timedelta(hours=8))


def _eng(r) -> int:
    return sum(int(r[k] or 0) for k in ("likes", "retweets", "replies", "quotes", "bookmarks"))


def _median(xs: list[int]) -> float:
    return statistics.median(xs) if xs else 0.0


def stats(days: int = 0, top: int = 8) -> None:
    cfg = load_config()
    db = DB(cfg.db_path)
    try:
        rows = list(db.latest_tweet_metrics())
    finally:
        db.close()

    if not rows:
        ui.info("No tweet metrics recorded yet — they are captured by the startup "
                "reconcile, so restart the bot (or wait for the next start) and re-run.")
        return

    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = [r for r in rows if (r["posted_at"] or "") >= cutoff]
        if not rows:
            ui.info(f"No posts with metrics in the last {days} day(s).")
            return

    impr = [int(r["impressions"] or 0) for r in rows]
    engaged = sum(1 for r in rows if _eng(r) > 0)
    fetched = max((r["fetched_at"] or "") for r in rows)

    print(f"\n=== ic2x stats — {len(rows)} post(s) · snapshot {fetched} ===")
    print(f"  impressions   total {sum(impr)} · median {_median(impr):.0f} · "
          f"mean {statistics.mean(impr):.1f} · max {max(impr)}")
    print(f"  engagement    {engaged}/{len(rows)} post(s) with any like/RT/reply/quote/bookmark")

    # ── by Beijing hour: the POST_WINDOW readout ──
    by_hour: dict[int, list[int]] = {}
    for r in rows:
        try:
            h = datetime.fromisoformat(r["posted_at"]).astimezone(_BEIJING).hour
        except Exception:  # noqa: BLE001 — a malformed timestamp must not kill the report
            continue
        by_hour.setdefault(h, []).append(int(r["impressions"] or 0))
    if by_hour:
        print("\n  median impressions by POSTING HOUR (Beijing):")
        for h in sorted(by_hour):
            xs = by_hour[h]
            bar = "█" * min(28, int(_median(xs)))
            print(f"    {h:02d}:00  n={len(xs):<3} median {_median(xs):>4.0f}  {bar}")

    # ── hashtag A/B ──
    tagged = [r for r in rows if "#" in (r["caption"] or "")]
    untag = [r for r in rows if "#" not in (r["caption"] or "")]
    if tagged and untag:
        a = [int(r["impressions"] or 0) for r in tagged]
        b = [int(r["impressions"] or 0) for r in untag]
        print(f"\n  hashtag A/B   tagged n={len(a)} median {_median(a):.0f}  ·  "
              f"untagged n={len(b)} median {_median(b):.0f}")
        if len(a) < 10 or len(b) < 10:
            print("                (too few posts per arm to conclude — keep collecting)")

    order = sorted(rows, key=lambda r: -int(r["impressions"] or 0))
    print(f"\n  TOP {min(top, len(order))} by impressions:")
    for r in order[:top]:
        head = (r["caption"] or "").split("\n")[0][:58]
        print(f"    {int(r['impressions'] or 0):>5} impr  {int(r['likes'] or 0)}L  "
              f"{(r['posted_at'] or '')[:10]}  {head}")

    ui.info("read-only snapshot of state.db — no X/API calls made. Metrics refresh "
            "on each bot start (reconcile).")
