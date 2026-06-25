"""
ic2x cost — show the bot's tracked AI spend per day (from run_stats.cost_rmb).

Estimated from token usage × the pricing table (data/ai_pricing_per_mtoken.json),
plus any flat per-call costs the bot records via db.add_run_cost (e.g. VIAPI
enhancement, once that's wired into the posting path).
"""

from __future__ import annotations

from ic2x.config import load_config
from ic2x.db import DB
from ic2x.utils import ui
from ic2x.utils.cost_report import pricing_currency


def cost(days: int = 14) -> None:
    cfg = load_config()
    db = DB(cfg.db_path)
    try:
        rows = db.recent_stats(days)
        month_colors = db.month_color_calls()
    finally:
        db.close()

    if not rows:
        ui.info("No run stats yet — the bot hasn't recorded any cost.")
        return

    cur = pricing_currency()
    sym = "¥" if cur == "RMB" else ""
    print(f"\n=== ic2x cost — estimated AI spend ({cur}, last {len(rows)} day(s)) ===")
    print(f"  {'date':<12}{'posts':>7}{'AI calls':>10}{'cost':>13}")
    print(f"  {'-' * 11:<12}{'-' * 6:>7}{'-' * 9:>10}{'-' * 12:>13}")
    t_cost = 0.0
    t_calls = t_posts = 0
    for r in rows:
        c = r.get("cost_rmb") or 0.0
        print(f"  {r['date']:<12}{r['images_posted']:>7}{r['ai_calls']:>10}{sym + format(c, '.4f'):>13}")
        t_cost += c
        t_calls += r["ai_calls"]
        t_posts += r["images_posted"]
    print(f"  {'-' * 11:<12}{'-' * 6:>7}{'-' * 9:>10}{'-' * 12:>13}")
    print(f"  {'TOTAL':<12}{t_posts:>7}{t_calls:>10}{sym + format(t_cost, '.4f'):>13}")
    free = cfg.color_enhance_free_quota
    print(f"  color-enhance: {month_colors} call(s) this month · "
          f"{max(0, free - month_colors)}/{free} free remaining")
    ui.info(f"estimate from the pricing table; color enhance is free for the first "
            f"{free}/month, then {sym}{cfg.color_enhance_cost_rmb:.2f}/call.")
