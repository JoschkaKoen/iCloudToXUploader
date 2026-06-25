"""
Offline tests for the bot's per-day cost tracking (run_stats.cost_rmb): the
add/read sink and the schema migration that adds the column to a pre-existing db.

Run: .venv/bin/python tests/test_cost.py   (or via pytest)
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.db import DB  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_cost_test_"))


def test_add_and_read_cost_accumulates():
    db = DB(_TMP / "a.db")
    db.add_run_cost(0.01)
    db.add_run_cost(0.025)
    db.add_run_cost(0)        # falsy → no-op
    s = db.get_today_stats()
    assert abs(s["cost_rmb"] - 0.035) < 1e-9, s
    rows = db.recent_stats(7)
    assert rows and abs(rows[0]["cost_rmb"] - 0.035) < 1e-9, rows
    db.close()


def test_color_call_monthly_counter():
    db = DB(_TMP / "color.db")
    assert db.increment_color_calls() == 1      # returns this month's running total
    assert db.increment_color_calls() == 2
    assert db.month_color_calls() == 2
    assert db.month_color_calls("2020-01") == 0  # a different month doesn't count
    db.close()


def test_migration_adds_columns_to_old_db():
    p = _TMP / "old.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE run_stats (date TEXT PRIMARY KEY, ai_calls INTEGER DEFAULT 0, "
                "images_posted INTEGER DEFAULT 0)")
    con.execute("INSERT INTO run_stats (date, ai_calls) VALUES ('2026-01-01', 5)")
    con.commit()
    con.close()

    db = DB(p)                       # __init__ runs _migrate → ADD cost_rmb + color_calls
    db.add_run_cost(0.5)             # would raise if cost_rmb were missing
    assert db.increment_color_calls() == 1   # would raise if color_calls were missing
    assert db.get_today_stats()["cost_rmb"] == 0.5
    old = db._conn.execute(
        "SELECT COALESCE(cost_rmb, 0) AS c FROM run_stats WHERE date='2026-01-01'").fetchone()
    assert old["c"] == 0             # pre-existing row backfilled to 0, preserved
    db.close()


def _main() -> int:
    failed = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_") and callable(t):
            try:
                t()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    print("OK" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
