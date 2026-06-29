"""
Offline tests for the `ic2x status` snapshot: DB.overview() counts and the
next-post countdown logic. No iCloud / X / AI.

Run: .venv/bin/python tests/test_status.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.db import DB  # noqa: E402
from ic2x.overview import _next_post, snapshot  # noqa: E402
from ic2x.status import Status  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_status_"))
_SEQ = 0


def _db() -> DB:
    global _SEQ
    _SEQ += 1
    return DB(_TMP / f"s{_SEQ}.db")


def test_overview_counts():
    db = _db()
    db.commit_burst(["a", "b", "c"], None)                       # 3 seen, no images
    db.commit_burst(["w"], {"sha256": "s1", "phash": "p1", "status": Status.POSTED})
    db.commit_burst(["x"], {"sha256": "s2", "phash": "p2", "status": Status.APPROVED})
    db.commit_burst(["y"], {"sha256": "s3", "phash": "p3", "status": Status.REJECTED,
                            "reject_stage": "screenshot"})
    ov = db.overview()
    db.close()
    assert ov["posted"] == 1
    assert ov["approved_pending"] == 1
    assert ov["rejected"] == 1
    assert ov["indexed"] == 6 and ov["seen"] == 6  # a,b,c,w,x,y


def test_overview_empty_db_is_zeros():
    ov = _db().overview()
    assert ov == {"posted": 0, "approved_pending": 0, "rejected": 0, "indexed": 0, "seen": 0}


def _cfg(interval=5, cap=6):
    return SimpleNamespace(post_interval_hours=interval, max_posts_per_day=cap)


def test_next_post_never_posted():
    now = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
    assert "due now" in _next_post(None, _cfg(), 0, now)


def test_next_post_countdown():
    now = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=2)            # 5h interval → 3h left
    msg = _next_post(last, _cfg(interval=5), 1, now)
    assert "~2h" in msg or "~3h" in msg        # ~3h00m → "~3h 00m"
    assert "3h 00m" in msg


def test_next_post_due_when_interval_elapsed():
    now = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=6)            # past the 5h interval
    assert _next_post(last, _cfg(interval=5), 1, now) == "due now"


def test_next_post_daily_cap_blocks():
    now = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
    msg = _next_post(now - timedelta(hours=10), _cfg(cap=6), 6, now)
    assert "cap reached" in msg


def test_snapshot_is_json_serializable_with_expected_fields():
    import json
    db = _db()
    db.commit_burst(["w"], {"sha256": "s1", "phash": "p1", "status": Status.POSTED})
    cfg = SimpleNamespace(x_dry_run=True, judge_model="qwen3.7-plus, 1000, 2000",
                          max_posts_per_day=6, post_interval_hours=5,
                          color_enhance_free_quota=100)
    snap = snapshot(cfg, db)
    db.close()
    assert snap["mode"] == "dry_run"
    assert snap["posted_total"] == 1
    assert snap["max_posts_per_day"] == 6
    expected = {"mode", "judge_model", "last_posted_at", "next_post_human",
                "seconds_until_next_post", "posts_24h", "posted_total", "today_cost",
                "currency", "assets_seen", "assets_indexed"}
    assert expected <= set(snap)
    json.dumps(snap)  # must be serializable (the --json path)


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback; print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
