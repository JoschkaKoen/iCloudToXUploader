"""
Offline tests for the commit → post → flush path (dry-run, no network/X).

Covers: atomic winner commit creates the images row; dry-run post_one advances
state + moves the file + bumps the rolling-24h count; flush_pending posts at most
one and respects MAX_POSTS_PER_DAY.

Run: .venv/bin/python tests/test_post.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.bot import flush_pending  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.post import post_one  # noqa: E402
from ic2x.status import Status  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_post_test_"))
_DB_SEQ = 0


def _cfg(max_posts: int = 6):
    c = SimpleNamespace(
        approved_dir=_TMP / "approved", posted_dir=_TMP / "posted",
        logs_dir=_TMP / "logs", x_dry_run=True, post_max_attempts=3,
        max_posts_per_day=max_posts,
    )
    for d in (c.approved_dir, c.posted_dir, c.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return c


def _fresh_db() -> DB:
    global _DB_SEQ
    _DB_SEQ += 1
    return DB(_TMP / f"db_{_DB_SEQ}.db")


def _stage_winner(db: DB, cfg, sha: str, ph: str, aid: str = "a") -> dict:
    db.commit_burst([aid], {"asset_id": aid, "sha256": sha, "phash": ph,
                            "filename": "x.HEIC", "status": Status.APPROVED, "caption": "hi"})
    (cfg.approved_dir / f"{ph}.jpg").write_bytes(b"fake-jpeg")
    return {"sha256": sha, "phash": ph, "caption": "hi"}


def test_commit_creates_approved_row():
    db = _fresh_db(); cfg = _cfg()
    _stage_winner(db, cfg, "sha_a", "ph_a")
    row = db.get_image_by_sha("sha_a")
    assert row is not None and row["status"] == Status.APPROVED.value
    assert len(db.get_approved()) == 1


def test_dry_run_post_advances_state():
    db = _fresh_db(); cfg = _cfg()
    row = _stage_winner(db, cfg, "sha_b", "ph_b")
    assert post_one(row, cfg, db, None, None) is True
    img = db.get_image_by_sha("sha_b")
    assert img["status"] == Status.POSTED.value and img["tweet_id"] == "DRYRUN"
    assert (cfg.posted_dir / "ph_b.jpg").exists()
    assert not (cfg.approved_dir / "ph_b.jpg").exists()
    assert db.count_posts_rolling_24h() == 1
    assert db.get_last_posted_at() is not None


def test_missing_file_is_rejected_not_wedged():
    db = _fresh_db(); cfg = _cfg()
    db.commit_burst(["a"], {"asset_id": "a", "sha256": "sha_c", "phash": "ph_c",
                            "status": Status.APPROVED, "caption": ""})
    # no file on disk
    ok = post_one({"sha256": "sha_c", "phash": "ph_c", "caption": ""}, cfg, db, None, None)
    assert ok is False
    assert db.get_image_by_sha("sha_c")["status"] == Status.REJECTED.value


def test_flush_posts_one_and_respects_cap():
    db = _fresh_db(); cfg = _cfg(max_posts=1)
    _stage_winner(db, cfg, "sha_d", "ph_d", aid="d")
    _stage_winner(db, cfg, "sha_e", "ph_e", aid="e")
    assert len(db.get_approved()) == 2
    assert flush_pending(db, cfg, (None, None)) is True   # posts one
    assert len(db.get_approved()) == 1
    # rolling cap (1) now reached → flush refuses the second
    assert flush_pending(db, cfg, (None, None)) is False
    assert len(db.get_approved()) == 1


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
