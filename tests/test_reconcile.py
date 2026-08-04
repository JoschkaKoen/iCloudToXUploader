"""
Offline tests for startup X reconciliation (`reconcile_with_x`). No network — a fake
`lookup_tweets(ids) -> (live, deleted)` stands in for the real Bearer get_tweets(ids)
existence check. Proves: a DB post X reports as deleted is re-queued (row deleted +
asset un-seen + re-postable); live posts are kept; missing captions are filled
(existing ones never clobbered); DRYRUN rows are ignored; the call is fail-open on any
X error; and the sanity cap blocks a mass re-queue.

Run: .venv/bin/python tests/test_reconcile.py   (or via pytest)
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.reconcile as reconcile  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.reconcile import reconcile_with_x  # noqa: E402
from ic2x.status import Status  # noqa: E402

reconcile._X_READ_RETRY_DELAY = 0  # no real sleeps between retries in tests

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_reconcile_test_"))
_SEQ = [0]
_NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)
_PH_A = "ffffffffffffffff"   # two 64-bit phashes 64 apart → never cross-match in dedup
_PH_B = "0000000000000000"


def _cfg(max_requeue=20, recent_n=50):
    _SEQ[0] += 1
    logs = _TMP / f"logs{_SEQ[0]}"; logs.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(logs_dir=logs, reconcile_recent_n=recent_n,
                           reconcile_max_requeue_per_run=max_requeue)


def _db():
    _SEQ[0] += 1
    return DB(_TMP / f"r{_SEQ[0]}.db")


def _seed(db, *, asset, sha, tid, ph, caption, posted_at=None):
    db._conn.execute(
        "INSERT INTO images (asset_id, sha256, phash, status, caption, tweet_id, posted_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (asset, sha, ph, Status.POSTED.value, caption, tid, (posted_at or _NOW).isoformat()),
    )
    db._conn.execute(
        "INSERT INTO asset_index (asset_id, seen) VALUES (?, 1) "
        "ON CONFLICT(asset_id) DO UPDATE SET seen = 1", (asset,),
    )
    db._conn.commit()
    return db._conn.execute("SELECT id FROM images WHERE sha256=?", (sha,)).fetchone()["id"]


def _lookup(live=None, deleted=None, raises=False):
    def fn(ids):
        if raises:
            raise RuntimeError("boom")
        return dict(live or {}), set(deleted or set())
    return fn


def _log_has(cfg, needle):
    return any(needle in p.read_text() for p in cfg.logs_dir.glob("*.jsonl"))


# ── tests ──────────────────────────────────────────────────────────────────────
def test_deleted_is_retired_and_live_kept():
    """The owner deleting a post is a verdict on that photo — it must never come back.
    Re-queuing it (the behaviour until 2026-08-04) meant deleting a post you disliked
    put the photo straight back in the pool."""
    db = _db(); cfg = _cfg()
    _seed(db, asset="A_DEL", sha="shaDEL", tid="100", ph=_PH_A, caption="deleted one")
    _seed(db, asset="A_LIVE", sha="shaLIVE", tid="200", ph=_PH_B, caption="live one")
    reconcile_with_x(db, cfg, _lookup(live={"200": "live one"}, deleted={"100"}))

    row = db.get_image_by_sha("shaDEL")
    assert row is not None, "the row was dropped — the photo can be picked up again"
    assert row["status"] == Status.REJECTED.value
    assert row["reject_stage"] == "deleted_on_x"
    assert db.seen_asset_id("A_DEL") is True, "asset went back in the pool"
    assert db.seen_sha256("shaDEL") is True, "the exact file is no longer blocked"
    # Near-duplicates are NOT blocked: rejecting one frame shouldn't retire the scene.
    assert db.seen_phash_similar(_PH_A, 12) is False
    assert db.get_image_by_sha("shaLIVE") is not None         # live kept
    assert db.seen_asset_id("A_LIVE") is True
    assert _log_has(cfg, "deleted_on_x")
    db.close()


def test_caption_filled_only_when_missing():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A1", sha="s1", tid="200", ph=_PH_A, caption="")
    _seed(db, asset="A2", sha="s2", tid="201", ph=_PH_B, caption="kept caption")
    reconcile_with_x(db, cfg, _lookup(live={"200": "from X", "201": "X edited"}))
    assert db.get_image_by_sha("s1")["caption"] == "from X"        # empty → filled
    assert db.get_image_by_sha("s2")["caption"] == "kept caption"  # non-empty → kept
    db.close()


def test_fail_open_on_x_error():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A", sha="s", tid="100", ph=_PH_A, caption="x")
    reconcile_with_x(db, cfg, _lookup(raises=True))
    assert db.get_image_by_sha("s") is not None and db.seen_asset_id("A") is True
    db.close()


def test_retries_in_close_succession_then_succeeds():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A_DEL", sha="shaDEL", tid="100", ph=_PH_A, caption="d")
    state = {"n": 0}

    def flaky(ids):  # fail twice, succeed on the 3rd attempt
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError(f"proxy 503 ({state['n']})")
        return {}, {"100"}

    reconcile_with_x(db, cfg, flaky)
    assert state["n"] == 3                            # one try + two retries
    assert db.get_image_by_sha("shaDEL")["status"] == Status.REJECTED.value, \
        "the 3rd-attempt result was not used"
    db.close()


def test_gives_up_after_three_attempts():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A", sha="s", tid="100", ph=_PH_A, caption="x")
    state = {"n": 0}

    def always_fail(ids):
        state["n"] += 1
        raise RuntimeError("down")

    reconcile_with_x(db, cfg, always_fail)
    assert state["n"] == 3                             # exactly 3 attempts, then move on
    assert db.get_image_by_sha("s") is not None        # fail-open: nothing changed
    db.close()


def test_deleting_most_recent_refreshes_post_timer():
    db = _db(); cfg = _cfg()
    old, new = _NOW - timedelta(hours=10), _NOW
    _seed(db, asset="OLD", sha="sOLD", tid="100", ph=_PH_A, caption="old live", posted_at=old)
    _seed(db, asset="NEW", sha="sNEW", tid="200", ph=_PH_B, caption="new deleted", posted_at=new)
    db.set_last_posted_at(new)                       # timer points at the soon-deleted post
    reconcile_with_x(db, cfg, _lookup(deleted={"200"}))
    assert db.get_last_posted_at() == old            # falls back to the most-recent LIVE post
    db.close()


def test_stale_timer_corrected_without_new_deletions():
    # the user's case: the deleted post was re-queued in a PRIOR run, so this run sees 0
    # deletions — but last_posted_at still points at that gone post and must be corrected.
    db = _db(); cfg = _cfg()
    live_time = _NOW - timedelta(hours=30)
    _seed(db, asset="LIVE", sha="sL", tid="100", ph=_PH_A, caption="old live", posted_at=live_time)
    db.set_last_posted_at(_NOW)                       # stale — points at a post no longer in the DB
    reconcile_with_x(db, cfg, _lookup(live={"100": "old live"}))   # 0 deletions this run
    assert db.get_last_posted_at() == live_time       # synced to the most-recent live post → due
    db.close()


def test_deleting_only_post_clears_timer():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A", sha="s", tid="100", ph=_PH_A, caption="x")
    db.set_last_posted_at(_NOW)
    reconcile_with_x(db, cfg, _lookup(deleted={"100"}))
    assert db.get_last_posted_at() is None           # nothing live → timer cleared → due now
    db.close()


def test_sanity_guard_skips_mass_deletion():
    db = _db(); cfg = _cfg(max_requeue=2)
    for i in range(4):
        _seed(db, asset=f"A{i}", sha=f"s{i}", tid=str(100 + i), ph=_PH_A, caption="x")
    reconcile_with_x(db, cfg, _lookup(deleted={"100", "101", "102", "103"}))
    for i in range(4):
        assert db.get_image_by_sha(f"s{i}") is not None         # guard tripped → all kept
    db.close()


def test_no_posted_rows_is_noop():
    db = _db(); cfg = _cfg()
    reconcile_with_x(db, cfg, _lookup(deleted={"x"}))           # nothing seeded → no crash
    db.close()


def test_reject_deleted_method():
    """What reconcile uses: the row survives as rejected and the asset stays seen."""
    db = _db()
    iid = _seed(db, asset="A", sha="s", tid="100", ph=_PH_A, caption="x")
    db.reject_deleted(iid, "A")
    row = db.get_image_by_sha("s")
    assert row["status"] == Status.REJECTED.value and row["tweet_id"] is None
    assert db.seen_asset_id("A") is True
    db.close()


def test_requeue_deleted_method_is_the_manual_escape_hatch():
    """Still available for deliberately giving a retired photo another chance."""
    db = _db()
    iid = _seed(db, asset="A", sha="s", tid="100", ph=_PH_A, caption="x")
    db.requeue_deleted(iid, "A")
    assert db.get_image_by_sha("s") is None and db.seen_asset_id("A") is False
    db.close()


def test_posted_for_reconcile_filters():
    db = _db()
    _seed(db, asset="P", sha="sp", tid="100", ph=_PH_A, caption="x")
    _seed(db, asset="D", sha="sd", tid="DRYRUN", ph=_PH_B, caption="x")
    tids = {r["tweet_id"] for r in db.posted_for_reconcile(50)}
    assert "100" in tids and "DRYRUN" not in tids               # real tweet_id only
    db.close()


def _main() -> int:
    failed = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_") and callable(t):
            try:
                t(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"FAIL {name}: {exc}"); traceback.print_exc()
    print("OK" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
