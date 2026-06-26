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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.db import DB  # noqa: E402
from ic2x.reconcile import reconcile_with_x  # noqa: E402
from ic2x.status import Status  # noqa: E402

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


def _seed(db, *, asset, sha, tid, ph, caption):
    db._conn.execute(
        "INSERT INTO images (asset_id, sha256, phash, status, caption, tweet_id, posted_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (asset, sha, ph, Status.POSTED.value, caption, tid, _NOW.isoformat()),
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
def test_deleted_is_requeued_and_live_kept():
    db = _db(); cfg = _cfg()
    _seed(db, asset="A_DEL", sha="shaDEL", tid="100", ph=_PH_A, caption="deleted one")
    _seed(db, asset="A_LIVE", sha="shaLIVE", tid="200", ph=_PH_B, caption="live one")
    reconcile_with_x(db, cfg, _lookup(live={"200": "live one"}, deleted={"100"}))
    assert db.get_image_by_sha("shaDEL") is None              # row gone
    assert db.seen_asset_id("A_DEL") is False                 # asset un-seen
    assert db.seen_sha256("shaDEL") is False                  # re-postable …
    assert db.seen_phash_similar(_PH_A, 12) is False          # … nothing blocks it
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


def test_requeue_deleted_method():
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
