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

import requests
import tweepy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.post as postmod  # noqa: E402
from ic2x.bot import flush_pending  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.post import _is_transient_post_error, _prune_scene_thumbs, post_one  # noqa: E402
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


def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f


def test_transient_network_error_defers_keeps_approved():
    """A network blip (X unreachable) must DEFER the post — keep it APPROVED, never
    burn the attempt budget, never drop the file. This is the lid-close / VPN-drop
    case that was permanently rejecting good photos."""
    db = _fresh_db(); cfg = _cfg(); cfg.x_dry_run = False
    row = _stage_winner(db, cfg, "sha_t", "ph_t", aid="t")
    boom = tweepy.TweepyException(
        "Failed to send request: HTTPSConnectionPool(host='upload.twitter.com', "
        "port=443): Max retries exceeded (Caused by ConnectTimeoutError())")
    orig = postmod._post_image
    postmod._post_image = _raise(boom)
    try:
        for _ in range(5):                 # hammer well past post_max_attempts (3)
            assert post_one(row, cfg, db, object(), object()) is False
    finally:
        postmod._post_image = orig
    img = db.get_image_by_sha("sha_t")
    assert img["status"] == Status.APPROVED.value, "network blip must NOT reject a good photo"
    assert img["post_attempts"] == 0, "transient errors must not burn the attempt budget"
    assert (cfg.approved_dir / "ph_t.jpg").exists(), "deferred photo's file must stay for retry"


def test_permanent_error_rejects_and_cleans_file():
    """A genuine rejection (not a network error) still rejects after the cap — and now
    cleans up the leftover approved/ file instead of orphaning it."""
    db = _fresh_db(); cfg = _cfg(); cfg.x_dry_run = False; cfg.post_max_attempts = 1
    row = _stage_winner(db, cfg, "sha_p", "ph_p", aid="p")
    orig = postmod._post_image
    postmod._post_image = _raise(ValueError("duplicate content not allowed"))
    try:
        assert post_one(row, cfg, db, object(), object()) is False
    finally:
        postmod._post_image = orig
    img = db.get_image_by_sha("sha_p")
    assert img["status"] == Status.REJECTED.value and img["reject_stage"] == "post_failed"
    assert not (cfg.approved_dir / "ph_p.jpg").exists(), "rejected photo's file must be cleaned up"


def test_is_transient_classifier():
    # the three real-world signatures we actually observed on upload.twitter.com
    for m in ("Failed to send request: ... Max retries exceeded (Caused by ConnectTimeoutError)",
              "Failed to send request: ... (Caused by NameResolutionError)",
              "Failed to send request: ... (Caused by SSLError)"):
        assert _is_transient_post_error(tweepy.TweepyException(m)) is True
    # a wrapped requests connection error (chained on __cause__) is transient
    try:
        try:
            raise requests.exceptions.ConnectionError("boom")
        except requests.exceptions.ConnectionError as e:
            raise tweepy.TweepyException("wrapped") from e
    except tweepy.TweepyException as e:
        assert _is_transient_post_error(e) is True
    # genuine rejections are NOT transient
    assert _is_transient_post_error(ValueError("duplicate content not allowed")) is False
    assert _is_transient_post_error(ValueError("media type unsupported")) is False


def test_prune_scene_thumbs_keeps_newest():
    import os

    d = _TMP / "scene_thumbs_prune"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        f = d / f"thumb{i:02d}.jpg"
        f.write_bytes(b"x")
        os.utime(f, (1_000_000 + i, 1_000_000 + i))  # increasing mtime: higher i = newer
    _prune_scene_thumbs(d, keep=5)
    remaining = sorted(p.name for p in d.glob("*.jpg"))
    assert remaining == [f"thumb{i:02d}.jpg" for i in range(15, 20)], remaining


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
