"""
Offline correctness tests for burst assembly — no iCloud, no network.

Proves: the seen-set stops re-assembly, two distinct scenes are both reachable
(no gap-drop), pHash grouping, capped-burst tail consumption, screenshot
dropping, and undecodable-asset handling. The bot now assembles from a live
(meta, asset) stream, so these drive `assemble_burst` with a fake stream + ic.

Run: .venv/bin/python tests/test_burst.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.bot import _Stream, assemble_burst  # noqa: E402
from ic2x.db import DB  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_burst_test_"))
_FIX = _TMP / "fix"; _FIX.mkdir(parents=True, exist_ok=True)
_DB_SEQ = 0


def _make_scene(path: Path, scene: int, variant: int = 0) -> None:
    img = Image.new("RGB", (256, 256), "black")
    d = ImageDraw.Draw(img)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")
    elif scene == 1:
        d.rectangle([0, 0, 256, 128], fill="white")
    elif scene == 2:
        d.ellipse([48, 48, 208, 208], fill="white")
    else:
        d.rectangle([160, 0, 256, 256], fill="white")
    if variant:
        d.point([(variant % 256, (variant * 7) % 256)], fill="gray")
    img.save(path, "JPEG", quality=92)


class _FakeAsset:
    def __init__(self, aid: str, path: Path | None) -> None:
        self.id = aid
        self._path = path  # None → unavailable/undecodable


class FakeIC:
    """Stands in for ICloudPhotos: download copies the fixture (or raises)."""

    def download(self, asset, version, dest: Path):
        if asset._path is None:
            raise RuntimeError("unavailable")
        shutil.copy(asset._path, dest)
        return dest


def _cfg(max_size: int = 3, ham: int = 8):
    return SimpleNamespace(burst_max_size=max_size, burst_hamming_threshold=ham,
                           thumb_version="thumb", work_dir=_TMP / "work",
                           prefetch_concurrency=4)


(_TMP / "work").mkdir(exist_ok=True)


def _fresh_db() -> DB:
    global _DB_SEQ
    _DB_SEQ += 1
    return DB(_TMP / f"db_{_DB_SEQ}.db")


def _asset(aid: str, scene: int, variant: int = 0, *, dead: bool = False):
    """Return (id, fixture_path|None) for the stream."""
    if dead:
        return (aid, None)
    p = _FIX / f"{aid}_{scene}_{variant}.jpg"
    _make_scene(p, scene, variant)
    return (aid, p)


def _stream(db: DB, items, cfg, screenshots=()):
    def gen():
        for aid, path in items:
            yield SimpleNamespace(id=aid), _FakeAsset(aid, path)
    return _Stream(gen(), db, cfg, FakeIC(), set(screenshots),
                   concurrency=getattr(cfg, "prefetch_concurrency", 4))


def _burst(db, items, cfg, screenshots=()):
    return assemble_burst(_stream(db, items, cfg, screenshots), cfg, FakeIC(), set(screenshots))


# ── tests ───────────────────────────────────────────────────────────────────────

def test_empty_returns_none():
    assert _burst(_fresh_db(), [], _cfg()) is None


def test_all_seen_returns_none():
    db = _fresh_db()
    db.commit_burst(["a"], None)  # mark seen
    assert _burst(db, [_asset("a", 0, 1)], _cfg()) is None


def test_single_asset_burst():
    b = _burst(_fresh_db(), [_asset("a", 0, 1)], _cfg())
    assert b is not None and [m.asset_id for m in b.members] == ["a"] and not b.aux_seen


def test_two_distinct_scenes_no_gap_drop():
    """Both scenes must be reachable as the shared stream advances (no drop)."""
    db = _fresh_db()
    stream = _stream(db, [_asset("A", 0, 1), _asset("B", 2, 1)], _cfg())  # far pHash
    b1 = assemble_burst(stream, _cfg(), FakeIC(), set())
    assert [m.asset_id for m in b1.members] == ["A"]
    db.commit_burst(["A"], None)
    b2 = assemble_burst(stream, _cfg(), FakeIC(), set())   # SAME stream continues
    assert [m.asset_id for m in b2.members] == ["B"]


def test_fresh_stream_skips_seen():
    db = _fresh_db()
    db.commit_burst(["A"], None)  # A already decided
    b = _burst(db, [_asset("A", 0, 1), _asset("B", 2, 1)], _cfg())
    assert [m.asset_id for m in b.members] == ["B"]  # seen A skipped, B assembled


def test_similar_run_groups():
    items = [_asset(a, 0, i + 1) for i, a in enumerate(["a", "b", "c"])]
    b = _burst(_fresh_db(), items, _cfg(max_size=5))
    assert sorted(m.asset_id for m in b.members) == ["a", "b", "c"]


def test_cap_spills_the_tail_into_the_next_burst():
    """A scene longer than BURST_MAX_SIZE must NOT lose its tail. The overflow used
    to be swallowed into aux_seen — marked decided without ever being judged or even
    downloaded — and 29.5% of real bursts hit the cap, so a 12-shot scene silently
    lost 7 frames. The tail now forms the following burst instead."""
    ids = [f"s{i}" for i in range(6)]
    items = [_asset(a, 1, i + 1) for i, a in enumerate(ids)]
    db, cfg = _fresh_db(), _cfg(max_size=3)
    stream = _stream(db, items, cfg, ())

    first = assemble_burst(stream, cfg, FakeIC(), set())
    assert len(first.members) == 3
    assert not first.aux_seen, f"tail was consumed unjudged: {first.aux_seen}"

    second = assemble_burst(stream, cfg, FakeIC(), set())
    assert second is not None, "the overflow vanished — this is the coverage leak"
    seen_ids = {m.asset_id for m in first.members} | {m.asset_id for m in second.members}
    assert seen_ids == set(ids), f"photos lost at the cap: {set(ids) - seen_ids}"


def test_screenshot_dropped_to_aux():
    b = _burst(_fresh_db(), [_asset("shot", 0, 1), _asset("real", 2, 1)], _cfg(),
               screenshots={"shot"})
    assert "shot" in b.aux_seen and [m.asset_id for m in b.members] == ["real"]


def test_undecodable_to_aux():
    # a file that downloads but can't be hashed (corrupt) is permanently aux-seen
    p = _FIX / "corrupt.jpg"
    p.write_bytes(b"definitely not a jpeg")
    b = _burst(_fresh_db(), [("bad", p), _asset("good", 2, 1)], _cfg())
    assert "bad" in b.aux_seen and [m.asset_id for m in b.members] == ["good"]


def test_download_failure_retries_until_attempts_exhausted():
    # a DOWNLOAD failure must NOT mark the photo seen — it retries next cycle
    # ("go back in time only when no newest unposted image was detected/downloaded")
    db = _fresh_db()
    cfg = _cfg()
    cfg.burst_max_attempts = 3
    items = [_asset("flaky", 0, dead=True), _asset("good", 2, 1)]

    for round_no in (1, 2):  # first two failures: retried, not seen
        b = assemble_burst(_stream(db, items, cfg), cfg, FakeIC(), set(), db)
        assert "flaky" not in b.aux_seen, round_no
        assert [m.asset_id for m in b.members] == ["good"]
        assert db.seen_asset_id("flaky") is False
        db._conn.execute("UPDATE asset_index SET seen=0 WHERE asset_id='good'")
        db._conn.commit()

    b = assemble_burst(_stream(db, items, cfg), cfg, FakeIC(), set(), db)  # 3rd → give up
    assert "flaky" in b.aux_seen


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
