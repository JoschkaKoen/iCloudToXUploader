"""
Offline tests for the chronological library walk-back (iter_image_assets).

No network — the album accessors are monkeypatched with stub assets. Proves:
recent window comes first; the deep walk skips bulk-imported archives (added ≫
captured) until the organic timeline is exhausted; imports then surface in the
final phase; ids never repeat across phases; broken assets are skipped.

Also covers ensure_catalog's incremental refresh, which must scan from the album
HEAD: the .all album is newest-first, so new photos land at offset 0. Scanning
from the tail (the 2004 imports) hit an all-known page and broke instantly, and
the catalog silently stopped growing — 69974 assets in the library against 67741
cataloged, with three weeks of photos invisible to the walk-back (2026-08-04).

Run: .venv/bin/python tests/test_icloud_walkback.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.icloud_photos as icp  # noqa: E402
from ic2x.icloud_photos import ICloudPhotos  # noqa: E402

NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _asset(aid, created_days_ago, added_days_ago=None, item_type="image"):
    created = NOW - timedelta(days=created_days_ago)
    added = NOW - timedelta(days=added_days_ago if added_days_ago is not None
                            else created_days_ago)
    return SimpleNamespace(id=aid, created=created, added_date=added,
                           filename=f"{aid}.jpg", is_live_photo=False,
                           dimensions=(100, 80), item_type=item_type)


def _ic(recent, full):
    ic = ICloudPhotos.__new__(ICloudPhotos)  # no session needed for iteration
    ic._recent_album = lambda: list(recent)
    ic._iter_all_backwards = lambda: iter(list(full))
    return ic


def test_archive_import_detection():
    organic = _asset("a", created_days_ago=3)                       # added ≈ captured
    late_sync = _asset("b", created_days_ago=10, added_days_ago=2)  # 8d gap — organic
    archive = _asset("c", created_days_ago=8000, added_days_ago=1)  # 2004 import
    assert ICloudPhotos._is_archive_import(organic) is False
    assert ICloudPhotos._is_archive_import(late_sync) is False
    assert ICloudPhotos._is_archive_import(archive) is True
    assert ICloudPhotos._is_archive_import(
        SimpleNamespace(created=None, added_date=None)) is False  # missing → organic


def test_walkback_defers_archive_imports_to_final_phase():
    recent = [_asset("new1", 0), _asset("skip_video", 1, item_type="video")]
    full = [
        _asset("new1", 0),                                  # duplicate of recent
        _asset("week", 7),
        _asset("ski2004", 8000, added_days_ago=1),          # import — must defer
        _asset("month", 30),
        _asset("prague2019", 2500, added_days_ago=2),       # import — must defer
        _asset("lastyear", 300),
    ]
    order = [meta.id for meta, _a in _ic(recent, full).iter_image_assets()]
    assert order == ["new1", "week", "month", "lastyear", "ski2004", "prague2019"]


def test_no_archives_means_no_third_pass():
    calls = {"n": 0}

    def counting_backwards():
        calls["n"] += 1
        return iter([_asset("a", 1), _asset("b", 2)])

    ic = ICloudPhotos.__new__(ICloudPhotos)
    ic._recent_album = lambda: []
    ic._iter_all_backwards = counting_backwards
    order = [m.id for m, _ in ic.iter_image_assets()]
    assert order == ["a", "b"]
    assert calls["n"] == 1  # nothing deferred → the archive pass never runs


def test_unreadable_asset_skipped():
    bad = SimpleNamespace(id="bad", item_type="image", created=NOW, added_date=NOW,
                          filename="x.jpg", is_live_photo=False)  # no .dimensions
    ok = _asset("ok", 1)
    order = [m.id for m, _ in _ic([], [bad, ok]).iter_image_assets()]
    assert order == ["ok"]


def test_catalog_chrono_iteration_and_seen_skip():
    import tempfile
    from ic2x.db import DB

    db = DB(Path(tempfile.mkdtemp(prefix="ic2x_cat_")) / "t.db")
    db.catalog_upsert_many([
        ("old2004", NOW.replace(year=2004).isoformat(), 5),
        ("lastweek", (NOW - timedelta(days=7)).isoformat(), 90),
        ("yesterday", (NOW - timedelta(days=1)).isoformat(), 100),
        ("seenone", (NOW - timedelta(days=2)).isoformat(), 95),
        ("nodate", None, 96),
    ])
    db.commit_burst(["seenone"], None)  # marks it seen
    order = [r["asset_id"] for r in db.catalog_unseen_desc()]
    assert order == ["yesterday", "lastweek", "old2004"]  # DESC, seen + dateless excluded
    assert db.catalog_count() == 5
    assert db.catalog_known(["lastweek", "ghost"]) == {"lastweek"}


def test_chrono_iterator_uses_catalog_order_and_recent_first():
    import tempfile
    from ic2x.db import DB

    db = DB(Path(tempfile.mkdtemp(prefix="ic2x_cat2_")) / "t.db")
    fresh = _asset("fresh", 0)
    anachron = _asset("import2004", 8000, added_days_ago=0)   # in recent window, deferred
    older = _asset("older", 5)
    oldest = _asset("oldest", 40)

    ic = ICloudPhotos.__new__(ICloudPhotos)
    ic.ensure_catalog = lambda _db: None
    ic._recent_album = lambda: [fresh, anachron]
    by_id = {"older": older, "oldest": oldest, "import2004": anachron}
    ic._fetch_by_rank = lambda aid, rank, db=None: by_id.get(aid)
    db.catalog_upsert_many([
        ("fresh", fresh.created.isoformat(), 500),        # dup of recent — must not repeat
        ("older", older.created.isoformat(), 490),
        ("oldest", oldest.created.isoformat(), 400),
        ("import2004", anachron.created.isoformat(), 510),
        ("gone", (NOW - timedelta(days=3)).isoformat(), 495),  # fetch returns None
    ])
    order = [m.id for m, _a in ic.iter_image_assets_chrono(db)]
    # recent first; then catalog strictly by capture DESC ("gone" skipped);
    # the 2004 import surfaces only at its true chronological position — last.
    assert order == ["fresh", "older", "oldest", "import2004"]


class _FakeAlbum:
    """Stand-in for the .all album: NEWEST-FIRST, so offset 0 is the most recent
    asset and the tail holds the oldest imports (verified against live iCloud)."""

    def __init__(self, assets):
        self._assets = list(assets)
        self.reads: list[tuple[int, int]] = []

    def __len__(self):
        return len(self._assets)

    def _get_photos_at(self, offset, direction, page):
        self.reads.append((offset, page))
        return self._assets[offset:offset + page]


def _catalog_ic(album):
    ic = ICloudPhotos.__new__(ICloudPhotos)   # no session needed for cataloging
    ic._all_album = lambda: album
    return ic


def _fresh_db():
    import tempfile
    from ic2x.db import DB
    return DB(Path(tempfile.mkdtemp(prefix="ic2x_catalog_")) / "state.db")


def test_catalog_refresh_indexes_new_photos_at_the_head():
    """New photos arrive at offset 0 and must be picked up at all — the 2026-08-04 bug
    was a tail-scan that hit an all-known page and indexed nothing, silently freezing
    the catalog. Rank semantics belong to
    test_catalog_refresh_leaves_existing_ranks_untouched."""
    db = _fresh_db()
    old = [_asset(f"old{i}", created_days_ago=10 + i) for i in range(5)]
    db.catalog_upsert_many([(a.id, a.created.isoformat(), i) for i, a in enumerate(old)])
    db.set_state("catalog_complete", "1")

    new = [_asset(f"new{i}", created_days_ago=i) for i in range(3)]   # newer than all
    album = _FakeAlbum(new + old)                                     # newest-first
    _catalog_ic(album).ensure_catalog(db, page=4)

    assert db.catalog_count() == 8, f"new head photos were not indexed ({db.catalog_count()})"
    cataloged = {r["asset_id"] for r in
                 db._conn.execute("SELECT asset_id FROM asset_catalog")}
    assert {f"new{i}" for i in range(3)} <= cataloged
    # They must sort ahead of the existing rows in the walk-back's capture order.
    order = [r["asset_id"] for r in db.catalog_unseen_desc()]
    assert order[:3] == ["new0", "new1", "new2"], order
    db.close()


def test_catalog_refresh_leaves_existing_ranks_untouched():
    """New assets must join the FROZEN baseline (negative ranks), not be given live
    album offsets with every existing rank shifted to match.

    Rewriting stored ranks broke _fetch_by_rank: it absorbs album movement with one
    learned `drift`, so with both the rank and the drift moving, probes miss — and a
    miss is read as "deleted from iCloud". `drift` only refreshes on a hit, so the
    first miss froze it and 67139 catalog rows were dropped in one run (2026-08-11)."""
    db = _fresh_db()
    old = [_asset(f"old{i}", created_days_ago=10 + i) for i in range(5)]
    db.catalog_upsert_many([(a.id, a.created.isoformat(), i) for i, a in enumerate(old)])
    db.set_state("catalog_complete", "1")

    new = [_asset(f"new{i}", created_days_ago=i) for i in range(3)]
    album = _FakeAlbum(new + old)
    _catalog_ic(album).ensure_catalog(db, page=4)

    ranks = {r["asset_id"]: r["rank"] for r in
             db._conn.execute("SELECT asset_id, rank FROM asset_catalog")}
    assert [ranks[f"old{i}"] for i in range(5)] == [0, 1, 2, 3, 4], \
        f"existing ranks were rewritten — this is what emptied the catalog: {ranks}"
    assert [ranks[f"new{i}"] for i in range(3)] == [-3, -2, -1], \
        f"new assets must sit before the frozen rank 0: {ranks}"
    assert db.catalog_count() == 8
    db.close()


def test_successive_refreshes_extend_one_continuous_rank_scale():
    """Each refresh must number new assets BELOW everything already there. Numbering
    every batch -n..-1 in isolation made successive batches collide at the wrong
    offsets — 1420 assets sharing 391 slots on 2026-09-07 — so every probe missed and,
    because drift only updates on a hit, the walk-back resolved nothing for 57 hours.
    One global drift only works if rank is a position on ONE continuous scale."""
    db = _fresh_db()
    old = [_asset(f"old{i}", created_days_ago=50 + i) for i in range(4)]
    db.catalog_upsert_many([(a.id, a.created.isoformat(), i) for i, a in enumerate(old)])
    db.set_state("catalog_complete", "1")

    batch1 = [_asset(f"b1_{i}", created_days_ago=20 + i) for i in range(3)]
    _catalog_ic(_FakeAlbum(batch1 + old)).ensure_catalog(db, page=10)
    # newer batch arrives later, so it sits AHEAD of batch1 in the album
    batch2 = [_asset(f"b2_{i}", created_days_ago=1 + i) for i in range(2)]
    _catalog_ic(_FakeAlbum(batch2 + batch1 + old)).ensure_catalog(db, page=10)

    ranks = {r["asset_id"]: r["rank"] for r in
             db._conn.execute("SELECT asset_id, rank FROM asset_catalog")}
    assert len(set(ranks.values())) == len(ranks), f"ranks collided: {sorted(ranks.values())}"
    assert [ranks[f"old{i}"] for i in range(4)] == [0, 1, 2, 3], "baseline ranks moved"
    b1 = [ranks[f"b1_{i}"] for i in range(3)]
    b2 = [ranks[f"b2_{i}"] for i in range(2)]
    assert max(b2) < min(b1) < 0, (
        f"the newer batch must sit below the older one on the scale: b2={b2} b1={b1}")
    db.close()


def test_probe_misses_cannot_empty_the_catalog():
    """A probe miss is evidence of deletion, not proof. A systematic mismatch makes
    EVERY probe miss, and drift cannot self-correct because only a hit refreshes it —
    so the drop path needs a hard cap or it deletes the whole library index."""
    db = _fresh_db()
    n = 200
    rows = [(f"a{i}", NOW.isoformat(), i) for i in range(n)]
    db.catalog_upsert_many(rows)
    assert db.catalog_count() == n

    ic = ICloudPhotos.__new__(ICloudPhotos)
    # An album that never contains the asset being probed, but always returns a window
    # covering the expected rank — i.e. every probe misses inside a covered window.
    ic._all_album = lambda: _FakeAlbum([_asset(f"ghost{i}", 1) for i in range(400)])
    for i in range(n):
        ic._fetch_by_rank(f"a{i}", i, db)

    remaining = db.catalog_count()
    assert remaining >= n - (icp._MAX_CONSECUTIVE_DROPS + 1), (
        f"only {remaining} of {n} rows survived a run of pure probe misses — "
        "the cap did not hold and the catalog can still be emptied")
    db.close()


def test_catalog_refresh_stops_at_known_territory():
    """It must stop at the first fully-known page, not re-walk the whole library —
    the refresh runs every cycle against a ~70k-asset album."""
    db = _fresh_db()
    old = [_asset(f"old{i}", created_days_ago=10 + i) for i in range(50)]
    db.catalog_upsert_many([(a.id, a.created.isoformat(), i) for i, a in enumerate(old)])
    db.set_state("catalog_complete", "1")

    album = _FakeAlbum(old)          # nothing new at all
    ic = _catalog_ic(album)
    ic.ensure_catalog(db, page=10)

    assert db.catalog_count() == 50
    assert len(album.reads) == 1, f"walked {len(album.reads)} pages for a no-op refresh"
    assert album.reads[0][0] == 0, f"refresh started at offset {album.reads[0][0]}, not the head"
    db.close()


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
