"""
Offline correctness tests for burst assembly — no iCloud, no network.

Proves the properties the redesign hinges on: newest-first by capture time, the
seen-set as single source of truth (no gap-drop across distinct scenes), burst
grouping by pHash, capped-burst tail consumption, screenshot dropping, and
undecodable-asset handling.

Run: .venv/bin/python tests/test_burst.py   (also works under pytest)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.bot import find_next_burst  # noqa: E402
from ic2x.db import DB  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_burst_test_"))
_FIXTURES = _TMP / "fixtures"
_FIXTURES.mkdir(parents=True, exist_ok=True)


def _make_scene(path: Path, scene: int, variant: int = 0) -> None:
    """Big distinct shapes per scene → far pHash; a single stray pixel per variant
    → negligible change → near-identical pHash within a scene."""
    img = Image.new("RGB", (256, 256), "black")
    d = ImageDraw.Draw(img)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")        # left half
    elif scene == 1:
        d.rectangle([0, 0, 256, 128], fill="white")        # top half
    elif scene == 2:
        d.ellipse([48, 48, 208, 208], fill="white")        # centre disc
    else:
        d.rectangle([160, 0, 256, 256], fill="white")      # right strip
    if variant:
        d.point([(variant % 256, (variant * 7) % 256)], fill="gray")
    img.save(path, "JPEG", quality=92)


class FakeSource:
    """Maps asset_id → a fixture image. Returns a disposable copy each call so the
    assembler's _unlink() never destroys the fixture. ids in `dead` → None."""

    def __init__(self, mapping: dict[str, Path], dead: set[str] | None = None) -> None:
        self._map = mapping
        self._dead = dead or set()
        self._n = 0

    def download_thumb(self, asset_id: str) -> Path | None:
        if asset_id in self._dead or asset_id not in self._map:
            return None
        self._n += 1
        dst = _TMP / f"thumb_{self._n}.jpg"
        shutil.copy(self._map[asset_id], dst)
        return dst

    def download_original(self, asset_id: str) -> Path | None:
        return self._map.get(asset_id)


def _cfg(max_size: int = 3, ham: int = 8):
    return SimpleNamespace(burst_max_size=max_size, burst_hamming_threshold=ham)


_DB_SEQ = 0


def _fresh_db() -> DB:
    global _DB_SEQ
    _DB_SEQ += 1
    return DB(_TMP / f"db_{_DB_SEQ}.db")


def _add(db: DB, source_map: dict, asset_id: str, scene: int, variant: int,
         when: datetime, *, screenshot: bool = False, dead: bool = False) -> None:
    if not dead:
        p = _FIXTURES / f"{asset_id}.jpg"
        _make_scene(p, scene, variant)
        source_map[asset_id] = p
    db.upsert_asset(asset_id, when, f"{asset_id}.HEIC")
    if screenshot:
        db.mark_screenshots([asset_id])


_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ── tests ───────────────────────────────────────────────────────────────────────

def test_empty_returns_none():
    db = _fresh_db()
    assert find_next_burst(db, FakeSource({}), _cfg(), set()) is None


def test_all_seen_returns_none():
    db = _fresh_db(); m = {}
    _add(db, m, "a", 0, 1, _BASE)
    db.commit_burst(["a"], None)  # mark seen
    assert find_next_burst(db, FakeSource(m), _cfg(), set()) is None


def test_single_asset_burst():
    db = _fresh_db(); m = {}
    _add(db, m, "a", 0, 1, _BASE)
    b = find_next_burst(db, FakeSource(m), _cfg(), set())
    assert b is not None and [x.asset_id for x in b.members] == ["a"] and not b.aux_seen


def test_newest_first_by_created():
    db = _fresh_db(); m = {}
    # insert out of capture order; b is newest by `created`
    _add(db, m, "old", 0, 1, _BASE)
    _add(db, m, "new", 1, 1, _BASE + timedelta(hours=5))
    b = find_next_burst(db, FakeSource(m), _cfg(), set())
    assert b.head == "new" and b.members[0].asset_id == "new"


def test_two_distinct_scenes_no_gap_drop():
    """The bug the old date-cursor had: two different scenes must BOTH be reachable."""
    db = _fresh_db(); m = {}
    _add(db, m, "A", 0, 1, _BASE + timedelta(hours=2))   # newest, scene 0
    _add(db, m, "B", 2, 1, _BASE + timedelta(hours=1))   # older, scene 2 (far pHash)
    b1 = find_next_burst(db, FakeSource(m), _cfg(), set())
    assert [x.asset_id for x in b1.members] == ["A"]
    db.commit_burst(["A"], None)                          # decide burst 1
    b2 = find_next_burst(db, FakeSource(m), _cfg(), set())
    assert [x.asset_id for x in b2.members] == ["B"]      # the older scene is NOT dropped


def test_similar_run_groups():
    db = _fresh_db(); m = {}
    for i, aid in enumerate(["a", "b", "c"]):            # same scene, near-identical
        _add(db, m, aid, 0, i + 1, _BASE + timedelta(minutes=30 - i))
    b = find_next_burst(db, FakeSource(m), _cfg(max_size=5), set())
    assert sorted(x.asset_id for x in b.members) == ["a", "b", "c"]


def test_cap_consumes_tail():
    db = _fresh_db(); m = {}
    ids = [f"s{i}" for i in range(6)]                    # 6 near-identical, cap=3
    for i, aid in enumerate(ids):
        _add(db, m, aid, 1, i + 1, _BASE + timedelta(minutes=60 - i))
    b = find_next_burst(db, FakeSource(m), _cfg(max_size=3), set())
    assert len(b.members) == 3
    seen = {x.asset_id for x in b.members} | set(b.aux_seen)
    assert seen == set(ids)                              # whole scene consumed, no fragment left


def test_screenshot_dropped_to_aux():
    db = _fresh_db(); m = {}
    _add(db, m, "shot", 0, 1, _BASE + timedelta(hours=2), screenshot=True)
    _add(db, m, "real", 2, 1, _BASE + timedelta(hours=1))
    b = find_next_burst(db, FakeSource(m), _cfg(), set())
    assert "shot" in b.aux_seen and [x.asset_id for x in b.members] == ["real"]


def test_undecodable_to_aux():
    db = _fresh_db(); m = {}
    _add(db, m, "bad", 0, 1, _BASE + timedelta(hours=2), dead=True)  # source returns None
    _add(db, m, "good", 2, 1, _BASE + timedelta(hours=1))
    b = find_next_burst(db, FakeSource(m, dead={"bad"}), _cfg(), set())
    assert "bad" in b.aux_seen and [x.asset_id for x in b.members] == ["good"]


def test_exclude_skips_burst():
    db = _fresh_db(); m = {}
    _add(db, m, "A", 0, 1, _BASE + timedelta(hours=2))
    _add(db, m, "B", 2, 1, _BASE + timedelta(hours=1))
    b = find_next_burst(db, FakeSource(m), _cfg(), exclude={"A"})
    assert [x.asset_id for x in b.members] == ["B"]      # excluded head skipped this cycle


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
