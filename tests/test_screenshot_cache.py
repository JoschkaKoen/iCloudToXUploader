"""
Offline test for ICloudPhotos.screenshot_ids_cached — the Screenshots-album
membership cache that refreshes only every N cycles (no iCloud, no network).

Proves: the underlying fetch runs once on the first call, is reused for the next
N-1 cycles, and re-fetches on the Nth; refresh_every<=1 disables caching; and a
fresh fetch reflects album changes.

Run: .venv/bin/python tests/test_screenshot_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.icloud_photos import ICloudPhotos  # noqa: E402


def _ic_with_counter(sets):
    """An ICloudPhotos whose screenshot_ids() returns sets[i] on the i-th call and
    records how many times it was hit."""
    ic = ICloudPhotos(SimpleNamespace())
    calls = {"n": 0}

    def fake_fetch():
        i = calls["n"]
        calls["n"] += 1
        return sets[min(i, len(sets) - 1)]

    ic.screenshot_ids = fake_fetch  # type: ignore[method-assign]
    return ic, calls


def test_refreshes_every_n_cycles():
    ic, calls = _ic_with_counter([{"a"}])
    # N=3 → fetch on calls 1 and 4 only
    for _ in range(3):
        assert ic.screenshot_ids_cached(3) == {"a"}
    assert calls["n"] == 1, "should have fetched exactly once across the first 3 cycles"
    ic.screenshot_ids_cached(3)            # 4th call → refresh
    assert calls["n"] == 2


def test_refresh_every_one_disables_cache():
    ic, calls = _ic_with_counter([{"a"}])
    for _ in range(4):
        ic.screenshot_ids_cached(1)
    assert calls["n"] == 4, "refresh_every<=1 must fetch every cycle"


def test_zero_or_negative_treated_as_one():
    ic, calls = _ic_with_counter([{"a"}])
    for _ in range(3):
        ic.screenshot_ids_cached(0)        # max(1, 0) → always refresh
    assert calls["n"] == 3


def test_fresh_fetch_picks_up_album_changes():
    ic, calls = _ic_with_counter([{"a"}, {"a", "b"}])
    assert ic.screenshot_ids_cached(2) == {"a"}   # call 1 → first set
    assert ic.screenshot_ids_cached(2) == {"a"}   # call 2 → cached
    assert ic.screenshot_ids_cached(2) == {"a", "b"}  # call 3 → refresh → new set
    assert calls["n"] == 2


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
