"""
Offline tests for geo.reverse_geocode caching (no real network — urlopen is mocked).

Proves: a successful lookup is cached (no second HTTP call), a genuine "no city"
result is also cached, but a TRANSIENT failure is NOT cached so the next call
retries. Guards the long-running-bot bug where a one-off blip permanently blocked
a location's geocode.

Also covers the in-call retry added 2026-08-05: a single blip used to cost the post
its 📍 line AND the caption model's location grounding, with no retry at all while
every other network call in a cycle retried.

Run: .venv/bin/python tests/test_geo.py
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x import geo  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


@contextmanager
def _mock_urlopen(side_effects):
    """side_effects: list of either a dict (→ JSON body) or an Exception (→ raised)."""
    calls = {"n": 0}
    orig = geo.urllib.request.urlopen

    def fake(url, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        eff = side_effects[min(i, len(side_effects) - 1)]
        if isinstance(eff, Exception):
            raise eff
        return _Resp(json.dumps(eff).encode())

    geo.urllib.request.urlopen = fake
    try:
        yield calls
    finally:
        geo.urllib.request.urlopen = orig


geo._GEOCODE_RETRY_DELAY = 0    # no real sleeps between retries in tests


def setup_function():  # pytest hook; also called manually by _main
    geo._cache.clear()


def test_success_is_cached():
    geo._cache.clear()
    with _mock_urlopen([{"city": "Ningbo", "countryName": "China"}]) as calls:
        assert geo.reverse_geocode(29.87, 121.55) == "Ningbo, China"
        assert geo.reverse_geocode(29.87, 121.55) == "Ningbo, China"  # cached
    assert calls["n"] == 1, "second call should hit the cache, not the network"


def test_no_city_result_is_cached_as_none():
    geo._cache.clear()
    with _mock_urlopen([{"countryName": ""}]) as calls:
        assert geo.reverse_geocode(0.0, 0.0001) is None
        assert geo.reverse_geocode(0.0, 0.0001) is None
    assert calls["n"] == 1, "a genuine empty result should still be cached"


def test_transient_blip_is_retried_within_the_call():
    """A one-off blip must not cost the post its location. Observed 2026-08-05: GPS
    read fine at 31.5778,120.2983 (Wuxi), the single lookup failed during a flaky
    window, and the post went out with no 📍 line and an ungrounded caption."""
    geo._cache.clear()
    effects = [TimeoutError("blip"), {"city": "Wuxi", "countryName": "China"}]
    with _mock_urlopen(effects) as calls:
        assert geo.reverse_geocode(31.5778, 120.2983) == "Wuxi, China"
    assert calls["n"] == 2, "the blip should have been retried inside the same call"


def test_gives_up_after_all_attempts_without_caching():
    """A sustained outage still fails open (no location beats a wrong one), and must
    not poison the cache — the next post from that area retries."""
    geo._cache.clear()
    effects = [TimeoutError("down")] * geo._GEOCODE_ATTEMPTS
    with _mock_urlopen(effects) as calls:
        assert geo.reverse_geocode(30.25, 120.16) is None
    assert calls["n"] == geo._GEOCODE_ATTEMPTS, f"expected {geo._GEOCODE_ATTEMPTS} tries"
    assert not geo._cache, "a failure must never be cached"

    with _mock_urlopen([{"city": "Hangzhou", "countryName": "China"}]) as calls:
        assert geo.reverse_geocode(30.25, 120.16) == "Hangzhou, China"
    assert calls["n"] == 1


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            geo._cache.clear()
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback; print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
