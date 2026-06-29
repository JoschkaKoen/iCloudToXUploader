"""
Offline tests for the iCloud request retry (`_icloud_retry`). No network. Proves a
transient "Request failed to iCloud" blip is retried in close succession (one try + two
retries) before giving up, while Reauth (needs 2FA) and throttling are re-raised at once
so the bot loop's own handlers take over instead of being retried here.

Run: .venv/bin/python tests/test_icloud_retry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.icloud_photos as ic  # noqa: E402
from ic2x.icloud_photos import PyiCloudThrottled, ReauthRequired, _icloud_retry  # noqa: E402

ic._ICLOUD_RETRY_DELAY = 0  # no real sleeps between retries in tests


def test_transient_retries_then_succeeds():
    n = {"c": 0}

    def fn():
        n["c"] += 1
        if n["c"] < 3:
            raise RuntimeError("Request failed to iCloud")
        return "ok"

    assert _icloud_retry(fn, "download") == "ok" and n["c"] == 3   # one try + two retries


def test_transient_gives_up_after_three():
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise RuntimeError("Request failed to iCloud")

    try:
        _icloud_retry(fn, "download")
        assert False, "should have raised after exhausting retries"
    except RuntimeError:
        pass
    assert n["c"] == 3


def test_reauth_is_not_retried():
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ReauthRequired("2fa due")

    try:
        _icloud_retry(fn, "download")
        assert False, "Reauth must propagate"
    except ReauthRequired:
        pass
    assert n["c"] == 1   # raised immediately, no retry → loop re-auths


def test_throttle_is_not_retried():
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise PyiCloudThrottled("429")

    try:
        _icloud_retry(fn, "download")
        assert False, "Throttle must propagate"
    except PyiCloudThrottled:
        pass
    assert n["c"] == 1   # raised immediately → loop backs off


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
