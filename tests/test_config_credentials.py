"""
Offline tests for the soft-credential model (ic2x/config.py).

Proves load_config() no longer hard-fails without iCloud/X creds (so offline
commands + `ic2x cost` run with none), while require_credentials() still raises a
clear error for the groups a given command actually needs — and skips X creds in
dry-run.

Run: .venv/bin/python tests/test_config_credentials.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.config import load_config, require_credentials  # noqa: E402

_CRED_VARS = (
    "ICLOUD_USERNAME", "ICLOUD_PASSWORD", "TWITTER_CONSUMER_KEY",
    "TWITTER_CONSUMER_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET",
)


def _without_creds(fn):
    saved = {k: os.environ.pop(k, None) for k in _CRED_VARS}
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _cfg(**creds):
    base = dict(icloud_username="", icloud_password="", twitter_consumer_key="",
                twitter_consumer_secret="", twitter_access_token="",
                twitter_access_token_secret="")
    base.update(creds)
    return SimpleNamespace(**base)


def test_load_config_succeeds_without_creds():
    cfg = _without_creds(load_config)
    assert cfg.icloud_username == "" and cfg.twitter_consumer_key == ""
    # non-credential config still populated
    assert cfg.post_interval_hours and cfg.judge_model


def test_require_icloud_raises_when_missing():
    try:
        require_credentials(_cfg(), icloud=True, x=False)
    except ValueError as e:
        assert "ICLOUD_USERNAME" in str(e) and "ICLOUD_PASSWORD" in str(e)
    else:
        raise AssertionError("expected ValueError for missing iCloud creds")


def test_require_x_raises_when_missing():
    cfg = _cfg(icloud_username="u", icloud_password="p")
    try:
        require_credentials(cfg, icloud=True, x=True)
    except ValueError as e:
        assert "TWITTER_CONSUMER_KEY" in str(e)
    else:
        raise AssertionError("expected ValueError for missing X creds")


def test_dry_run_skips_x_creds():
    # iCloud present, X absent, x=False (dry-run) → must NOT raise
    cfg = _cfg(icloud_username="u", icloud_password="p")
    require_credentials(cfg, icloud=True, x=False)


def test_all_present_passes():
    cfg = _cfg(icloud_username="u", icloud_password="p", twitter_consumer_key="a",
               twitter_consumer_secret="b", twitter_access_token="c",
               twitter_access_token_secret="d")
    require_credentials(cfg, icloud=True, x=True)


def test_icloud_build_guard_raises_clear_error_without_creds():
    # The iCloud session build must give a clear creds error BEFORE importing pyicloud,
    # so it works even where pyicloud isn't installed (this test never touches it).
    from ic2x.icloud_photos import ICloudPhotos, ReauthRequired
    cfg = SimpleNamespace(icloud_username="", icloud_password="",
                          icloud_with_family=False, icloud_family_override=False)
    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as e:
        assert "ICLOUD_USERNAME" in str(e)
    else:
        raise AssertionError("expected ReauthRequired when iCloud creds are empty")


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
