"""
Offline tests for surviving an expired iCloud 2FA session. No network, no iCloud.

Regression guard for the 2026-07-20 outage: the bot went down for 15 days after a
routine ~30-day session expiry. The loop's own reauth handler is built to stay alive
and poll so a later `ic2x login` revives it — but repeated errors escalated to the
errors>=6 re-exec, and the fresh process hit a startup preflight that returned on
ReauthRequired. A recoverable stall became a permanent death.

Proves: (1) a continuous run whose startup session is dead enters the poll loop
instead of exiting, (2) `--once` still fails fast with one clear line, (3) while 2FA
is pending the loop NEVER re-execs, and (4) it recovers on its own once the session
comes back (i.e. after the user runs `ic2x login`).

Run: .venv/bin/python tests/test_reauth_survival.py   (or via pytest)
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Real Config via load_config() needs credential env vars; no network is touched
# (every client is monkeypatched below). Mirrors tests/conftest.py so this file also
# runs standalone. setdefault means a real .env / environment always wins.
for _k, _v in {
    "ICLOUD_USERNAME": "test@example.com", "ICLOUD_PASSWORD": "test-password",
    "TWITTER_CONSUMER_KEY": "test-ck", "TWITTER_CONSUMER_SECRET": "test-cs",
    "TWITTER_ACCESS_TOKEN": "test-at", "TWITTER_ACCESS_TOKEN_SECRET": "test-ats",
}.items():
    os.environ.setdefault(_k, _v)

import ic2x.bot as bot  # noqa: E402
from ic2x.icloud_photos import ReauthRequired  # noqa: E402

_REAUTH = "interactive 2FA required — run `ic2x login`"


class ReauthIC:
    """iCloud stub: the first `ok_before` ensure_session() calls succeed, the next
    `fail_times` need interactive 2FA, and anything after that is healthy again —
    enough to model both "dead at startup" and "died mid-run, revived by `ic2x login`"."""

    def __init__(self, fail_times: int = 10**9, ok_before: int = 0):
        self.fail_times = fail_times
        self.ok_before = ok_before
        self.calls = 0

    def ensure_session(self):
        self.calls += 1
        if self.ok_before < self.calls <= self.ok_before + self.fail_times:
            raise ReauthRequired(_REAUTH)

    def screenshot_ids(self):
        return set()


class _Harness:
    """Runs bot() offline against a throwaway state dir, stopping the loop after a
    bounded number of sleeps so a poll loop is testable without waiting on wall time."""

    def __init__(self, ic, cycle_exc, max_sleeps: int = 9):
        self.ic = ic
        self.cycle_exc = cycle_exc          # what run_one_cycle raises (None → posts)
        self.max_sleeps = max_sleeps
        self.sleeps = 0
        self.cycles = 0
        self.notified = 0
        self.execv_calls = 0
        self.tmp = Path(tempfile.mkdtemp(prefix="ic2x_reauth_"))

    def _sleep(self, seconds, heartbeat=None, heartbeat_every=300.0):
        self.sleeps += 1
        if self.sleeps >= self.max_sleeps:
            bot._stop = True                # end the loop deterministically

    def _cycle(self, db, cfg, ic, clients):
        self.cycles += 1
        if self.cycle_exc is not None:
            raise self.cycle_exc
        return "posted"

    def run(self, **kwargs) -> None:
        cfg = bot.load_config()
        cfg.test_mode = False               # a REAL continuous run, not a soak
        cfg.post_window = ""                # this test is about REAUTH, not the clock:
                                            # leaving POST_WINDOW set made it pass inside
                                            # the window and fail outside it
        cfg.x_dry_run = True                # no X creds needed
        cfg.color_enhance_enabled = False
        cfg.reconcile_on_startup = False
        cfg.db_path = self.tmp / "state.db"
        cfg.logs_dir = self.tmp / "logs"
        for name in ("work", "queue", "approved", "posted", "scene_thumbs", "reviewed"):
            setattr(cfg, f"{name}_dir", self.tmp / name)

        stop_before = bot._stop
        bot._stop = False
        sigint, sigterm = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        orig = (bot.load_config, bot.require_vision_api_credentials, bot.ICloudPhotos,
                bot.make_clients, bot.run_one_cycle, bot.flush_pending,
                bot._sleep_interruptible, bot._notify_reauth, os.execv)

        def _notify(cfg_, exc):
            self.notified += 1

        def _execv(path, argv):
            self.execv_calls += 1
            raise AssertionError("bot re-execed while 2FA was pending")

        bot.load_config = lambda: cfg
        bot.require_vision_api_credentials = lambda *a, **k: None
        bot.ICloudPhotos = lambda c: self.ic
        bot.make_clients = lambda c: (None, None)
        bot.run_one_cycle = self._cycle
        bot.flush_pending = lambda db, cfg_, clients: False
        bot._sleep_interruptible = self._sleep
        bot._notify_reauth = _notify
        os.execv = _execv
        try:
            bot.bot(**kwargs)
        finally:
            (bot.load_config, bot.require_vision_api_credentials, bot.ICloudPhotos,
             bot.make_clients, bot.run_one_cycle, bot.flush_pending,
             bot._sleep_interruptible, bot._notify_reauth, os.execv) = orig
            signal.signal(signal.SIGINT, sigint)
            signal.signal(signal.SIGTERM, sigterm)
            bot._stop = stop_before
            shutil.rmtree(self.tmp, ignore_errors=True)


def test_no_session_raises_reauth_not_a_bare_runtimeerror():
    """A failed ensure_session() leaves _api None. The next photo access must report
    that as ReauthRequired so the loop's notify-and-poll handles it — as a bare
    RuntimeError it fell through to the generic error path, counted toward the
    errors>=6 re-exec, and the bot restarted 14 times (2026-09-06)."""
    from ic2x.icloud_photos import ICloudPhotos

    ic = ICloudPhotos.__new__(ICloudPhotos)   # never authenticated
    ic._api = None
    try:
        _ = ic._photos
    except ReauthRequired:
        pass
    except Exception as exc:                                        # noqa: BLE001
        raise AssertionError(
            f"no-session raised {type(exc).__name__}, not ReauthRequired — the loop "
            "will treat it as a generic error and escalate to the re-exec") from exc
    else:
        raise AssertionError("accessing photos without a session should raise")


def test_startup_reauth_does_not_kill_a_continuous_run():
    # session dead at startup AND every cycle — the bot must poll, not exit.
    h = _Harness(ReauthIC(), cycle_exc=ReauthRequired(_REAUTH))
    h.run()
    assert h.sleeps > 0, "bot exited at startup instead of polling for `ic2x login`"
    assert h.notified == 1, f"expected exactly one reauth notice, got {h.notified}"


def test_once_still_fails_fast():
    # `--once` is a one-shot command: it must still exit with one clear line and
    # never start a cycle against a dead session.
    h = _Harness(ReauthIC(), cycle_exc=ReauthRequired(_REAUTH))
    h.run(once=True)
    assert h.cycles == 0, "--once ran a cycle despite a dead session"
    assert h.sleeps == 0, "--once entered a poll loop instead of exiting"
    assert h.notified == 1


def test_pending_2fa_never_re_execs():
    # THE regression, in the exact shape of 2026-07-19: the bot starts healthy, then
    # the session expires mid-run and iCloud answers with generic "Request failed to
    # iCloud" errors rather than a classified ReauthRequired. Those used to pile up to
    # 6 and re-exec, dropping the loop into the fatal startup path. With 2FA pending,
    # escalation must not happen no matter how many errors accumulate.
    h = _Harness(ReauthIC(ok_before=1), cycle_exc=RuntimeError("Request failed to iCloud"),
                 max_sleeps=12)
    h.run()
    assert h.execv_calls == 0, "re-execed while 2FA was pending — this is the 15-day outage"
    assert h.cycles >= 7, f"needed >6 errors to exercise the escalation, got {h.cycles}"
    assert h.notified == 1


def test_recovers_once_login_restores_the_session():
    # Session dead for the first 2 ensure_session() calls, then healthy — the shape of
    # a user running `ic2x login` while the bot polls. It must resume and post.
    ic = ReauthIC(fail_times=2)
    posted = {"n": 0}

    class _Recovering(_Harness):
        def _cycle(self, db, cfg, ic_, clients):
            self.cycles += 1
            if ic.calls <= ic.fail_times:
                raise ReauthRequired(_REAUTH)
            posted["n"] += 1
            return "posted"

    h = _Recovering(ic, cycle_exc=None, max_sleeps=6)
    h.run()
    assert posted["n"] > 0, "bot never resumed posting after the session came back"


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
