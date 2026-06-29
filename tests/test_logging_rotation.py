"""
Offline test for DatedFileHandler — the daily-rolling log file handler. Proves it
writes to logs/<UTC-date>.log and rolls to a new dated file when the UTC day
changes (the bug being a weeks-long run piling everything into the start-date file).

Run: .venv/bin/python tests/test_logging_rotation.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.utils.logging_setup as ls  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_logroll_"))


def _emit(handler, msg):
    handler.emit(logging.makeLogRecord({"msg": msg, "levelno": logging.INFO, "levelname": "INFO"}))
    handler.flush()


def test_rolls_to_new_dated_file_on_utc_day_change():
    logs = _TMP / "logs"
    logs.mkdir()
    orig = ls.DatedFileHandler._utc_day
    clock = {"day": "2026-06-27"}
    ls.DatedFileHandler._utc_day = staticmethod(lambda: clock["day"])
    try:
        h = ls.DatedFileHandler(logs)
        h.setFormatter(logging.Formatter("%(message)s"))
        _emit(h, "day-one-line")
        assert (logs / "2026-06-27.log").exists()

        clock["day"] = "2026-06-28"          # cross UTC midnight
        _emit(h, "day-two-line")
        h.close()
    finally:
        ls.DatedFileHandler._utc_day = orig

    f1, f2 = logs / "2026-06-27.log", logs / "2026-06-28.log"
    assert f1.exists() and f2.exists(), "should have one file per UTC day"
    assert "day-one-line" in f1.read_text() and "day-one-line" not in f2.read_text()
    assert "day-two-line" in f2.read_text() and "day-two-line" not in f1.read_text()


def test_same_day_appends_one_file():
    logs = _TMP / "logs2"
    logs.mkdir()
    orig = ls.DatedFileHandler._utc_day
    ls.DatedFileHandler._utc_day = staticmethod(lambda: "2026-01-01")
    try:
        h = ls.DatedFileHandler(logs)
        h.setFormatter(logging.Formatter("%(message)s"))
        _emit(h, "a")
        _emit(h, "b")
        h.close()
    finally:
        ls.DatedFileHandler._utc_day = orig
    files = list(logs.glob("*.log"))
    assert len(files) == 1 and files[0].name == "2026-01-01.log"
    assert files[0].read_text().count("\n") == 2


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
