#!/usr/bin/env python3
"""ic2x — how to run.  Three commands, no flags:

    python3 Xup.py                  Normal bot (THIS file). Real X posting on the
                                    normal cadence (1 post every POST_INTERVAL_HOURS,
                                    capped at MAX_POSTS_PER_DAY). Long-running.

    python3 nXtest_no_posting.py    Speed-run test, NO posting. Full pipeline runs
                                    (iCloud → judge → dedup → color enhance) but
                                    nothing is tweeted; output → test_run/<timestamp>/.
                                    Fast (no waits, no interval/cap). Until Ctrl-C.

    python3 wXtest_with_posting.py  Speed-run test that POSTS to X for real (fast),
                                    on the real state.db so posts are tracked (no
                                    duplicate re-posts). ⚠ Lower the posts' visibility
                                    first. Until Ctrl-C.

All three run the real pipeline — iCloud + AI calls are ALWAYS real and cost money.
Stop any of them with Ctrl-C.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_PY = _ROOT / ".venv" / "bin" / "python3"
if not _VENV_PY.exists():
    _VENV_PY = _ROOT / ".venv" / "bin" / "python"
# Re-exec under the project's venv python (which has the dependencies) so plain
# `python3 Xup.py` works no matter which python3 is on PATH.
if _VENV_PY.exists() and Path(sys.prefix).resolve() != (_ROOT / ".venv").resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])

os.environ["X_DRY_RUN"] = "false"            # production: post for real
sys.path.insert(0, str(_ROOT / "src"))
from ic2x.bot import bot  # noqa: E402

bot()
