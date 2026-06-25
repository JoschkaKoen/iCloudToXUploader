#!/usr/bin/env python3
"""Speed-run the bot WITHOUT posting to X.

The full pipeline runs for real (iCloud download → judge → dedup/grouping → color
enhance), but nothing is posted: decisions + output land in test_run/<timestamp>/,
the real state.db is untouched, and it's re-runnable. Fast (no waits, no 5h interval
or daily cap). iCloud + AI calls are REAL and cost money. Ctrl-C to stop.

    python3 nXtest_no_posting.py
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_PY = _ROOT / ".venv" / "bin" / "python3"
if not _VENV_PY.exists():
    _VENV_PY = _ROOT / ".venv" / "bin" / "python"
# Re-exec under the project's venv python (which has the dependencies) so plain
# `python3 nXtest_no_posting.py` works no matter which python3 is on PATH.
if _VENV_PY.exists() and Path(sys.prefix).resolve() != (_ROOT / ".venv").resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])

sys.path.insert(0, str(_ROOT / "src"))
from ic2x.bot import bot  # noqa: E402

bot(test=True, post=False, test_cycles=0)
