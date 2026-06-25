#!/usr/bin/env python3
"""Speed-run the bot WITH real X posting.  ⚠ LOWER THE POSTS' VISIBILITY FIRST.

Posts to X for real, fast (no waits, no 5h interval or daily cap), using the real
state.db so posts are tracked (no duplicate re-posts later). Runs until Ctrl-C (or
the library / daily AI cap runs out). iCloud + AI calls are REAL and cost money.

    python3 wXtest_with_posting.py
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_PY = _ROOT / ".venv" / "bin" / "python3"
if not _VENV_PY.exists():
    _VENV_PY = _ROOT / ".venv" / "bin" / "python"
# Re-exec under the project's venv python (which has the dependencies) so plain
# `python3 wXtest_with_posting.py` works no matter which python3 is on PATH.
if _VENV_PY.exists() and Path(sys.prefix).resolve() != (_ROOT / ".venv").resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])

sys.path.insert(0, str(_ROOT / "src"))
from ic2x.bot import bot  # noqa: E402

bot(test=True, post=True, test_cycles=0)
