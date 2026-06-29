#!/usr/bin/env python3
"""
Run the whole offline test suite with NO third-party deps (no pytest needed).

Each tests/test_*.py is a standalone script that exits non-zero on failure; this
runs them one per subprocess (isolated module state + a clean process per file)
and prints a single PASS/FAIL summary. The worktree's src/ is forced onto
PYTHONPATH so the tests exercise THIS checkout, not any pip-installed ic2x.

    python tests/run_all.py            # run all
    python tests/run_all.py burst cost # run only files matching these substrings

Equivalent to `pytest` if you have it (uv sync --extra dev); this is the
dependency-free fallback that matches the repo's standalone-script convention.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_SRC = _TESTS.parent / "src"


def main(argv: list[str]) -> int:
    files = sorted(p for p in _TESTS.glob("test_*.py"))
    if argv:
        files = [p for p in files if any(a in p.name for a in argv)]
    if not files:
        print("no matching test files")
        return 1

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    passed, failed = [], []
    for f in files:
        print(f"\n{'=' * 12} {f.name} {'=' * 12}")
        rc = subprocess.run([sys.executable, str(f)], env=env).returncode
        (passed if rc == 0 else failed).append(f.name)

    print("\n" + "=" * 40)
    print(f"  {len(passed)}/{len(files)} files passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
