#!/usr/bin/env python3
"""Export AI API costs from eXercise's spreadsheet to ic2x JSON.

Reads the same columns as xScore cost_report._load_pricing (model, input, output).

Usage:
  python scripts/export_pricing_from_exercise_xlsx.py [path/to/AI API costs.xlsx]

Paths (first match wins):
  1. CLI argument
  2. Env IC2X_AI_PRICING_SOURCE_XLSX
  3. Env EXERCISE_REPO_ROOT / \"AI API costs.xlsx\"
  4. Sibling repo: <this-repo>/../eXercise/AI API costs.xlsx

Requires: pip install openpyxl  (or: uv run --with openpyxl python scripts/...)

Duplicate model ids in the sheet: last row wins (same as typical last-write semantics).
Exit: 0 on success, 1 on error (missing file, no rows, openpyxl import error).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _resolve_xlsx_path() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    env = os.environ.get("IC2X_AI_PRICING_SOURCE_XLSX", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = os.environ.get("EXERCISE_REPO_ROOT", "").strip()
    if root:
        p = Path(root).expanduser() / "AI API costs.xlsx"
        if p.is_file():
            return p.resolve()
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    sibling = repo_root.parent / "eXercise" / "AI API costs.xlsx"
    if sibling.is_file():
        return sibling.resolve()
    return None


def _load_xlsx_rows(path: Path) -> dict[str, tuple[float, float]]:
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError:
        print("error: openpyxl not installed. Try: pip install openpyxl", file=sys.stderr)
        raise SystemExit(1) from None

    result: dict[str, tuple[float, float]] = {}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = iter(ws.rows)
        headers = [str(c.value).strip() if c.value else "" for c in next(rows)]
        model_col = next((i for i, h in enumerate(headers) if "model" in h.lower()), None)
        inp_col = next((i for i, h in enumerate(headers) if "input" in h.lower()), None)
        out_col = next((i for i, h in enumerate(headers) if "output" in h.lower()), None)
        if None in (model_col, inp_col, out_col):
            print("error: xlsx missing model/input/output columns", file=sys.stderr)
            raise SystemExit(1)
        n = 0
        for row in rows:
            model = str(row[model_col].value or "").strip()
            if not model:
                continue
            try:
                inp = float(row[inp_col].value or 0)
                out = float(row[out_col].value or 0)
            except (TypeError, ValueError):
                inp, out = 0.0, 0.0
            result[model] = (inp, out)
            n += 1
        if n == 0:
            print("error: no data rows in spreadsheet", file=sys.stderr)
            raise SystemExit(1)
    finally:
        wb.close()
    return result


def main() -> None:
    xlsx = _resolve_xlsx_path()
    if xlsx is None or not xlsx.is_file():
        print(
            "error: AI API costs.xlsx not found. Set IC2X_AI_PRICING_SOURCE_XLSX "
            "or pass path as argv[1].",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows = _load_xlsx_rows(xlsx)
    out_path = Path(__file__).resolve().parent.parent / "data" / "ai_pricing_per_mtoken.json"
    models: dict[str, dict[str, float]] = {}
    for name, (inp, out) in sorted(rows.items()):
        models[name] = {"input_per_million": inp, "output_per_million": out}

    payload = {
        "currency": "RMB",
        "note": "Generated from eXercise AI API costs.xlsx — rates per 1M tokens (RMB). "
        "Regenerate: python scripts/export_pricing_from_exercise_xlsx.py",
        "source_xlsx": str(xlsx),
        "models": models,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(models)} models) from {xlsx}")


if __name__ == "__main__":
    main()
