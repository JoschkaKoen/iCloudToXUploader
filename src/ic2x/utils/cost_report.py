"""Estimate API cost from accumulated token usage and a pricing table."""

from __future__ import annotations

import json
import os
from pathlib import Path

_PRICING_CACHE: dict[str, tuple[float, float]] | None = None
_PRICING_CURRENCY = "USD"


def _default_pricing_path() -> Path:
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "data" / "ai_pricing_per_mtoken.json"
        if cand.is_file():
            return cand
    return Path.cwd() / "data" / "ai_pricing_per_mtoken.json"


def _load_pricing_json(path: Path) -> dict[str, tuple[float, float]]:
    global _PRICING_CURRENCY
    result: dict[str, tuple[float, float]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _PRICING_CURRENCY = str(data.get("currency", "USD"))
        models = data.get("models", {})
        for model, rates in models.items():
            if not isinstance(rates, dict):
                continue
            inp = float(rates.get("input_per_million", 0) or 0)
            out = float(rates.get("output_per_million", 0) or 0)
            result[str(model)] = (inp, out)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return result


def _load_pricing_xlsx(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    try:
        import openpyxl  # noqa: PLC0415

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = iter(ws.rows)
        headers = [str(c.value).strip() if c.value else "" for c in next(rows)]
        model_col = next((i for i, h in enumerate(headers) if "model" in h.lower()), None)
        inp_col = next((i for i, h in enumerate(headers) if "input" in h.lower()), None)
        out_col = next((i for i, h in enumerate(headers) if "output" in h.lower()), None)
        if None not in (model_col, inp_col, out_col):
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
        wb.close()
    except Exception:
        pass
    return result


def load_pricing() -> dict[str, tuple[float, float]]:
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE

    override = os.environ.get("IC2X_AI_PRICING_JSON", "").strip()
    path = Path(override).expanduser() if override else _default_pricing_path()
    if path.suffix.lower() == ".xlsx":
        _PRICING_CACHE = _load_pricing_xlsx(path)
    else:
        _PRICING_CACHE = _load_pricing_json(path)
    return _PRICING_CACHE


def pricing_currency() -> str:
    load_pricing()
    return _PRICING_CURRENCY


def compute_cost(
    usage: dict[str, dict[str, int]],
) -> tuple[float, dict[str, dict]]:
    """Return (total_cost, per_model rows). Rates are per 1M tokens in pricing file currency."""
    pricing = load_pricing()
    breakdown: dict[str, dict] = {}
    total = 0.0
    for model, counts in usage.items():
        inp_rate, out_rate = pricing.get(model, (0.0, 0.0))
        cost = (
            counts["input"] / 1_000_000 * inp_rate
            + counts["output"] / 1_000_000 * out_rate
        )
        total += cost
        breakdown[model] = {
            "input_tokens": counts["input"],
            "output_tokens": counts["output"],
            "thinking_tokens": counts.get("thinking", 0),
            "cost": round(cost, 6),
        }
    return round(total, 6), breakdown


def format_usage_log(
    usage: dict[str, dict[str, int]],
    *,
    cost_line: str | None = None,
) -> str:
    parts: list[str] = []
    for model, u in sorted(usage.items()):
        parts.append(
            f"{model}: in={u['input']} out={u['output']} think={u.get('thinking', 0)}"
        )
    body = "; ".join(parts) if parts else "(no token usage recorded)"
    if cost_line:
        return f"{body} | {cost_line}"
    return body
