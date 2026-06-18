"""Append one JSON line per decision to logs/YYYY-MM-DD.jsonl — the burn-in
audit trail for the autonomous bot."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ic2x.decision_log")


def log_decision(logs_dir: Path, *, outcome: str, asset_id: str | None = None,
                 detail: Any = None) -> None:
    now = datetime.now(timezone.utc)
    rec = {"ts": now.isoformat(), "outcome": outcome, "asset_id": asset_id, "detail": detail}
    try:
        path = logs_dir / f"{now:%Y-%m-%d}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 — logging must never crash the loop
        logger.warning("decision_log: could not write: %s", exc)
