"""
Configure root logging once per process: file handler under logs/YYYY-MM-DD.log
plus a Rich console handler. Idempotent — calling twice is a no-op so daemon
ticks and CLI dispatch can both invoke it safely.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(logs_dir: Path) -> None:
    if logging.getLogger().handlers:
        return  # already configured — don't leak a second FileHandler
    from rich.logging import RichHandler

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)s  %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            file_handler,
            RichHandler(rich_tracebacks=True, show_path=False),
        ],
    )
