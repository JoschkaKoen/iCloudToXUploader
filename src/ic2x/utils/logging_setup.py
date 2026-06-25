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
    # Console shows ONLY warnings/errors from the logger — the clean ui.* trace
    # (console.print) carries the normal narrative. ic2x's own INFO lines
    # (post:/cycle:/rotation:) still go to the file, just not the terminal.
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[file_handler, console_handler],
    )
    # Silence noisy third-party libraries. Their per-request INFO chatter
    # (httpx "HTTP Request: POST … 200 OK", urllib3, the OpenAI/Gemini/DashScope
    # SDKs, pyicloud, PIL) otherwise drowns ic2x's own output in the terminal.
    # WARNING+ still surfaces real problems (throttling, auth failures).
    for _noisy in (
        "httpx", "httpcore", "urllib3", "requests",
        "openai", "google", "google_genai", "google.genai", "google.generativeai",
        "dashscope", "pyicloud", "PIL", "tweepy", "oauthlib", "requests_oauthlib",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Mirror ui.* console activity into the same log file so a background run is
    # fully reviewable. File-only (propagate=False → not re-printed to the terminal
    # by the RichHandler); ui.py writes the plain text of each console line here.
    clog = logging.getLogger("ic2x.console")
    clog.setLevel(logging.INFO)
    clog.propagate = False
    if file_handler not in clog.handlers:
        clog.addHandler(file_handler)
