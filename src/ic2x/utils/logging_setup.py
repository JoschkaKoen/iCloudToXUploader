"""
Configure root logging once per process: a daily-rolling file handler writing to
logs/YYYY-MM-DD.log (UTC) plus a Rich console handler. Idempotent — calling twice
is a no-op so daemon ticks and CLI dispatch can both invoke it safely.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path


class DatedFileHandler(logging.FileHandler):
    """Writes to ``logs/<UTC-date>.log`` and rolls to a fresh dated file when the UTC
    day changes. The previous handler created one file named for the process START
    date and appended forever, so a bot running for weeks piled every day's logs into
    day-1's file (and the filename's date became a lie). This keeps one file per UTC
    day with no external rotation config."""

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._day = self._utc_day()
        super().__init__(self._path(self._day), encoding="utf-8")

    @staticmethod
    def _utc_day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _path(self, day: str) -> str:
        return str(self._logs_dir / f"{day}.log")

    def emit(self, record: logging.LogRecord) -> None:
        day = self._utc_day()
        if day != self._day:               # crossed UTC midnight → reopen a new file
            self._day = day
            self.baseFilename = os.path.abspath(self._path(day))
            if self.stream:
                self.stream.close()
                self.stream = self._open()
        super().emit(record)


def setup_logging(logs_dir: Path) -> None:
    if logging.getLogger().handlers:
        return  # already configured — don't leak a second FileHandler
    from rich.logging import RichHandler

    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = DatedFileHandler(logs_dir)
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
