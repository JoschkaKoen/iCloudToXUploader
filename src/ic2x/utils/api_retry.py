"""Retry/backoff for transient model API errors."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("ic2x.api_retry")

_NON_RETRY_HTTP = frozenset({400, 401, 403, 404, 408, 413, 422})


class ContentFilterRefusal(Exception):
    """Raised when the model returns a policy / content_filter outcome (do not retry)."""


_NEVER_RETRY: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    json.JSONDecodeError,
    ContentFilterRefusal,
)


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, _NEVER_RETRY):
        return False

    try:
        from pydantic import ValidationError  # noqa: PLC0415

        if isinstance(exc, ValidationError):
            return False
    except ImportError:
        pass

    try:
        from openai import APIStatusError  # noqa: PLC0415

        if isinstance(exc, APIStatusError) and exc.status_code in _NON_RETRY_HTTP:
            return False
    except ImportError:
        pass

    return True


def _retry_max_attempts() -> int:
    raw = os.environ.get("IC2X_AI_RETRY_MAX_ATTEMPTS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 4


def _retry_base_sleep() -> float:
    raw = os.environ.get("IC2X_AI_RETRY_BASE_SLEEP", "").strip()
    try:
        return float(raw) if raw else 0.1
    except ValueError:
        return 0.1


def retry_api_call(
    fn: Callable[[], T],
    *,
    label: str,
    max_attempts: int | None = None,
    base_sleep: float | None = None,
    backoff_factor: float = 2.0,
    max_sleep: float = 5.0,
    jitter: float = 0.25,
    is_retryable: Callable[[BaseException], bool] = is_retryable_error,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call *fn* with exponential backoff on transient failures."""
    attempts = max_attempts if max_attempts is not None else _retry_max_attempts()
    base = base_sleep if base_sleep is not None else _retry_base_sleep()
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:
            if attempt >= attempts or not is_retryable(exc):
                suffix = ", no more retries" if attempt >= attempts else ""
                logger.warning(
                    "%s: API error (attempt %d/%d%s) — %s",
                    label, attempt, attempts, suffix, exc,
                )
                raise

            sleep_s = min(base * (backoff_factor ** (attempt - 1)), max_sleep)
            if jitter:
                sleep_s *= random.uniform(1.0 - jitter, 1.0 + jitter)
            logger.warning(
                "%s: API error (attempt %d/%d) — %s",
                label, attempt, attempts, exc,
            )
            logger.info("%s: retrying in %.2fs …", label, sleep_s)
            if on_retry is not None:
                on_retry(attempt, exc, sleep_s)
            time.sleep(sleep_s)
