"""
Retry utilities — decorator and inline helper with configurable delay.

Usage:
    @with_retry(max_attempts=3, base_delay=0.1, backoff=1.0, label="my_call")
    def my_fn(...): ...

backoff=1.0 → fixed delay (default).
backoff=2.0 → exponential back-off.
"""

import time
import functools
import logging
from typing import Callable, Tuple, Type

logger = logging.getLogger("ic2x.retry")


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error("%s failed after %d attempts: %s", name, max_attempts, exc)
                        raise
                    exc_str = str(exc)
                    if len(exc_str) > 120:
                        exc_str = exc_str[:120] + " …"
                    logger.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %.1fs …",
                        name, attempt, max_attempts, exc_str, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


def retry_call(
    fn: Callable,
    *args,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
    **kwargs,
):
    decorated = with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff=backoff,
        exceptions=exceptions,
        label=label or fn.__name__,
    )(fn)
    return decorated(*args, **kwargs)
