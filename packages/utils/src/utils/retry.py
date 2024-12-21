import logging
from collections.abc import Callable
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: float | None = None,
    min_seconds: float = 4,
    max_seconds: float = 60,
) -> Callable[[Any], Any]:
    "Tenacity retry decorator."
    logger = logging.getLogger(__name__)

    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=retry_if_exception_type(),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
