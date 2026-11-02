# Copyright (c) 2025 The AutoDialect Authors
# SPDX-License-Identifier: MIT

import random
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """
    Decorator that adds retry logic with randomized backoff.

    Args:
        func: The function to decorate
        provider_name: The name of the model provider being called
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries:
                    # Last attempt, re-raise the exception
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                print(
                    f"{provider_name} API call failed: {this_error_message}. Will sleep for {sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )
                # Randomly sleep for 3-30 seconds
                time.sleep(sleep_time)

        # This should never be reached, but just in case
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper


def async_retry_with(
    func: Callable[..., Any],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
    min_sleep_seconds: int = 3,
    max_sleep_seconds: int = 30,
) -> Callable[..., Any]:
    """
    Async variant of retry_with for async provider calls.

    `max_retries` is retry count, matching retry_with: total attempts are
    max_retries + 1.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        import asyncio

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries:
                    raise

                sleep_time = random.randint(min_sleep_seconds, max_sleep_seconds)
                this_error_message = str(e)
                print(
                    f"{provider_name} API call failed: {this_error_message}. "
                    f"Will sleep for {sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )
                await asyncio.sleep(sleep_time)

        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper
