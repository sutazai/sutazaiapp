"""
A simple retry decorator module for SutazAI.

This module provides a decorator that wraps a function with retry logic.
"""

import time
from functools import wraps
from typing import Any, Callable


def retry(*, tries: int, delay: float) -> Callable:
    """
    A retry decorator that reattempts a function upon encountering an exception.

    Args:
        tries (int): Maximum number of attempts for the decorated function.
        delay (float): Time delay (in seconds) between each retry.

    Returns:
        Callable: The wrapped function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = tries
            last_exception = None
            while attempts > 0:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    attempts -= 1
                    if attempts == 0:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


if __name__ == "__main__":

    @retry(tries=3, delay=1)
    def test_func():
        print("Attempting function execution...")
        raise ValueError("Test error")

    try:
        test_func()
    except Exception as e:
        print("Function failed after all retries:", e)
