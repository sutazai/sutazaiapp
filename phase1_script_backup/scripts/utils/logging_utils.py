"""
Shared logging utilities to reduce duplication across scripts.
"""
import logging
from typing import Optional


def setup_logging(verbose: Optional[bool] = None, level: Optional[str] = None) -> None:
    """
    Configure root logger in a consistent format.

    - If level is provided (e.g., "INFO", "DEBUG"), use it.
    - Else, if verbose is provided, DEBUG when True else INFO.
    - Defaults to INFO.
    """
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    elif verbose is not None:
        log_level = logging.DEBUG if verbose else logging.INFO
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

