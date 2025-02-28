#!/usr/bin/env python3.11
import logging
from typing import Optional

logger = logging.getLogger("advanced_logger")


def log_info(message: str) -> None:    """Log an info message.

    Args:    message: The message to log
    """


logger.info(message)


def log_error(message: str) -> None:    """Log an error message.

    Args:    message: The message to log
    """


logger.error(message)


def log_warning(message: str) -> None:    """Log a warning message.

    Args:    message: The message to log
    """


logger.warning(message)

def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
    ) -> None:    """Setup logging configuration.

    Args:    log_level: The logging level to use
    log_file: Optional file path to write logs to
    """
handlers = [logging.StreamHandler()]
if log_file:        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
