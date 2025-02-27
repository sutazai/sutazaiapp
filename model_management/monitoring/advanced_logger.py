import logging

logger = logging.getLogger("advanced_logger")


def log_info(message: str):
    logger.info(message)


def log_error(message: str):
    logger.error(message)
