import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict, Optional


class LoggingConfigurator:
    """
    Comprehensive logging configuration and management utility.
    Provides flexible, secure, and performance-optimized logging.
    """

    @staticmethod
    def configure_logging(
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_log_size_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        log_format: Optional[str] = None,
    ) -> Dict[str, logging.Logger]:
        """
        Configure comprehensive logging with multiple handlers.

        Args:
            log_dir (str): Directory to store log files
            log_level (str): Logging level
            max_log_size_bytes (int): Maximum log file size before rotation
            backup_count (int): Number of backup log files to keep
            log_format (Optional[str]): Custom log format

        Returns:
            Dict[str, logging.Logger]: Configured loggers
        """
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Default log format if not provided
        if log_format is None:
            log_format = (
                "%(asctime)s | %(levelname)8s | " "%(name)s:%(lineno)d | %(message)s"
            )

        # Create formatters
        formatter = logging.Formatter(log_format)

        # Logging levels mapping
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # Configure root logger
        logging.root.setLevel(level_map.get(log_level.upper(), logging.INFO))

        # Loggers dictionary
        loggers = {}

        # System loggers
        system_loggers = [
            "SutazAI.Core",
            "SutazAI.Security",
            "SutazAI.Performance",
            "SutazAI.Exceptions",
        ]

        for logger_name in system_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level_map.get(log_level.upper(), logging.INFO))

            # File Handler with Rotation
            log_file_path = os.path.join(
                log_dir,
                f"{logger_name.lower().replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log",
            )
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_log_size_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            loggers[logger_name] = logger

        return loggers

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Retrieve a configured logger.

        Args:
            name (str): Logger name

        Returns:
            logging.Logger: Configured logger
        """
        return logging.getLogger(name)

    @staticmethod
    def log_system_startup(logger: logging.Logger) -> None:
        """
        Log system startup information.

        Args:
            logger (logging.Logger): Logger to use
        """
        logger.info("SutazAI System Initializing")
        logger.info(f"Python Version: {__import__('sys').version}")
        logger.info(f"Startup Timestamp: {datetime.now().isoformat()}")

    @staticmethod
    def configure_exception_logging() -> None:
        """
        Configure global exception logging.
        """

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                logging.getLogger("SutazAI.Core").info("System Shutdown Requested")
                return

            logging.getLogger("SutazAI.Exceptions").critical(
                "Unhandled Exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

