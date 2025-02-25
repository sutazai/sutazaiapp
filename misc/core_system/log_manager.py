import logging
import logging.handlers
import os


class LogManager:
    """Manages logging configuration and setup for the application."""

    def __init__(self):
        self.log_dir = "/var/log/sutazai"
        self.max_size = 10485760  # 10MB
        self.backup_count = 5

    def setup_logging(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.handlers.RotatingFileHandler(
                    os.path.join(self.log_dir, "sutazai.log"),
                    maxBytes=self.max_size,
                    backupCount=self.backup_count,
                ),
                logging.StreamHandler(),
            ],
        )
