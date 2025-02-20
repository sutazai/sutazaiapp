import logging
from pathlib import Path
from .settings import settings


def setup_logging():
    """Set up logging configuration for the project."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "ai_system.log"),
            logging.StreamHandler()
        ]
    )
    # Suppress noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.INFO)


if __name__ == '__main__':
    setup_logging()
    logging.info("Logging is set up.") 