"""
Storage initialization module.

This module handles storage initialization for the application.
"""

import os
import logging
from pathlib import Path
from .config import get_settings

# Get settings
settings = get_settings()

# Set up logging
logger = logging.getLogger("storage")


def init_storage():
    """
    Initialize storage directories for the application.

    This function creates all necessary directories for storing
    uploaded files, documents, and temporary files.
    """
    logger.info("Initializing storage directories...")

    # Create directories
    directories = [
        settings.UPLOAD_DIR,
        settings.DOCUMENT_STORE_PATH,
        os.path.join("data", "vectors"),
        os.path.join("data", "models"),
        os.path.join("data", "documents"),
        os.path.join("data", "qdrant"),
        os.path.join("storage"),
        os.path.join("outputs"),
        os.path.join("tmp", "document_parser"),
        os.path.join("workspace"),
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

    # Check write permissions
    for directory in directories:
        if not os.access(directory, os.W_OK):
            logger.warning(f"Directory {directory} is not writable")

    logger.info("Storage initialization complete")


def get_upload_path():
    """
    Get the upload directory path.

    Returns:
        str: Path to the upload directory
    """
    return settings.UPLOAD_DIR


def get_document_store_path():
    """
    Get the document store path.

    Returns:
        str: Path to the document store
    """
    return settings.DOCUMENT_STORE_PATH
