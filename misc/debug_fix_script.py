#!/usr/bin/env python3
"""
Debug Script for Core System File Fixing
"""

import os
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("fix_debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Determine the core system directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_system_path = os.path.join(current_dir, "core_system")

        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Core system path: {core_system_path}")

        # Check if core system directory exists
        if not os.path.exists(core_system_path):
            logger.error(
                f"Core system directory not found: {core_system_path}"
            )
            return

        # List all Python files
        py_files = []
        for root, _, files in os.walk(core_system_path):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        logger.info(f"Found {len(py_files)} Python files")

        # Try to process a few files
        for file_path in py_files[
            :10
        ]:  # Limit to first 10 files for debugging
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Store content but use it for validation only
                    file_content = f.read()
                    # Validate the file can be read
                    if file_content:
                        pass
                logger.info(f"Successfully read file: {file_path}")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
