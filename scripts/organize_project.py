#!/usr/bin/env python3
"""
Automatic Project Organizer for SutazAI

This script checks the project root and subdirectories for files and folders that are not part of the expected structure.
Any extraneous files found in the root will be moved to a 'misc' directory for further inspection.

Expected structure at project root:
Directories: ai_agents, model_management, backend, web_ui, scripts, packages, logs, doc_data, venv
Files: README.md, requirements.txt

Usage:
python3 scripts/organize_project.py

Logs operations to logs/organize.log
"""

import logging
import os
import shutil

# Configure logging
LOG_DIR = os.path.join("logs")
    if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    LOG_FILE = os.path.join(LOG_DIR, "organize.log")
    
    logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Define the expected structure in the project root
EXPECTED_DIRECTORIES = {
"ai_agents",
"model_management",
"backend",
"web_ui",
"scripts",
"packages",
"logs",
"doc_data",
"venv",
}
EXPECTED_FILES = {"README.md", "requirements.txt"}

# Directory to move unexpected files
MISC_DIR = "misc"


    def organize_root():
    logging.info("Starting organization of project root...")
    # Create misc directory if it doesn't exist
        if not os.path.exists(MISC_DIR):
        os.makedirs(MISC_DIR)
        logging.info(f"Created misc directory: {MISC_DIR}")
        
        # List items in root
        items = os.listdir(".")
            for item in items:
            # Skip expected directories and files
                if item in EXPECTED_DIRECTORIES or item in EXPECTED_FILES or item == MISC_DIR:
            continue
            # Also skip hidden files and directories
                if item.startswith("."):
            continue
            # If item is a file or directory and not expected, move to misc
                try:
                dest = os.path.join(MISC_DIR, item)
                shutil.move(item, dest)
                logging.info(f"Moved '{item}' to '{dest}'")
                except Exception as e:
                logging.exception(f"Error moving {item} to {MISC_DIR}: {e}")
                
                
                    if __name__ == "__main__":
                    logging.info("Starting automatic organization of project files...")
                    organize_root()
                    logging.info("Organization completed. Check logs/organize.log for details.")
                    