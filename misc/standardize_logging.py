#!/usr/bin/env python3
"""
SutazAI Logging Configuration Standardizer

This script identifies and standardizes logging configurations throughout
the codebase to ensure consistent logging behavior and reduce conflicts.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Pattern, Set

# Logging import patterns to detect
LOGGING_IMPORT_PATTERNS = [
    re.compile(r"import\s+logging"),
    re.compile(r"from\s+logging\s+import"),
    re.compile(r"from\s+loguru\s+import\s+logger"),
    re.compile(r"import\s+loguru"),
]

# Logging config patterns to detect and replace
LOGGING_CONFIG_PATTERNS = [
    # Basic logging.basicConfig pattern
    (
        re.compile(r"logging\.basicConfig\s*\(\s*.*?\s*\)", re.DOTALL),
        "logging.basicConfig(\n"
        "    level=logging.INFO,\n"
        '    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",\n'
        "    handlers=[\n"
        "        logging.StreamHandler(),\n"
        '        logging.FileHandler(os.path.join(os.getenv("SUTAZAI_LOG_DIR", "/opt/SutazAI/logs"), "app.log"))\n'
        "    ]\n"
        ")",
    ),
    # Loguru logger.add pattern
    (
        re.compile(r"logger\.add\s*\(\s*.*?\s*\)", re.DOTALL),
        "logger.add(\n"
        '    os.path.join(os.getenv("SUTAZAI_LOG_DIR", "/opt/SutazAI/logs"), "{}_{{time}}.log".format(os.path.basename(__file__))),\n'
        '    rotation="10 MB",\n'
        '    level="INFO"\n'
        ")",
    ),
]

# Standard imports to add if logging is used but imports are missing
STANDARD_LOGGING_IMPORTS = {
    "logging": "import logging\nimport os",
    "loguru": "from loguru import logger\nimport os",
}


class LoggingStandardizer:
    """Standardizes logging configurations throughout the codebase."""

    def __init__(self, root_dir: str):
        """
        Initialize the logging standardizer.

        Args:
            root_dir: Root directory to process
        """
        self.root_dir = root_dir
        self.python_files: List[str] = []
        self.files_processed: int = 0
        self.files_modified: int = 0
        self.files_with_logging: Set[str] = set()
        self.logger_types: Dict[str, str] = (
            {}
        )  # file -> logger type (logging/loguru)

    def find_python_files(self) -> None:
        """Find all Python files in the project."""
        for root, _, files in os.walk(self.root_dir):
            # Skip virtual environments, .git, etc.
            if any(
                part
                in [
                    ".venv",
                    "venv",
                    ".git",
                    ".pytest_cache",
                    "__pycache__",
                    ".mypy_cache",
                ]
                for part in Path(root).parts
            ):
                continue

            for file in files:
                if file.endswith(".py"):
                    self.python_files.append(os.path.join(root, file))

        print(f"Found {len(self.python_files)} Python files to process.")

    def analyze_logging_usage(self) -> None:
        """Analyze how logging is used across the codebase."""
        for file_path in self.python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for logging imports
                has_logging = False
                for pattern in LOGGING_IMPORT_PATTERNS:
                    if pattern.search(content):
                        has_logging = True

                        # Determine logger type (logging or loguru)
                        if "loguru" in pattern.pattern:
                            self.logger_types[file_path] = "loguru"
                        else:
                            self.logger_types[file_path] = "logging"

                if has_logging:
                    self.files_with_logging.add(file_path)

            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {str(e)}")

        print(f"Found {len(self.files_with_logging)} files using logging:")
        print(
            f"  - Standard logging: {sum(1 for v in self.logger_types.values() if v == 'logging')}"
        )
        print(
            f"  - Loguru: {sum(1 for v in self.logger_types.values() if v == 'loguru')}"
        )

    def process_files(self) -> None:
        """Process files to standardize logging configurations."""
        for file_path in self.files_with_logging:
            self.files_processed += 1
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content
                logger_type = self.logger_types.get(file_path)

                # Add missing imports if needed
                if "import os" not in content and logger_type:
                    # Find the position after imports
                    import_section_end = 0
                    for match in re.finditer(
                        r"(^import .*$|^from .* import .*$)",
                        content,
                        re.MULTILINE,
                    ):
                        if match.end() > import_section_end:
                            import_section_end = match.end()

                    if import_section_end > 0:
                        # Insert 'import os' after the last import
                        content = (
                            content[:import_section_end]
                            + "\nimport os"
                            + content[import_section_end:]
                        )

                # Replace logging configurations
                for pattern, replacement in LOGGING_CONFIG_PATTERNS:
                    if (
                        logger_type == "logging"
                        and "logging.basicConfig" in pattern.pattern
                    ):
                        content = pattern.sub(replacement, content)
                    elif (
                        logger_type == "loguru"
                        and "logger.add" in pattern.pattern
                    ):
                        content = pattern.sub(replacement, content)

                # Write changes if content was modified
                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.files_modified += 1
                    print(
                        f"✅ Updated logging in {os.path.relpath(file_path, self.root_dir)}"
                    )

            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")

    def create_logs_directory(self) -> None:
        """Ensure the logs directory exists."""
        logs_dir = os.path.join(self.root_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"✅ Created logs directory: {logs_dir}")
        else:
            print(f"ℹ️ Logs directory already exists: {logs_dir}")

    def print_summary(self) -> None:
        """Print summary of changes made."""
        print("\n=== Logging Standardization Summary ===")
        print(f"Files processed: {self.files_processed}")
        print(f"Files modified: {self.files_modified}")
        print(f"Files with logging: {len(self.files_with_logging)}")


def main():
    """Main function to run the logging standardizer."""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Standardizing logging configurations in: {root_dir}")
    standardizer = LoggingStandardizer(root_dir)

    standardizer.find_python_files()
    standardizer.analyze_logging_usage()

    # Create logs directory if it doesn't exist
    standardizer.create_logs_directory()

    # Process files to standardize logging
    standardizer.process_files()
    standardizer.print_summary()

    print("\nLogging standardization complete! 🎉")
    print("Note: All logs will now be stored in /opt/SutazAI/logs")


if __name__ == "__main__":
    main()
