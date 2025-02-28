#!/usr/bin/env python3.11
"""
Core System Fixer for SutazAI

This script scans all Python files under the core_system directory for syntax errors.
For any file that fails to compile, it creates a backup (.bak) and replaces the file with a stub.
The stub contains a module docstring and a basic main() function.
"""

import ast
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
    logging.FileHandler("/opt/sutazaiapp/logs/core_system_fix.log"),
    logging.StreamHandler(),
],
)
logger = logging.getLogger(__name__)

class CoreSystemFixer:    """Core system fixer utility."""

    def __init__(self, base_dir: str = "/opt/sutazaiapp/core_system"):        """
        Initialize the core system fixer.

        Args:        base_dir: Base directory containing core system files
        """
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        def validate_python_file(
                self, file_path: Path) -> Tuple[bool, Optional[str]]:            """
            Validate a Python file for syntax errors.

            Args:            file_path: Path to the Python file

            Returns:            Tuple[bool, Optional[str]]: (is_valid, error_message)
            """
            try:                    with open(file_path, encoding="utf-8") as f:                    code = f.read()
                    ast.parse(code)
                return True, None
                except SyntaxError as e:                return False, str(e)
                except Exception as e:                return False, str(e)

                def create_backup(self, file_path: Path) -> Path:                    """
                    Create a backup of a file.

                    Args:                    file_path: Path to the file to backup

                    Returns:                    Path: Path to the backup file
                    """
                    backup_path = self.backup_dir / \
                        f"{file_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                    shutil.copy2(file_path, backup_path)
                    logger.info("Created backup: %s", backup_path)
                return backup_path

                def create_stub(self, file_path: Path) -> None:                    """
                    Create a stub Python file.

                    Args:                    file_path: Path where the stub should be created
                    """
                    stub_content = f'''#!/usr/bin/env python3.11
                    """
                    {file_path.name} stub generated due to syntax errors.
                    Please implement logic as needed.
                    """

                        def main() -> None:                        """Main function stub."""
                        print("Stub for {file_path.name}")


                            if __name__ == "__main__":                            main()
                            '''
                    with open(file_path, "w", encoding="utf-8") as f:                        f.write(stub_content)
                        logger.info("Created stub: %s", file_path)

                        def fix_core_system_files(self) -> Dict[str, List[str]]:                            """
                                    Fix core system files by creating backups and stubs.

                                    Returns:                                    Dict[str, List[str]]: Report of fixed files and errors
                                    """
                            report = {
                                "fixed_files": [],
                                "errors": [],
                            }

                            logger.info(
    "Scanning directory: %s", self.base_dir)

                            try:                                    for file_path in self.base_dir.rglob(
                                        "*.py"):                                        if file_path.is_file():                                        is_valid, error = self.validate_python_file(
                                            file_path)
                                        if not is_valid:                                            logger.warning(
    "Error in %s: %s", file_path, error)
                                            try:                                                    # Create backup
                                                backup_path = self.create_backup(
                                                    file_path)
                                                # Create stub
                                                self.create_stub(file_path)
                                                report["fixed_files"].append(
                                                    str(file_path))
                                                except Exception as e:                                                    error_msg = f"Failed to fix {file_path}: {e!s}"
                                                    logger.error(error_msg)
                                                    report["errors"].append(
                                                        error_msg)
                                                    else:                                                    logger.info(
                                                        "Compiled OK: %s", file_path)

                                                    except Exception as e:                                                        error_msg = f"Error scanning directory: {e!s}"
                                                        logger.error(error_msg)
                                                        report["errors"].append(
                                                            error_msg)

                                                    return report

                                                    def main() -> None:                                                        """Main function to run core system fixes."""
                                                        try:                                                            fixer = CoreSystemFixer()
                                                            report = fixer.fix_core_system_files()

                                                            # Print summary
                                                            print("\nSummary:")
                                                            print(
                                                                f"Fixed files: {len(report['fixed_files'])}")
                                                            print(
                                                                f"Errors: {len(report['errors'])}")

                                                            if report["fixed_files"]:                                                                print(
                                                                    "\nFixed files:")
                                                                for file in report["fixed_files"]:                                                                    print(
                                                                        f"  - {file}")

                                                                    if report["errors"]:                                                                        print(
                                                                            "\nErrors:")
                                                                        for error in report["errors"]:                                                                            print(
                                                                                f"  - {error}")

                                                                            except Exception as e:                                                                                logger.exception(
                                                                                    "Core system fix failed: %s", str(e))
                                                                                sys.exit(
                                                                                    1)

                                                                                if __name__ == "__main__":                                                                                    main()
