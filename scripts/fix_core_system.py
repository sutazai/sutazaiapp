#!/usr/bin/env python3.11"""Core System Fixer for SutazAIThis script scans all Python files under the core_system directory for syntax errors.If a file fails to compile, it attempts to fix the syntax errors in place."""import astimport loggingimport osimport sysfrom pathlib import Pathfrom typing import Dict, List, Optional, Tuple# Configure logginglogging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.FileHandler("/opt/sutazaiapp/logs/core_system_fix.log"),logging.StreamHandler(),],)logger = logging.getLogger(__name__)class CoreSystemFixer:    """Core system fixer utility."""
def __init__(self, base_dir: str = "/opt/sutazaiapp/core_system"):        """
    Initialize the core system fixer.
    Args:
    base_dir: Base directory containing core system files
    """
    self.base_dir = Path(base_dir)
    def validate_python_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:            """
        Validate a Python file for syntax errors.
        Args:
        file_path: Path to the Python file
        Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            with open(file_path, encoding="utf-8") as f:
            code = f.read()
            ast.parse(code)
            return True, None
        except SyntaxError as e:                return False, str(e)            except Exception as e:                return False, str(e)            def fix_syntax_errors(self, file_path: Path) -> bool:                """
            Attempt to fix syntax errors in a file.
            Args:
            file_path: Path to the file to fix
            Returns:
            bool: True if fixes were successful
            """
            try:
                with open(file_path, encoding="utf-8") as f:
                code = f.read()
                # Basic syntax fixes
                lines = code.split("\n")
                fixed_lines = []
                indent_level = 0
                for line in lines:
                stripped = line.strip()
                if not stripped:  # Empty line
                    fixed_lines.append("")
                    continue
                # Adjust indentation
                if stripped.startswith(("def ", "class ", "if ", "elif ", "else:", "try:", "except", "finally:")):
                    fixed_lines.append(" " * (4 * indent_level) + stripped)
                    indent_level += 1
                    elif stripped in ("pass", "break", "continue") or stripped.startswith("return"):
                        fixed_lines.append(" " * (4 * indent_level) + stripped)
                        indent_level = max(0, indent_level - 1)
                        else:
                            fixed_lines.append(" " * (4 * indent_level) + stripped)
                            # Write fixed code back to file
                            fixed_code = "\n".join(fixed_lines)
                            with open(file_path, "w", encoding="utf-8") as f:
                            f.write(fixed_code)
                            return True
                        except Exception as e:                                logger.error(f"Failed to fix syntax in {file_path}: {e}")                                return False                            def fix_core_system_files(self) -> Dict[str, List[str]]:                                """
                            Fix core system files with syntax errors.
                            Returns:
                            Dict[str, List[str]]: Report of fixed files and errors
                            """
                            report = {
                            "fixed_files": [],
                            "errors": [],
                            }
                            logger.info("Scanning directory: %s", self.base_dir)
                            try:
                                for file_path in self.base_dir.rglob("*.py"):
                                if file_path.is_file():
                                    is_valid, error = self.validate_python_file(file_path)
                                    if not is_valid:
                                        logger.warning("Error in %s: %s", file_path, error)
                                        try:
                                            if self.fix_syntax_errors(file_path):
                                                report["fixed_files"].append(str(file_path))
                                                else:
                                                    report["errors"].append(f"Failed to fix {file_path}")
                                                    except Exception as e:
                                                        error_msg = f"Failed to fix {file_path}: {e!s}"
                                                        logger.error(error_msg)
                                                        report["errors"].append(error_msg)
                                                        else:
                                                            logger.info("Compiled OK: %s", file_path)
                                                            except Exception as e:
                                                                error_msg = f"Error scanning directory: {e!s}"
                                                                logger.error(error_msg)
                                                                report["errors"].append(error_msg)
                                                                return report
                                                            def main() -> None:                                                                    """Main function to run core system fixes."""
                                                                try:
                                                                    fixer = CoreSystemFixer()
                                                                    report = fixer.fix_core_system_files()
                                                                    # Print summary
                                                                    print("\nSummary:")
                                                                    print(f"Fixed files: {len(report['fixed_files'])}")
                                                                    print(f"Errors: {len(report['errors'])}")
                                                                    if report["fixed_files"]:
                                                                        print("\nFixed files:")
                                                                        for file in report["fixed_files"]:
                                                                        print(f"  - {file}")
                                                                        if report["errors"]:
                                                                            print("\nErrors:")
                                                                            for error in report["errors"]:
                                                                            print(f"  - {error}")
                                                                            except Exception as e:
                                                                                logger.exception("Core system fix failed: %s", str(e))
                                                                                sys.exit(1)
                                                                                if __name__ == "__main__":
                                                                                    main()

