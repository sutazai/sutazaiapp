#!/usr/bin/env python3.11
"""
Python 3.11 Compatibility Fixer

This script automatically fixes common Python 3.11 compatibility issues across the codebase,
including:
1. Indentation problems
2. Deprecated typing imports (Dict, List, Tuple)
3. Missing type annotations
4. f-string issues in logging
5. Other common linting issues

Usage:
python3.11 python311_compatibility_fixer.py [--directory=/path/to/fix]
"""

import argparse
import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s | %(levelname)s | %(message)s",
handlers=[
logging.FileHandler("python311_compatibility_fixes.log"),
logging.StreamHandler(sys.stdout),
],
)
logger = logging.getLogger("Python311CompatibilityFixer")


class Python311CompatibilityFixer:
    """Class for fixing Python 3.11 compatibility issues."""

    def __init__(self, directory: str = "/opt/sutazaiapp"):
        """
        Initialize the fixer with the directory to process.

        Args:
        directory: The directory to process
        """
        self.directory = directory
        self.files_processed = 0
        self.files_fixed = 0
        self.skipped_files: set[str] = set()

        def fix_all_files(self) -> None:
            """Process all Python files in the specified directory."""
            logger.info(
                "Starting Python 3.11 compatibility fixes in %s",
                self.directory)

            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.endswith(".py") and not self._should_skip(file):
                        file_path = os.path.join(root, file)
                        self.fix_file(file_path)

                        logger.info("Completed processing %d files. Fixed: %d, Skipped: %d",
                        self.files_processed, self.files_fixed, len(
                            self.skipped_files))

                        def _should_skip(self, filename: str) -> bool:
                            """
                            Determine if a file should be skipped.

                            Args:
                            filename: The filename to check

                            Returns:
                            bool: True if the file should be skipped, False otherwise
                            """
                            skip_patterns = [
                            "__pycache__",
                            ".egg-info",
                            "venv",
                            ".git",
                            "dist",
                            "build",
                            ]
                        return any(
                            pattern in filename for pattern in skip_patterns)

                        def fix_file(self, file_path: str) -> None:
                            """
                            Fix compatibility issues in a single Python file.

                            Args:
                            file_path: Path to the Python file to fix
                            """
                            self.files_processed += 1
                            logger.info("Processing file: %s", file_path)

                            try:
                                # Read the file
                                with open(file_path, encoding="utf-8") as f:
                                content = f.read()

                                # Apply fixes
                                original_content = content
                                content = self._fix_shebang(content)
                                content = self._fix_typing_imports(content)
                                content = self._fix_f_string_logging(content)
                                content = self._fix_indentation(content)

                                # Only write if changed
                                if content != original_content:
                                    with open(
                                        file_path,
                                        "w",
                                        encoding="utf-8") as f:
                                    f.write(content)
                                    self.files_fixed += 1
                                    logger.info(
                                        "Fixed issues in %s",
                                        file_path)

                                    except Exception as e:
                                        logger.error(
                                            "Error processing %s: %s",
                                            file_path,
                                            e)
                                        self.skipped_files.add(file_path)

                                        def _fix_shebang(
                                            self,
                                            content: str) -> str:
                                            """
                                            Fix the shebang line to use Python 3.11.

                                            Args:
                                            content: The Python file content

                                            Returns:
                                            str: The fixed content
                                            """
                                            # Replace Python 3 shebang with Python 3.11
                                            if content.startswith(
                                                "#!/usr/bin/env python3\n"):
                                            return content.replace(
                                                "#!/usr/bin/env python3\n",
                                                "#!/usr/bin/env python3.11\n")

                                            # Add shebang if missing and file is executable
                                            if not content.startswith("#!/"):
                                                                                        return "#!/usr/bin/env python3.11\n" + \
                                                content

                                        return content

                                        def _fix_typing_imports(
                                            self,
                                            content: str) -> str:
                                            """
                                            Fix deprecated typing imports (
                                                Dict -> dict,
                                                List -> list,
                                                etc.).

                                            Args:
                                            content: The Python file content

                                            Returns:
                                            str: The fixed content
                                            """
                                            # Regular expression to find typing imports
                                            pattern = r"from\s+typing\s+import\s+(
                                                [^#\n]*)"

                                            def replace_typing(match):
                                                imports = match.group(
                                                    1).split(",
                                                    ")
                                                fixed_imports = []

                                                for imp in imports:
                                                    imp = imp.strip()
                                                    if imp == "Dict":
                                                        fixed_imports.append(
                                                            "dict")
                                                        elif imp == "List":
                                                        fixed_imports.append(
                                                            "list")
                                                        elif imp == "Tuple":
                                                        fixed_imports.append(
                                                            "tuple")
                                                        elif imp == "Set":
                                                        fixed_imports.append(
                                                            "set")
                                                        else:
                                                        fixed_imports.append(
                                                            imp)

                                                    return f"from typing import {', '.join(
                                                        fixed_imports)}"

                                                return re.sub(
                                                    pattern,
                                                    replace_typing,
                                                    content)

                                                def _fix_f_string_logging(
                                                    self,
                                                    content: str) -> str:
                                                    """
                                                    Fix logging statements that use f-strings instead of % formatting.

                                                    Args:
                                                    content: The Python file content

                                                    Returns:
                                                    str: The fixed content
                                                    """
                                                    # Regular expression to find logging f-strings
                                                    pattern = r'logger\.(
                                                        debug|info|warning|error|critical)\(f"([^"]*?)({[^}]*?})([^"]*?)"\)'

                                                    def replace_logging(match):
                                                        level, pre, var, post = match.groups()
                                                        var = var[1:-1]  # Remove the curly braces
                                                    return f'logger.{level}(
                                                        "{pre}%s{post}",
                                                        {var})'

                                                    # Apply multiple times to catch nested replacements
                                                    for _ in range(3):
                                                        content = re.sub(
                                                            pattern,
                                                            replace_logging,
                                                            content)

                                                    return content

                                                    def _fix_indentation(
                                                        self,
                                                        content: str) -> str:
                                                        """
                                                        Fix indentation issues.

                                                        Args:
                                                        content: The Python file content

                                                        Returns:
                                                        str: The fixed content
                                                        """
                                                        try:
                                                            # Try to parse the content
                                                            ast.parse(content)
                                                        return content  # If parsing succeeds, no need to fix indentation
                                                        except SyntaxError:
                                                            # If parsing fails, try to fix indentation
                                                            lines = content.splitlines()
                                                            fixed_lines = []
                                                            indent_level = 0

                                                            for line in lines:
                                                                stripped = line.strip()

                                                                # Skip empty lines
                                                                if not stripped:
                                                                    fixed_lines.append(
                                                                        "")
                                                                continue

                                                                # Check for indentation markers
                                                                if stripped.endswith(
                                                                    ":"):
                                                                    fixed_lines.append(
                                                                        " " * (4 * indent_level) + stripped)
                                                                    indent_level += 1
                                                                    elif stripped in (
                                                                        "break",
                                                                        "continue",
                                                                        "pass",
                                                                        "return",
                                                                        "raise"):
                                                                    fixed_lines.append(
                                                                        " " * (4 * indent_level) + stripped)
                                                                    if indent_level > 0 and not any(
                                                                        l.strip().startswith(("elif", "else", "except", "finally")) for l in lines[lines.index(line)+1:lines.index(line)+5] if lines.index(line)+5 < len(lines)):
                                                                        indent_level -= 1
                                                                        else:
                                                                        fixed_lines.append(
                                                                            " " * (4 * indent_level) + stripped)

                                                                    return "\n".join(
                                                                        fixed_lines)


                                                                    def main() -> None:
                                                                        """Main function to run the fixer."""
                                                                        parser = argparse.ArgumentParser(
                                                                            description="Fix Python 3.11 compatibility issues")
                                                                        parser.add_argument(
                                                                            "--directory",
                                                                            default="/opt/sutazaiapp",
                                                                            help="Directory to process")
                                                                        args = parser.parse_args()

                                                                        fixer = Python311CompatibilityFixer(
                                                                            args.directory)
                                                                        fixer.fix_all_files()


                                                                        if __name__ == "__main__":
                                                                            main()
