#!/usr/bin/env python3.11
"""
Python 3.11 Typing Compatibility Fixer for SutazAI Project

This script updates typing annotations to be compatible with Python 3.11,
addressing common issues identified by the python311_compatibility_checker.py.
"""

import logging
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("typing_compatibility_fixer")

# Patterns to find potential typing issues
TYPING_PATTERNS = {
    # Old style A | B → A | B or A | B
    r"Union\[([\w\s,\.]+)\]": "union_pattern",

    # Old style A | None → A | None or A | None
    r"Optional\[([\w\.]+)\]": "optional_pattern",

    # Type comments (# type: X) → proper annotations
    r"#\s*type:\s*(.+)$": "type_comment_pattern",

    # TypedDict without total parameter in Python 3.11
    r"class\s+(\w+)\s*\(\s*TypedDict\s*\)": "typeddict_pattern",
}


def union_pattern(match, line: str) -> str:
    """
    Convert A | B to use the | operator where appropriate.

    Args:
        match: Regex match object
        line: The line of code

    Returns:
        Updated line of code
    """
    # We'll keep Union for complex cases but use | for simple ones
    union_types = match.group(1).split(",")

    # Only convert simple cases to | syntax
    if len(union_types) <= 2 and all(t.strip().isalnum() for t in union_types):
        types_str = " | ".join(t.strip() for t in union_types)
        return line.replace(match.group(0), types_str)

    return line


def optional_pattern(match, line: str) -> str:
    """
    Convert A | None to A | None where appropriate.

    Args:
        match: Regex match object
        line: The line of code

    Returns:
        Updated line of code
    """
    # Simple case: use | None
    type_name = match.group(1).strip()

    # Only convert simple cases to | syntax
    if type_name.isalnum():
        return line.replace(match.group(0), f"{type_name} | None")

    return line


def type_comment_pattern(match, line: str) -> str:
    """
    Convert type comments to annotations where possible.

    Args:
        match: Regex match object
        line: The line of code

    Returns:
        Updated line of code
    """
    # This is more complex as it requires context
    # For this script, we'll just add a warning
    logger.warning(f"Type comment found, manual conversion recommended: {line.strip()}")
    return line


def typeddict_pattern(match, line: str) -> str:
    """
    Update TypedDict for Python 3.11 compatibility.

    Args:
        match: Regex match object
        line: The line of code

    Returns:
        Updated line of code
    """
    # In Python 3.11, TypedDict should include a total parameter
    class_name = match.group(1)
    return line.replace(f"({class_name}(TypedDict)", f"({class_name}(TypedDict, total=True)")


def process_typing_issues(file_path: str) -> int:
    """
    Process and fix typing issues in a file.

    Args:
        file_path: Path to the file to process

    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        fixed_lines = []
        fixes_applied = 0

        # Process each line for typing issues
        for line in lines:
            original_line = line

            # Handle each pattern
            for pattern, handler_name in TYPING_PATTERNS.items():
                for match in re.finditer(pattern, line):
                    # Get the handler function dynamically
                    handler = globals().get(handler_name)
                    if handler and callable(handler):
                        line = handler(match, line)

            # Check for Self typing (new in Python 3.11)
            if "self" in line and "->" in line and re.search(r"-> .*self\.", line):
                # Replace the return type to use Self
                line = re.sub(r"-> .*self\.(\w+)", r"-> Self", line)
                if "from typing import Self" not in content:
                    fixed_lines.insert(0, "from typing import Self")
                    fixes_applied += 1

            # Add the line (possibly modified)
            fixed_lines.append(line)

            # Count fixes
            if line != original_line:
                fixes_applied += 1

        # Only write back if changes were made
        if fixes_applied > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines))

            logger.info(f"Applied {fixes_applied} typing fixes to {file_path}")

        return fixes_applied

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0


def add_typing_imports(file_path: str) -> bool:
    """
    Add necessary typing imports for Python 3.11 compatibility.

    Args:
        file_path: Path to the file to update

    Returns:
        True if imports were added, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for patterns that might need imports
    imports_to_add = []

    # Check for | operator in type annotations
    if re.search(r":\s*\w+\s*\|", content) and "from typing import Union" not in content:
        imports_to_add.append("from typing import Union")

    # Check for Optional-like patterns with None
    if re.search(r":\s*\w+\s*\|\s*None", content) and "from typing import Optional" not in content:
        imports_to_add.append("from typing import Optional")

    # Check for TypedDict
    if "TypedDict" in content and "from typing import TypedDict" not in content:
        imports_to_add.append("from typing import TypedDict")

    # Apply changes if needed
    if imports_to_add:
        lines = content.splitlines()

        # Find where to insert the imports
        import_section_end = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_section_end = i + 1

        # Insert the imports
        for import_line in imports_to_add:
            lines.insert(import_section_end, import_line)
            import_section_end += 1

        # Write back the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Added typing imports to {file_path}: {', '.join(imports_to_add)}")
        return True

    return False


def main() -> None:
    """Main function to run the typing compatibility fixer."""
    project_path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    # Get files with typing issues from the compatibility checker
    typing_issue_files = []

    # Load the output from stdin if piped
    if not sys.stdin.isatty():
        report_lines = sys.stdin.readlines()

        # Extract files with typing issues
        current_file = None
        for line in report_lines:
            line = line.strip()

            # Look for file paths
            if line.endswith(":") and "/" in line:
                current_file = line[:-1]

            # Check for typing-related issues
            elif current_file and "typing" in line:
                typing_issue_files.append(current_file)
                current_file = None  # Reset to avoid duplicates
    else:
        # Find Python files with typing imports
        logger.info("Scanning for files with typing imports")
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Skip virtual environment files
                    if any(p in file_path for p in ["venv/", ".venv/", "__pycache__/"]):
                        continue

                    # Check if the file imports typing
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(1000)  # Just check the beginning
                        if "import typing" in content or "from typing import" in content:
                            rel_path = os.path.relpath(file_path, project_path)
                            typing_issue_files.append(rel_path)

    # Process each file with typing issues
    total_files = len(typing_issue_files)
    fixed_files = 0
    total_fixes = 0

    logger.info(f"Found {total_files} files with potential typing issues")

    for file_path in typing_issue_files:
        full_path = os.path.join(project_path, file_path)

        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue

        logger.info(f"Processing {file_path}")

        # Process typing issues
        fixes = process_typing_issues(full_path)

        # Add necessary imports
        if add_typing_imports(full_path):
            fixes += 1

        if fixes > 0:
            total_fixes += fixes
            fixed_files += 1

    logger.info(
        f"Applied {total_fixes} typing fixes to {fixed_files}/{total_files} files"
    )


if __name__ == "__main__":
    main()
