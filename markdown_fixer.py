#!/usr/bin/env python3
"""
Advanced Markdown Documentation Quality Improvement Script

Automatically fixes complex markdown documentation issues:
- Adds blank lines around headings
- Removes trailing spaces
- Ensures single trailing newline
- Adds language specifications for code blocks
- Corrects list formatting
- Handles nested lists and complex markdown structures
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def fix_markdown_file(file_path: str) -> Dict[str, Any]:
    """
    Comprehensively fix markdown file formatting issues.

    Args:
        file_path (str): Path to the markdown file

    Returns:
        Dict[str, Any]: Detailed information about fixes applied
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    fixes_applied: Dict[str, int] = {
        "headings_fixed": 0,
        "trailing_spaces_removed": 0,
        "code_blocks_fixed": 0,
        "list_formatting_fixed": 0,
    }

    # Add blank lines around headings (multiple levels)
    content, heading_fixes = _fix_headings(content)
    fixes_applied["headings_fixed"] = heading_fixes

    # Remove trailing spaces
    content, trailing_space_fixes = _remove_trailing_spaces(content)
    fixes_applied["trailing_spaces_removed"] = trailing_space_fixes

    # Ensure single trailing newline
    content = content.rstrip() + "\n"

    # Add language to code blocks
    content, code_block_fixes = _fix_code_blocks(content)
    fixes_applied["code_blocks_fixed"] = code_block_fixes

    # Fix list formatting
    content, list_fixes = _fix_list_formatting(content)
    fixes_applied["list_formatting_fixed"] = list_fixes

    # Check if changes were made
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"modified": True, "fixes_applied": fixes_applied}

    return {"modified": False, "fixes_applied": fixes_applied}


def _fix_headings(content: str) -> Tuple[str, int]:
    """
    Fix heading formatting with blank lines.

    Args:
        content (str): Markdown content

    Returns:
        Tuple[str, int]: Modified content and number of fixes
    """
    fixes = 0
    # Add blank lines before and after headings
    content = re.sub(r"(^|\n)([#]{1,6} )", r"\1\n\2", content)
    content = re.sub(r"([#]{1,6} .*)\n([^#\n])", r"\1\n\n\2", content)
    fixes = len(re.findall(r"\n\n[#]{1,6} ", content))

    return content, fixes


def _remove_trailing_spaces(content: str) -> Tuple[str, int]:
    """
    Remove trailing spaces from each line.

    Args:
        content (str): Markdown content

    Returns:
        Tuple[str, int]: Modified content and number of fixes
    """
    lines = content.split("\n")
    fixed_lines = []
    fixes = 0

    for line in lines:
        stripped_line = line.rstrip()
        if line != stripped_line:
            fixes += 1
        fixed_lines.append(stripped_line)

    return "\n".join(fixed_lines), fixes


def _fix_code_blocks(content: str) -> Tuple[str, int]:
    """
    Add language specifications to code blocks.

    Args:
        content (str): Markdown content

    Returns:
        Tuple[str, int]: Modified content and number of fixes
    """
    fixes = 0

    # Add default language (python) to code blocks without language
    def _replace_code_block(match):
        nonlocal fixes
        fixes += 1
        return "```python\n"

    content = re.sub(r"(```)\n", _replace_code_block, content)

    return content, fixes


def _fix_list_formatting(content: str) -> Tuple[str, int]:
    """
    Improve list formatting with blank lines and consistent indentation.

    Args:
        content (str): Markdown content

    Returns:
        Tuple[str, int]: Modified content and number of fixes
    """
    fixes = 0
    # Add blank lines around lists
    content = re.sub(r"([^\n])\n([-*+])", r"\1\n\n\2", content)
    content = re.sub(r"([-*+].*)\n([^-*+\n])", r"\1\n\n\2", content)
    fixes = len(re.findall(r"\n\n[-*+] ", content))

    return content, fixes


def process_markdown_files(base_dir: str) -> List[Dict[str, Any]]:
    """
    Process all markdown files in the given directory.

    Args:
        base_dir (str): Base directory to search for markdown files

    Returns:
        List[Dict[str, Any]]: List of files that were modified with fix details
    """
    modified_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                result = fix_markdown_file(full_path)

                if result["modified"]:
                    result["file_path"] = full_path
                    modified_files.append(result)

    return modified_files


def main():
    base_dir = os.getcwd()
    modified = process_markdown_files(base_dir)

    if modified:
        print("🔧 Markdown files fixed:")
        for file_result in modified:
            print(f"  - {file_result['file_path']}")
            for fix_type, count in file_result["fixes_applied"].items():
                if count > 0:
                    print(
                        f"    • {fix_type.replace('_', ' ').title()}: {count}"
                    )
    else:
        print("✅ No markdown formatting issues found.")


if __name__ == "__main__":
    main()
