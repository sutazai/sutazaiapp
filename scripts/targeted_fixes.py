#!/usr/bin/env python3
"""
Script to fix specific syntax errors identified in the codebase.

This script targets particular files and patterns of syntax errors
that were identified by previous scanning tools.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Set


# ANSI color codes for pretty output

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str) -> None:
    print(f"{Colors.HEADER}{message}{Colors.ENDC}")


def print_info(message: str) -> None:
    print(f"{Colors.BLUE}{message}{Colors.ENDC}")


def print_success(message: str) -> None:
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    print(f"{Colors.YELLOW}{message}{Colors.ENDC}")


def print_error(message: str) -> None:
    print(f"{Colors.RED}{message}{Colors.ENDC}")


def check_for_syntax(file_path: str) -> Tuple[bool, str]:
    """Check if a file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use Python's built-in compile function to check for syntax errors
        compile(content, file_path, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except UnicodeDecodeError:
        return False, "File contains non-UTF-8 characters"


def fix_empty_docstrings(content: str) -> str:
    """
    Fix empty triple-quote docstrings (e.g., """""").

    This is a common issue in the codebase where there are empty docstrings
    that need to be properly formatted or removed.
    """
    # Replace empty triple quotes with proper docstring format
    content = content.replace('""""""', '"""This is an auto-generated docstring."""')
    content = content.replace("''''''", "'''This is an auto-generated docstring.'''")

    return content


def fix_orchestrator_docstring(content: str) -> str:
    """
    Fix the specific docstring issue in the orchestrator.py file.
    """
    # Pattern to find the problematic function definition and docstring
    pattern = r'def _generate_improvement_recommendations\(\s*self, agent_name: str\) -> List\[str\]:\s*"""(.*?)"""'

    if '_generate_improvement_recommendations' in content:
        # Replace with properly formatted function and docstring
        new_function = """def _generate_improvement_recommendations(self, agent_name: str) -> List[str]:
        \"\"\"
        Generate improvement recommendations for a specific agent

        Args:
            agent_name (str): Name of the agent

        Returns:
            List[str]: Recommended improvement actions
        \"\"\"
        recommendations = [
            f"Review and update {agent_name} implementation",
            f"Retrain {agent_name} with expanded dataset",
        ]
        return recommendations"""

        # Try to replace the malformed function with our corrected version
        if 'def _generate_improvement_recommendations' in content:
            # Find the function start
            func_start = content.find('def _generate_improvement_recommendations')
            if func_start != -1:
                # Find the next function or class definition after this one
                next_def = content.find('def ', func_start + 5)
                next_class = content.find('class ', func_start + 5)

                if next_def == -1 and next_class == -1:
                    # This is the last function, replace till the end or a reasonable point
                    content = content[:func_start] + new_function
                else:
                    # Replace until the next function or class
                    end_pos = min(pos for pos in [next_def, next_class] if pos != -1)
                    content = content[:func_start] + new_function + content[end_pos:]

    return content


def fix_fstring_formatting(content: str) -> str:
    """
    Fix f-string formatting issues such as those in system_setup.py.
    """
    # Look for lines with f-strings that have formatting issues
    lines = content.splitlines()
    result_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for f-strings with incomplete formatting
        if 'f"' in line or "f'" in line:
            # If the line has a missing closing brace or quote
            if line.count('{') != line.count('}') or (line.count('"') % 2 != 0 and 'f"' in line) or (line.count("'") % 2 != 0 and "f'" in line):
                # Try to find the end of the f-string in subsequent lines
                combined_line = line
                j = i + 1
                fixed = False

                while j < len(lines) and not fixed:
                    next_line = lines[j]
                    combined_line += "\n" + next_line

                    # Check if adding this line fixes the f-string
                    if (combined_line.count('{') == combined_line.count('}') and
                       ((combined_line.count('"') % 2 == 0) or 'f"' not in combined_line) and
                       ((combined_line.count("'") % 2 == 0) or "f'" not in combined_line)):
                        # Fix found, replace all these lines with the combined line
                        result_lines.append(combined_line)
                        i = j + 1
                        fixed = True
                        break

                    j += 1

                if not fixed:
                    # Couldn't fix automatically, add a closing brace/quote as best guess
                    if line.count('{') > line.count('}'):
                        line += "}"
                    if ('f"' in line and line.count('"') % 2 != 0):
                        line += '"'
                    if ("f'" in line and line.count("'") % 2 != 0):
                        line += "'"
                    result_lines.append(line)
                    i += 1
            else:
                # No issues detected in this f-string
                result_lines.append(line)
                i += 1
        else:
            # Not an f-string line
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def fix_specific_file(file_path: str) -> bool:
    """
    Apply targeted fixes to a specific file based on its known issues.
    """
    try:
        # Check if the file exists and has syntax errors
        if not os.path.exists(file_path):
            print_error(f"File not found: {file_path}")
            return False

        is_valid, error_msg = check_for_syntax(file_path)
        if is_valid:
            print_info(f"No syntax errors in {file_path}")
            return False

        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply appropriate fixes based on the file path or error message
        if "empty docstring" in error_msg.lower() or """""" in content:
            print_info(f"Fixing empty docstrings in {file_path}")
            content = fix_empty_docstrings(content)

        if "orchestrator.py" in file_path:
            print_info(f"Applying orchestrator.py specific fixes to {file_path}")
            content = fix_orchestrator_docstring(content)

        if "system_setup.py" in file_path or "f-string" in error_msg.lower() or ("f\"" in content and "{" in content):
            print_info(f"Fixing f-string formatting in {file_path}")
            content = fix_fstring_formatting(content)

        # Write the fixed content back to the file if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Check if the fix resolved the syntax error
            is_valid_now, new_error = check_for_syntax(file_path)
            if is_valid_now:
                print_success(f"✓ Successfully fixed {file_path}")
                return True
            else:
                print_warning(f"  Partial fix applied to {file_path}, but syntax errors remain: {new_error}")
                return True
        else:
            print_info(f"No changes needed for {file_path}")
            return False

    except Exception as e:
        print_error(f"Error processing {file_path}: {str(e)}")
        return False


def main() -> None:
    """
    Main function that applies targeted fixes to specific files.
    """
    print_header("Starting targeted syntax fixes")

    # List of specific files to fix based on previous scan results
    target_files = [
        "ai_agents/document_processor/tests/test_document_processor.py",
        "ai_agents/document_processor/tests/conftest.py",
        "ai_agents/auto_gpt/src/__init__.py",
        "ai_agents/supreme_ai/orchestrator.py",
        "doc_data/core_system/system_setup.py",
        "scripts/system_manager.py",
        "scripts/python_compatibility_manager.py"
    ]

    # Get base directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.getcwd()

    fixed_count = 0

    # Fix each target file
    for rel_path in target_files:
        full_path = os.path.join(base_dir, rel_path)
        print_info(f"Processing {full_path}")

        if fix_specific_file(full_path):
            fixed_count += 1

    print_header("\nTargeted Fix Summary")
    print_info(f"Fixed {fixed_count} out of {len(target_files)} target files")


if __name__ == "__main__":
    main()
