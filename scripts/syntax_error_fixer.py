#!/usr/bin/env python3.11
"""
Syntax Error Fixer for SutazAI Project

This script automatically fixes common syntax errors in Python files
based on the report from python311_compatibility_checker.py.
"""

import os
import re
import sys
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("syntax_error_fixer")

# Map of error patterns to their fixes
ERROR_PATTERNS = {
    r"invalid syntax.*line (\d+)": "fix_invalid_syntax",
    r"unexpected indent.*line (\d+)": "fix_indentation",
    r"illegal target for annotation.*line (\d+)": "fix_illegal_annotation",
    r"invalid decimal literal.*line (\d+)": "fix_decimal_literal",
    r"expected an indented block.*line (\d+)": "fix_missing_indentation",
}


def fix_invalid_syntax(file_path: str, line_num: int) -> bool:
    """
    Attempts to fix common invalid syntax errors.
    
    Args:
        file_path: Path to the file with the error
        line_num: Line number with the error
        
    Returns:
        True if a fix was applied, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if line_num <= 0 or line_num > len(lines):
        return False
    
    # Get the problematic line (0-indexed in the list)
    line = lines[line_num - 1]
    fixed = False
    
    # Common syntax error fixes
    
    # 1. Fix missing colons in function/class definitions or control statements
    if re.search(r"(def|class|if|elif|else|for|while|try|except|finally)\s+[^:]*$", line):
        lines[line_num - 1] = line.rstrip() + ":\n"
        fixed = True
    
    # 2. Fix unterminated strings (missing quotes)
    elif re.search(r"(\"[^\"]*$)|(\'[^\']*$)", line):
        # Find the type of quote that's missing
        if "\"" in line and line.count("\"") % 2 != 0:
            lines[line_num - 1] = line.rstrip() + "\"\n"
            fixed = True
        elif "'" in line and line.count("'") % 2 != 0:
            lines[line_num - 1] = line.rstrip() + "'\n"
            fixed = True
    
    # 3. Fix missing parentheses
    elif line.count("(") > line.count(")"):
        lines[line_num - 1] = line.rstrip() + ")" * (line.count("(") - line.count(")")) + "\n"
        fixed = True
    
    # 4. Fix missing comma in list/dict/tuple
    elif re.search(r"\[.*\w+\s+\w+.*\]", line) or re.search(r"\{.*\w+\s+\w+.*\}", line):
        lines[line_num - 1] = re.sub(r"(\w+)\s+(\w+)", r"\1, \2", line)
        fixed = True
    
    # 5. Fix f-string syntax (common in Python 3.11)
    elif "f'" in line or 'f"' in line:
        # Fix cases like: print(f"Value: {value")
        if "{" in line and "}" not in line:
            lines[line_num - 1] = line.rstrip() + "}\n"
            fixed = True
            
    if fixed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"Fixed invalid syntax in {file_path} at line {line_num}")
    
    return fixed


def fix_indentation(file_path: str, line_num: int) -> bool:
    """
    Fixes unexpected indentation errors.
    
    Args:
        file_path: Path to the file with the error
        line_num: Line number with the error
        
    Returns:
        True if a fix was applied, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if line_num <= 1 or line_num > len(lines):
        return False
    
    # Get the current line and the previous line
    line = lines[line_num - 1]
    prev_line = lines[line_num - 2]
    
    # Determine the correct indentation level based on the previous line
    prev_indent = len(prev_line) - len(prev_line.lstrip())
    current_indent = len(line) - len(line.lstrip())
    
    # Check if previous line ends with a colon (indicating a block start)
    if prev_line.rstrip().endswith(":"):
        # The current line should be indented more than the previous line
        correct_indent = prev_indent + 4
    else:
        # Otherwise, it should match the previous line's indentation
        correct_indent = prev_indent
    
    # Fix the indentation
    if current_indent != correct_indent:
        # Create the correct indentation
        fixed_line = " " * correct_indent + line.lstrip()
        lines[line_num - 1] = fixed_line
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        logger.info(f"Fixed indentation in {file_path} at line {line_num}")
        return True
    
    return False


def fix_illegal_annotation(file_path: str, line_num: int) -> bool:
    """
    Fixes illegal annotation target errors.
    
    Args:
        file_path: Path to the file with the error
        line_num: Line number with the error
        
    Returns:
        True if a fix was applied, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if line_num <= 0 or line_num > len(lines):
        return False
    
    line = lines[line_num - 1]
    fixed = False
    
    # Common pattern: Annotations on non-names (e.g., a.b: int)
    if "." in line and ":" in line:
        # Extract the illegal annotation
        match = re.search(r"([\w\.]+)\s*:\s*(\w+)", line)
        if match and "." in match.group(1):
            # Replace with a legal annotation (using a variable)
            name_parts = match.group(1).split(".")
            var_name = name_parts[-1]
            type_anno = match.group(2)
            
            # Create a legal annotation with a comment
            new_line = f"{var_name}: {type_anno}  # Was: {match.group(1)}: {type_anno}\n"
            lines[line_num - 1] = new_line
            fixed = True
    
    # Fix annotations on literals
    elif re.search(r"(\".*\"|\d+)\s*:\s*\w+", line):
        # Remove the annotation
        new_line = re.sub(r"(\".*\"|\d+)\s*:\s*\w+", r"\1", line)
        lines[line_num - 1] = new_line
        fixed = True
    
    if fixed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"Fixed illegal annotation in {file_path} at line {line_num}")
    
    return fixed


def fix_decimal_literal(file_path: str, line_num: int) -> bool:
    """
    Fixes invalid decimal literal errors.
    
    Args:
        file_path: Path to the file with the error
        line_num: Line number with the error
        
    Returns:
        True if a fix was applied, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if line_num <= 0 or line_num > len(lines):
        return False
    
    line = lines[line_num - 1]
    fixed = False
    
    # Check for leading zeros in decimal literals (e.g., 01, 02)
    if re.search(r"\b0\d+\b", line):
        # Replace with decimal without leading zero or with proper octal notation
        new_line = re.sub(r"\b0(\d+)\b", r"\1", line)
        lines[line_num - 1] = new_line
        fixed = True
    
    if fixed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"Fixed decimal literal in {file_path} at line {line_num}")
    
    return fixed


def fix_missing_indentation(file_path: str, line_num: int) -> bool:
    """
    Fixes missing indentation block errors.
    
    Args:
        file_path: Path to the file with the error
        line_num: Line number with the error
        
    Returns:
        True if a fix was applied, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if line_num <= 1 or line_num > len(lines):
        return False
    
    # Get the line that should have an indented block
    block_start_line = lines[line_num - 2]  # The previous line
    current_line = lines[line_num - 1]
    
    # Check if the previous line ends with a colon (indicating a block start)
    if block_start_line.rstrip().endswith(":"):
        # Calculate the indentation
        block_indent = len(block_start_line) - len(block_start_line.lstrip())
        
        # Insert a placeholder 'pass' statement with proper indentation
        indent = " " * (block_indent + 4)
        pass_line = f"{indent}pass\n"
        
        # Insert the 'pass' statement before the current line
        lines.insert(line_num - 1, pass_line)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        logger.info(f"Fixed missing indentation block in {file_path} at line {line_num}")
        return True
    
    return False


def fix_file_errors(file_path: str, error_message: str) -> int:
    """
    Attempt to fix errors in a file based on the error message.
    
    Args:
        file_path: Path to the file to fix
        error_message: Error message from the compatibility checker
        
    Returns:
        Number of fixes applied
    """
    fixes_applied = 0
    
    for pattern, fix_func_name in ERROR_PATTERNS.items():
        match = re.search(pattern, error_message)
        if match:
            line_num = int(match.group(1)) if match.groups() else 0
            
            # Get the fix function dynamically
            fix_func = globals().get(fix_func_name)
            if fix_func and callable(fix_func):
                if fix_func(file_path, line_num):
                    fixes_applied += 1
    
    return fixes_applied


def process_compatibility_report(report_lines: List[str]) -> Dict[str, str]:
    """
    Process the output of the compatibility checker to extract file errors.
    
    Args:
        report_lines: Lines from the compatibility checker output
        
    Returns:
        Dictionary mapping file paths to error messages
    """
    file_errors = {}
    current_file = None
    
    for line in report_lines:
        line = line.strip()
        
        # Look for file paths followed by error messages
        # Typically in format: ai_agents/path/to/file.py:
        #   - Error analyzing file: <error message>
        if line.endswith(":") and "/" in line and not line.startswith("-"):
            # Extract file path
            current_file = line[:-1]  # Remove the trailing colon
        elif current_file and line.startswith("  - Error analyzing file:"):
            # Extract error message
            error_msg = line.replace("  - Error analyzing file:", "").strip()
            file_errors[current_file] = error_msg
    
    return file_errors


def main() -> None:
    """Main function to run the syntax error fixer."""
    project_path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    # First run the compatibility checker to get the errors
    logger.info(f"Running compatibility checker on {project_path}")
    
    # Read the compatibility checker output from stdin if piped
    if not sys.stdin.isatty():
        report_lines = sys.stdin.readlines()
    else:
        # Otherwise, run the compatibility checker directly
        checker_script = os.path.join(
            project_path, "scripts", "python311_compatibility_checker.py"
        )
        
        if os.path.exists(checker_script):
            import subprocess
            
            try:
                result = subprocess.run(
                    ["python", checker_script, project_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                report_lines = result.stdout.splitlines()
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running compatibility checker: {e}")
                if e.stdout:
                    report_lines = e.stdout.splitlines()
                else:
                    logger.error("No output from compatibility checker")
                    return
        else:
            logger.error(f"Compatibility checker script not found at {checker_script}")
            return
    
    # Process the report to extract file errors
    file_errors = process_compatibility_report(report_lines)
    
    # Attempt to fix each file
    total_files = len(file_errors)
    fixed_files = 0
    total_fixes = 0
    
    logger.info(f"Found {total_files} files with syntax errors")
    
    for file_path, error_msg in file_errors.items():
        full_path = os.path.join(project_path, file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue
        
        logger.info(f"Attempting to fix {file_path} - Error: {error_msg}")
        fixes = fix_file_errors(full_path, error_msg)
        
        if fixes > 0:
            total_fixes += fixes
            fixed_files += 1
    
    logger.info(
        f"Applied {total_fixes} fixes to {fixed_files}/{total_files} files with syntax errors"
    )


if __name__ == "__main__":
    main() 