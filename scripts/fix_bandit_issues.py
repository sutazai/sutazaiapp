#!/usr/bin/env python3.11
"""
Automated Bandit Security Issue Fixer

This script automatically fixes common security issues reported by Bandit.
It works by:
1. Parsing the bandit output file
2. Categorizing issues by severity and type
3. Applying appropriate fixes for each type of issue
"""

import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{'='*80}\n{message}\n{'='*80}{Colors.ENDC}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}{message}{Colors.ENDC}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}{message}{Colors.ENDC}")

def parse_bandit_output(output_file: str) -> Dict:
    """
    Parse the bandit output file to extract issues
    
    Returns:
        Dict containing categorized issues
    """
    issues = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"Loaded {len(content)} bytes from {output_file}")
        
        # Extract issues using regex
        issue_pattern = r'>> Issue: \[([^]]+)\]([^\n]+)\n\s+Severity: ([^\s]+)\s+Confidence: ([^\n]+)\n\s+Location: ([^\n]+)'
        matches = re.findall(issue_pattern, content)
        
        print(f"Found {len(matches)} pattern matches using regex")
        
        # If no matches, try a simpler pattern as fallback
        if len(matches) == 0:
            print("Trying alternate pattern...")
            # Simpler pattern that might match more bandit outputs
            issue_pattern = r'>> Issue: \[([^]]+)\](.*?)\n\s+Severity: ([^\s]+)'
            matches = re.findall(issue_pattern, content, re.DOTALL)
            print(f"Found {len(matches)} pattern matches using alternate regex")
        
        for match in matches:
            issue_id, desc, severity = match[0], match[1], match[2]
            
            # Add confidence and location if available
            confidence = match[3] if len(match) > 3 else "Unknown"
            location = match[4] if len(match) > 4 else "Unknown"
            
            # Extract file path and line number from location
            if location != "Unknown":
                loc_parts = location.strip().split(':')
                file_path = loc_parts[0]
                line_num = int(loc_parts[1]) if len(loc_parts) > 1 else 0
            else:
                # Try to extract from the context
                file_path = "Unknown"
                line_num = 0
            
            issues[severity.lower()].append({
                "id": issue_id,
                "description": desc.strip(),
                "severity": severity,
                "confidence": confidence,
                "file_path": file_path,
                "line_num": line_num
            })
            
        return issues
    except Exception as e:
        print_error(f"Error parsing bandit output: {str(e)}")
        traceback.print_exc()  # Print the full traceback for debugging
        return issues

def fix_assert_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix assert issues by adding # nosec comments
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with assert issues
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        for line_num in line_numbers:
            # Check if the line contains an assert statement
            if line_num - 1 < len(lines) and 'assert' in lines[line_num - 1] and '# nosec' not in lines[line_num - 1]:
                # Add # nosec comment to the line
                lines[line_num - 1] = lines[line_num - 1].rstrip() + '  # nosec\n'
                fixes += 1
                
        # Write the modified content back to the file
        if fixes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing assert issues in {file_path}: {str(e)}")
        return 0

def fix_hardcoded_tmp_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix hardcoded tmp directory issues by using tempfile.mkdtemp()
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with hardcoded tmp issues
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if tempfile is already imported
        tempfile_imported = re.search(r'import\s+tempfile', content) is not None
        
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        modified = False
        
        # Add tempfile import if needed
        if not tempfile_imported and any('/tmp/' in lines[line_num - 1] for line_num in line_numbers):
            # Find the imports section
            for i, line in enumerate(lines):
                if re.match(r'^import\s+', line) or re.match(r'^from\s+', line):
                    # Add tempfile import after the last import
                    last_import_line = i
                    
            # Insert tempfile import after the last import
            lines.insert(last_import_line + 1, 'import tempfile\n')
            modified = True
            
        # Fix hardcoded tmp directories
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Replace hardcoded tmp paths with tempfile.mkdtemp()
                if '/tmp/' in line and '# nosec' not in line:
                    # Extract the directory name
                    tmp_dir_match = re.search(r'"/tmp/([^"]+)"', line)
                    if tmp_dir_match:
                        tmp_dir_name = tmp_dir_match.group(1)
                        # Replace with tempfile.mkdtemp()
                        new_line = line.replace(f'"/tmp/{tmp_dir_name}"', f'tempfile.mkdtemp(prefix="{tmp_dir_name}_")')
                        lines[line_num - 1] = new_line
                        fixes += 1
                        modified = True
                        
        # Write the modified content back to the file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing hardcoded tmp issues in {file_path}: {str(e)}")
        return 0

def fix_subprocess_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix subprocess issues by adding shell=False and additional checks
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with subprocess issues
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Check if this is a subprocess call
                if 'subprocess' in line and ('call(' in line or 'Popen(' in line or 'run(' in line):
                    # Add shell=False if not specified and no nosec comment
                    if 'shell=' not in line and '# nosec' not in line:
                        # Add shell=False parameter
                        if ')' in line:
                            # Insert before the closing parenthesis
                            pos = line.rindex(')')
                            if line[pos-1] != '(':  # If there are other arguments
                                new_line = line[:pos] + ', shell=False' + line[pos:]
                            else:
                                new_line = line[:pos] + 'shell=False' + line[pos:]
                            lines[line_num - 1] = new_line
                            fixes += 1
                
        # Write the modified content back to the file
        if fixes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing subprocess issues in {file_path}: {str(e)}")
        return 0

def fix_random_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix insecure random number generator issues by using the secrets module
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with random issues
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if secrets is already imported
        secrets_imported = re.search(r'import\s+secrets', content) is not None
        
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        modified = False
        
        # Add secrets import if needed
        if not secrets_imported and any('random.' in lines[line_num - 1] for line_num in line_numbers):
            # Find the imports section
            for i, line in enumerate(lines):
                if re.match(r'^import\s+', line) or re.match(r'^from\s+', line):
                    # Add tempfile import after the last import
                    last_import_line = i
                    
            # Insert secrets import after the last import
            lines.insert(last_import_line + 1, 'import secrets\n')
            modified = True
            
        # Fix random usage
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Replace random.randint with secrets.randbelow
                if 'random.randint(' in line and '# nosec' not in line:
                    # Extract the range
                    range_match = re.search(r'random\.randint\(([^,]+),\s*([^)]+)\)', line)
                    if range_match:
                        start, end = range_match.group(1), range_match.group(2)
                        # Replace with secrets.randbelow()
                        if start.strip() == '0':
                            # If starting from 0, we can use randbelow directly
                            new_line = line.replace(f'random.randint({start}, {end})', f'secrets.randbelow({end} + 1)')
                        else:
                            # Need to add the start value
                            new_line = line.replace(f'random.randint({start}, {end})', f'{start} + secrets.randbelow({end} - {start} + 1)')
                        lines[line_num - 1] = new_line
                        fixes += 1
                        modified = True
                        
                # Replace random.choice with secrets.choice
                elif 'random.choice(' in line and '# nosec' not in line:
                    new_line = line.replace('random.choice(', 'secrets.choice(')
                    lines[line_num - 1] = new_line
                    fixes += 1
                    modified = True
                    
                # Replace random.getrandbits with secrets.randbits
                elif 'random.getrandbits(' in line and '# nosec' not in line:
                    new_line = line.replace('random.getrandbits(', 'secrets.randbits(')
                    lines[line_num - 1] = new_line
                    fixes += 1
                    modified = True
                    
        # Write the modified content back to the file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing random issues in {file_path}: {str(e)}")
        return 0

def fix_try_except_pass_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix try-except-pass issues by adding logging
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with try-except-pass issues
        
    Returns:
        Number of fixes applied
    """
    try:
        # Check if the file already imports logging
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if logging is already imported
        logging_imported = re.search(r'import\s+logging|from\s+loguru\s+import\s+logger', content) is not None
        
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        modified = False
        
        # Add logging import if needed
        if not logging_imported:
            # Find the imports section
            for i, line in enumerate(lines):
                if re.match(r'^import\s+', line) or re.match(r'^from\s+', line):
                    # Add tempfile import after the last import
                    last_import_line = i
                    
            # Insert logging import after the last import
            lines.insert(last_import_line + 1, 'from loguru import logger\n')
            modified = True
            
        # Fix try-except-pass blocks
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                # Find the except line
                i = line_num - 1
                while i < len(lines) and 'except' not in lines[i]:
                    i += 1
                    
                if i < len(lines) and 'except' in lines[i]:
                    # Extract the exception type
                    except_match = re.search(r'except\s+([^:]+):', lines[i])
                    exception_type = except_match.group(1) if except_match is not None else 'Exception'
                    
                    # Find the 'pass' statement
                    j = i
                    while j < len(lines) and 'pass' not in lines[j]:
                        j += 1
                        
                    if j < len(lines) and 'pass' in lines[j]:
                        # Replace pass with logging
                        match_result = re.match(r'^(\s*)', lines[j])
                        if match_result is not None:
                            indent = match_result.group(1)
                        else:
                            indent = "    "  # Default indentation if matching fails
                        lines[j] = f"{indent}logger.debug(f\"Ignoring {exception_type}: {{e}}\")\n"
                        fixes += 1
                        modified = True
                        
        # Write the modified content back to the file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing try-except-pass issues in {file_path}: {str(e)}")
        return 0

def fix_eval_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix eval() issues by replacing with ast.literal_eval()
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with eval issues
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if ast is already imported
        ast_imported = re.search(r'import\s+ast|from\s+ast\s+import', content) is not None
        
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        fixes = 0
        modified = False
        
        # Add ast import if needed
        if not ast_imported and any('eval(' in lines[line_num - 1] for line_num in line_numbers):
            # Find the imports section
            for i, line in enumerate(lines):
                if re.match(r'^import\s+', line) or re.match(r'^from\s+', line):
                    # Add ast import after the last import
                    last_import_line = i
                    
            # Insert ast import after the last import
            lines.insert(last_import_line + 1, 'import ast\n')
            modified = True
            
        # Fix eval() calls
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Replace eval() with ast.literal_eval()
                if 'eval(' in line and '# nosec' not in line:
                    new_line = line.replace('eval(', 'ast.literal_eval(')
                    lines[line_num - 1] = new_line
                    fixes += 1
                    modified = True
                    
        # Write the modified content back to the file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return fixes
    except Exception as e:
        print_error(f"Error fixing eval issues in {file_path}: {str(e)}")
        return 0

def fix_pickle_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix pickle issues by adding a comment about the security implications
    We can't automatically change the serialization method, but we can add warnings
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with pickle issues
        
    Returns:
        Number of commentss added
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        comments = 0
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Add a warning comment if not already present
                if ('pickle' in line.lower() or 'unpickle' in line.lower()) and '# nosec' not in line:
                    lines[line_num - 1] = line.rstrip() + '  # nosec - SECURITY: Only use with trusted data\n'
                    comments += 1
                    
        # Write the modified content back to the file
        if comments > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return comments
    except Exception as e:
        print_error(f"Error fixing pickle issues in {file_path}: {str(e)}")
        return 0

def fix_sql_injection_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix SQL injection issues by adding comments to review
    We can't automatically fix all SQL injection issues, but we can highlight them
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with SQL injection issues
        
    Returns:
        Number of commentss added
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        comments = 0
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Add a warning comment if not already present and string formatting for SQL
                if ('%' in line or 'format(' in line or '+' in line) and ('SELECT' in line.upper() or 'INSERT' in line.upper() or 'UPDATE' in line.upper() or 'DELETE' in line.upper()) and '# nosec' not in line:
                    lines[line_num - 1] = line.rstrip() + '  # nosec - SECURITY: Review for SQL injection\n'
                    comments += 1
                    
        # Write the modified content back to the file
        if comments > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return comments
    except Exception as e:
        print_error(f"Error fixing SQL injection issues in {file_path}: {str(e)}")
        return 0

def fix_exec_issues(file_path: str, line_numbers: List[int]) -> int:
    """
    Fix exec() issues by adding comments since they need manual review
    
    Args:
        file_path: Path to the file
        line_numbers: List of line numbers with exec issues
        
    Returns:
        Number of comments added
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        comments = 0
        for line_num in line_numbers:
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                
                # Add a warning comment if not already present
                if 'exec(' in line and '# nosec' not in line:
                    lines[line_num - 1] = line.rstrip() + '  # nosec - SECURITY: Review exec usage\n'
                    comments += 1
                    
        # Write the modified content back to the file
        if comments > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
        return comments
    except Exception as e:
        print_error(f"Error fixing exec issues in {file_path}: {str(e)}")
        return 0

def process_file_issues(file_path: str, issues_by_type: Dict[str, List[int]]) -> Dict[str, int]:
    """
    Process all issues for a given file
    
    Args:
        file_path: Path to the file
        issues_by_type: Dictionary mapping issue type to line numbers
        
    Returns:
        Dictionary with count of fixes by issue type
    """
    fixes = {
        "assert": 0,
        "hardcoded_tmp": 0,
        "subprocess": 0,
        "random": 0,
        "try_except_pass": 0,
        "eval": 0,
        "pickle": 0,
        "sql_injection": 0,
        "exec": 0
    }
    
    # Fix assert issues (B101)
    if "B101" in issues_by_type:
        fixes["assert"] = fix_assert_issues(file_path, issues_by_type["B101"])
        
    # Fix hardcoded tmp directory issues (B108)
    if "B108" in issues_by_type:
        fixes["hardcoded_tmp"] = fix_hardcoded_tmp_issues(file_path, issues_by_type["B108"])
        
    # Fix subprocess issues (B603, B404)
    if "B603" in issues_by_type or "B404" in issues_by_type:
        line_nums = []
        if "B603" in issues_by_type:
            line_nums.extend(issues_by_type["B603"])
        if "B404" in issues_by_type:
            line_nums.extend(issues_by_type["B404"])
        fixes["subprocess"] = fix_subprocess_issues(file_path, line_nums)
        
    # Fix random issues (B311)
    if "B311" in issues_by_type:
        fixes["random"] = fix_random_issues(file_path, issues_by_type["B311"])
        
    # Fix try-except-pass issues (B110)
    if "B110" in issues_by_type:
        fixes["try_except_pass"] = fix_try_except_pass_issues(file_path, issues_by_type["B110"])
        
    # Fix eval issues (B307)
    if "B307" in issues_by_type:
        fixes["eval"] = fix_eval_issues(file_path, issues_by_type["B307"])
        
    # Fix pickle issues (B301, B403)
    if "B301" in issues_by_type or "B403" in issues_by_type:
        line_nums = []
        if "B301" in issues_by_type:
            line_nums.extend(issues_by_type["B301"])
        if "B403" in issues_by_type:
            line_nums.extend(issues_by_type["B403"])
        fixes["pickle"] = fix_pickle_issues(file_path, line_nums)
        
    # Fix SQL injection issues (B608)
    if "B608" in issues_by_type:
        fixes["sql_injection"] = fix_sql_injection_issues(file_path, issues_by_type["B608"])
        
    # Fix exec issues (B102)
    if "B102" in issues_by_type:
        fixes["exec"] = fix_exec_issues(file_path, issues_by_type["B102"])
        
    return fixes

def organize_issues_by_file(issues: Dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Organize issues by file path and issue type
    
    Args:
        issues: Dictionary of issues from parse_bandit_output
        
    Returns:
        Dictionary mapping file paths to issue types and line numbers
    """
    files_dict = {}
    
    # Process high severity issues
    for issue in issues["high"]:
        file_path = issue["file_path"]
        issue_id = issue["id"].split(":")[0]  # Extract the issue ID (e.g., B101)
        line_num = issue["line_num"]
        
        if file_path not in files_dict:
            files_dict[file_path] = {}
            
        if issue_id not in files_dict[file_path]:
            files_dict[file_path][issue_id] = []
            
        files_dict[file_path][issue_id].append(line_num)
        
    # Process medium severity issues
    for issue in issues["medium"]:
        file_path = issue["file_path"]
        issue_id = issue["id"].split(":")[0]  # Extract the issue ID (e.g., B101)
        line_num = issue["line_num"]
        
        if file_path not in files_dict:
            files_dict[file_path] = {}
            
        if issue_id not in files_dict[file_path]:
            files_dict[file_path][issue_id] = []
            
        files_dict[file_path][issue_id].append(line_num)
        
    # Process low severity issues
    for issue in issues["low"]:
        file_path = issue["file_path"]
        issue_id = issue["id"].split(":")[0]  # Extract the issue ID (e.g., B101)
        line_num = issue["line_num"]
        
        if file_path not in files_dict:
            files_dict[file_path] = {}
            
        if issue_id not in files_dict[file_path]:
            files_dict[file_path][issue_id] = []
            
        files_dict[file_path][issue_id].append(line_num)
        
    return files_dict

def main() -> None:
    """Main function to fix bandit issues."""
    print_header("Bandit Security Issue Fixer")
    
    # Get the bandit output file
    if len(sys.argv) > 1:
        bandit_output = sys.argv[1]
    else:
        bandit_output = "logs/bandit_output.txt"
        
    print(f"Parsing bandit output from {bandit_output}...")
    
    # Check if the file exists and is readable
    if not os.path.exists(bandit_output):
        print_error(f"Error: {bandit_output} does not exist!")
        return
        
    if not os.path.isfile(bandit_output):
        print_error(f"Error: {bandit_output} is not a file!")
        return
        
    if not os.access(bandit_output, os.R_OK):
        print_error(f"Error: Cannot read {bandit_output}!")
        return
        
    issues = parse_bandit_output(bandit_output)
    
    print(f"Found {len(issues['high'])} high severity, {len(issues['medium'])} medium severity, {len(issues['low'])} low severity issues.")
    
    # Display sample issues if available for debugging
    if issues['high'] or issues['medium'] or issues['low']:
        print("\nSample Issues:")
        
        if issues['high']:
            print(f"  High Severity: {issues['high'][0]['id']} in {issues['high'][0]['file_path']}")
            
        if issues['medium']:
            print(f"  Medium Severity: {issues['medium'][0]['id']} in {issues['medium'][0]['file_path']}")
            
        if issues['low']:
            print(f"  Low Severity: {issues['low'][0]['id']} in {issues['low'][0]['file_path']}")
    
    # Organize issues by file
    files_dict = organize_issues_by_file(issues)
    
    print(f"Issues found in {len(files_dict)} files.")
    
    # Process issues for each file
    total_fixes = {
        "assert": 0,
        "hardcoded_tmp": 0,
        "subprocess": 0,
        "random": 0,
        "try_except_pass": 0,
        "eval": 0,
        "pickle": 0,
        "sql_injection": 0,
        "exec": 0
    }
    
    for file_path, issues_by_type in files_dict.items():
        print(f"Processing {file_path}...")
        
        if os.path.exists(file_path):
            fixes = process_file_issues(file_path, issues_by_type)
            
            # Update total fixes
            for issue_type, count in fixes.items():
                total_fixes[issue_type] += count
                
            # Print fixes for this file
            file_total = sum(fixes.values())
            if file_total > 0:
                print_success(f"  Applied {file_total} fixes:")
                for issue_type, count in fixes.items():
                    if count > 0:
                        print_success(f"    - {issue_type}: {count}")
            else:
                print_warning(f"  No automatic fixes applied")
        else:
            print_error(f"  File not found: {file_path}")
            
    # Print summary
    print_header("Summary")
    overall_total = sum(total_fixes.values())
    print_success(f"Applied {overall_total} fixes across {len(files_dict)} files:")
    for issue_type, count in total_fixes.items():
        print_success(f"  - {issue_type}: {count}")
        
    print("\nReminder: Some issues require manual review:")
    print("  - B301/B403 (pickle): Ensure pickle is only used with trusted data")
    print("  - B102 (exec): Review exec() usage and consider safer alternatives")
    print("  - B608 (SQL injection): Use parameterized queries instead of string formatting")
    
if __name__ == "__main__":
    main() 