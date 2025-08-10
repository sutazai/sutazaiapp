#!/usr/bin/env python3
"""
Purpose: Detect and prevent garbage files from being committed (Rule 13)
Usage: python check-garbage-files.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import re
from pathlib import Path

# Garbage file patterns
GARBAGE_PATTERNS = [
    # Backup files
    r'.*\.backup\d*$',
    r'.*\.bak$',
    r'.*\.tmp$',
    r'.*\.temp$',
    r'.*~$',
    r'.*\.old$',
    r'.*\.orig$',
    r'.*\.save$',
    r'.*\.swp$',
    r'.*\.swo$',
    
    # Copy variants
    r'.*[-_]copy(\d+)?\..*$',
    r'.*[-_]backup(\d+)?\..*$',
    r'.*[-_]old(\d+)?\..*$',
    r'.*[-_]temp(\d+)?\..*$',
    r'.*\s+copy(\s+\d+)?\..*$',
    r'Copy\s+of\s+.*',
    
    # Version variants
    r'.*[-_]v\d+\..*$',
    r'.*[-_]final[-_]?v?\d*\..*$',
    r'.*[-_]test\d*\..*$',
    r'.*[-_]draft\d*\..*$',
    
    # WIP/TODO files
    r'.*[-_]wip\..*$',
    r'.*[-_]todo\..*$',
    r'.*[-_]fixme\..*$',
    r'.*[-_]hack\..*$',
    r'.*[-_]quick[-_]?fix\..*$',
    
    # Personal/debug files
    r'test\d+\.(py|js|sh)$',
    r'debug\d*\.(py|js|sh)$',
    r'scratch\.(py|js|sh)$',
    r'untitled\d*\..*$',
    r'asdf\..*$',
    r'foo\..*$',
    r'bar\..*$',
    r'temp_.*',
    
    # IDE/editor files
    r'\.DS_Store$',
    r'Thumbs\.db$',
    r'desktop\.ini$',
    r'\.idea/.*',
    r'\.vscode/.*',
    r'\._.*',  # macOS resource forks
    
    # Log files (unless in designated log directory)
    r'.*\.log\.\d+$',
    r'.*\.log\.gz$',
    
    # Archive files in source
    r'.*\.(zip|tar|gz|bz2|7z|rar)$',
    
    # Compiled/generated files
    r'.*\.pyc$',
    r'.*\.pyo$',
    r'.*\.class$',
    r'.*\.o$',
    r'.*\.so$',
    r'.*\.dll$',
    r'.*\.exe$',
    
    # AGI backup files (specific to this project)
    r'.*\.agi_backup$',
    r'.*\.fantasy.*$',
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in GARBAGE_PATTERNS]

# Allowed exceptions
ALLOWED_PATHS = {
    'archive/',
    '.git/',
    'node_modules/',
    'venv/',
    '.venv/',
    '__pycache__/',
    'build/',
    'dist/',
    '.pytest_cache/',
}

def is_garbage_file(filepath: Path) -> bool:
    """Check if a file matches garbage patterns."""
    # Check if in allowed directory
    path_str = str(filepath)
    for allowed in ALLOWED_PATHS:
        if allowed in path_str:
            return False
    
    # Check against patterns
    filename = filepath.name
    for pattern in COMPILED_PATTERNS:
        if pattern.match(filename):
            return True
    
    # Check for multiple extensions (common in backups)
    parts = filename.split('.')
    if len(parts) > 3:  # e.g., file.txt.bak.old
        return True
    
    # Check for date stamps in filenames (often indicates copies)
    date_pattern = re.compile(r'.*(\d{4}[-_]?\d{2}[-_]?\d{2}|\d{8}).*')
    if date_pattern.match(filename) and not any(
        ok in filename for ok in ['changelog', 'release', 'version']
    ):
        # Likely a dated backup
        return True
    
    return False

def check_file_content(filepath: Path) -> List[str]:
    """Check file content for garbage indicators."""
    issues = []
    
    # Skip binary files
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars
    except (IOError, OSError, FileNotFoundError) as e:
        logger.warning(f"Exception caught, returning: {e}")
        return issues
    
    # Check for obvious test/debug content
    garbage_content_patterns = [
        r'asdfasdf',
        r'testtest',
        r'xxxxxx',
        r'DELETE\s*ME',
        r'REMOVE\s*THIS',
        r'TEMPORARY\s*HACK',
    ]
    
    for pattern in garbage_content_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Contains garbage content pattern: {pattern}")
    
    return issues

def main():
    """Main function to check for garbage files."""
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    garbage_files = []
    content_issues = {}
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        
        # Check if file exists
        if not filepath.exists():
            continue
        
        # Check filename
        if is_garbage_file(filepath):
            garbage_files.append(filepath)
        
        # Check content for certain file types
        if filepath.suffix in ['.py', '.js', '.sh', '.txt', '.md']:
            issues = check_file_content(filepath)
            if issues:
                content_issues[filepath] = issues
    
    # Report violations
    if garbage_files or content_issues:
        print("❌ Rule 13 Violation: Garbage files detected")
        
        if garbage_files:
            print("\n📋 Garbage files that must not be committed:")
            for filepath in garbage_files:
                print(f"  - {filepath}")
                
                # Suggest action
                if 'backup' in str(filepath).lower() or 'old' in str(filepath).lower():
                    print("    → Archive this file if needed, don't commit")
                elif 'tmp' in str(filepath).lower() or 'temp' in str(filepath).lower():
                    print("    → Delete this temporary file")
                elif re.search(r'test\d+', str(filepath)):
                    print("    → Use proper test framework, not numbered test files")
        
        if content_issues:
            print("\n📋 Files with garbage content:")
            for filepath, issues in content_issues.items():
                print(f"  - {filepath}")
                for issue in issues:
                    print(f"    → {issue}")
        
        print("\n📋 How to fix:")
        print("  1. Delete temporary, backup, and test files")
        print("  2. Use git for version control, not file copies")
        print("  3. Archive old files separately if needed")
        print("  4. Clean up debug code and test data")
        print("  5. Use .gitignore for files that shouldn't be tracked")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())