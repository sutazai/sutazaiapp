#!/usr/bin/env python3
"""
Check for hardcoded secrets in the codebase
Part of CLAUDE.md hygiene enforcement
"""

import os
import re
import sys
from pathlib import Path

# Patterns that indicate potential secrets
SECRET_PATTERNS = [
    r'password\s*=\s*["\'][^"\']+["\']',
    r'api_key\s*=\s*["\'][^"\']+["\']',
    r'secret\s*=\s*["\'][^"\']+["\']',
    r'token\s*=\s*["\'][^"\']+["\']',
    r'[A-Z_]+_PASSWORD\s*=\s*["\'][^"\']+["\']',
    r'[A-Z_]+_KEY\s*=\s*["\'][^"\']+["\']',
    r'[A-Z_]+_SECRET\s*=\s*["\'][^"\']+["\']',
]

# Allowed patterns (environment variables, placeholders)
ALLOWED_PATTERNS = [
    r'os\.environ',
    r'os\.getenv',
    r'\$\{.*\}',
    r'<.*>',
    r'xxx+',
    r'your[_-]?password',
    r'your[_-]?key',
    r'example',
    r'placeholder',
    r'password_field',
    r'secret_key',
    r'api_key',
    r'\*{3,}',  # Masked secrets like "*****"
    r'"\.\.\."',  # Example patterns like "..."
    r'ghp_\.\.\.',  # Example GitHub token pattern
    r'change_me',  # Placeholder values
    r'LATEST_SECURE',  # Placeholder for dependency versions
    r'# Never expose secrets',  # Comment indicating proper masking
    r'# Example pattern',  # Comment for regex examples
]

def check_file(filepath):
    """Check a single file for hardcoded secrets"""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern in SECRET_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's an allowed pattern
                    is_allowed = any(re.search(allowed, line, re.IGNORECASE) 
                                   for allowed in ALLOWED_PATTERNS)
                    
                    if not is_allowed:
                        violations.append({
                            'file': filepath,
                            'line': line_num,
                            'content': line.strip()
                        })
    except Exception:
        pass  # Skip files that can't be read
    
    return violations

def main():
    """Check all Python and config files for secrets"""
    violations = []
    
    # File patterns to check
    patterns = ['**/*.py', '**/*.json', '**/*.yaml', '**/*.yml', 
                '**/*.env', '**/*.conf', '**/*.config']
    
    for pattern in patterns:
        for filepath in Path('.').glob(pattern):
            # Skip virtual environments, dependencies, and backup directories
            if any(part in str(filepath) for part in 
                   ['venv', 'node_modules', '.git', '__pycache__', 'backup', 'archive', 
                    'security-scan-results/backups', 'semgrep_custom_rules.yaml']):
                continue
                
            file_violations = check_file(filepath)
            violations.extend(file_violations)
    
    if violations:
        print("ERROR: Hardcoded secrets detected!")
        print("-" * 60)
        for v in violations:
            print(f"{v['file']}:{v['line']} - {v['content']}")
        print("-" * 60)
        print(f"Total violations: {len(violations)}")
        print("\nUse environment variables instead of hardcoded values.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())