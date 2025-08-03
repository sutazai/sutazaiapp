#!/usr/bin/env python3
"""
Check for duplicate files and functionality
Part of CLAUDE.md hygiene enforcement
"""

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

def get_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def find_duplicate_files():
    """Find files with identical content"""
    file_hashes = defaultdict(list)
    
    # Scan all files
    for filepath in Path('.').rglob('*'):
        if filepath.is_file():
            # Skip certain directories
            if any(part in str(filepath) for part in 
                   ['venv', 'node_modules', '.git', '__pycache__', 'archive']):
                continue
            
            file_hash = get_file_hash(filepath)
            if file_hash:
                file_hashes[file_hash].append(str(filepath))
    
    # Find duplicates
    duplicates = {hash_val: files for hash_val, files in file_hashes.items() 
                  if len(files) > 1}
    
    return duplicates

def find_similar_names():
    """Find files with similar names that might be duplicates"""
    similar_groups = defaultdict(list)
    
    # Common duplicate patterns
    patterns = [
        (r'(.*)\.backup.*', 'backup file'),
        (r'(.*)\.old.*', 'old file'),
        (r'(.*)\.copy.*', 'copy file'),
        (r'(.*)_v\d+.*', 'versioned file'),
        (r'(.*)_backup.*', 'backup file'),
        (r'(.*)_old.*', 'old file'),
        (r'(.*)_copy.*', 'copy file'),
        (r'(.*)\.fantasy.*', 'fantasy file'),
    ]
    
    for filepath in Path('.').rglob('*'):
        if filepath.is_file():
            filename = filepath.name
            for pattern, group_type in patterns:
                import re
                match = re.match(pattern, filename)
                if match:
                    base_name = match.group(1)
                    similar_groups[base_name].append((str(filepath), group_type))
    
    return similar_groups

def find_duplicate_functions():
    """Find duplicate function definitions in Python files"""
    function_defs = defaultdict(list)
    
    import re
    func_pattern = re.compile(r'^def\s+(\w+)\s*\(')
    
    for filepath in Path('.').glob('**/*.py'):
        # Skip virtual environments
        if any(part in str(filepath) for part in 
               ['venv', 'node_modules', '.git', '__pycache__']):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    match = func_pattern.match(line.strip())
                    if match:
                        func_name = match.group(1)
                        function_defs[func_name].append((str(filepath), line_num))
        except Exception:
            pass
    
    # Find duplicates (same function name in multiple files)
    duplicate_funcs = {name: locations for name, locations in function_defs.items() 
                       if len(locations) > 1 and name not in ['__init__', 'main', 'setup']}
    
    return duplicate_funcs

def main():
    """Check for all types of duplicates"""
    print("Checking for duplicate files and functionality...")
    
    violations = []
    
    # Check for identical files
    duplicate_files = find_duplicate_files()
    if duplicate_files:
        violations.append("DUPLICATE FILES FOUND:")
        for hash_val, files in duplicate_files.items():
            violations.append(f"\nIdentical content in:")
            for f in files:
                violations.append(f"  - {f}")
    
    # Check for similar names (backups, copies, etc.)
    similar_names = find_similar_names()
    if similar_names:
        violations.append("\nBACKUP/COPY FILES FOUND:")
        for base_name, files in similar_names.items():
            violations.append(f"\nSimilar to '{base_name}':")
            for f, file_type in files:
                violations.append(f"  - {f} ({file_type})")
    
    # Check for duplicate functions
    duplicate_funcs = find_duplicate_functions()
    if duplicate_funcs:
        violations.append("\nDUPLICATE FUNCTIONS FOUND:")
        for func_name, locations in duplicate_funcs.items():
            violations.append(f"\nFunction '{func_name}' defined in:")
            for filepath, line_num in locations:
                violations.append(f"  - {filepath}:{line_num}")
    
    if violations:
        print("ERROR: Duplicates detected!")
        print("-" * 60)
        for v in violations:
            print(v)
        print("-" * 60)
        print("\nActions required:")
        print("  - Remove backup and copy files")
        print("  - Consolidate duplicate implementations")
        print("  - Use single source of truth")
        return 1
    
    print("✓ No duplicates found")
    return 0

if __name__ == "__main__":
    sys.exit(main())