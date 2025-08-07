#!/usr/bin/env python3
"""Debug safety path checking"""

import os

protected_paths = {'/etc', '/boot', '/usr', '/bin', '/sbin', '/lib', '/proc', '/sys', '/dev'}
user_protected_patterns = {'/home/*/Documents', '/home/*/Desktop', '/home/*/Pictures'}

def _is_safe_path(path: str) -> bool:
    """Check if a path is safe to analyze/modify"""
    path = os.path.abspath(path)
    print(f"Checking path: {path}")
    
    # Never touch protected system paths
    for protected in protected_paths:
        if path.startswith(protected):
            print(f"  Blocked by protected path: {protected}")
            return False
    
    # Check user protected patterns
    for pattern in user_protected_patterns:
        if '*' in pattern:
            pattern_parts = pattern.split('*')
            if len(pattern_parts) == 2 and path.startswith(pattern_parts[0]) and path.endswith(pattern_parts[1]):
                print(f"  Blocked by pattern: {pattern}")
                return False
        elif path.startswith(pattern):
            print(f"  Blocked by pattern: {pattern}")
            return False
    
    print(f"  Path is safe: True")
    return True

test_paths = [
    '/tmp/storage_test_environment/duplicates',
    '/var/tmp/storage_test/duplicates', 
    '/opt/sutazaiapp/storage_test_data/duplicates',
    '/tmp/test',
    '/var/log/test',
    '/home/user/test'
]

print("Testing path safety:")
for path in test_paths:
    result = _is_safe_path(path)
    print(f"{path}: {'SAFE' if result else 'BLOCKED'}")
    print()