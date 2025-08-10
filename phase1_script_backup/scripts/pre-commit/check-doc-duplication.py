#!/usr/bin/env python3
"""
Purpose: Check for documentation duplication (Rule 15 enforcement)
Usage: python check-doc-duplication.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import hashlib
from pathlib import Path
import re

def calculate_content_hash(content: str) -> str:
    """Calculate hash of normalized content."""
    # Normalize whitespace and case
    normalized = re.sub(r'\s+', ' ', content.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def check_files(files: List[Path]) -> int:
    """Check for duplicate documentation content."""
    content_hashes = {}
    duplicates = []
    
    for filepath in files:
        if filepath.suffix not in ['.md', '.rst', '.txt']:
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content_hash = calculate_content_hash(content)
            
            if content_hash in content_hashes:
                duplicates.append((filepath, content_hashes[content_hash]))
            else:
                content_hashes[content_hash] = filepath
                
        except (IOError, OSError, FileNotFoundError) as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
    
    if duplicates:
        print("❌ Rule 15 Violation: Documentation duplication found")
        print("\n📋 Duplicate documentation files:")
        for file1, file2 in duplicates:
            print(f"  - {file1} duplicates {file2}")
        return 1
    
    print("✅ Rule 15: No documentation duplication detected")
    return 0

def main():
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
        
    files = [Path(f) for f in sys.argv[1:]]
    return check_files(files)

if __name__ == "__main__":
    sys.exit(main())