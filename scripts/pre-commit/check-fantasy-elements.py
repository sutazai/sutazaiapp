#!/usr/bin/env python3
"""
Purpose: Check for fantasy/placeholder code elements (Rule 1 enforcement)
Usage: python check-fantasy-elements.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

# Fantasy/placeholder patterns to detect
FANTASY_PATTERNS = [
    # Magic/wizard terms
    r'\b(magic|wizard|teleport|telekinesis|supernatural)\b',
    r'\bmagic[A-Z]\w*\b',  # magicHandler, magicService, etc.
    r'\bwizard[A-Z]\w*\b',  # wizardService, wizardModule, etc.
    
    # Placeholder functions
    r'#\s*TODO:\s*(magically|imagine|someday|future\s+AI)',
    r'#\s*FIXME:\s*(magic|wizard|teleport)',
    r'\bsuper(?:Intuitive|Smart|Intelligent)AI\b',
    r'\b(?:magic|wizard|fantasy)(?:Service|Handler|Module|Function)\b',
    
    # Unrealistic abstractions
    r'\bblack[_-]?box\b',
    r'\bauto[_-]?magic\b',
    r'\bself[_-]?healing[_-]?magic\b',
    
    # Placeholder data
    r'dummy_(?:data|function|module)',
    r'fake_(?:implementation|service)',
    r'placeholder_\w+',
    r'TODO:\s*(?:add|implement)\s+(?:real|actual)',
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FANTASY_PATTERNS]

def check_file_for_fantasy_elements(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a single file for fantasy elements."""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for pattern in COMPILED_PATTERNS:
                match = pattern.search(line)
                if match:
                    violations.append((
                        line_num,
                        match.group(0),
                        line.strip()
                    ))
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        
    return violations

def main():
    """Main function to check files passed as arguments."""
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
        
    total_violations = 0
    files_with_violations = []
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        
        # Skip non-source files
        if filepath.suffix not in ['.py', '.js', '.ts', '.java', '.go', '.cpp', '.c']:
            continue
            
        violations = check_file_for_fantasy_elements(filepath)
        
        if violations:
            files_with_violations.append(filepath)
            total_violations += len(violations)
            
            print(f"\n❌ Fantasy elements found in {filepath}:")
            for line_num, match, line in violations:
                print(f"  Line {line_num}: '{match}' in: {line}")
    
    if total_violations > 0:
        print(f"\n❌ Rule 1 Violation: Found {total_violations} fantasy elements in {len(files_with_violations)} files")
        print("\n📋 How to fix:")
        print("  1. Replace fantasy terms with concrete, real implementations")
        print("  2. Remove placeholder comments and implement actual functionality")
        print("  3. Use descriptive, non-metaphorical names (e.g., 'emailSender' not 'magicMailer')")
        print("  4. Document real libraries and APIs being used")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())