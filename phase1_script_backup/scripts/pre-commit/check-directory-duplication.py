#!/usr/bin/env python3
"""
Purpose: Check for duplicate directories (Rule 9 enforcement)
Usage: python check-directory-duplication.py
Requirements: Python 3.8+
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import os
from pathlib import Path
from typing import List, Dict, Set
import re

def find_duplicate_patterns(project_root: Path) -> Dict[str, List[Path]]:
    """Find directories that might be duplicates based on naming patterns."""
    duplicates = {}
    
    # Patterns that indicate potential duplication
    duplicate_patterns = [
        (r'(.+?)[-_]?(old|new|copy|backup|temp|tmp|orig|v\d+)$', 'version/backup variants'),
        (r'(.+?)[-_]?(backend|frontend|api|web|app)[-_]?(old|new|v\d+)$', 'component variants'),
        (r'(.+?)[-_]?(dev|test|prod|staging)$', 'environment variants'),
        (r'(.+?)[-_]?\d+$', 'numbered variants'),
    ]
    
    # Walk through all directories
    all_dirs = []
    for root, dirs, _ in os.walk(project_root):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
        
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            relative_path = dir_path.relative_to(project_root)
            
            # Skip archive directory
            if 'archive' in str(relative_path).split(os.sep):
                continue
                
            all_dirs.append((dir_name, dir_path))
    
    # Check for pattern-based duplicates
    for pattern_regex, pattern_type in duplicate_patterns:
        pattern = re.compile(pattern_regex, re.IGNORECASE)
        matched_groups = {}
        
        for dir_name, dir_path in all_dirs:
            match = pattern.match(dir_name)
            if match:
                base_name = match.group(1)
                if base_name not in matched_groups:
                    matched_groups[base_name] = []
                matched_groups[base_name].append(dir_path)
        
        # Add groups with multiple matches
        for base_name, paths in matched_groups.items():
            if len(paths) > 1:
                key = f"{base_name} ({pattern_type})"
                if key not in duplicates:
                    duplicates[key] = []
                duplicates[key].extend(paths)
    
    return duplicates

def check_content_similarity(dir1: Path, dir2: Path) -> float:
    """Check how similar two directories are based on their contents."""
    def get_dir_structure(path: Path) -> Set[str]:
        structure = set()
        for item in path.rglob('*'):
            if item.is_file():
                relative = item.relative_to(path)
                structure.add(str(relative))
        return structure
    
    try:
        struct1 = get_dir_structure(dir1)
        struct2 = get_dir_structure(dir2)
        
        if not struct1 or not struct2:
            return 0.0
            
        intersection = len(struct1 & struct2)
        union = len(struct1 | struct2)
        
        return intersection / union if union > 0 else 0.0
    except (IOError, OSError, FileNotFoundError) as e:
        logger.warning(f"Exception caught, returning: {e}")
        return 0.0

def check_specific_duplications(project_root: Path) -> List[str]:
    """Check for specific known duplication patterns."""
    violations = []
    
    # Check for multiple backend/frontend directories
    backend_dirs = list(project_root.glob('**/backend*'))
    frontend_dirs = list(project_root.glob('**/frontend*'))
    
    # Filter out legitimate uses
    backend_dirs = [d for d in backend_dirs if d.is_dir() and 'archive' not in str(d)]
    frontend_dirs = [d for d in frontend_dirs if d.is_dir() and 'archive' not in str(d)]
    
    if len(backend_dirs) > 1:
        violations.append(f"Multiple backend directories found: {len(backend_dirs)}")
        for dir_path in backend_dirs:
            violations.append(f"  - {dir_path.relative_to(project_root)}")
    
    if len(frontend_dirs) > 1:
        violations.append(f"Multiple frontend directories found: {len(frontend_dirs)}")
        for dir_path in frontend_dirs:
            violations.append(f"  - {dir_path.relative_to(project_root)}")
    
    # Check for script duplications
    script_dirs = []
    for pattern in ['scripts*', 'script*', 'utils*', 'tools*', 'bin*']:
        found = list(project_root.glob(f'**/{pattern}'))
        script_dirs.extend([d for d in found if d.is_dir() and 'archive' not in str(d)])
    
    if len(script_dirs) > 2:  # Allow scripts/ and bin/ as standard
        violations.append(f"Multiple script/utility directories found: {len(script_dirs)}")
        for dir_path in sorted(set(script_dirs)):
            violations.append(f"  - {dir_path.relative_to(project_root)}")
    
    return violations

def main():
    """Main function to check for directory duplications."""
    project_root = Path("/opt/sutazaiapp")
    
    print("🔍 Checking for duplicate directories (Rule 9)...")
    
    # Find potential duplicates
    duplicates = find_duplicate_patterns(project_root)
    violations = check_specific_duplications(project_root)
    
    if duplicates or violations:
        print("\n❌ Rule 9 Violation: Directory duplication detected")
        
        if violations:
            print("\n📋 Specific violations:")
            for violation in violations:
                print(violation)
        
        if duplicates:
            print("\n📋 Potential duplicate directories:")
            for pattern, paths in duplicates.items():
                print(f"\n  Pattern: {pattern}")
                for path in sorted(paths):
                    print(f"    - {path.relative_to(project_root)}")
                
                # Check content similarity for small groups
                if len(paths) == 2:
                    similarity = check_content_similarity(paths[0], paths[1])
                    if similarity > 0.5:
                        print(f"    ⚠️  High content similarity: {similarity:.0%}")
        
        print("\n📋 How to fix:")
        print("  1. Consolidate duplicate directories into a single location")
        print("  2. Remove version-specific directories (use git for versioning)")
        print("  3. Use branches instead of duplicate directories for different environments")
        print("  4. Archive old versions if necessary, don't keep in main codebase")
        print("  5. Update all references to point to the consolidated location")
        
        return 1
    
    print("✅ Rule 9: No directory duplication detected")
    return 0

if __name__ == "__main__":
    sys.exit(main())