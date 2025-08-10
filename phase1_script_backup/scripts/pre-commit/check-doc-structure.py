#!/usr/bin/env python3
"""
Purpose: Validate documentation structure and organization (Rule 6 enforcement)
Usage: python check-doc-structure.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import re
from pathlib import Path

class DocStructureChecker:
    """Validates documentation structure according to Rule 6."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []
        
    def check_doc_location(self, filepath: Path) -> bool:
        """Check if documentation is in appropriate location."""
        relative_path = filepath.relative_to(self.project_root)
        path_parts = relative_path.parts
        
        # Allowed documentation locations
        allowed_locations = ['docs/', 'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE']
        
        # Check if it's in an allowed location
        if len(path_parts) == 1:
            # Root level - only specific files allowed
            if filepath.name not in ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE', 'LICENSE.md']:
                if filepath.suffix == '.md':
                    self.violations.append((
                        filepath,
                        "Documentation file in project root",
                        "Move to docs/ directory"
                    ))
                    return False
        elif path_parts[0] != 'docs':
            # Not in docs directory - check if it's a valid README
            if filepath.name == 'README.md':
                # README files allowed in subdirectories
                return True
            else:
                self.violations.append((
                    filepath,
                    "Documentation scattered outside docs/",
                    "Centralize in docs/ directory"
                ))
                return False
        
        return True
    
    def check_doc_naming(self, filepath: Path) -> bool:
        """Check documentation naming conventions."""
        filename = filepath.name
        
        # Check for proper naming
        if not re.match(r'^[a-z0-9\-_]+\.(md|rst|txt)$', filename.lower()):
            if filename not in ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE', 'LICENSE.md']:
                self.violations.append((
                    filepath,
                    f"Non-standard documentation naming: {filename}",
                    "Use lowercase with hyphens (e.g., api-guide.md)"
                ))
                return False
        
        # Check for version numbers in filenames
        if re.search(r'v\d+|_v\d+|_old|_new|_backup', filename.lower()):
            self.violations.append((
                filepath,
                "Version/backup indicator in filename",
                "Use git for versioning, not filename suffixes"
            ))
            return False
        
        return True
    
    def check_doc_structure(self, filepath: Path) -> bool:
        """Check internal documentation structure."""
        if not filepath.exists() or filepath.suffix not in ['.md', '.rst']:
            return True
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Check for title
            has_title = False
            if lines:
                if lines[0].startswith('# ') or (len(lines) > 1 and lines[1].startswith('===')):
                    has_title = True
            
            if not has_title:
                self.warnings.append((
                    filepath,
                    "Missing document title",
                    "Add # Title as first line"
                ))
            
            # Check for sections
            headings = [line for line in lines if line.startswith('#')]
            if len(content) > 500 and len(headings) < 2:
                self.warnings.append((
                    filepath,
                    "Long document without sections",
                    "Break into sections with ## headings"
                ))
            
            # Check for TOC in long documents
            if len(content) > 2000 and 'table of contents' not in content.lower():
                self.warnings.append((
                    filepath,
                    "Long document without table of contents",
                    "Add TOC for documents over 2000 chars"
                ))
                
        except Exception as e:
            self.warnings.append((
                filepath,
                f"Could not read documentation: {e}",
                "Ensure file is valid"
            ))
        
        return True

def check_doc_duplication(files: List[Path]) -> List[str]:
    """Check for potential documentation duplication."""
    duplicates = []
    
    # Group files by similar names
    name_groups = {}
    for filepath in files:
        base_name = filepath.stem.lower()
        # Remove common suffixes
        base_name = re.sub(r'[-_](guide|manual|doc|documentation|reference|api|readme)$', '', base_name)
        
        if base_name not in name_groups:
            name_groups[base_name] = []
        name_groups[base_name].append(filepath)
    
    # Find groups with multiple files
    for base_name, group_files in name_groups.items():
        if len(group_files) > 1:
            duplicates.append(f"Potential duplicates for '{base_name}': " + 
                            ", ".join(str(f) for f in group_files))
    
    return duplicates

def main():
    """Main function to check documentation structure."""
    project_root = Path("/opt/sutazaiapp")
    checker = DocStructureChecker(project_root)
    
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    print("🔍 Checking documentation structure (Rule 6)...")
    
    doc_files = []
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        
        # Only check documentation files
        if filepath.suffix not in ['.md', '.rst', '.txt']:
            continue
            
        doc_files.append(filepath)
        
        # Run checks
        checker.check_doc_location(filepath)
        checker.check_doc_naming(filepath)
        checker.check_doc_structure(filepath)
    
    # Check for duplication across all doc files
    if len(doc_files) > 1:
        duplicates = check_doc_duplication(doc_files)
        for dup in duplicates:
            checker.warnings.append((
                Path("docs/"),
                dup,
                "Consider consolidating duplicate documentation"
            ))
    
    # Report results
    if checker.violations:
        print(f"\n❌ Rule 6 Violations: Documentation structure issues")
        print("\n📋 Issues that must be fixed:")
        
        for filepath, issue, fix in checker.violations:
            print(f"\n  File: {filepath}")
            print(f"  Issue: {issue}")
            print(f"  Fix: {fix}")
        
        return 1
    
    if checker.warnings:
        print("\n⚠️  Documentation warnings:")
        for filepath, warning, suggestion in checker.warnings:
            print(f"\n  File: {filepath}")
            print(f"  Warning: {warning}")
            print(f"  Suggestion: {suggestion}")
    
    print("✅ Rule 6: Documentation structure check passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())