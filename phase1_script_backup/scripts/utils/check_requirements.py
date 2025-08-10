#!/usr/bin/env python3
"""
Check and validate requirements files
Part of CLAUDE.md hygiene enforcement
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_requirements(filepath):
    """Parse a requirements file and return list of packages"""
    packages = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Skip special directives
                if line.startswith('-'):
                    continue
                # Extract package name
                match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if match:
                    packages.append(match.group(1).lower())
    except Exception:
        pass
    return packages

def check_requirements_files():
    """Check all requirements files for issues"""
    violations = []
    all_packages = defaultdict(list)
    requirements_files = []
    
    # Find all requirements files
    for pattern in ['**/requirements*.txt', '**/requirements/*.txt']:
        for filepath in Path('.').glob(pattern):
            # Skip virtual environments
            if any(part in str(filepath) for part in 
                   ['venv', 'node_modules', '.git', '__pycache__']):
                continue
            requirements_files.append(filepath)
    
    # Check each file
    for filepath in requirements_files:
        packages = parse_requirements(filepath)
        
        # Track where each package is defined
        for pkg in packages:
            all_packages[pkg].append(str(filepath))
        
        # Check for issues in this file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded versions that might conflict
            hardcoded_versions = re.findall(r'^([a-zA-Z0-9_-]+)==([0-9.]+)', 
                                           content, re.MULTILINE)
            
            # Check for git dependencies (security risk)
            git_deps = re.findall(r'git\+https?://.*', content)
            if git_deps:
                violations.append(f"{filepath}: Contains git dependencies (security risk)")
            
            # Check for local file dependencies
            local_deps = re.findall(r'file://.*', content)
            if local_deps:
                violations.append(f"{filepath}: Contains local file dependencies")
                
        except Exception:
            pass
    
    # Check for duplicate requirements files
    if len(requirements_files) > 3:  # Allow main, dev, and test
        violations.append(f"Too many requirements files ({len(requirements_files)}). Should consolidate.")
        violations.append("Files found:")
        for f in requirements_files:
            violations.append(f"  - {f}")
    
    # Check for packages defined in multiple places
    duplicates = {pkg: files for pkg, files in all_packages.items() 
                  if len(files) > 1}
    
    if duplicates:
        violations.append("\nPackages defined in multiple files:")
        for pkg, files in duplicates.items():
            violations.append(f"\n'{pkg}' appears in:")
            for f in files:
                violations.append(f"  - {f}")
    
    return violations

def suggest_consolidation():
    """Suggest how to consolidate requirements"""
    suggestions = [
        "\nSuggested requirements structure:",
        "  - requirements.txt          # Production dependencies only",
        "  - requirements-dev.txt      # Development tools (includes -r requirements.txt)",
        "  - requirements-test.txt     # Testing tools (includes -r requirements.txt)",
        "",
        "Use pip-tools to manage dependencies:",
        "  1. Create requirements.in with direct dependencies",
        "  2. Run: pip-compile requirements.in",
        "  3. This generates requirements.txt with pinned versions",
        "",
        "Example requirements.in:",
        "  fastapi",
        "  uvicorn[standard]",
        "  pydantic>=2.0",
        "  # Add direct dependencies only, not sub-dependencies"
    ]
    return suggestions

def main():
    """Check requirements files for hygiene violations"""
    print("Checking requirements files...")
    
    violations = check_requirements_files()
    
    if violations:
        print("ERROR: Requirements file issues detected!")
        print("-" * 60)
        for v in violations:
            print(v)
        print("-" * 60)
        
        suggestions = suggest_consolidation()
        for s in suggestions:
            print(s)
        
        return 1
    
    print("✓ Requirements files are properly organized")
    return 0

if __name__ == "__main__":
    sys.exit(main())