#!/usr/bin/env python3
"""
Purpose: Validate script organization and structure (Rule 7 enforcement)
Usage: python check-script-organization.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple

class ScriptOrganizationChecker:
    """Validates script organization according to Rule 7."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []
        
        # Expected script directory structure
        self.expected_structure = {
            'scripts/': {
                'dev/': 'Local development tools',
                'deploy/': 'Deployment and release scripts',
                'data/': 'Data manipulation and migration scripts',
                'utils/': 'General utility scripts',
                'test/': 'Testing and validation scripts',
                'pre-commit/': 'Pre-commit hook scripts',
                'agents/': 'AI agent-related scripts'
            }
        }
        
        # Script naming patterns
        self.naming_patterns = {
            'good': [
                r'^[a-z]+[-_]?[a-z0-9]*\.(?:py|sh)$',  # lowercase with hyphens/underscores
                r'^[a-z]+[-_]?[a-z0-9]*[-_]?[a-z0-9]*\.(?:py|sh)$'
            ],
            'bad': [
                r'^test\d+\.(?:py|sh)$',  # test1.py, test2.sh
                r'^temp.*\.(?:py|sh)$',   # temp.py, temporary.sh
                r'^[A-Z].*\.(?:py|sh)$',  # UpperCase.py
                r'^\d+.*\.(?:py|sh)$',    # 1_script.py
                r'^.*\s+.*\.(?:py|sh)$'   # spaces in filename
            ]
        }
    
    def check_script_location(self, filepath: Path) -> bool:
        """Check if script is in appropriate location."""
        relative_path = filepath.relative_to(self.project_root)
        path_parts = relative_path.parts
        
        # Check if it's a script file
        if filepath.suffix not in ['.py', '.sh']:
            return True
        
        # Special allowed locations
        allowed_root_scripts = ['setup.py', 'manage.py', 'deploy.sh']
        if len(path_parts) == 1 and filepath.name in allowed_root_scripts:
            return True
        
        # Scripts should be in scripts/ directory
        if len(path_parts) > 1 and path_parts[0] != 'scripts':
            # Allow scripts in specific directories
            allowed_dirs = ['tests', 'backend', 'agents', 'workflows', '.github']
            if path_parts[0] not in allowed_dirs:
                self.violations.append((
                    filepath,
                    f"Script not in scripts/ directory",
                    f"Move to scripts/{self._suggest_subdirectory(filepath)}/"
                ))
                return False
        
        # Check if in appropriate subdirectory
        if len(path_parts) > 1 and path_parts[0] == 'scripts':
            if len(path_parts) == 2:
                # Script directly in scripts/
                self.warnings.append((
                    filepath,
                    "Script in scripts/ root",
                    f"Consider moving to scripts/{self._suggest_subdirectory(filepath)}/"
                ))
            elif len(path_parts) > 2 and path_parts[1] not in self.expected_structure['scripts/']:
                self.warnings.append((
                    filepath,
                    f"Script in non-standard subdirectory: {path_parts[1]}",
                    "Use standard subdirectories: dev/, deploy/, data/, utils/, test/"
                ))
        
        return True
    
    def check_script_naming(self, filepath: Path) -> bool:
        """Check if script follows naming conventions."""
        filename = filepath.name
        
        # Check against bad patterns
        import re
        for pattern in self.naming_patterns['bad']:
            if re.match(pattern, filename):
                self.violations.append((
                    filepath,
                    f"Poor script naming: {filename}",
                    "Use descriptive lowercase names with hyphens (e.g., process-data.py)"
                ))
                return False
        
        # Check for descriptive names
        if len(filename.split('.')[0]) < 4:
            self.warnings.append((
                filepath,
                "Script name too short",
                "Use descriptive names that explain the script's purpose"
            ))
        
        return True
    
    def check_script_header(self, filepath: Path) -> bool:
        """Check if script has proper header."""
        if not filepath.exists():
            return True
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # Check first 10 lines
            
            has_shebang = False
            has_purpose = False
            has_usage = False
            
            for i, line in enumerate(lines):
                if i == 0 and filepath.suffix == '.sh' and line.startswith('#!'):
                    has_shebang = True
                if 'purpose:' in line.lower() or 'description:' in line.lower():
                    has_purpose = True
                if 'usage:' in line.lower():
                    has_usage = True
            
            if filepath.suffix == '.sh' and not has_shebang:
                self.violations.append((
                    filepath,
                    "Shell script missing shebang",
                    "Add #!/bin/bash or #!/bin/sh as first line"
                ))
            
            if not has_purpose:
                self.violations.append((
                    filepath,
                    "Script missing purpose/description",
                    "Add comment: # Purpose: <description>"
                ))
            
            if not has_usage:
                self.warnings.append((
                    filepath,
                    "Script missing usage information",
                    "Add comment: # Usage: <how to run>"
                ))
                
        except Exception as e:
            self.warnings.append((
                filepath,
                f"Could not read script: {e}",
                "Ensure script is readable"
            ))
        
        return True
    
    def check_script_permissions(self, filepath: Path) -> bool:
        """Check script permissions."""
        if not filepath.exists():
            return True
            
        try:
            mode = filepath.stat().st_mode
            is_executable = bool(mode & 0o111)
            
            if filepath.suffix == '.sh' and not is_executable:
                self.warnings.append((
                    filepath,
                    "Shell script not executable",
                    "Run: chmod +x " + str(filepath)
                ))
            elif filepath.suffix == '.py' and is_executable and 'scripts/' in str(filepath):
                # Python scripts in scripts/ dir should be executable if they're meant to be run directly
                with open(filepath, 'r') as f:
                    first_line = f.readline()
                    if not first_line.startswith('#!/usr/bin/env python'):
                        self.warnings.append((
                            filepath,
                            "Executable Python script missing shebang",
                            "Add #!/usr/bin/env python3 as first line"
                        ))
        except:
            pass
        
        return True
    
    def _suggest_subdirectory(self, filepath: Path) -> str:
        """Suggest appropriate subdirectory based on script name/content."""
        name_lower = filepath.stem.lower()
        
        if any(keyword in name_lower for keyword in ['deploy', 'release', 'provision']):
            return 'deploy'
        elif any(keyword in name_lower for keyword in ['test', 'check', 'validate', 'verify']):
            return 'test'
        elif any(keyword in name_lower for keyword in ['data', 'migrate', 'seed', 'export', 'import']):
            return 'data'
        elif any(keyword in name_lower for keyword in ['dev', 'local', 'setup', 'init']):
            return 'dev'
        else:
            return 'utils'

def main():
    """Main function to check script organization."""
    project_root = Path("/opt/sutazaiapp")
    checker = ScriptOrganizationChecker(project_root)
    
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    print("🔍 Checking script organization (Rule 7)...")
    
    scripts_checked = 0
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        
        # Only check script files
        if filepath.suffix not in ['.py', '.sh']:
            continue
            
        scripts_checked += 1
        
        # Run checks
        checker.check_script_location(filepath)
        checker.check_script_naming(filepath)
        checker.check_script_header(filepath)
        checker.check_script_permissions(filepath)
    
    # Report results
    if checker.violations:
        print(f"\n❌ Rule 7 Violations: Script organization issues in {len(checker.violations)} files")
        print("\n📋 Issues that must be fixed:")
        
        for filepath, issue, fix in checker.violations:
            print(f"\n  File: {filepath}")
            print(f"  Issue: {issue}")
            print(f"  Fix: {fix}")
        
        print("\n📋 Script organization requirements:")
        print("  1. All scripts must be in /scripts/ directory (with few exceptions)")
        print("  2. Use subdirectories: dev/, deploy/, data/, utils/, test/")
        print("  3. Use lowercase hyphenated names (e.g., process-data.py)")
        print("  4. Include purpose and usage in header comments")
        print("  5. Shell scripts need shebang and execute permissions")
        
        return 1
    
    if checker.warnings:
        print("\n⚠️  Script organization warnings:")
        for filepath, warning, suggestion in checker.warnings:
            print(f"\n  File: {filepath}")
            print(f"  Warning: {warning}")
            print(f"  Suggestion: {suggestion}")
    
    if scripts_checked > 0:
        print(f"✅ Rule 7: Script organization check passed for {scripts_checked} files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())