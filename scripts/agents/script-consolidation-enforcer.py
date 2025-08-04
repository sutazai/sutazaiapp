#!/usr/bin/env python3
"""
Script Consolidation and Python Sanity Enforcer

Purpose: Enforces CLAUDE.md Rules 7 & 8 by auditing, consolidating, and standardizing
         all scripts across the codebase to eliminate duplication, enforce naming
         conventions, and ensure Python scripts follow production standards.

Usage: python script-consolidation-enforcer.py [--mode MODE] [--fix] [--dry-run]
Requirements: pathlib, ast, difflib, shutil, logging, argparse, hashlib, subprocess
"""

import argparse
import ast
import difflib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import textwrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/script-consolidation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ScriptViolation:
    """Represents a script standards violation"""
    
    def __init__(self, file_path: Path, violation_type: str, description: str, 
                 severity: str = "medium", fix_suggestion: str = None):
        self.file_path = file_path
        self.violation_type = violation_type
        self.description = description
        self.severity = severity
        self.fix_suggestion = fix_suggestion
        self.timestamp = datetime.now()


class ScriptAnalyzer:
    """Analyzes scripts for standards compliance and duplication"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.violations: List[ScriptViolation] = []
        self.script_fingerprints: Dict[str, List[Path]] = defaultdict(list)
        self.scripts_by_functionality: Dict[str, List[Path]] = defaultdict(list)
        
    def compute_script_fingerprint(self, file_path: Path) -> str:
        """Compute a fingerprint based on script functionality, not exact content"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Remove comments and blank lines for comparison
            lines = []
            in_multiline_comment = False
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Handle Python multiline strings/comments
                if file_path.suffix == '.py':
                    if '"""' in line or "'''" in line:
                        in_multiline_comment = not in_multiline_comment
                        continue
                    if in_multiline_comment:
                        continue
                    if line.startswith('#'):
                        continue
                        
                # Handle shell comments
                elif file_path.suffix == '.sh':
                    if line.startswith('#'):
                        continue
                
                # Normalize variable names and common patterns
                normalized_line = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', line)
                normalized_line = re.sub(r'"[^"]*"', '"STRING"', normalized_line)
                normalized_line = re.sub(r"'[^']*'", "'STRING'", normalized_line)
                
                lines.append(normalized_line)
            
            # Create fingerprint from normalized functional content
            functional_content = '\n'.join(lines)
            return hashlib.md5(functional_content.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not fingerprint {file_path}: {e}")
            return ""
    
    def analyze_python_script_standards(self, file_path: Path) -> List[ScriptViolation]:
        """Analyze Python script for Rule 8 compliance"""
        violations = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Check for docstring at top
            has_module_docstring = False
            try:
                tree = ast.parse(content, filename=str(file_path))
                if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                    isinstance(tree.body[0].value, ast.Constant) and 
                    isinstance(tree.body[0].value.value, str)):
                    docstring = tree.body[0].value.value
                    if ('Purpose:' in docstring and 'Usage:' in docstring and 
                        'Requirements:' in docstring):
                        has_module_docstring = True
            except SyntaxError:
                violations.append(ScriptViolation(
                    file_path, "syntax_error", "Python syntax error in script",
                    "high", "Fix syntax errors before proceeding"
                ))
                return violations
            
            if not has_module_docstring:
                violations.append(ScriptViolation(
                    file_path, "missing_docstring", 
                    "Missing proper module docstring with Purpose/Usage/Requirements",
                    "medium", "Add module docstring with Purpose, Usage, and Requirements sections"
                ))
            
            # Check for hardcoded values
            hardcoded_patterns = [
                (r'/opt/sutazaiapp/[^"\']*', "hardcoded_path"),
                (r'localhost:\d+', "hardcoded_localhost"),
                (r'127\.0\.0\.1', "hardcoded_ip"),
                (r'password\s*=\s*["\'][^"\']+["\']', "hardcoded_password"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "hardcoded_api_key"),
            ]
            
            for i, line in enumerate(lines, 1):
                for pattern, violation_type in hardcoded_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(ScriptViolation(
                            file_path, violation_type,
                            f"Hardcoded value found at line {i}: {line.strip()}",
                            "medium", "Use configuration files or command-line arguments"
                        ))
            
            # Check for print() usage instead of logging
            print_usage = [i+1 for i, line in enumerate(lines) if 'print(' in line]
            if print_usage:
                violations.append(ScriptViolation(
                    file_path, "print_usage",
                    f"Using print() at lines: {print_usage}. Should use logging.",
                    "low", "Replace print() with proper logging"
                ))
            
            # Check for CLI argument handling
            if '__main__' in content:
                has_argparse = 'argparse' in content or 'click' in content
                if not has_argparse:
                    violations.append(ScriptViolation(
                        file_path, "no_cli_args",
                        "Script appears executable but lacks CLI argument handling",
                        "medium", "Add argparse or click for CLI argument handling"
                    ))
            
            # Check for proper __name__ == "__main__" guard
            if '__main__' not in content and file_path.name not in ['__init__.py']:
                violations.append(ScriptViolation(
                    file_path, "no_main_guard",
                    "Missing __name__ == '__main__' guard",
                    "low", "Add if __name__ == '__main__': guard"
                ))
        
        except Exception as e:
            logger.error(f"Error analyzing Python script {file_path}: {e}")
            violations.append(ScriptViolation(
                file_path, "analysis_error", f"Could not analyze script: {e}",
                "high", "Fix script format or encoding issues"
            ))
        
        return violations
    
    def analyze_shell_script_standards(self, file_path: Path) -> List[ScriptViolation]:
        """Analyze shell script for Rule 7 compliance"""
        violations = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Check for header comment with Purpose/Usage/Requires
            has_proper_header = False
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                if ('Purpose:' in line and any('Usage:' in l for l in lines[i:i+5]) and 
                    any('Requires:' in l for l in lines[i:i+5])):
                    has_proper_header = True
                    break
            
            if not has_proper_header:
                violations.append(ScriptViolation(
                    file_path, "missing_header",
                    "Missing proper header comment with Purpose/Usage/Requires",
                    "medium", "Add header comment with Purpose, Usage, and Requires information"
                ))
            
            # Check for hardcoded values
            hardcoded_patterns = [
                (r'/opt/sutazaiapp/[^\s]*', "hardcoded_path"),
                (r'localhost:\d+', "hardcoded_localhost"),
                # (r'password="[^"]*"', "hardcoded_password"), # Example pattern
                # (r"password='[^']*'", "hardcoded_password"), # Example pattern
            ]
            
            for i, line in enumerate(lines, 1):
                for pattern, violation_type in hardcoded_patterns:
                    if re.search(pattern, line):
                        violations.append(ScriptViolation(
                            file_path, violation_type,
                            f"Hardcoded value found at line {i}: {line.strip()}",
                            "medium", "Use environment variables or script arguments"
                        ))
            
            # Check for proper error handling
            has_error_handling = any('exit ' in line or 'return ' in line for line in lines)
            if not has_error_handling and len(lines) > 10:
                violations.append(ScriptViolation(
                    file_path, "no_error_handling",
                    "Script lacks proper error handling with exit codes",
                    "medium", "Add proper error handling with exit codes"
                ))
        
        except Exception as e:
            logger.error(f"Error analyzing shell script {file_path}: {e}")
            violations.append(ScriptViolation(
                file_path, "analysis_error", f"Could not analyze script: {e}",
                "high", "Fix script format or encoding issues"
            ))
        
        return violations
    
    def check_naming_conventions(self, file_path: Path) -> List[ScriptViolation]:
        """Check if script follows proper naming conventions"""
        violations = []
        
        # Check for lowercase, hyphenated naming
        filename = file_path.name
        
        # Remove extension for checking
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Check for problematic patterns
        problematic_patterns = [
            (r'[A-Z]', "uppercase_letters", "Contains uppercase letters"),
            (r'_+', "underscores", "Uses underscores instead of hyphens"),
            (r'^(test|temp|copy|old|backup|bak)\d*$', "bad_names", "Uses temporary/debug naming"),
            (r'(final|v\d+|copy|backup)$', "version_suffix", "Has version suffix"),
        ]
        
        for pattern, violation_type, description in problematic_patterns:
            if re.search(pattern, name_without_ext):
                violations.append(ScriptViolation(
                    file_path, violation_type,
                    f"Naming violation: {description}",
                    "medium", "Rename to lowercase, hyphenated format (e.g., script-name.py)"
                ))
        
        # Check if name is descriptive enough
        if len(name_without_ext) < 3:
            violations.append(ScriptViolation(
                file_path, "non_descriptive",
                "Script name is too short/non-descriptive",
                "low", "Use a more descriptive name explaining the script's purpose"
            ))
        
        return violations
    
    def find_duplicate_scripts(self) -> Dict[str, List[Path]]:
        """Find potentially duplicate scripts based on functionality"""
        script_files = []
        
        # Find all script files
        for pattern in ['**/*.py', '**/*.sh']:
            script_files.extend(self.root_path.glob(pattern))
        
        # Filter out virtual environments and __pycache__
        script_files = [
            f for f in script_files 
            if not any(skip in str(f) for skip in ['venv', '__pycache__', '.git', 'node_modules'])
        ]
        
        # Group by fingerprint
        fingerprint_groups = defaultdict(list)
        for script_file in script_files:
            fingerprint = self.compute_script_fingerprint(script_file)
            if fingerprint:  # Only add if we could compute fingerprint
                fingerprint_groups[fingerprint].append(script_file)
        
        # Return only groups with multiple files (potential duplicates)
        return {fp: files for fp, files in fingerprint_groups.items() if len(files) > 1}
    
    def analyze_all_scripts(self) -> None:
        """Run comprehensive analysis of all scripts"""
        logger.info("Starting comprehensive script analysis...")
        
        # Find all scripts
        script_files = []
        for pattern in ['**/*.py', '**/*.sh']:
            found_files = list(self.root_path.glob(pattern))
            script_files.extend(found_files)
        
        # Filter out unwanted directories
        script_files = [
            f for f in script_files 
            if not any(skip in str(f) for skip in ['venv', '__pycache__', '.git', 'node_modules'])
        ]
        
        logger.info(f"Found {len(script_files)} scripts to analyze")
        
        # Analyze each script
        for script_file in script_files:
            try:
                # Check naming conventions
                violations = self.check_naming_conventions(script_file)
                self.violations.extend(violations)
                
                # Analyze based on file type
                if script_file.suffix == '.py':
                    py_violations = self.analyze_python_script_standards(script_file)
                    self.violations.extend(py_violations)
                elif script_file.suffix == '.sh':
                    sh_violations = self.analyze_shell_script_standards(script_file)
                    self.violations.extend(sh_violations)
                
            except Exception as e:
                logger.error(f"Error analyzing {script_file}: {e}")
        
        logger.info(f"Analysis complete. Found {len(self.violations)} violations")


class ScriptConsolidator:
    """Handles safe script consolidation and reorganization"""
    
    def __init__(self, root_path: Path, dry_run: bool = True):
        self.root_path = root_path
        self.dry_run = dry_run
        self.archive_dir = root_path / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.actions_taken: List[str] = []
        
    def create_archive_structure(self) -> None:
        """Create archive directory structure"""
        if not self.dry_run:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            (self.archive_dir / "README.md").write_text(
                f"# Script Archive - {datetime.now().isoformat()}\n\n"
                "This directory contains scripts that were archived during consolidation.\n"
                "Each script is preserved with its original path structure.\n\n"
                "## Archived Scripts\n\n"
            )
    
    def archive_script(self, script_path: Path, reason: str) -> Path:
        """Archive a script before modification/removal"""
        relative_path = script_path.relative_to(self.root_path)
        archive_path = self.archive_dir / relative_path
        
        if not self.dry_run:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(script_path, archive_path)
            
            # Update archive README
            readme_path = self.archive_dir / "README.md"
            with readme_path.open('a') as f:
                f.write(f"- `{relative_path}`: {reason}\n")
        
        action = f"ARCHIVE: {script_path} -> {archive_path} (Reason: {reason})"
        self.actions_taken.append(action)
        logger.info(action)
        return archive_path
    
    def consolidate_duplicates(self, duplicate_groups: Dict[str, List[Path]]) -> None:
        """Consolidate duplicate scripts"""
        logger.info(f"Consolidating {len(duplicate_groups)} groups of duplicate scripts...")
        
        for fingerprint, scripts in duplicate_groups.items():
            if len(scripts) < 2:
                continue
                
            logger.info(f"Processing duplicate group with {len(scripts)} scripts:")
            for script in scripts:
                logger.info(f"  - {script}")
            
            # Choose the "best" script to keep (prefer scripts/ directory, then by name)
            scripts_priority = []
            for script in scripts:
                priority = 0
                if '/scripts/' in str(script):
                    priority += 100
                if not any(bad in script.name.lower() for bad in ['temp', 'test', 'old', 'copy']):
                    priority += 50
                if '-' in script.name:  # Proper naming convention
                    priority += 25
                scripts_priority.append((priority, script))
            
            # Sort by priority (highest first)
            scripts_priority.sort(key=lambda x: x[0], reverse=True)
            keeper = scripts_priority[0][1]
            duplicates = [s[1] for s in scripts_priority[1:]]
            
            logger.info(f"  Keeping: {keeper}")
            logger.info(f"  Removing: {[str(d) for d in duplicates]}")
            
            # Archive duplicates
            for duplicate in duplicates:
                self.archive_script(duplicate, f"Duplicate of {keeper}")
                if not self.dry_run:
                    duplicate.unlink()
    
    def reorganize_scripts(self, violations: List[ScriptViolation]) -> None:
        """Reorganize scripts to proper directory structure"""
        scripts_dir = self.root_path / "scripts"
        
        # Define proper structure
        proper_structure = {
            'dev': 'Development and debugging tools',
            'deploy': 'Deployment and release scripts',
            'data': 'Data processing and migration scripts',
            'utils': 'Utility and helper scripts',
            'test': 'Testing and validation scripts',
            'agents': 'AI agents and orchestration',
            'monitoring': 'System monitoring and health checks',
        }
        
        # Create proper structure
        if not self.dry_run:
            for subdir, description in proper_structure.items():
                subdir_path = scripts_dir / subdir
                subdir_path.mkdir(exist_ok=True)
                
                readme_path = subdir_path / "README.md"
                if not readme_path.exists():
                    readme_path.write_text(f"# {subdir.title()} Scripts\n\n{description}\n\n")
        
        # Move misplaced scripts
        misplaced_scripts = []
        for script_file in self.root_path.rglob('*.py'):
            if ('/scripts/' not in str(script_file) and 
                not any(skip in str(script_file) for skip in ['venv', '__pycache__', '.git'])):
                misplaced_scripts.append(script_file)
        
        for script_file in misplaced_scripts:
            # Determine proper location based on name/content
            target_subdir = 'utils'  # default
            
            script_name = script_file.name.lower()
            if any(keyword in script_name for keyword in ['test', 'validate', 'check']):
                target_subdir = 'test'
            elif any(keyword in script_name for keyword in ['deploy', 'build', 'release']):
                target_subdir = 'deploy'
            elif any(keyword in script_name for keyword in ['data', 'migrate', 'seed']):
                target_subdir = 'data'
            elif any(keyword in script_name for keyword in ['monitor', 'health', 'status']):
                target_subdir = 'monitoring'
            elif any(keyword in script_name for keyword in ['agent', 'orchestrat']):
                target_subdir = 'agents'
            
            target_path = scripts_dir / target_subdir / script_file.name
            
            if target_path != script_file:
                action = f"MOVE: {script_file} -> {target_path}"
                self.actions_taken.append(action)
                logger.info(action)
                
                if not self.dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(script_file), str(target_path))


class ScriptStandardsFixer:
    """Fixes common script standards violations automatically"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.fixes_applied: List[str] = []
    
    def fix_python_docstring(self, file_path: Path) -> bool:
        """Add proper docstring to Python script"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check if already has proper docstring
            if '"""' in content and 'Purpose:' in content:
                return False
            
            # Find shebang line if it exists
            insert_index = 0
            if lines and lines[0].startswith('#!'):
                insert_index = 1
            
            # Create template docstring
            script_name = file_path.name
            docstring = f'''"""
{script_name} - Brief description

Purpose: Describe what this script does in 1-2 sentences.
Usage: python {file_path.name} [--options]
Requirements: List any external dependencies or env vars.
"""'''
            
            # Insert docstring
            lines.insert(insert_index, docstring)
            
            if not self.dry_run:
                file_path.write_text('\n'.join(lines), encoding='utf-8')
            
            fix = f"DOCSTRING: Added module docstring to {file_path}"
            self.fixes_applied.append(fix)
            logger.info(fix)
            return True
            
        except Exception as e:
            logger.error(f"Could not fix docstring for {file_path}: {e}")
            return False
    
    def fix_shell_header(self, file_path: Path) -> bool:
        """Add proper header comment to shell script"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check if already has proper header
            if any('Purpose:' in line for line in lines[:10]):
                return False
            
            # Find shebang line if it exists
            insert_index = 0
            if lines and lines[0].startswith('#!'):
                insert_index = 1
            
            # Create template header
            script_name = file_path.name
            header = f'''# Purpose: Describe what this script does
# Usage: ./{script_name} [options]
# Requires: List any external dependencies'''
            
            # Insert header
            for i, line in enumerate(header.split('\n')):
                lines.insert(insert_index + i, line)
            
            if not self.dry_run:
                file_path.write_text('\n'.join(lines), encoding='utf-8')
            
            fix = f"HEADER: Added proper header to {file_path}"
            self.fixes_applied.append(fix)
            logger.info(fix)
            return True
            
        except Exception as e:
            logger.error(f"Could not fix header for {file_path}: {e}")
            return False
    
    def fix_naming_convention(self, file_path: Path) -> bool:
        """Fix script naming convention"""
        try:
            # Convert to lowercase, hyphenated format
            name_parts = file_path.stem.split('_')
            new_name = '-'.join(part.lower() for part in name_parts) + file_path.suffix
            
            if new_name == file_path.name:
                return False  # Already correct
            
            new_path = file_path.parent / new_name
            
            if not self.dry_run and not new_path.exists():
                file_path.rename(new_path)
            
            fix = f"RENAME: {file_path.name} -> {new_name}"
            self.fixes_applied.append(fix)
            logger.info(fix)
            return True
            
        except Exception as e:
            logger.error(f"Could not fix naming for {file_path}: {e}")
            return False
    
    def apply_fixes(self, violations: List[ScriptViolation]) -> None:
        """Apply automated fixes for violations"""
        logger.info(f"Applying automated fixes for {len(violations)} violations...")
        
        for violation in violations:
            if violation.violation_type == "missing_docstring":
                self.fix_python_docstring(violation.file_path)
            elif violation.violation_type == "missing_header":
                self.fix_shell_header(violation.file_path)
            elif violation.violation_type in ["uppercase_letters", "underscores"]:
                self.fix_naming_convention(violation.file_path)


class ScriptConsolidationEnforcer:
    """Main enforcer class orchestrating all consolidation activities"""
    
    def __init__(self, root_path: Path = Path("/opt/sutazaiapp")):
        self.root_path = root_path
        self.analyzer = ScriptAnalyzer(root_path)
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'violations': [],
            'duplicates': {},
            'actions_taken': [],
            'fixes_applied': []
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive consolidation report"""
        report = f"""
# Script Consolidation and Python Sanity Enforcement Report
Generated: {self.report_data['timestamp']}

## Summary
- Total violations found: {len(self.report_data['violations'])}
- Duplicate groups found: {len(self.report_data['duplicates'])}
- Actions taken: {len(self.report_data['actions_taken'])}
- Fixes applied: {len(self.report_data['fixes_applied'])}

## Violations by Type
"""
        
        # Group violations by type
        violations_by_type = defaultdict(list)
        for violation in self.report_data['violations']:
            violations_by_type[violation['type']].append(violation)
        
        for violation_type, violations in violations_by_type.items():
            report += f"\n### {violation_type.replace('_', ' ').title()} ({len(violations)})\n"
            for violation in violations[:10]:  # Show first 10
                report += f"- {violation['file']}: {violation['description']}\n"
            if len(violations) > 10:
                report += f"... and {len(violations) - 10} more\n"
        
        # Duplicate groups
        if self.report_data['duplicates']:
            report += "\n## Duplicate Script Groups\n"
            for i, (fingerprint, scripts) in enumerate(self.report_data['duplicates'].items(), 1):
                report += f"\n### Group {i} ({len(scripts)} scripts)\n"
                for script in scripts:
                    report += f"- {script}\n"
        
        # Actions taken
        if self.report_data['actions_taken']:
            report += "\n## Actions Taken\n"
            for action in self.report_data['actions_taken']:
                report += f"- {action}\n"
        
        # Fixes applied
        if self.report_data['fixes_applied']:
            report += "\n## Automated Fixes Applied\n"
            for fix in self.report_data['fixes_applied']:
                report += f"- {fix}\n"
        
        return report
    
    def update_scripts_readme(self) -> None:
        """Update or create scripts/README.md with current organization"""
        scripts_dir = self.root_path / "scripts"
        readme_path = scripts_dir / "README.md"
        
        readme_content = """# Scripts Directory

This directory contains all scripts organized by functionality and purpose.
All scripts follow CLAUDE.md Rules 7 & 8 for consolidation and Python sanity.

## Directory Structure

### agents/
AI agents and orchestration scripts for automated system management.

### data/
Data processing, migration, and model management scripts.

### deploy/
Deployment, build, and release automation scripts.

### dev/
Development tools and debugging utilities.

### monitoring/
System monitoring, health checks, and performance scripts.

### test/
Testing, validation, and quality assurance scripts.

### utils/
General utility scripts and shared helper functions.

## Script Standards

All scripts in this directory must follow these standards:

### Python Scripts (.py)
- Module docstring with Purpose, Usage, and Requirements sections
- CLI argument handling using argparse or click
- Proper logging instead of print() statements
- __name__ == "__main__" guard for executable scripts
- No hardcoded values - use configuration or arguments
- Proper error handling with meaningful exit codes

### Shell Scripts (.sh)
- Header comment with Purpose, Usage, and Requires information
- Proper error handling with exit codes
- No hardcoded secrets or paths
- Parameterized inputs instead of hardcoded values

### Naming Conventions
- Lowercase, hyphenated filenames (e.g., script-name.py)
- Descriptive names explaining the script's purpose
- No temporary or version suffixes (temp, v1, copy, etc.)

## Usage

Before adding new scripts:
1. Check if similar functionality already exists
2. Follow the directory structure above
3. Ensure compliance with naming conventions
4. Include proper documentation and error handling

For script consolidation and standards enforcement:
```bash
python scripts/agents/script-consolidation-enforcer.py --mode audit
python scripts/agents/script-consolidation-enforcer.py --mode fix --dry-run
python scripts/agents/script-consolidation-enforcer.py --mode consolidate
```
"""
        
        # Add current script inventory
        readme_content += "\n## Current Script Inventory\n\n"
        
        for subdir in sorted(scripts_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('.'):
                readme_content += f"### {subdir.name}/\n"
                
                scripts = list(subdir.glob('*.py')) + list(subdir.glob('*.sh'))
                if scripts:
                    for script in sorted(scripts):
                        # Try to extract purpose from docstring/header
                        purpose = "No description available"
                        try:
                            content = script.read_text(encoding='utf-8', errors='ignore')
                            
                            if script.suffix == '.py':
                                # Extract from docstring
                                if 'Purpose:' in content:
                                    purpose_line = [line for line in content.split('\n') if 'Purpose:' in line]
                                    if purpose_line:
                                        purpose = purpose_line[0].split('Purpose:', 1)[1].strip()
                            elif script.suffix == '.sh':
                                # Extract from header comment
                                if '# Purpose:' in content:
                                    purpose_line = [line for line in content.split('\n') if '# Purpose:' in line]
                                    if purpose_line:
                                        purpose = purpose_line[0].split('# Purpose:', 1)[1].strip()
                        except:
                            pass
                        
                        readme_content += f"- `{script.name}`: {purpose}\n"
                else:
                    readme_content += "- No scripts in this directory\n"
                
                readme_content += "\n"
        
        readme_path.write_text(readme_content)
        logger.info(f"Updated {readme_path}")
    
    def run_audit(self) -> None:
        """Run comprehensive audit of all scripts"""
        logger.info("Starting script consolidation audit...")
        
        # Analyze all scripts
        self.analyzer.analyze_all_scripts()
        
        # Find duplicates
        duplicates = self.analyzer.find_duplicate_scripts()
        
        # Prepare report data
        self.report_data['violations'] = [
            {
                'file': str(v.file_path),
                'type': v.violation_type,
                'description': v.description,
                'severity': v.severity,
                'suggestion': v.fix_suggestion
            }
            for v in self.analyzer.violations
        ]
        
        self.report_data['duplicates'] = {
            fp: [str(f) for f in files] 
            for fp, files in duplicates.items()
        }
        
        logger.info("Audit complete")
    
    def run_consolidation(self, dry_run: bool = True) -> None:
        """Run script consolidation process"""
        logger.info(f"Starting script consolidation (dry_run={dry_run})...")
        
        # First run audit
        self.run_audit()
        
        consolidator = ScriptConsolidator(self.root_path, dry_run=dry_run)
        fixer = ScriptStandardsFixer(dry_run=dry_run)
        
        # Create archive structure
        consolidator.create_archive_structure()
        
        # Find and consolidate duplicates
        duplicates = self.analyzer.find_duplicate_scripts()
        consolidator.consolidate_duplicates(duplicates)
        
        # Reorganize misplaced scripts
        consolidator.reorganize_scripts(self.analyzer.violations)
        
        # Apply automated fixes
        fixer.apply_fixes(self.analyzer.violations)
        
        # Update report data
        self.report_data['actions_taken'] = consolidator.actions_taken
        self.report_data['fixes_applied'] = fixer.fixes_applied
        
        # Update scripts README
        if not dry_run:
            self.update_scripts_readme()
        
        logger.info("Consolidation complete")
    
    def save_report(self, output_path: Path = None) -> None:
        """Save detailed report to file"""
        if output_path is None:
            output_path = self.root_path / f"script-consolidation-report-{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_content = self.generate_report()
        output_path.write_text(report_content)
        
        # Also save JSON data
        json_path = output_path.with_suffix('.json')
        with json_path.open('w') as f:
            json.dump(self.report_data, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        logger.info(f"JSON data saved to {json_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Script Consolidation and Python Sanity Enforcer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Run audit only
          python script-consolidation-enforcer.py --mode audit
          
          # Test consolidation (dry run)
          python script-consolidation-enforcer.py --mode consolidate --dry-run
          
          # Apply fixes and consolidation
          python script-consolidation-enforcer.py --mode consolidate --fix
          
          # Generate report only
          python script-consolidation-enforcer.py --mode report
        """)
    )
    
    parser.add_argument(
        '--mode', 
        choices=['audit', 'consolidate', 'report'], 
        default='audit',
        help='Operation mode (default: audit)'
    )
    
    parser.add_argument(
        '--fix', 
        action='store_true',
        help='Apply fixes (not dry-run)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        default=True,
        help='Show what would be done without making changes'
    )
    
    parser.add_argument(
        '--root-path',
        type=Path,
        default=Path("/opt/sutazaiapp"),
        help='Root path to analyze (default: /opt/sutazaiapp)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for reports'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize enforcer
    enforcer = ScriptConsolidationEnforcer(args.root_path)
    
    try:
        if args.mode == 'audit':
            enforcer.run_audit()
            print("\n" + enforcer.generate_report())
            
        elif args.mode == 'consolidate':
            dry_run = not args.fix  # If --fix is specified, don't do dry run
            enforcer.run_consolidation(dry_run=dry_run)
            
            if dry_run:
                print("\nDRY RUN - No changes were made. Use --fix to apply changes.")
            
            print("\n" + enforcer.generate_report())
            
        elif args.mode == 'report':
            enforcer.run_audit()
            enforcer.save_report(args.output)
        
        # Always save report
        enforcer.save_report(args.output)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()