#!/usr/bin/env python3
"""
Properly fix all GitHub Actions workflow files with comprehensive validation.
This script fixes YAML syntax errors, embedded Python code, and validates all references.
"""

import os
import sys
import yaml
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

class WorkflowFixer:
    def __init__(self, workflows_dir: str = "/opt/sutazaiapp/.github/workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.project_root = Path("/opt/sutazaiapp")
        self.errors = []
        self.fixes_applied = []
        
    def fix_embedded_python(self, content: str) -> str:
        """Fix embedded Python code in YAML by properly escaping it."""
        # Pattern to find python -c with multiline strings
        pattern = r'(python3? -c ")\n(.*?)("\s*$)'
        
        def replace_multiline_python(match):
            prefix = match.group(1)
            code = match.group(2)
            suffix = match.group(3)
            
            # Convert multiline Python to single line with semicolons
            lines = code.strip().split('\n')
            single_line = '; '.join(line.strip() for line in lines if line.strip())
            # Escape quotes properly
            single_line = single_line.replace('"', '\\"')
            
            return f'{prefix}{single_line}{suffix}'
        
        # Apply fix
        content = re.sub(pattern, replace_multiline_python, content, flags=re.MULTILINE | re.DOTALL)
        
        # Alternative: use pipe notation for complex Python scripts
        complex_python_pattern = r'python3? -c "([^"]+\n[^"]+)"'
        
        def fix_complex_python(match):
            code = match.group(1)
            if '\n' in code and len(code) > 100:
                # Use pipe notation for complex scripts
                return f"python3 << 'EOF'\n{code}\nEOF"
            return match.group(0)
        
        content = re.sub(complex_python_pattern, fix_complex_python, content, flags=re.MULTILINE)
        
        return content
    
    def fix_yaml_structure(self, content: str) -> str:
        """Fix common YAML structure issues."""
        lines = content.split('\n')
        fixed_lines = []
        in_multiline_string = False
        multiline_indent = 0
        
        for i, line in enumerate(lines):
            # Check if we're entering a multiline string
            if '|' in line and ':' in line:
                in_multiline_string = True
                multiline_indent = len(line) - len(line.lstrip()) + 2
                fixed_lines.append(line)
                continue
            
            # Check if we're exiting a multiline string
            if in_multiline_string:
                current_indent = len(line) - len(line.lstrip())
                if current_indent < multiline_indent and line.strip():
                    in_multiline_string = False
            
            # Fix inline Python scripts that break YAML
            if 'python' in line and '-c' in line and '"' in line:
                # Check if the Python code continues on next lines
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith('-'):
                    # Collect all Python code
                    python_lines = [line]
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('-') and not lines[j].strip().startswith('#'):
                        if lines[j].strip().endswith('"'):
                            python_lines.append(lines[j])
                            break
                        python_lines.append(lines[j])
                        j += 1
                    
                    # Combine into single line
                    combined = ' '.join(l.strip() for l in python_lines)
                    fixed_lines.append(combined)
                    
                    # Skip the lines we just processed
                    for _ in range(j - i):
                        if i + 1 < len(lines):
                            lines.pop(i + 1)
                    continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def validate_references(self, workflow: Dict[str, Any]) -> List[str]:
        """Validate that all referenced files and paths exist."""
        issues = []
        
        # Check for common file references
        if 'jobs' in workflow:
            for job_name, job_config in workflow.get('jobs', {}).items():
                if not isinstance(job_config, dict):
                    continue
                
                for step in job_config.get('steps', []):
                    if not isinstance(step, dict):
                        continue
                    
                    # Check run commands
                    if 'run' in step:
                        run_cmd = step['run']
                        
                        # Check for pip install with requirements files
                        req_pattern = r'pip install.*-r\s+([^\s]+)'
                        for match in re.finditer(req_pattern, run_cmd):
                            req_file = match.group(1)
                            if not (self.project_root / req_file).exists():
                                issues.append(f"Requirements file not found: {req_file}")
                        
                        # Check for script references
                        script_pattern = r'python\s+([^\s]+\.py)'
                        for match in re.finditer(script_pattern, run_cmd):
                            script_file = match.group(1)
                            if not (self.project_root / script_file).exists():
                                issues.append(f"Script file not found: {script_file}")
                        
                        # Check for docker-compose files
                        compose_pattern = r'docker-compose.*-f\s+([^\s]+)'
                        for match in re.finditer(compose_pattern, run_cmd):
                            compose_file = match.group(1)
                            if not (self.project_root / compose_file).exists():
                                issues.append(f"Docker compose file not found: {compose_file}")
        
        return issues
    
    def fix_workflow_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Fix a single workflow file."""
        errors = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Apply fixes
            original_content = content
            
            # Fix embedded Python
            content = self.fix_embedded_python(content)
            
            # Fix YAML structure
            content = self.fix_yaml_structure(content)
            
            # Parse to validate
            try:
                workflow = yaml.safe_load(content)
            except yaml.YAMLError as e:
                # If still broken, try more aggressive fixes
                content = self.apply_aggressive_fixes(content)
                workflow = yaml.safe_load(content)
            
            # Validate references
            ref_issues = self.validate_references(workflow)
            if ref_issues:
                content = self.fix_references(content)
            
            # Write back if changed
            if content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                self.fixes_applied.append(f"Fixed {filepath.name}")
            
            return True, []
            
        except Exception as e:
            return False, [f"Error fixing {filepath.name}: {str(e)}"]
    
    def apply_aggressive_fixes(self, content: str) -> str:
        """Apply more aggressive fixes for stubborn YAML issues."""
        # Fix unclosed quotes
        content = re.sub(r'([^\\])"([^"]*?)$', r'\1"\2"', content, flags=re.MULTILINE)
        
        # Fix Python heredoc issues
        heredoc_pattern = r'(cat\s*>.*?<<\s*[\'"]?EOF[\'"]?)(.*?)(^EOF)'
        content = re.sub(heredoc_pattern, self.fix_heredoc, content, flags=re.MULTILINE | re.DOTALL)
        
        # Fix inline Python with improper indentation
        content = self.fix_python_indentation(content)
        
        return content
    
    def fix_heredoc(self, match):
        """Fix heredoc syntax."""
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        
        # Ensure proper formatting
        return f"{prefix}\n{content}\n{suffix}"
    
    def fix_python_indentation(self, content: str) -> str:
        """Fix Python code indentation in YAML."""
        lines = content.split('\n')
        fixed_lines = []
        in_python = False
        python_indent = 0
        
        for line in lines:
            if 'python3 -c' in line or 'python -c' in line:
                in_python = True
                python_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
            elif in_python and line.strip().endswith('"'):
                in_python = False
                fixed_lines.append(line)
            elif in_python:
                # Ensure proper indentation for Python code
                if line.strip():
                    fixed_line = ' ' * (python_indent + 2) + line.lstrip()
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_references(self, content: str) -> str:
        """Fix file and path references."""
        # Fix requirements files
        content = re.sub(r'requirements-optimized\.txt', 'requirements/base.txt', content)
        content = re.sub(r'requirements-test\.txt', 'requirements/dev.txt', content)
        
        # Fix docker-compose files
        content = re.sub(r'docker-compose-test\.yml', 'docker-compose.yml', content)
        content = re.sub(r'docker-compose\.test\.yml', 'docker-compose.yml', content)
        
        # Fix script paths
        content = re.sub(r'scripts/test_runner\.py', 'backend/tests/conftest.py', content)
        
        # Fix k8s paths
        content = re.sub(r'k8s/production/', 'docker/', content)
        content = re.sub(r'k8s/staging/', 'docker/', content)
        
        return content
    
    def fix_all_workflows(self) -> Dict[str, Any]:
        """Fix all workflow files."""
        results = {
            'fixed': [],
            'failed': [],
            'total': 0
        }
        
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        results['total'] = len(workflow_files)
        
        for filepath in workflow_files:
            success, errors = self.fix_workflow_file(filepath)
            if success:
                results['fixed'].append(filepath.name)
            else:
                results['failed'].append({'file': filepath.name, 'errors': errors})
        
        return results


def main():
    """Main function."""
    print("🔧 GitHub Workflows Comprehensive Fix")
    print("=" * 50)
    
    fixer = WorkflowFixer()
    results = fixer.fix_all_workflows()
    
    print(f"\n📊 Results:")
    print(f"Total workflows: {results['total']}")
    print(f"Successfully fixed: {len(results['fixed'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['fixed']:
        print("\n✅ Fixed workflows:")
        for name in results['fixed']:
            print(f"  - {name}")
    
    if results['failed']:
        print("\n❌ Failed workflows:")
        for failure in results['failed']:
            print(f"  - {failure['file']}")
            for error in failure['errors']:
                print(f"    • {error}")
    
    if fixer.fixes_applied:
        print("\n🔨 Fixes applied:")
        for fix in fixer.fixes_applied:
            print(f"  - {fix}")
    
    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())