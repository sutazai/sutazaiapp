#!/usr/bin/env python3
"""
Fix multiple syntax errors found in the codebase
"""

import os
import re
import glob


def fix_setup_pytest_config():
    """Fix the setup_pytest_config.py file"""
    file_path = "setup_pytest_config.py"
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the problematic line and fix it
    content = re.sub(
        r'"""Create an instance of the default event loop for each test\."""',
        '    """Create an instance of the default event loop for each test."""',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Fixed {file_path}")


def fix_script_files():
    """Fix various script files with syntax errors"""
    script_files = [
        "scripts/test_pipeline.py",
        "scripts/system_diagnostics.py",
        "scripts/organize_project.py",
        "scripts/sutazai_script_cleanup.py",
        "scripts/process_optimizer.py",
        "scripts/system_maintenance_pro.py",
        "scripts/log_rotation.py"
    ]
    
    for file_path in script_files:
        if not os.path.exists(file_path):
            continue
            
        print(f"Checking {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Skip if file is empty or very problematic
            if len(content.strip()) < 50:
                continue
                
            # Create a backup
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Basic fixes for common issues
            fixed_content = fix_common_syntax_issues(content)
            
            # Only write if we made changes
            if fixed_content != content:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                print(f"Fixed syntax issues in {file_path}")
                
        except Exception as e:
            print(f"Could not fix {file_path}: {e}")


def fix_common_syntax_issues(content):
    """Fix common syntax issues in Python files"""
    # Fix missing colons after function definitions
    content = re.sub(
        r'def\s+(\w+)\s*\([^)]*\)\s*->\s*[^:]+(?!\s*:)',
        r'def \1(\2) -> \3:',
        content
    )
    
    # Fix missing indentation after try/except
    content = re.sub(
        r'except\s+\w+\s+as\s+\w+:\s*$',
        r'except Exception as e:\n    pass',
        content,
        flags=re.MULTILINE
    )
    
    return content


def fix_requirements_conflicts():
    """Fix the requirements.txt conflicts"""
    requirements_path = "requirements.txt"
    if not os.path.exists(requirements_path):
        return
        
    with open(requirements_path, 'r') as f:
        lines = f.readlines()
    
    # Remove conflicting versions - keep the ones that work together
    fixed_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            fixed_lines.append(line)
        elif 'black==' in line:
            # Use a compatible version
            fixed_lines.append('black==22.12.0')
        elif 'safety==' in line:
            # Update to a version that works with newer packaging
            fixed_lines.append('safety==3.0.1')
        else:
            fixed_lines.append(line)
    
    with open(requirements_path, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print("Fixed requirements.txt conflicts")


def main():
    """Main function to fix all syntax errors"""
    print("🔧 Starting syntax error fixes...")
    
    # Fix specific files
    fix_setup_pytest_config()
    fix_script_files()
    fix_requirements_conflicts()
    
    print("✅ Syntax error fixes completed!")
    print("\nNote: Some files with severe formatting issues were already manually fixed:")
    print("- ai_agents/auto_gpt/tests/test_auto_gpt.py")
    print("- ai_agents/auto_gpt/tests/test_task.py") 
    print("- ai_agents/document_processor/src/__init__.py")
    print("- ai_agents/document_processor/tests/performance_benchmark.py")
    print("- ai_agents/auto_gpt/src/config.py")


if __name__ == "__main__":
    main()
