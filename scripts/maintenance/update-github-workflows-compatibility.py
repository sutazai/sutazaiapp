#!/usr/bin/env python3
"""
GitHub Actions Workflow Compatibility Updater
Updates workflows to use flexible script discovery during migration
"""

import os
import re
import yaml
import json
from pathlib import Path

def update_workflow_file(workflow_path):
    """Update a single workflow file with flexible script paths."""
    print(f"📝 Updating workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{workflow_path}.backup"
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Replace hardcoded script paths with flexible discovery
    replacements = {
        r'python scripts/check_secrets\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script check_secrets.py',
        r'python scripts/check_naming\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script check_naming.py',
        r'python scripts/check_duplicates\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script check_duplicates.py',
        r'python scripts/validate_agents\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script validate_agents.py',
        r'python scripts/check_requirements\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script check_requirements.py',
        r'python scripts/enforce_claude_md_simple\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script enforce_claude_md_simple.py',
        r'python scripts/testing/test_runner\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script test_runner.py',
        r'python scripts/coverage_reporter\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script coverage_reporter.py',
        r'python scripts/export_openapi\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script export_openapi.py',
        r'python scripts/summarize_openapi\.py': 'bash scripts/script-discovery-bootstrap.sh exec_script summarize_openapi.py',
    }
    
    updated_content = content
    changes_made = False
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, updated_content):
            updated_content = re.sub(pattern, replacement, updated_content)
            changes_made = True
            print(f"  ✅ Updated pattern: {pattern}")
    
    if changes_made:
        with open(workflow_path, 'w') as f:
            f.write(updated_content)
        print(f"  💾 Updated workflow: {workflow_path}")
    else:
        # Remove backup if no changes
        os.remove(backup_path)
        print(f"  ℹ️  No changes needed: {workflow_path}")
    
    return changes_made

def main():
    """Update all GitHub Actions workflows with compatibility layer."""
    print("🚀 GITHUB ACTIONS WORKFLOW COMPATIBILITY UPDATER")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent.parent
    workflows_dir = project_root / '.github' / 'workflows'
    
    if not workflows_dir.exists():
        print(f"❌ Workflows directory not found: {workflows_dir}")
        return
    
    workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
    
    print(f"📋 Found {len(workflow_files)} workflow files")
    
    updated_count = 0
    for workflow_file in workflow_files:
        if update_workflow_file(workflow_file):
            updated_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Updated {updated_count} workflow files")
    print(f"📁 Backups created for modified files (.backup extension)")
    
    # Create summary report
    report = {
        "timestamp": "2025-08-10T17:00:00Z",
        "updated_workflows": updated_count,
        "total_workflows": len(workflow_files),
        "backup_files_created": True,
        "compatibility_layer": "scripts/script-discovery-bootstrap.sh"
    }
    
    with open(project_root / 'github-workflows-update-report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved: github-workflows-update-report.json")

if __name__ == "__main__":
    main()
