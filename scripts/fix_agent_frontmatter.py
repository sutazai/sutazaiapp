#!/usr/bin/env python3
"""Fix missing frontmatter in agent files."""

import os
import re

def add_frontmatter_to_file(filepath):
    """Add frontmatter to files missing it."""
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if frontmatter already exists
    if content.startswith('---'):
        # Check if it's properly formatted
        if re.match(r'^---\s*\n.*?name:.*?\n.*?---', content, re.DOTALL):
            print(f"✓ {filepath} already has proper frontmatter")
            return
    
    # Determine the name based on file type and path
    filename = os.path.basename(filepath)
    dirname = os.path.basename(os.path.dirname(filepath))
    
    if filename == 'CHANGELOG.md':
        name = f"{dirname}-changelog"
        description = f"Change log for {dirname} directory"
    elif filename == 'README.md':
        name = f"{dirname}-readme"
        description = f"Documentation for {dirname} directory"
    elif filename == 'MIGRATION_SUMMARY.md':
        name = "migration-summary"
        description = "Summary of agent migration and consolidation"
    else:
        # For other .md files without proper frontmatter
        name = os.path.splitext(filename)[0]
        description = f"Configuration for {name}"
    
    # Create frontmatter
    frontmatter = f"""---
name: {name}
description: {description}
model: opus
tools: Read, Write
---

"""
    
    # Add frontmatter to content
    new_content = frontmatter + content
    
    # Write back to file
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Fixed {filepath}")

# List of files to fix (from the doctor output)
files_to_fix = [
    '/opt/sutazaiapp/.claude/agents/MIGRATION_SUMMARY.md',
    '/opt/sutazaiapp/.claude/agents/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/devops/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/specialized/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/core/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/analysis/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/devops/ci-cd/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/specialized/mobile/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/architecture/system-design/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/architecture/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/data/ml/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/data/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/consensus/README.md',
    '/opt/sutazaiapp/.claude/agents/consensus/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/testing/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/testing/validation/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/testing/unit/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/templates/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/github/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/README.md',
    '/opt/sutazaiapp/.claude/agents/swarm/README.md',
    '/opt/sutazaiapp/.claude/agents/analysis/code-review/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/swarm/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/development/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/documentation/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/.claude-flow/metrics/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/hive-mind/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/optimization/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/optimization/README.md',
    '/opt/sutazaiapp/.claude/agents/sparc/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/documentation/api-docs/CHANGELOG.md',
    '/opt/sutazaiapp/.claude/agents/development/backend/CHANGELOG.md',
]

# Process each file
for filepath in files_to_fix:
    if os.path.exists(filepath):
        add_frontmatter_to_file(filepath)
    else:
        print(f"⚠️  File not found: {filepath}")

print("\n✨ Frontmatter fixing complete!")