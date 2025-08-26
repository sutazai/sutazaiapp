#!/usr/bin/env python3
"""Remove frontmatter from documentation files that aren't agents."""

import os
import re

def remove_frontmatter(filepath):
    """Remove frontmatter from documentation files."""
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if file has frontmatter
    if content.startswith('---'):
        # Find the closing ---
        match = re.match(r'^---\s*\n.*?---\s*\n', content, re.DOTALL)
        if match:
            # Remove the frontmatter
            new_content = content[match.end():]
            
            # Write back to file
            with open(filepath, 'w') as f:
                f.write(new_content)
            
            print(f"✅ Removed frontmatter from {filepath}")
        else:
            print(f"⚠️  No closing frontmatter found in {filepath}")
    else:
        print(f"✓ {filepath} has no frontmatter")

# List of documentation files that shouldn't have frontmatter
docs_files = [
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
for filepath in docs_files:
    if os.path.exists(filepath):
        remove_frontmatter(filepath)
    else:
        print(f"⚠️  File not found: {filepath}")

print("\n✨ Documentation files cleaned!")