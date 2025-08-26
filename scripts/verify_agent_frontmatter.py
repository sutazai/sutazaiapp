#!/usr/bin/env python3
"""Verify agent frontmatter format."""

import os
import re
import yaml

def check_agent_file(filepath):
    """Check if an agent file has proper frontmatter."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for frontmatter
        if not content.startswith('---'):
            return f"Missing frontmatter start: {filepath}"
        
        # Extract frontmatter
        match = re.match(r'^---\s*\n(.*?)---\s*\n', content, re.DOTALL)
        if not match:
            return f"Malformed frontmatter (no closing ---): {filepath}"
        
        # Parse YAML
        try:
            frontmatter = yaml.safe_load(match.group(1))
            if not frontmatter:
                return f"Empty frontmatter: {filepath}"
            
            # Check required fields
            if 'name' not in frontmatter:
                return f"Missing 'name' field: {filepath}"
            
            return None  # Success
            
        except yaml.YAMLError as e:
            return f"Invalid YAML in frontmatter: {filepath} - {e}"
            
    except Exception as e:
        return f"Error reading file: {filepath} - {e}"

# Find all agent files (excluding docs)
agent_dir = '/opt/sutazaiapp/.claude/agents'
issues = []

for root, dirs, files in os.walk(agent_dir):
    for file in files:
        if file.endswith('.md'):
            # Skip documentation files
            if file in ['CHANGELOG.md', 'README.md', 'MIGRATION_SUMMARY.md']:
                continue
            if 'CHANGELOG' in file or 'README' in file or 'MIGRATION' in file:
                continue
                
            filepath = os.path.join(root, file)
            issue = check_agent_file(filepath)
            if issue:
                issues.append(issue)

if issues:
    print(f"Found {len(issues)} issues:\n")
    for issue in issues[:20]:  # Show first 20 issues
        print(f"  ❌ {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more issues")
else:
    print("✅ All agent files have proper frontmatter!")