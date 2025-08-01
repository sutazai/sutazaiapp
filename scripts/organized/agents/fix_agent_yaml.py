#!/usr/bin/env python3
"""
Fix YAML formatting issues in Claude agent files.
The issue is that multi-line descriptions need to be properly formatted in YAML.
"""

import os
import re
from pathlib import Path

def fix_agent_yaml(file_path):
    """Fix YAML formatting in a single agent file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Find the description line
    fixed_lines = []
    in_description = False
    description_indent = 0
    
    for i, line in enumerate(lines):
        if line.startswith('description: Use this agent'):
            # Start of multi-line description
            fixed_lines.append('description: |')
            fixed_lines.append('  Use this agent when you need to:')
            in_description = True
            description_indent = 2
        elif in_description and line and not line.startswith(' '):
            # End of description block (new field starting)
            in_description = False
            fixed_lines.append(line)
        elif in_description and line.strip():
            # Part of description - ensure proper indentation
            if line.startswith('- '):
                fixed_lines.append('  ' + line)
            else:
                fixed_lines.append('  ' + line.strip())
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Fixed: {file_path}")

def main():
    """Fix all agent files with YAML issues."""
    agents_dir = Path('/opt/sutazaiapp/.claude/agents')
    
    # Find all .md files with the problematic pattern
    for agent_file in agents_dir.glob('*.md'):
        with open(agent_file, 'r') as f:
            content = f.read()
        
        if 'description: Use this agent' in content:
            fix_agent_yaml(agent_file)
    
    print("\nAll agent YAML files have been fixed!")

if __name__ == '__main__':
    main()