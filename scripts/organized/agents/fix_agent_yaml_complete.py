#!/usr/bin/env python3
"""
Complete fix for YAML formatting issues in Claude agent files.
This handles both the description formatting and proper indentation.
"""

import os
import re
from pathlib import Path

def fix_agent_yaml_complete(file_path):
    """Fix all YAML formatting issues in a single agent file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into frontmatter and body
    parts = content.split('---', 2)
    if len(parts) < 3:
        print(f"Warning: {file_path} doesn't have proper frontmatter delimiters")
        return
    
    frontmatter = parts[1].strip()
    body = parts[2]
    
    # Process frontmatter line by line
    lines = frontmatter.split('\n')
    fixed_lines = []
    in_description = False
    in_list = False
    
    for i, line in enumerate(lines):
        # Handle the description field
        if line.startswith('description: |'):
            fixed_lines.append(line)
            in_description = True
            in_list = False
        elif line.startswith('description: Use this agent'):
            # Fix description that wasn't caught before
            fixed_lines.append('description: |')
            fixed_lines.append('  Use this agent when you need to:')
            in_description = True
            in_list = True
        elif line.startswith('description:') and not line.strip().endswith('|'):
            # Handle other description formats
            desc_content = line[12:].strip()
            if desc_content:
                fixed_lines.append('description: |')
                fixed_lines.append(f'  {desc_content}')
                in_description = True
            else:
                fixed_lines.append(line)
        elif in_description:
            # Check if we've hit a new field (not indented)
            if line and not line.startswith(' ') and not line.startswith('-') and ':' in line:
                in_description = False
                in_list = False
                fixed_lines.append(line)
            elif line.strip().startswith('- '):
                # List item - ensure proper indentation (2 spaces)
                fixed_lines.append('  ' + line.strip())
                in_list = True
            elif line.strip() == '':
                # Empty line
                if in_list:
                    fixed_lines.append('')
                else:
                    fixed_lines.append(line)
            elif line.strip() and in_list:
                # Continuation of list item or new paragraph
                if line.startswith('  '):
                    fixed_lines.append(line)
                else:
                    fixed_lines.append('  ' + line.strip())
            else:
                # Other content in description
                if line.startswith('  '):
                    fixed_lines.append(line)
                else:
                    fixed_lines.append('  ' + line.strip())
        else:
            # Not in description, keep as is
            fixed_lines.append(line)
    
    # Reconstruct the file
    fixed_content = '---\n' + '\n'.join(fixed_lines) + '\n---' + body
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed: {file_path}")

def validate_yaml_frontmatter(file_path):
    """Validate that the YAML frontmatter is properly formatted."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for proper frontmatter delimiters
    if not content.startswith('---'):
        return False, "Missing opening frontmatter delimiter"
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return False, "Missing closing frontmatter delimiter"
    
    # Try to parse the YAML
    try:
        import yaml
        yaml.safe_load(parts[1])
        return True, "Valid YAML"
    except Exception as e:
        return False, f"YAML parsing error: {str(e)}"

def main():
    """Fix all agent files with YAML issues."""
    agents_dir = Path('/opt/sutazaiapp/.claude/agents')
    
    # Process all .md files
    for agent_file in agents_dir.glob('*.md'):
        # First check if it's valid
        is_valid, msg = validate_yaml_frontmatter(agent_file)
        if not is_valid:
            print(f"\nFixing {agent_file.name}: {msg}")
            fix_agent_yaml_complete(agent_file)
            
            # Validate again
            is_valid, msg = validate_yaml_frontmatter(agent_file)
            if is_valid:
                print(f"  ✓ Fixed successfully")
            else:
                print(f"  ✗ Still has issues: {msg}")
    
    print("\nAll agent YAML files have been processed!")

if __name__ == '__main__':
    main()