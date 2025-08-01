#!/usr/bin/env python3
"""
Final fix for YAML frontmatter issues in Claude agent files.
This handles all edge cases including content after the YAML section.
"""

import os
import re
from pathlib import Path
import yaml

def fix_yaml_frontmatter(file_path):
    """Fix all YAML frontmatter issues comprehensively."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file has frontmatter
    if not content.startswith('---'):
        print(f"Skipping {file_path} - no frontmatter")
        return False
    
    # Split into parts
    parts = content.split('---', 2)
    if len(parts) < 3:
        # Missing closing delimiter
        lines = content.split('\n')
        yaml_lines = []
        body_lines = []
        in_yaml = True
        
        for i, line in enumerate(lines[1:]):  # Skip first ---
            if in_yaml and line and not line.startswith(' ') and not line.startswith('-'):
                # Check if this looks like the start of body content
                if any(keyword in line for keyword in ['You are', 'This agent', '## ', '### ']):
                    in_yaml = False
                    body_lines = lines[i+1:]
                    break
            if in_yaml:
                yaml_lines.append(line)
        
        yaml_content = '\n'.join(yaml_lines)
        body_content = '\n'.join(body_lines)
    else:
        yaml_content = parts[1].strip()
        body_content = parts[2]
    
    # Fix YAML content
    try:
        # First, try to parse as-is
        yaml_data = yaml.safe_load(yaml_content)
    except:
        # If that fails, fix common issues
        lines = yaml_content.split('\n')
        fixed_lines = []
        in_description = False
        
        for line in lines:
            # Handle description field
            if line.startswith('description:'):
                if ' |' not in line and line.strip() != 'description:':
                    # Single line description, keep as is
                    fixed_lines.append(line)
                else:
                    # Multi-line description
                    fixed_lines.append('description: |')
                    in_description = True
            elif in_description:
                # Check if we've hit a new field
                if line and not line.startswith(' ') and ':' in line:
                    in_description = False
                    fixed_lines.append(line)
                elif line.strip() == '':
                    fixed_lines.append('')
                else:
                    # Ensure proper indentation for description content
                    if line.startswith('-'):
                        fixed_lines.append('  ' + line)
                    elif line.strip():
                        fixed_lines.append('  ' + line.strip())
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        yaml_content = '\n'.join(fixed_lines)
        
        # Try to parse again
        try:
            yaml_data = yaml.safe_load(yaml_content)
        except Exception as e:
            print(f"Failed to fix YAML in {file_path}: {e}")
            return False
    
    # Ensure there's a newline between frontmatter and body
    if body_content and not body_content.startswith('\n'):
        body_content = '\n' + body_content
    
    # Reconstruct file
    fixed_content = f"---\n{yaml_content}\n---{body_content}"
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    return True

def main():
    """Fix all agent files."""
    agents_dir = Path('/opt/sutazaiapp/.claude/agents')
    fixed_count = 0
    failed_count = 0
    
    for agent_file in agents_dir.glob('*.md'):
        print(f"\nProcessing {agent_file.name}...")
        
        # Skip non-agent files
        if agent_file.name in ['COMPREHENSIVE_INVESTIGATION_PROTOCOL.md', 'AGENT_CLEANUP_SUMMARY.md', 'essential_agents.txt', 'missing_agents_list.txt']:
            print(f"  Skipping non-agent file")
            continue
            
        # Skip -detailed.md files as they don't have frontmatter
        if agent_file.name.endswith('-detailed.md'):
            print(f"  Skipping detailed file")
            continue
        
        try:
            if fix_yaml_frontmatter(agent_file):
                # Validate the fix
                with open(agent_file, 'r') as f:
                    content = f.read()
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        try:
                            yaml.safe_load(parts[1])
                            print(f"  ✓ Fixed successfully")
                            fixed_count += 1
                        except Exception as e:
                            print(f"  ✗ Still has YAML issues: {e}")
                            failed_count += 1
                    else:
                        print(f"  ✗ Missing closing delimiter")
                        failed_count += 1
                else:
                    print(f"  ✗ No frontmatter found")
                    failed_count += 1
        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
            failed_count += 1
    
    print(f"\n\n=== Summary ===")
    print(f"Fixed: {fixed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"\nYAML frontmatter fixes complete!")

if __name__ == '__main__':
    main()