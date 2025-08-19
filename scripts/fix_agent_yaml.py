#!/usr/bin/env python3
"""Fix YAML frontmatter in agent files by properly quoting descriptions."""

import os
import re

def fix_agent_yaml(filepath):
    """Fix YAML frontmatter by quoting description field."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if file starts with frontmatter
        if not content.startswith('---'):
            return False
        
        # Find the frontmatter section
        match = re.match(r'^(---\s*\n)(.*?)(---\s*\n)', content, re.DOTALL)
        if not match:
            return False
        
        frontmatter = match.group(2)
        rest = content[match.end():]
        
        # Fix description field by adding quotes if it contains colons or special chars
        # Look for description line
        lines = frontmatter.split('\n')
        new_lines = []
        
        for line in lines:
            if line.startswith('description:'):
                # Extract the description value
                desc_match = re.match(r'^description:\s*(.+)$', line)
                if desc_match:
                    desc_value = desc_match.group(1).strip()
                    # If not already quoted and contains special chars, quote it
                    if not (desc_value.startswith('"') or desc_value.startswith("'")):
                        if ':' in desc_value or ',' in desc_value or '‑' in desc_value:
                            # Escape any quotes in the description
                            desc_value = desc_value.replace('"', '\\"')
                            line = f'description: "{desc_value}"'
            new_lines.append(line)
        
        # Reconstruct the file
        new_frontmatter = '\n'.join(new_lines)
        new_content = f"---\n{new_frontmatter}---\n{rest}"
        
        # Write back to file
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        return True
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

# Process all agent files
agent_dir = '/opt/sutazaiapp/.claude/agents'
fixed = 0
skipped = 0

for root, dirs, files in os.walk(agent_dir):
    for file in files:
        if file.endswith('.md'):
            # Skip documentation files
            if 'CHANGELOG' in file or 'README' in file or 'MIGRATION' in file:
                skipped += 1
                continue
                
            filepath = os.path.join(root, file)
            if fix_agent_yaml(filepath):
                fixed += 1
                print(f"✅ Fixed: {os.path.basename(filepath)}")

print(f"\n✨ Fixed {fixed} files, skipped {skipped} documentation files")