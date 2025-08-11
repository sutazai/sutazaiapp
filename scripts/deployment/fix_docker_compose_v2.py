#!/usr/bin/env python3
"""
More comprehensive script to fix docker-compose.yml by removing ALL deploy: sections
"""

import re
import sys

def fix_docker_compose(file_path):
    """Fix docker-compose.yml by removing all deploy: sections"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} characters")
    
    # More aggressive pattern to match deploy sections
    # This matches the entire deploy: block including all nested content
    lines = content.split('\n')
    new_lines = []
    skip_until_indent = None
    
    for i, line in enumerate(lines):
        # Check if this line starts a deploy: section
        if re.match(r'(\s*)deploy:\s*$', line):
            indent_level = len(line) - len(line.lstrip())
            skip_until_indent = indent_level
            print(f"Found deploy: section at line {i+1}, indentation {indent_level}")
            continue
            
        # If we're skipping (inside a deploy section)
        if skip_until_indent is not None:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else float('inf')
            
            # If we encounter a line with same or less indentation than the deploy:, stop skipping
            if line.strip() and current_indent <= skip_until_indent:
                skip_until_indent = None
                new_lines.append(line)
            else:
                # Skip this line (it's part of the deploy section)
                continue
        else:
            new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    print(f"New file size: {len(new_content)} characters")
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Fixed docker-compose.yml by removing all deploy: sections")

if __name__ == "__main__":
    fix_docker_compose('/opt/sutazaiapp/docker-compose.yml')