#!/usr/bin/env python3
"""
Fix Docker Compose YAML merge key conflicts
"""

import re
import sys

def fix_yaml_merge_conflicts(file_path):
    """Fix YAML merge key conflicts by replacing conflicting <<: *resource-limits with explicit deploy sections"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find services with both agent-defaults and resource-limits merge keys
    # This pattern matches the problematic structure
    pattern = r'(\s+)<<: \*agent-defaults\n(.*?\n)(\s+)<<: \*resource-limits'
    
    def replacement(match):
        indent = match.group(1)
        middle_content = match.group(2)
        
        # Replace with explicit deploy section
        return (f'{indent}<<: *agent-defaults\n'
                f'{middle_content}'
                f'{indent}deploy:\n'
                f'{indent}  resources:\n'
                f'{indent}    limits:\n'
                f'{indent}      cpus: \'2\'\n'
                f'{indent}      memory: 2G\n'
                f'{indent}    reservations:\n'
                f'{indent}      cpus: \'1\'\n'
                f'{indent}      memory: 1G')
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back the fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed YAML merge conflicts in {file_path}")

if __name__ == "__main__":
    file_path = "/opt/sutazaiapp/docker-compose-complete-agents.yml"
    fix_yaml_merge_conflicts(file_path)
    print("YAML merge conflict fixes applied!")