#!/usr/bin/env python3
"""
Compliance Fix for dify-automation-specialist-detailed

This script fixes the following issues:
- Insufficient rules enforcement integration
"""

import re
from pathlib import Path

def fix_dify_automation_specialist_detailed():
    """Fix compliance issues for dify-automation-specialist-detailed"""
    
    agent_file = Path('/opt/sutazaiapp/.claude/agents/dify-automation-specialist-detailed.md')
    
    if not agent_file.exists():
        print(f"❌ Agent file not found: {agent_file}")
        return False
    
    # Read current content
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_file = agent_file.with_suffix('.md.backup')
    if not backup_file.exists():
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📦 Backup created: {backup_file}")
    
    modified = False
    
    # Fix: Add environment variables if missing
    if 'environment:' not in content and 'env_file:' not in content:
        print("🔧 Adding environment variables...")
        
        # Find YAML front matter end
        yaml_end = content.find('---', 3)  # Find second ---
        if yaml_end != -1:
            env_section = '''
environment:
  - CLAUDE_RULES_ENABLED=true
  - CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
  - AGENT_NAME=dify-automation-specialist-detailed
  - LOG_LEVEL=INFO
'''
            content = content[:yaml_end] + env_section + content[yaml_end:]
            modified = True
    
    # Fix: Add rules integration section if missing
    if 'claude_rules_checker' not in content.lower():
        print("🔧 Adding rules integration section...")
        
        rules_section = '''

## CLAUDE.md Rules Integration

This agent enforces CLAUDE.md rules through the following integration:

```python
# Import the rules checker
import sys
import os
sys.path.append('/opt/sutazaiapp/.claude/agents')

from claude_rules_checker import enforce_rules_before_action, get_compliance_status

# Before any action, check CLAUDE.md compliance
def safe_execute(action_description: str, action_function):
    """Safely execute an action with CLAUDE.md rules checking"""
    
    # Check rules compliance
    if not enforce_rules_before_action(action_description):
        return False
    
    # Execute the action
    try:
        return action_function()
    except Exception as e:
        print(f"❌ Action failed: {e}")
        return False

# Example usage in agent
def example_agent_action():
    action_desc = "Analyzing codebase for dify-automation-specialist-detailed tasks"
    
    def analyze_code():
        # Your actual analysis code here
        print("✅ Performing analysis...")
        return True
    
    return safe_execute(action_desc, analyze_code)
```

### Required Environment Variables:
- `CLAUDE_RULES_ENABLED=true` - Enable rules checking
- `CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md` - Path to rules file
- `AGENT_NAME=dify-automation-specialist-detailed` - This agent's name for logging

### Startup Compliance Check:
```bash
# Use the startup wrapper to ensure compliance
python3 /opt/sutazaiapp/.claude/agents/agent_startup_wrapper.py dify-automation-specialist-detailed
```
'''
        
        # Add after the compliance header
        insert_pos = content.find('This file contains critical rules')
        if insert_pos != -1:
            # Find end of that section
            insert_pos = content.find('\n\n', insert_pos)
            if insert_pos != -1:
                content = content[:insert_pos + 2] + rules_section + content[insert_pos + 2:]
                modified = True
        else:
            # Add at the end
            content += rules_section
            modified = True
    
    # Write updated content
    if modified:
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed dify-automation-specialist-detailed")
        return True
    else:
        print(f"ℹ️  dify-automation-specialist-detailed already compliant")
        return True

if __name__ == "__main__":
    success = fix_dify_automation_specialist_detailed()
    exit(0 if success else 1)
