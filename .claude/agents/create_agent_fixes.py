#!/usr/bin/env python3
"""
Create Agent Fixes Script

This script creates specific fixes for individual non-compliant agents
"""

import os
import json
from pathlib import Path
from typing import List, Dict

class AgentFixCreator:
    """Creates specific fixes for non-compliant agents"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
    
    def create_environment_section(self, agent_name: str) -> str:
        """Create environment section for agent YAML"""
        return f"""environment:
  - CLAUDE_RULES_ENABLED=true
  - CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
  - AGENT_NAME={agent_name}
  - LOG_LEVEL=INFO"""
    
    def create_rules_integration_section(self, agent_name: str) -> str:
        """Create rules integration section for agent"""
        return f"""

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
    \"\"\"Safely execute an action with CLAUDE.md rules checking\"\"\"
    
    # Check rules compliance
    if not enforce_rules_before_action(action_description):
        return False
    
    # Execute the action
    try:
        return action_function()
    except Exception as e:
        print(f"❌ Action failed: {{e}}")
        return False

# Example usage in agent
def example_agent_action():
    action_desc = "Analyzing codebase for {agent_name} tasks"
    
    def analyze_code():
        # Your actual analysis code here
        print("✅ Performing analysis...")
        return True
    
    return safe_execute(action_desc, analyze_code)
```

### Required Environment Variables:
- `CLAUDE_RULES_ENABLED=true` - Enable rules checking
- `CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md` - Path to rules file
- `AGENT_NAME={agent_name}` - This agent's name for logging

### Startup Compliance Check:
```bash
# Use the startup wrapper to ensure compliance
python3 /opt/sutazaiapp/.claude/agents/agent_startup_wrapper.py {agent_name}
```
"""
    
    def generate_agent_fix(self, agent_name: str, issues: List[str]) -> str:
        """Generate a fix script for a specific agent"""
        
        fix_content = f"""#!/usr/bin/env python3
\"\"\"
Compliance Fix for {agent_name}

This script fixes the following issues:
{chr(10).join(f'- {issue}' for issue in issues)}
\"\"\"

import re
from pathlib import Path

def fix_{agent_name.replace('-', '_')}():
    \"\"\"Fix compliance issues for {agent_name}\"\"\"
    
    agent_file = Path('/opt/sutazaiapp/.claude/agents/{agent_name}.md')
    
    if not agent_file.exists():
        print(f"❌ Agent file not found: {{agent_file}}")
        return False
    
    # Read current content
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_file = agent_file.with_suffix('.md.backup')
    if not backup_file.exists():
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📦 Backup created: {{backup_file}}")
    
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
  - AGENT_NAME={agent_name}
  - LOG_LEVEL=INFO
'''
            content = content[:yaml_end] + env_section + content[yaml_end:]
            modified = True
    
    # Fix: Add rules integration section if missing
    if 'claude_rules_checker' not in content.lower():
        print("🔧 Adding rules integration section...")
        
        rules_section = '''{self.create_rules_integration_section(agent_name)}'''
        
        # Add after the compliance header
        insert_pos = content.find('This file contains critical rules')
        if insert_pos != -1:
            # Find end of that section
            insert_pos = content.find('\\n\\n', insert_pos)
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
        print(f"✅ Fixed {agent_name}")
        return True
    else:
        print(f"ℹ️  {agent_name} already compliant")
        return True

if __name__ == "__main__":
    success = fix_{agent_name.replace('-', '_')}()
    exit(0 if success else 1)
"""
        
        return fix_content
    
    def create_fixes_for_non_compliant_agents(self, compliance_report_file: str) -> Dict[str, bool]:
        """Create fix scripts for all non-compliant agents"""
        
        # Load compliance report
        with open(compliance_report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        fixes_dir = self.agents_dir / "fixes"
        fixes_dir.mkdir(exist_ok=True)
        
        created_fixes = {}
        
        for agent_file, analysis in report['agents'].items():
            if not analysis['compliant']:
                agent_name = agent_file.replace('.md', '')
                issues = analysis['issues']
                
                print(f"🔧 Creating fix for {agent_name}...")
                
                # Generate fix script
                fix_content = self.generate_agent_fix(agent_name, issues)
                
                # Save fix script
                fix_file = fixes_dir / f"fix_{agent_name.replace('-', '_')}.py"
                
                with open(fix_file, 'w', encoding='utf-8') as f:
                    f.write(fix_content)
                
                # Make executable
                os.chmod(fix_file, 0o755)
                
                created_fixes[agent_name] = str(fix_file)
                print(f"  ✅ Fix created: {fix_file}")
        
        return created_fixes
    
    def create_master_fix_script(self, created_fixes: Dict[str, str]):
        """Create a master script to run all fixes"""
        
        master_script_content = f"""#!/bin/bash
# Master Agent Compliance Fix Script
# Auto-generated on {Path(__file__).name}

set -e

echo "🚀 Running all agent compliance fixes..."
echo "============================================"

FIXES_DIR="/opt/sutazaiapp/.claude/agents/fixes"
TOTAL_FIXES={len(created_fixes)}
SUCCESS_COUNT=0

"""
        
        for agent_name, fix_file in created_fixes.items():
            master_script_content += f"""
echo "🔧 Fixing {agent_name}..."
if python3 "$FIXES_DIR/fix_{agent_name.replace('-', '_')}.py"; then
    echo "  ✅ {agent_name} fixed successfully"
    ((SUCCESS_COUNT++))
else
    echo "  ❌ Failed to fix {agent_name}"
fi
echo
"""
        
        master_script_content += """
echo "============================================"
echo "📊 Fix Summary: $SUCCESS_COUNT/$TOTAL_FIXES agents fixed"

if [ $SUCCESS_COUNT -eq $TOTAL_FIXES ]; then
    echo "🎉 All agents fixed successfully!"
    exit 0
else
    echo "⚠️  Some agents could not be fixed"
    exit 1
fi
"""
        
        master_script_path = self.agents_dir / "run_all_agent_fixes.sh"
        
        with open(master_script_path, 'w', encoding='utf-8') as f:
            f.write(master_script_content)
        
        os.chmod(master_script_path, 0o755)
        
        print(f"✅ Master fix script created: {master_script_path}")
        return master_script_path

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 create_agent_fixes.py <compliance_report.json>")
        sys.exit(1)
    
    report_file = sys.argv[1]
    
    if not Path(report_file).exists():
        print(f"❌ Compliance report not found: {report_file}")
        sys.exit(1)
    
    creator = AgentFixCreator()
    
    # Create individual fix scripts
    fixes = creator.create_fixes_for_non_compliant_agents(report_file)
    
    print(f"\\n📊 Created {len(fixes)} fix scripts")
    
    if fixes:
        # Create master fix script
        master_script = creator.create_master_fix_script(fixes)
        print(f"\\n🎯 Master fix script: {master_script}")
        print("\\n💡 To run all fixes:")
        print(f"   bash {master_script}")

if __name__ == "__main__":
    main()