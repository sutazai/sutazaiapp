#!/usr/bin/env python3
"""
Agent Compliance Update Script

This script updates individual agent .md files to ensure they comply with CLAUDE.md rules
"""

import os
import re
import yaml
from typing import Dict, List, Optional
from pathlib import Path

class AgentComplianceUpdater:
    """Updates agent files for CLAUDE.md compliance"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        
    def backup_agent_file(self, file_path: Path) -> Path:
        """Create a backup of the agent file before modification"""
        backup_dir = self.agents_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / file_path.name
        
        # If backup already exists, don't overwrite it
        if not backup_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  📦 Backup created: {backup_path}")
        
        return backup_path
    
    def add_rules_integration_section(self, content: str, agent_name: str) -> str:
        """Add rules integration section to agent content"""
        
        # Find the end of the YAML front matter
        yaml_end_match = re.search(r'^---\s*$', content, re.MULTILINE)
        if not yaml_end_match:
            # No YAML front matter found, add at the beginning
            yaml_start = 0
        else:
            yaml_start = yaml_end_match.end()
        
        # Extract the YAML section and the rest
        yaml_section = content[:yaml_start]
        remaining_content = content[yaml_start:]
        
        # Add environment variables to YAML if missing
        if 'environment:' not in yaml_section and 'env_file:' not in yaml_section:
            # Find where to insert environment section
            if yaml_section.strip().endswith('---'):
                # Insert before the closing ---
                yaml_insert_pos = yaml_section.rfind('---')
                environment_section = """environment:
  - CLAUDE_RULES_ENABLED=true
  - CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
  - AGENT_NAME={agent_name}
""".format(agent_name=agent_name)
                yaml_section = yaml_section[:yaml_insert_pos] + environment_section + yaml_section[yaml_insert_pos:]
        
        # Add rules integration to the prompt/description if not present
        rules_integration_text = """

## Rules Enforcement Integration

This agent integrates with the CLAUDE.md rules checker to ensure compliance:

```python
# Import the rules checker
from claude_rules_checker import check_action, get_rules_summary

# Before executing any action, check compliance
def execute_action(action_description: str):
    can_proceed, issues = check_action(action_description)
    
    if not can_proceed:
        print("❌ BLOCKED: Action violates CLAUDE.md rules:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    if issues:  # Warnings
        print("⚠️  WARNING: Please review:")
        for issue in issues:
            print(f"  • {issue}")
    
    # Proceed with action
    return True
```

## Required Environment Variables:
- CLAUDE_RULES_ENABLED=true
- CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
- AGENT_NAME={agent_name}

""".format(agent_name=agent_name)
        
        # Check if rules integration is already present
        if 'claude_rules_checker' not in remaining_content.lower() and 'rules enforcement' not in remaining_content.lower():
            # Add after the compliance header if it exists, otherwise at the end
            if 'This file contains critical rules' in remaining_content:
                insert_pos = remaining_content.find('This file contains critical rules')
                # Find the end of that paragraph
                insert_pos = remaining_content.find('\n\n', insert_pos)
                if insert_pos == -1:
                    insert_pos = len(remaining_content)
                else:
                    insert_pos += 2  # After the double newline
            else:
                insert_pos = len(remaining_content)
            
            remaining_content = remaining_content[:insert_pos] + rules_integration_text + remaining_content[insert_pos:]
        
        return yaml_section + remaining_content
    
    def update_agent_file(self, file_path: Path) -> bool:
        """Update a single agent file for compliance"""
        try:
            # Create backup first
            self.backup_agent_file(file_path)
            
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            agent_name = file_path.stem
            
            # Add rules integration
            updated_content = self.add_rules_integration_section(content, agent_name)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"  ✅ Updated: {file_path.name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error updating {file_path.name}: {str(e)}")
            return False
    
    def update_agents_by_list(self, agent_names: List[str]) -> Dict[str, bool]:
        """Update specific agents by name"""
        results = {}
        
        for agent_name in agent_names:
            file_path = self.agents_dir / f"{agent_name}.md"
            
            if not file_path.exists():
                print(f"❌ Agent file not found: {file_path}")
                results[agent_name] = False
                continue
            
            print(f"🔧 Updating: {agent_name}")
            results[agent_name] = self.update_agent_file(file_path)
        
        return results
    
    def update_non_compliant_agents(self, compliance_report_file: str) -> Dict[str, bool]:
        """Update all non-compliant agents from compliance report"""
        import json
        
        # Load compliance report
        with open(compliance_report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Find non-compliant agents
        non_compliant_agents = []
        for agent_name, analysis in report['agents'].items():
            if not analysis['compliant']:
                # Remove .md extension to get agent name
                agent_base_name = agent_name.replace('.md', '')
                non_compliant_agents.append(agent_base_name)
        
        print(f"Found {len(non_compliant_agents)} non-compliant agents to update")
        
        return self.update_agents_by_list(non_compliant_agents)

def main():
    """Main function"""
    import sys
    
    updater = AgentComplianceUpdater()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--from-report" and len(sys.argv) > 2:
            # Update from compliance report
            report_file = sys.argv[2]
            results = updater.update_non_compliant_agents(report_file)
        else:
            # Update specific agents
            agent_names = sys.argv[1:]
            results = updater.update_agents_by_list(agent_names)
    else:
        print("Usage:")
        print("  python3 update_agent_compliance.py agent1 agent2 ...")
        print("  python3 update_agent_compliance.py --from-report report.json")
        return
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Update Summary: {successful}/{total} agents updated successfully")
    
    if successful < total:
        print("\n❌ Failed agents:")
        for agent_name, success in results.items():
            if not success:
                print(f"  • {agent_name}")

if __name__ == "__main__":
    main()