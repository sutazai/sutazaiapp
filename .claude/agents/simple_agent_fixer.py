#!/usr/bin/env python3
"""
Simple Agent Fixer - Direct approach to fix agent compliance

This script directly adds the missing rules integration to agent files
"""

import os
import re
from pathlib import Path

class SimpleAgentFixer:
    """Simple, direct agent fixer"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        
    def add_rules_integration_to_agent(self, agent_file: Path) -> bool:
        """Add rules integration section directly to an agent file"""
        
        try:
            # Read current content
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already has rules integration
            if 'claude_rules_checker' in content.lower():
                print(f"  ℹ️  {agent_file.name} already has rules integration")
                return True
            
            # Create backup
            backup_file = agent_file.with_suffix('.md.backup')
            if not backup_file.exists():
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  📦 Backup: {backup_file.name}")
            
            agent_name = agent_file.stem
            
            # Rules integration section to add
            rules_section = f"""

## CLAUDE.md Rules Integration

This agent enforces CLAUDE.md rules through integrated compliance checking:

```python
# Import rules checker
import sys
import os
sys.path.append('/opt/sutazaiapp/.claude/agents')

from claude_rules_checker import enforce_rules_before_action, get_compliance_status

# Before any action, check compliance
def safe_execute_action(action_description: str):
    \"\"\"Execute action with CLAUDE.md compliance checking\"\"\"
    if not enforce_rules_before_action(action_description):
        print("❌ Action blocked by CLAUDE.md rules")
        return False
    print("✅ Action approved by CLAUDE.md compliance")
    return True

# Example usage
def example_task():
    if safe_execute_action("Analyzing codebase for {agent_name}"):
        # Your actual task code here
        pass
```

**Environment Variables:**
- `CLAUDE_RULES_ENABLED=true`
- `CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md`
- `AGENT_NAME={agent_name}`

**Startup Check:**
```bash
python3 /opt/sutazaiapp/.claude/agents/agent_startup_wrapper.py {agent_name}
```
"""
            
            # Add environment section to YAML if missing
            if 'environment:' not in content and 'env_file:' not in content:
                # Find where to add environment section in YAML
                yaml_end_match = re.search(r'^---\s*$', content, re.MULTILINE)
                if yaml_end_match:
                    yaml_end_pos = yaml_end_match.start()
                    env_section = f"""environment:
  - CLAUDE_RULES_ENABLED=true
  - CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
  - AGENT_NAME={agent_name}
"""
                    content = content[:yaml_end_pos] + env_section + content[yaml_end_pos:]
            
            # Add rules integration section at the end
            content += rules_section
            
            # Write updated content
            with open(agent_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✅ Fixed: {agent_file.name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error fixing {agent_file.name}: {str(e)}")
            return False
    
    def fix_all_non_compliant_agents(self) -> dict:
        """Fix all agents that need rules integration"""
        
        # Get all agent .md files
        agent_files = []
        exclude_patterns = [
            'AGENT_CLEANUP_SUMMARY.md',
            'AGENT_COMPLIANCE_REPORT.md', 
            'COMPREHENSIVE_INVESTIGATION_PROTOCOL.md',
            'COMPLETE_CLEANUP_STATUS.md',
            'DUPLICATE_AGENTS_REPORT.md',
            'FINAL_COMPLIANCE_SUMMARY.md',
            'team_collaboration_standards.md'
        ]
        
        for md_file in self.agents_dir.glob("*.md"):
            if md_file.name not in exclude_patterns:
                agent_files.append(md_file)
        
        print(f"Found {len(agent_files)} agent files to check")
        
        results = {"fixed": 0, "already_compliant": 0, "failed": 0, "details": []}
        
        for agent_file in agent_files:
            print(f"🔧 Processing: {agent_file.name}")
            
            # Check if it needs fixing
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'claude_rules_checker' in content.lower():
                print(f"  ✅ Already compliant: {agent_file.name}")
                results["already_compliant"] += 1
                results["details"].append({"file": agent_file.name, "status": "already_compliant"})
            else:
                # Try to fix it
                if self.add_rules_integration_to_agent(agent_file):
                    results["fixed"] += 1
                    results["details"].append({"file": agent_file.name, "status": "fixed"})
                else:
                    results["failed"] += 1
                    results["details"].append({"file": agent_file.name, "status": "failed"})
        
        return results
    
    def fix_specific_agents(self, agent_names: list) -> dict:
        """Fix specific agents by name"""
        results = {"fixed": 0, "already_compliant": 0, "failed": 0, "details": []}
        
        for agent_name in agent_names:
            agent_file = self.agents_dir / f"{agent_name}.md"
            
            if not agent_file.exists():
                print(f"❌ Agent file not found: {agent_file}")
                results["failed"] += 1
                results["details"].append({"file": agent_name, "status": "file_not_found"})
                continue
            
            print(f"🔧 Processing: {agent_name}")
            
            # Check if it needs fixing
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'claude_rules_checker' in content.lower():
                print(f"  ✅ Already compliant: {agent_name}")
                results["already_compliant"] += 1
                results["details"].append({"file": agent_name, "status": "already_compliant"})
            else:
                # Try to fix it
                if self.add_rules_integration_to_agent(agent_file):
                    results["fixed"] += 1
                    results["details"].append({"file": agent_name, "status": "fixed"})
                else:
                    results["failed"] += 1
                    results["details"].append({"file": agent_name, "status": "failed"})
        
        return results

def main():
    """Main function"""
    import sys
    
    fixer = SimpleAgentFixer()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            print("🚀 Fixing all non-compliant agents...")
            results = fixer.fix_all_non_compliant_agents()
        else:
            # Fix specific agents
            agent_names = sys.argv[1:]
            print(f"🚀 Fixing specific agents: {', '.join(agent_names)}")
            results = fixer.fix_specific_agents(agent_names)
    else:
        print("Usage:")
        print("  python3 simple_agent_fixer.py --all")
        print("  python3 simple_agent_fixer.py agent1 agent2 ...")
        return
    
    # Print summary
    total = results["fixed"] + results["already_compliant"] + results["failed"]
    print(f"\n📊 SUMMARY:")
    print(f"  Total processed: {total}")
    print(f"  Fixed: {results['fixed']}")
    print(f"  Already compliant: {results['already_compliant']}")
    print(f"  Failed: {results['failed']}")
    
    if results["failed"] > 0:
        print(f"\n❌ Failed agents:")
        for detail in results["details"]:
            if detail["status"] == "failed":
                print(f"  • {detail['file']}")

if __name__ == "__main__":
    main()