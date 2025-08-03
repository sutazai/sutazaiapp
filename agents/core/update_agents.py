#!/usr/bin/env python3
"""
Purpose: Update all agents to use BaseAgentV2 with Ollama integration
Usage: python update_agents.py
Requirements: None - uses only standard library
"""

import os
import re
from pathlib import Path

def update_agent_file(file_path: Path) -> bool:
    """Update a single agent file to use BaseAgentV2"""
    try:
        content = file_path.read_text()
        
        # Check if already updated
        if "BaseAgentV2" in content or "base_agent_v2" in content:
            return True
            
        # Update imports
        replacements = [
            (r'from agents\.agent_base import BaseAgent', 'from agents.core.base_agent_v2 import BaseAgentV2'),
            (r'from agent_base import BaseAgent', 'from agents.core.base_agent_v2 import BaseAgentV2'),
            (r'from \.agent_base import BaseAgent', 'from agents.core.base_agent_v2 import BaseAgentV2'),
            (r'class\s+(\w+)\s*\(\s*BaseAgent\s*\)', r'class \1(BaseAgentV2)'),
        ]
        
        modified = False
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                
        if modified:
            file_path.write_text(content)
            print(f"✅ Updated: {file_path.parent.name}/{file_path.name}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    agents_dir = Path("/opt/sutazaiapp/agents")
    updated = 0
    
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
            # Check app.py or agent.py
            for filename in ["app.py", "agent.py"]:
                agent_file = agent_dir / filename
                if agent_file.exists():
                    if update_agent_file(agent_file):
                        updated += 1
                    break
                    
    print(f"\n✅ Updated {updated} agents to use BaseAgentV2")

if __name__ == "__main__":
    main()