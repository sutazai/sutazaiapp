#!/usr/bin/env python3
"""List all agents with their configuration"""

import os
import yaml
from pathlib import Path

def list_agents():
    # Check both directories
    dirs = [
        Path("/opt/sutazaiapp/.claude/agents"),
        Path("/root/.claude/agents")
    ]
    
    all_agents = {}
    
    for agent_dir in dirs:
        if not agent_dir.exists():
            continue
            
        print(f"\n=== Checking {agent_dir} ===")
        
        for agent_file in sorted(agent_dir.glob("*.md")):
            with open(agent_file, 'r') as f:
                content = f.read()
                
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        name = frontmatter.get('name', 'unknown')
                        model = frontmatter.get('model', 'unknown')
                        
                        if name not in all_agents:
                            all_agents[name] = {
                                'model': model,
                                'path': str(agent_file),
                                'locations': [str(agent_dir)]
                            }
                        else:
                            all_agents[name]['locations'].append(str(agent_dir))
                            
                    except Exception as e:
                        print(f"Error parsing {agent_file}: {e}")
    
    # Display results
    print(f"\n=== Total unique agents: {len(all_agents)} ===\n")
    
    print(f"{'Agent Name':<40} {'Model':<10} {'Locations'}")
    print("-" * 100)
    
    for name, info in sorted(all_agents.items()):
        locations = ', '.join([loc.replace('/opt/sutazaiapp/.claude/agents', 'PROJECT')
                                  .replace('/root/.claude/agents', 'PERSONAL') 
                              for loc in info['locations']])
        print(f"{name:<40} {info['model']:<10} {locations}")
    
    # Show only the ones visible in Claude
    visible_agents = [
        'infrastructure-devops-manager',
        'agentzero-coordinator',
        'flowiseai-flow-manager',
        'bigagi-system-manager',
        'localagi-orchestration-manager',
        'langflow-workflow-designer',
        'agentgpt-autonomous-executor',
        'dify-automation-specialist',
        'opendevin-code-generator',
        'semgrep-security-analyzer',
        'private-data-analyst'
    ]
    
    print(f"\n=== Agents visible in Claude UI ({len(visible_agents)}) ===")
    for agent in visible_agents:
        if agent in all_agents:
            print(f"✓ {agent} - {all_agents[agent]['model']}")
        else:
            print(f"✗ {agent} - NOT FOUND")
    
    print(f"\n=== Missing from Claude UI ({len(all_agents) - len(visible_agents)}) ===")
    for name in sorted(all_agents.keys()):
        if name not in visible_agents:
            print(f"• {name} - {all_agents[name]['model']}")

if __name__ == "__main__":
    list_agents()