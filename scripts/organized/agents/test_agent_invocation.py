#!/usr/bin/env python3
"""Test script to verify agent invocation through Task tool"""

import json
import os
from pathlib import Path

def test_agents():
    print("=== Testing Agent Invocation ===\n")
    
    # List of agents to test
    test_agents = [
        'ai-agent-orchestrator',
        'deployment-automation-master',
        'testing-qa-validator',
        'agi-system-architect',
        'autonomous-system-controller',
        'system-optimizer-reorganizer'
    ]
    
    print("Agents configured in .claude/agents/:")
    agent_dir = Path("/opt/sutazaiapp/.claude/agents")
    for agent_file in sorted(agent_dir.glob("*.md")):
        print(f"  ✓ {agent_file.stem}")
    
    print(f"\nTotal agents found: {len(list(agent_dir.glob('*.md')))}")
    
    print("\n=== MCP Server Configuration ===")
    mcp_config = Path("/opt/sutazaiapp/.mcp.json")
    if mcp_config.exists():
        with open(mcp_config) as f:
            config = json.load(f)
            for server, details in config.get('mcpServers', {}).items():
                print(f"  • {server}: {details.get('command')} {' '.join(details.get('args', []))}")
    
    print("\n=== Claude Settings ===")
    settings = Path("/opt/sutazaiapp/.claude/settings.local.json")
    if settings.exists():
        with open(settings) as f:
            config = json.load(f)
            enabled = config.get('enabledMcpjsonServers', [])
            disabled = config.get('disabledMcpjsonServers', [])
            print(f"  Enabled MCP servers: {enabled}")
            print(f"  Disabled MCP servers: {disabled}")
    
    print("\n=== Instructions for Using Agents ===")
    print("\n1. The agents are available in the Claude UI agent selection")
    print("2. To use an agent programmatically, Claude needs the Task tool")
    print("3. The Task tool is provided by the task-master-ai MCP server")
    print("4. If the Task tool is not available, restart Claude to reload MCP servers")
    print("\n5. Example usage when Task tool is available:")
    print("   - Use the Task tool with subagent_type parameter")
    print("   - Example: Task(subagent_type='ai-agent-orchestrator', prompt='Help coordinate agents')")

if __name__ == "__main__":
    test_agents()