#!/usr/bin/env python3
"""
Analyze all AI agents in the SutazAI system
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def parse_agent_file(filepath):
    """Parse an agent markdown file to extract metadata"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract name and model
    name_match = re.search(r'^name:\s*(.+)$', content, re.MULTILINE)
    model_match = re.search(r'^model:\s*(.+)$', content, re.MULTILINE)
    
    name = name_match.group(1).strip() if name_match else Path(filepath).stem
    model = model_match.group(1).strip() if model_match else 'unknown'
    
    return name, model

def main():
    agent_dir = Path("/opt/sutazaiapp/.claude/agents")
    agents = {}
    
    # Parse all agent files
    for agent_file in agent_dir.glob("*.md"):
        name, model = parse_agent_file(agent_file)
        agents[name] = model
    
    # Count agents by model
    model_counts = defaultdict(int)
    for model in agents.values():
        model_counts[model] += 1
    
    # Display results
    print("🤖 COMPLETE SUTAZAI AGENT REGISTRY")
    print("=" * 80)
    print(f"Total Agents: {len(agents)}")
    print("=" * 80)
    
    # Group by model
    print("\n📊 AGENTS BY MODEL:")
    print("-" * 60)
    
    for model in sorted(model_counts.keys()):
        print(f"\n🧠 {model.upper()} Model ({model_counts[model]} agents):")
        model_agents = [name for name, m in agents.items() if m == model]
        for agent in sorted(model_agents):
            print(f"  • {agent}")
    
    # Display all agents with models
    print("\n\n📋 COMPLETE AGENT LIST:")
    print("-" * 60)
    print(f"{'Agent Name':<45} {'Model':<10}")
    print("-" * 60)
    
    for name in sorted(agents.keys()):
        print(f"{name:<45} {agents[name]:<10}")
    
    # Summary statistics
    print("\n\n📈 SUMMARY:")
    print("-" * 60)
    print(f"Total Agents: {len(agents)}")
    for model, count in sorted(model_counts.items()):
        percentage = (count / len(agents)) * 100
        print(f"{model.capitalize()} agents: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()