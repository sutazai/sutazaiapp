#!/usr/bin/env python3
"""
Demonstrate the properly consolidated agent configuration system
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.core.unified_agent_registry import get_registry

def demonstrate_consolidation():
    """Show how agent configurations are now properly consolidated"""
    
    print("=" * 80)
    print("UNIFIED AGENT REGISTRY - CONFIGURATION CONSOLIDATION DEMO")
    print("=" * 80)
    
    registry = get_registry()
    
    # Example 1: Find specialized agents
    print("\n📋 Example 1: Finding Specialized Agents")
    print("-" * 40)
    
    tasks = [
        ("Deploy my application to production", ["deployment"]),
        ("Optimize system performance", ["optimization"]),
        ("Run security audit", ["security_analysis"]),
        ("Create test suite", ["testing"]),
        ("Orchestrate multiple agents", ["orchestration"])
    ]
    
    for task, caps in tasks:
        agent = registry.find_best_agent(task, caps)
        if agent:
            print(f"✓ Task: '{task[:40]}...'")
            print(f"  → Agent: {agent.name} ({agent.type})")
            print(f"  → Capabilities: {', '.join(agent.capabilities[:3])}")
        print()
    
    # Example 2: Show consolidation statistics
    print("\n📊 Example 2: Consolidation Statistics")
    print("-" * 40)
    
    stats = registry.get_statistics()
    
    print(f"Total Agents Consolidated: {stats['total_agents']}")
    print(f"  • Claude Agents: {stats['claude_agents']} (from .claude/agents/)")
    print(f"  • Container Agents: {stats['container_agents']} (from agent_registry.json)")
    
    # Example 3: Show top capabilities
    print("\n🎯 Example 3: Top Agent Capabilities")
    print("-" * 40)
    
    sorted_caps = sorted(stats['capabilities'].items(), key=lambda x: x[1], reverse=True)
    for cap, count in sorted_caps[:5]:
        bar = "█" * min(50, count // 5)
        print(f"{cap:20} {bar} ({count})")
    
    # Example 4: Verify no fantasy files
    print("\n✅ Example 4: File Reference Validation")
    print("-" * 40)
    
    invalid_count = 0
    valid_count = 0
    
    for agent in registry.agents.values():
        if agent.deployment_info:
            agent_file = agent.deployment_info.get("agent_file")
            config_path = agent.deployment_info.get("config_path")
            
            if agent_file and Path(agent_file).exists():
                valid_count += 1
            elif agent_file:
                invalid_count += 1
                
            if config_path and Path(config_path).exists():
                valid_count += 1
    
    print(f"Valid File References: {valid_count}")
    print(f"Invalid File References: {invalid_count}")
    
    if invalid_count == 0:
        print("🎉 ALL FILE REFERENCES ARE VALID - RULE 1 COMPLIANT!")
    
    # Example 5: Show agent selection intelligence
    print("\n🧠 Example 5: Intelligent Agent Selection")
    print("-" * 40)
    
    complex_task = """
    I need to build a complete AI-powered web application with a React frontend,
    FastAPI backend, proper testing, security auditing, and automated deployment
    to production with monitoring.
    """
    
    print(f"Complex Task: {complex_task[:100]}...")
    print("\nRecommended Agent Team:")
    
    subtasks = [
        ("Frontend development", ["frontend", "react", "ui"]),
        ("Backend API", ["backend", "api", "fastapi"]),
        ("AI integration", ["ai", "ml", "ollama"]),
        ("Security audit", ["security", "pentesting", "audit"]),
        ("Testing", ["testing", "qa", "validation"]),
        ("Deployment", ["deployment", "cicd", "production"]),
        ("Monitoring", ["monitoring", "observability", "metrics"])
    ]
    
    team = []
    for subtask, keywords in subtasks:
        agent = registry.find_best_agent(subtask, keywords)
        if agent and agent.name not in [a.name for a in team]:
            team.append(agent)
            print(f"  • {subtask:20} → {agent.name}")
    
    print(f"\nTotal team size: {len(team)} specialized agents")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE - AGENT CONFIGURATION PROPERLY CONSOLIDATED")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_consolidation()