#!/usr/bin/env python3
"""Debug container name mapping issue"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

def get_agent_registry():
    """Load agent registry"""
    registry_path = Path('/opt/sutazaiapp/agents/agent_registry.json')
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return json.load(f)
    return {}

def get_actual_containers() -> List[str]:
    """Get actual running containers"""
    result = subprocess.run(
        ['docker', 'ps', '-a', '--filter', 'name=sutazai-', '--format', '{{.Names}}'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().split('\n')
    return []

def find_container_matches(agent_id: str, containers: List[str]) -> List[str]:
    """Find potential container matches for an agent ID"""
    matches = []
    
    # Exact pattern matches
    patterns = [
        f'sutazai-{agent_id}',
        f'{agent_id}',
        f'sutazaiapp-{agent_id}',
        f'{agent_id}-1'
    ]
    
    for pattern in patterns:
        if pattern in containers:
            matches.append(pattern)
    
    # Fuzzy matches - contains part of agent name
    agent_parts = agent_id.split('-')
    for container in containers:
        if len(agent_parts) > 1:
            # Check if any significant part of agent name is in container
            for part in agent_parts:
                if len(part) > 4 and part in container:  # Only meaningful parts
                    if container not in matches:
                        matches.append(f"{container} (fuzzy match on '{part}')")
    
    return matches

def main():
    print("=== Agent Container Mapping Debug ===\n")
    
    # Get data
    registry = get_agent_registry()
    containers = get_actual_containers()
    
    print(f"Total agents in registry: {len(registry.get('agents', {}))}")
    print(f"Total sutazai containers: {len(containers)}")
    print()
    
    # Focus on the 6 agents from the monitor
    test_agents = [
        'document-knowledge-manager',
        'ollama-integration-specialist', 
        'code-generation-improver',
        'semgrep-security-analyzer',
        'senior-ai-engineer',
        'hardware-resource-optimizer'
    ]
    
    print("=== Container Mapping Analysis ===")
    for agent_id in test_agents:
        print(f"\nAgent: {agent_id}")
        matches = find_container_matches(agent_id, containers)
        if matches:
            print(f"  ✓ Found matches: {matches}")
        else:
            print(f"  ✗ No matches found")
            
            # Look for partial matches in all containers
            partial_matches = []
            agent_words = agent_id.replace('-', ' ').split()
            for container in containers:
                container_clean = container.replace('sutazai-', '').replace('-', ' ')
                for word in agent_words:
                    if len(word) > 3 and word in container_clean:
                        partial_matches.append(f"{container} (contains '{word}')")
                        break
            
            if partial_matches:
                print(f"  ? Possible partial matches: {partial_matches[:3]}")  # Show max 3
    
    print("\n=== All Container Names ===")
    for container in sorted(containers):
        if container.strip():
            print(f"  - {container}")

if __name__ == "__main__":
    main()