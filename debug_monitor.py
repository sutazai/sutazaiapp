#!/usr/bin/env python3
"""Debug monitor agent detection"""

import subprocess
import json
from pathlib import Path

# Load agent registry
registry_path = Path('/opt/sutazaiapp/agents/agent_registry.json')
with open(registry_path, 'r') as f:
    registry = json.load(f)

print("Registry has", len(registry.get('agents', {})), "agents")

# Get running containers
result = subprocess.run(
    ['docker', 'ps', '--filter', 'name=sutazai-', '--format', '{{.Names}}'],
    capture_output=True, text=True
)

containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
print("\nFound", len(containers), "running containers")

# Show what the monitor is looking for vs what's actually running
print("\nMonitor display names vs actual containers:")
print("-" * 60)

# These are the display names shown in the monitor output
monitor_names = {
    'document-manager': ['document-knowledge-manager'],
    'ollama-specialist': ['ollama-integration-specialist'],
    'code-improver': ['code-generation-improver'],
    'semgrep-analyzer': ['semgrep-security-analyzer'],
    'senior-ai-engineer': ['senior-ai-engineer', 'ai-senior-engineer', 'senior-engineer'],
    'hw-resource-optim': ['hardware-resource-optimizer']
}

for display_name, possible_ids in monitor_names.items():
    print(f"\n{display_name}:")
    for agent_id in possible_ids:
        # Check registry
        in_registry = agent_id in registry.get('agents', {})
        # Check containers
        container_name = f'sutazai-{agent_id}'
        container_running = container_name in containers
        
        print(f"  {agent_id}: registry={in_registry}, container={container_running}")

print("\n\nActual running agent containers:")
for container in sorted(containers):
    if container.startswith('sutazai-'):
        agent_id = container[8:]
        if agent_id not in ['postgres', 'redis', 'ollama', 'chromadb', 'qdrant', 'neo4j', 'backend']:
            print(f"  {container} -> {agent_id}")