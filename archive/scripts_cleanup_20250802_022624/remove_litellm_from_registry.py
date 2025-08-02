#!/usr/bin/env python3
"""
Remove litellm-proxy-manager from agent registry
"""
import json

# Read the agent registry
with open('/opt/sutazaiapp/agents/agent_registry.json', 'r') as f:
    registry = json.load(f)

# Remove litellm-proxy-manager if it exists
if 'litellm-proxy-manager' in registry['agents']:
    del registry['agents']['litellm-proxy-manager']
    print("✅ Removed litellm-proxy-manager from agent registry")
else:
    print("ℹ️  litellm-proxy-manager not found in registry")

# Write back the updated registry
with open('/opt/sutazaiapp/agents/agent_registry.json', 'w') as f:
    json.dump(registry, f, indent=2)

print(f"📊 Total agents in registry: {len(registry['agents'])}")