# Agent Configuration

All agent configurations have been consolidated into a single unified registry:

**Location:** `/opt/sutazaiapp/config/agents/unified_agent_registry.json`

## Migration Notice

This directory previously contained 140+ individual configuration files:
- 70+ `*_universal.json` files  
- 30+ `*_ollama.json` files
- 40+ `.modelfile` files

These have all been consolidated into the unified registry for:
- Easier maintenance
- Reduced duplication
- Centralized configuration management
- Better consistency across agents

## How to Access Agent Configs

```python
import json

# Load unified registry
with open('/opt/sutazaiapp/config/agents/unified_agent_registry.json') as f:
    registry = json.load(f)

# Access specific agent config
agent_config = registry['agents']['ai-agent-orchestrator']
```

## Archived Configs

Original individual config files have been archived to:
`/opt/sutazaiapp/archive/agent_configs_20250815/`