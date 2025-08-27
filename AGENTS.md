# SutazAI Agent System Documentation

## Overview
The SutazAI system includes a comprehensive agent framework designed for AI orchestration. This document provides accurate information about the agent system based on actual code inspection.

## Agent Registry Status (2025-08-26)

### Unified Agent Registry
- **Location**: `/opt/sutazaiapp/backend/app/core/unified_agent_registry.py`
- **Loaded Agents**: 117 unified agents (verified from backend startup logs)
- **Claude Agent Files**: 14 files found and loaded

### Default Agents (Currently Active)
The system currently loads 5 default agents in emergency mode:

1. **Text Analysis Agent**
   - ID: `text-analysis`
   - Status: Module missing (`agents.core` not found)
   - Capabilities: Text processing (intended)

2. **Code Generation Agent**
   - ID: `code-gen`
   - Status: Unknown
   - Capabilities: Code generation (intended)

3. **Data Processing Agent**
   - ID: `data-proc`
   - Status: Unknown
   - Capabilities: Data transformation (intended)

4. **Research Agent**
   - ID: `research`
   - Status: Unknown
   - Capabilities: Information gathering (intended)

5. **System Monitoring Agent**
   - ID: `monitor`
   - Status: Unknown
   - Capabilities: System health monitoring (intended)

### Agent Definition Files
**Total Agent Files**: 243 (found in various locations)

#### Directory Structure
```
/opt/sutazaiapp/
├── .claude/agents/           # 117 Claude agent definitions
├── backend/
│   ├── ai_agents/           # Core agent implementations
│   └── app/core/            # Agent registry and management
├── agents/                  # Additional agent modules
└── mcp-servers/            # MCP-based agents
```

### MCP-Based Agents
MCP servers that act as specialized agents:

| Agent | Location | Status |
|-------|----------|--------|
| Sequential Thinking | `/scripts/mcp/wrappers/sequentialthinking.sh` | ✅ Working |
| Files | `/scripts/mcp/wrappers/files.sh` | ✅ Working |
| Code Index | `/scripts/mcp/wrappers/code-index-mcp.sh` | ✅ Working |
| GitHub | `/scripts/mcp/wrappers/github.sh` | ✅ Working |
| Knowledge Graph | `/scripts/mcp/wrappers/knowledge-graph-mcp.sh` | ✅ Working |

### Agent Capabilities (Intended)
Based on code analysis, agents are designed to provide:

1. **Task Orchestration**: Multi-agent coordination
2. **Code Generation**: AI-powered code creation
3. **Data Processing**: ETL and transformation
4. **Research & Analysis**: Information gathering
5. **System Monitoring**: Health and performance tracking
6. **Vector Operations**: Embedding and similarity search
7. **Mesh Integration**: Service discovery and routing

### Current Issues

1. **Module Import Errors**: 
   - Text Analysis Agent cannot load due to missing `agents.core` module
   - Many agent implementations reference non-existent modules

2. **Hardcoded URLs**:
   - Default agents use `http://internal-agent-url` placeholder
   - No actual agent endpoints configured

3. **Missing Implementations**:
   - Agent definition files exist but lack actual implementation
   - Many `.md` files describe agents that don't have corresponding code

### Agent API Endpoints

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/v1/agents/list` | GET | ✅ Working |
| `/api/v1/agents/{agent_id}/status` | GET | ❓ Untested |
| `/api/v1/agents/{agent_id}/execute` | POST | ❓ Untested |
| `/api/v1/agents/register` | POST | ❓ Untested |

### How to Test Agents

```bash
# List all registered agents
curl http://localhost:10010/api/v1/agents/list

# Check specific agent status
curl http://localhost:10010/api/v1/agents/text-analysis/status

# Execute agent task (if implemented)
curl -X POST http://localhost:10010/api/v1/agents/code-gen/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "generate Python function"}'
```

## Agent Development Status

### Completed
- ✅ Agent registry framework
- ✅ Unified agent loading system
- ✅ MCP agent wrappers
- ✅ Basic API endpoints

### In Progress
- 🔄 Agent implementation modules
- 🔄 Service mesh integration
- 🔄 Agent-to-agent communication

### Not Started
- ❌ Agent UI in frontend
- ❌ Agent marketplace
- ❌ Agent performance monitoring
- ❌ Agent versioning system

## Contributing

To add a new agent:

1. Create agent definition in `.claude/agents/`
2. Implement agent logic in `backend/ai_agents/`
3. Register in `unified_agent_registry.py`
4. Add API endpoints if needed
5. Test with curl commands

## Notes

- Most agent functionality is aspirational - framework exists but implementations are incomplete
- The "200+ agents" claim is based on definition files, not working implementations
- MCP agents are the most functional, using external npm packages
- Focus should be on implementing core agents before expanding

---
*Last updated: 2025-08-26 23:50 UTC*
*Based on actual code inspection and testing*