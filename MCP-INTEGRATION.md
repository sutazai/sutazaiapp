# MCP Integration Guide for SutazAI

## Current Status (2025-08-26)
- **26 of 28 MCP servers connected** (93% success rate)
- **claude-flow v2.0.0-alpha.101** installed and functional
- **Multiple integration methods available**

## Available MCP Servers

### ✅ Working Servers (26)
| Server | Purpose | Access Method |
|--------|---------|---------------|
| claude-flow | Orchestration | `npx claude-flow@alpha` |
| github | GitHub API | Shell wrapper |
| sequential-thinking | Complex reasoning | Shell wrapper |
| context7 | Documentation | Shell wrapper |
| code-index | Code search | Shell wrapper |
| ultimatecoder | Advanced coding | Shell wrapper |
| extended-memory | Persistent memory | Shell wrapper |
| files | File operations | Shell wrapper |
| search/ddg | Web search | Shell wrapper |
| http/http_fetch | HTTP requests | Shell wrapper |
| playwright | Browser automation | Shell wrapper |
| knowledge-graph | Graph operations | Shell wrapper |
| compass-mcp | Navigation | Shell wrapper |
| git-mcp | Git operations | Shell wrapper |
| nx-mcp | Nx workspace | Shell wrapper |
| memory-bank | Memory storage | Shell wrapper |
| language-server | LSP operations | Shell wrapper |

### ❌ Failed Servers (1)
- `task-runner` (one variant) - Other variants working

## Integration Methods

### Method 1: Direct MCP Tool Usage (In Claude Code)
```javascript
// Use MCP tools directly when available
mcp__claude-flow__sparc_mode {
  mode: "mcp",
  task_description: "your task",
  options: {
    namespace: "mcp",
    non_interactive: false
  }
}

// Memory operations
mcp__extended-memory__save_context {
  content: "important data",
  importance_level: 8,
  project_id: "sutazai-mcp",
  tags: ["mcp", "integration"]
}
```

### Method 2: NPX CLI Commands
```bash
# SPARC modes (17 available)
npx claude-flow@alpha sparc run mcp "integrate with API"
npx claude-flow@alpha sparc run architecture "design system"
npx claude-flow@alpha sparc run coding "implement feature"

# Swarm intelligence
npx claude-flow@alpha swarm "build REST API"
npx claude-flow@alpha swarm "create microservice" --claude

# Hive mind coordination
npx claude-flow@alpha hive-mind init
npx claude-flow@alpha hive-mind spawn "objective"
npx claude-flow@alpha hive-mind status
```

### Method 3: Shell Wrapper Scripts
```bash
# Direct wrapper invocation
/opt/sutazaiapp/scripts/mcp/wrappers/claude-flow.sh --selfcheck
/opt/sutazaiapp/scripts/mcp/wrappers/github.sh --help
/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh --selfcheck
```

## Quick Start Commands

### Test MCP Connection
```bash
# Test all MCP servers
/opt/sutazaiapp/scripts/mcp/test-all-mcp.sh

# Test specific server
/opt/sutazaiapp/scripts/mcp/wrappers/claude-flow.sh --selfcheck

# Check Claude MCP list
claude mcp list
```

### Start Services
```bash
# Backend API (port 10010)
cd /opt/sutazaiapp/backend
export JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
export PYTHONPATH=/opt/sutazaiapp/backend
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 10010

# Check health
curl http://localhost:10010/health
```

### Memory Integration
```bash
# Store context
npx claude-flow@alpha memory store "mcp_context" "integration complete"

# Query memory
npx claude-flow@alpha memory query "mcp" --limit 5

# List all memories
npx claude-flow@alpha memory list
```

## SPARC Modes Available

1. **specification** - Requirements analysis
2. **pseudocode** - Algorithm design  
3. **architecture** - System design
4. **refinement** - Code improvement
5. **coding** - Implementation
6. **debugging** - Bug fixing
7. **testing** - Test creation
8. **optimization** - Performance tuning
9. **documentation** - Doc generation
10. **review** - Code review
11. **deployment** - Deploy automation
12. **monitoring** - System monitoring
13. **security** - Security analysis
14. **mcp** - MCP integration (this mode)
15. **migration** - Code migration
16. **refactoring** - Code refactoring
17. **analysis** - Code analysis

## Integration Workflow

### Step 1: Initialize
```bash
# Initialize claude-flow
npx claude-flow@alpha init

# Initialize hive-mind for swarm coordination
npx claude-flow@alpha hive-mind init
```

### Step 2: Configure
```bash
# Set up monitoring
npx claude-flow@alpha config set monitoring true

# Configure swarm parameters
npx claude-flow@alpha config set swarm.agents 5
```

### Step 3: Deploy
```bash
# Start orchestration with UI
npx claude-flow@alpha start --ui --swarm

# Deploy specific task
npx claude-flow@alpha swarm "integrate external API"
```

### Step 4: Monitor
```bash
# Check status
npx claude-flow@alpha status

# View metrics
npx claude-flow@alpha hive-mind metrics

# Check agent performance
npx claude-flow@alpha analysis performance
```

## Troubleshooting

### Common Issues

1. **"--dangerously-skip-permissions cannot be used with root"**
   - Solution: Run without sudo or use different user
   - Alternative: Use non-interactive mode without permissions flag

2. **"postgres MCP connection failed"**
   - Container is running but MCP wrapper needs adjustment
   - Check: `docker exec -it sutazai-postgres psql -U postgres`
   - Fix wrapper: `/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh`

3. **"Hive Mind not initialized"**
   - Run: `npx claude-flow@alpha hive-mind init`
   - Then: `npx claude-flow@alpha hive-mind wizard`

## Next Steps

1. **Fix postgres MCP wrapper** - Update connection parameters
2. **Initialize hive-mind** - Enable swarm coordination
3. **Deploy frontend** - Streamlit UI on port 10011
4. **Create AGENTS.md** - Document agent configurations
5. **Setup monitoring** - Enable token tracking and analytics

## Resources

- **Documentation**: https://github.com/ruvnet/claude-flow
- **Hive Mind Guide**: https://github.com/ruvnet/claude-flow/tree/main/docs/hive-mind
- **ruv-swarm**: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- **Discord**: https://discord.agentics.org

---
*Generated: 2025-08-26 23:58 UTC*
*MCP Integration Status: 93% operational*