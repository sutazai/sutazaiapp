# Claude Code Configuration - VERIFIED STATE (2025-08-21)

## ⚠️ CRITICAL: Previous Documentation Was Inaccurate
This file has been updated with ONLY verified facts. No assumptions.

## 📚 Area-Specific Documentation
- **[CLAUDE-RULES.md](CLAUDE-RULES.md)** - Critical rules and anti-hallucination
- **[CLAUDE-FRONTEND.md](CLAUDE-FRONTEND.md)** - Frontend status
- **[CLAUDE-BACKEND.md](CLAUDE-BACKEND.md)** - Backend and databases
- **[CLAUDE-INFRASTRUCTURE.md](CLAUDE-INFRASTRUCTURE.md)** - Containers and services
- **[CLAUDE-WORKFLOW.md](CLAUDE-WORKFLOW.md)** - Development workflow

## Actual System Status (2025-08-21 10:30 UTC)

### Verified Facts Only
- **Containers**: 38 running (not 42 or 49)
- **Healthy Containers**: 23 (60% have health checks)
- **Technical Debt**: 7,189 markers across codebase
- **Backend**: http://localhost:10010 - Returns healthy
- **Frontend**: http://localhost:10011 - Returns HTML
- **Redis**: Port 10001 - WORKING (PONG response)
- **MCP Servers**: Only 3 server.js files found

### Unknown/Untested
- PostgreSQL connection method
- Neo4j authentication
- ChromaDB v2 API endpoints
- Qdrant functionality
- Most MCP server implementations
- Actual Playwright test status

### Known Issues
- 15 containers without health checks
- Many unnamed containers (poor hygiene)
- MCP servers mostly missing implementations
- No containers found inside MCP orchestrator

## Most Critical Rules
1. **VERIFY EVERYTHING** - Test before claiming
2. **NO ASSUMPTIONS** - Check actual state
3. **SHOW EVIDENCE** - Include command outputs
4. **ADMIT UNKNOWNS** - Say "not tested" when uncertain

## Quick Verification Commands
```bash
docker ps | wc -l                    # Count containers: 38
docker ps | grep healthy | wc -l     # Count healthy: 23
curl http://localhost:10010/health   # Backend health
curl http://localhost:10011          # Frontend HTML
redis-cli -p 10001 ping              # Redis: PONG
```

---
*Previous claims of 100% operational, 19 MCP servers, and 22 TODOs were FALSE.*
*This documentation based on actual testing at 2025-08-21 10:30 UTC.*