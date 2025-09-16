# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Full Stack Deployment
```bash
# Complete system deployment (recommended first run)
cd /opt/sutazaiapp
./deploy.sh

# Start specific service groups
docker compose -f docker-compose-core.yml up -d      # Databases and infrastructure
docker compose -f docker-compose-backend.yml up -d   # Backend API
docker compose -f docker-compose-frontend.yml up -d  # Streamlit UI
docker compose -f docker-compose-vectors.yml up -d   # Vector databases
```

### Service Management
```bash
# Check all services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Restart specific service
docker compose restart sutazai-backend

# View logs
./scripts/monitoring/live_logs.sh live              # Live aggregated logs
docker logs sutazai-backend --tail 100 -f          # Specific service logs

# Stop everything
docker compose down
```

### Testing Commands
```bash
# Backend tests (Python/FastAPI)
cd backend
./venv/bin/pytest tests/ -v --cov=app

# Frontend tests (if configured)
cd frontend
./venv/bin/python -m pytest tests/

# MCP server tests
for wrapper in /opt/sutazaiapp/scripts/mcp/wrappers/*.sh; do
    basename "$wrapper" .sh
    "$wrapper" --selfcheck
done

# Integration tests
docker exec sutazai-backend pytest tests/integration/
```

### Database Access
```bash
# PostgreSQL (main database)
PGPASSWORD=sutazai_secure_2024 psql -h localhost -p 10000 -U jarvis -d jarvis_ai

# Redis (cache/sessions)
redis-cli -h localhost -p 10001

# Neo4j (graph database)
# Browser: http://localhost:10002
# Credentials: neo4j/sutazai_secure_2024

# RabbitMQ (message queue)
# Management UI: http://localhost:10005
# Credentials: sutazai/sutazai_secure_2024
```

## Architecture Overview

### System Design
**Type**: Hybrid Microservices with Event-Driven Multi-Agent Orchestration

```
User Interface (Streamlit:11000)
        ↓ HTTP/WebSocket
API Gateway (Kong:10008-10009)
        ↓ REST/gRPC
Backend Service (FastAPI:10200)
        ↓ Service Connections Manager
┌─────────────────────────────────┐
│  PostgreSQL (10000) - Relational │
│  Redis (10001) - Cache/PubSub    │
│  Neo4j (10002-10003) - Graph     │
│  RabbitMQ (10004-10005) - Queue  │
│  ChromaDB (10100) - Vectors      │
│  Qdrant (10101-10102) - Vectors  │
│  FAISS (10103) - Vectors         │
└─────────────────────────────────┘
        ↓ AMQP/HTTP
MCP Bridge (11100) - Agent Orchestration
        ↓ Routing
AI Agents (11401-11801)
```

### Network Configuration
- **Docker Network**: sutazai-network (172.20.0.0/16)
- **Backend Services**: 172.20.0.10-29
- **Vector Services**: 172.20.0.20-22
- **Frontend**: 172.20.0.30 (WARNING: Duplicate with backend)
- **Agents**: 172.20.0.100-199

### Key Files and Patterns

#### Backend Structure (FastAPI)
```
backend/
├── app/
│   ├── api/v1/          # API endpoints
│   │   ├── router.py    # Main router registration
│   │   └── endpoints/   # Individual endpoint modules
│   ├── core/            # Core configurations
│   │   ├── config.py    # Settings management
│   │   ├── database.py  # Database connections
│   │   └── security.py  # JWT authentication
│   ├── models/          # SQLAlchemy models
│   └── services/        # Business logic
│       └── connections.py # Service connections manager (SINGLETON)
└── tests/               # Test files
```

#### Adding New API Endpoint
```python
# 1. Create endpoint file: app/api/v1/endpoints/new_feature.py
from fastapi import APIRouter, Depends
from app.api.dependencies.auth import get_current_user

router = APIRouter()

@router.post("/create")
async def create_feature(
    data: FeatureCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Implementation
    return {"status": "created"}

# 2. Register in app/api/v1/router.py
from app.api.v1.endpoints import new_feature
api_router.include_router(new_feature.router, prefix="/features", tags=["features"])
```

## MCP Server Infrastructure

### Available MCP Servers (18 total)
| Server | Purpose | Command |
|--------|---------|---------|
| filesystem | File system access | `/scripts/mcp/wrappers/filesystem.sh` |
| memory | Session memory | `/scripts/mcp/wrappers/memory.sh` |
| extended-memory | Persistent memory | `/scripts/mcp/wrappers/extended-memory.sh` |
| memory-bank | Project memory | `/scripts/mcp/wrappers/memory-bank.sh` |
| github | GitHub API | `/scripts/mcp/wrappers/github.sh` |
| github-project-manager | Project management | `/scripts/mcp/wrappers/github-project-manager.sh` |
| claude-flow | Agent orchestration | `/scripts/mcp/wrappers/claude-flow.sh` |
| ruv-swarm | Swarm management | `/scripts/mcp/wrappers/ruv-swarm.sh` |
| context7 | Documentation lookup | `/scripts/mcp/wrappers/context7.sh` |
| playwright | Browser automation | `/scripts/mcp/wrappers/playwright.sh` |
| sequential-thinking | Multi-step reasoning | `/scripts/mcp/wrappers/sequential-thinking.sh` |
| code-index | Code search (uses uvx) | `/scripts/mcp/wrappers/code-index.sh` |
| ddg | Web search | `/scripts/mcp/wrappers/ddg.sh` |
| http-fetch | HTTP requests | `/scripts/mcp/wrappers/http-fetch.sh` |
| gitmcp-sutazai | Git operations | `/scripts/mcp/wrappers/gitmcp-sutazai.sh` |
| gitmcp-anthropic | Claude docs | `/scripts/mcp/wrappers/gitmcp-anthropic.sh` |
| gitmcp-docs | Generic docs | `/scripts/mcp/wrappers/gitmcp-docs.sh` |
| everything | Utility server | `/scripts/mcp/wrappers/everything.sh` |

### MCP Server Management
```bash
# Test individual server
/opt/sutazaiapp/scripts/mcp/wrappers/[server-name].sh --selfcheck

# Debug MCP server
export DEBUG=mcp:*
/opt/sutazaiapp/scripts/mcp/wrappers/[server-name].sh 2>&1 | tee debug.log

# Add new MCP server
# 1. Create wrapper in /scripts/mcp/wrappers/
# 2. Add to .mcp.json
# 3. Update .claude/settings.local.json enabledMcpjsonServers
```

## Code Conventions

### Python (Backend/Services)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Async**: Use async/await throughout FastAPI
- **Type Hints**: Always include (e.g., `Optional[Dict[str, Any]]`)
- **Error Handling**: Re-raise HTTPException, log others
- **Imports**: stdlib → third-party → local

### TypeScript (MCP Servers)
- **Naming**: camelCase for functions/variables, PascalCase for classes/types
- **Modules**: ES modules with .js extensions in imports
- **Strict Mode**: Enabled in tsconfig.json
- **Error Handling**: Typed errors with McpError

### Shell Scripts
- **Shebang**: `#!/bin/bash`
- **Error Handling**: `set -e` at start
- **Logging**: Use color-coded output functions
- **Self-check**: Support `--selfcheck` flag

## Common Operations

### Adding New Agent
```python
# 1. Create wrapper in agents/wrappers/new_agent_local.py
from base_agent_wrapper import BaseAgentWrapper

class NewAgentLocal(BaseAgentWrapper):
    def __init__(self):
        super().__init__()
        self.capabilities = ["new-capability"]

# 2. Register in AGENT_REGISTRY (mcp_bridge_server.py)
AGENT_REGISTRY["new_agent"] = {
    "name": "New Agent",
    "capabilities": ["new-capability"],
    "port": 11XXX,
    "status": "offline"
}
```

### Database Migrations
```bash
# Create migration
cd backend
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Health Checks
```bash
# System audit
./scripts/comprehensive_system_audit.sh

# Service health endpoints
curl http://localhost:10200/health          # Backend
curl http://localhost:11000/_stcore/health  # Frontend
curl http://localhost:11100/health          # MCP Bridge
curl http://localhost:10100/api/v1/heartbeat # ChromaDB
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker logs sutazai-[service] --tail 50

# Check port conflicts
netstat -tulpn | grep [port]

# Force recreate
docker compose up -d --force-recreate [service]
```

### MCP Server Issues
```bash
# Check wrapper exists
ls -la /opt/sutazaiapp/scripts/mcp/wrappers/[server].sh

# Test connectivity
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"1.0.0"},"id":1}' | \
    /opt/sutazaiapp/scripts/mcp/wrappers/[server].sh

# Kill stuck process
pkill -f "scripts/mcp/wrappers/[server]"
```

### Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec sutazai-backend pg_isready -h sutazai-postgres -p 5432

# Check Redis
docker exec sutazai-backend redis-cli -h sutazai-redis ping

# Verify credentials
grep PASSWORD /opt/sutazaiapp/.env
```

### Resource Issues
```bash
# Check disk space
df -h

# Check memory
free -h

# Docker cleanup
docker system prune -a --volumes

# Check container resources
docker stats --no-stream
```

## Critical Issues to Address

1. **Duplicate IP Assignment**: Frontend and Backend both use 172.20.0.30
2. **Resource Misallocation**: 
   - Ollama: Using 24MB of 23GB allocated
   - Neo4j: At 96% memory utilization
   - Vector DBs: Over-provisioned
3. **Missing Infrastructure**:
   - No CI/CD pipeline
   - No SSL/TLS termination
   - No centralized logging
4. **Security Gaps**:
   - Credentials in plain environment variables
   - No network policies
   - Missing mTLS between services

## Git Workflow

- **Current Branch**: v4
- **Never work on**: main/master
- **Feature branches**: Create from v4
- **Commit pattern**: Descriptive messages, no "fix" or "update"
- **Before committing**: Run linters and tests

## Testing Requirements

- **Coverage Target**: 80% for core, 95% for critical paths
- **Test Naming**: 
  - Python: `test_<function_name>`
  - TypeScript: `should <behavior>`
- **Test Locations**:
  - Backend: `/backend/tests/`
  - MCP: `/mcp-servers/*/tests/`
  - Integration: Docker exec into containers

## Performance Notes

- Backend uses async/await throughout
- Redis caching with TTL
- Connection pooling for all databases
- Resource limits enforced via Docker
- Health checks prevent cascading failures

## Development Tips

1. Always check existing patterns before implementing
2. Use the service connections manager (singleton) for backend
3. MCP servers auto-restart on failure
4. Logs are your friend - use live_logs.sh
5. Test MCP servers with --selfcheck first
6. Feature branches only, never commit to main
7. Resource usage is critical - monitor with docker stats