# Infrastructure Configuration (DEEP DIVE - 2025-08-21)

## Docker Architecture
- **Main Compose**: `/docker-compose.yml`
- **Additional Configs**: 7 active Docker files
- **Networks**: sutazai-network
- **Volumes**: Persistent data for all DBs

## Services Defined in docker-compose.yml
1. **PostgreSQL**: 16-alpine, 256MB limit
2. **Redis**: 7-alpine, 64MB limit
3. **Neo4j**: 5-community, 512MB limit
4. **RabbitMQ**: 3-management-alpine
5. **Kong Gateway**: API gateway
6. **Consul**: Service discovery
7. **ChromaDB**: Vector database
8. **Qdrant**: Vector search
9. **Ollama**: LLM service
10. **Backend**: FastAPI service
11. **Frontend**: Streamlit UI
12. **Monitoring**: Prometheus, Grafana, Loki

## MCP Infrastructure
- **Orchestrator**: DinD container for MCP servers
- **Manager**: MCP service management
- **Extended Memory**: SQLite-based persistence
- **Bridge Services**: 20 mesh integration files

## Container Health Status (Actual)
```bash
Total: 38 running
Healthy: 23 (60%)
No health checks: 15 (40%)
Unnamed containers: 10+ (poor hygiene)
```

## Actual MCP Servers Found
1. `/scripts/mcp/servers/memory/server.js`
2. `/scripts/mcp/servers/files/server.js`
3. Extended Memory container (working)

## Missing MCP Implementations
- context (folder exists, no server.js)
- docs (folder exists, no server.js)
- search (folder exists, no server.js)
- Most others claimed to exist

## Resource Limits
- PostgreSQL: 256MB max, 128MB reserved
- Redis: 64MB max, 32MB reserved
- Neo4j: 512MB max, 256MB reserved
- Backend: Defined in docker-compose
- Frontend: Defined in docker-compose

## Port Mappings
- 10000: PostgreSQL
- 10001: Redis
- 10002-10003: Neo4j
- 10005-10015: Kong Gateway
- 10007-10008: RabbitMQ
- 10010: Backend API
- 10011: Frontend UI
- 10100: ChromaDB
- 10101-10102: Qdrant
- 10104: Ollama
- 10200: Prometheus
- 10201: Grafana