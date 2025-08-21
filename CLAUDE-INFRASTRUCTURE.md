# Infrastructure Configuration (VERIFIED 2025-08-21)

## Container Status (ACTUAL)
- **Total Running**: 38 containers (not 42 or 49)
- **Healthy**: 23 containers
- **Without Health Checks**: 15 containers
- **Unnamed Containers**: Multiple (youthful_vaughan, crazy_northcutt, etc.)

## Key Services
- **MCP Orchestrator**: Up 17 hours (healthy)
- **MCP Manager**: Up 17 hours (healthy)
- **Extended Memory**: Up 15 hours (healthy)
- **ChromaDB**: Up 15 hours (healthy)
- **Neo4j**: Up 8 minutes (healthy)
- **Backend**: Up 11 hours (healthy)

## MCP Servers (ACTUAL)
- **Found server.js files**: 3 total
  - /scripts/mcp/servers/memory/server.js
  - /scripts/mcp/servers/files/server.js
  - /node_modules/@playwright/mcp/lib/mcp/server.js
- **MCP in DinD**: No containers found inside orchestrator
- **Extended Memory**: Only confirmed working MCP server

## Ports in Use
- 10000-10003: Databases
- 10010: Backend API
- 10011: Frontend
- 10100: ChromaDB
- 12375-12376: MCP Orchestrator
- 3009: Extended Memory

## Issues Found
- 15 containers without health checks
- Many unnamed containers (poor hygiene)
- MCP servers mostly missing implementations