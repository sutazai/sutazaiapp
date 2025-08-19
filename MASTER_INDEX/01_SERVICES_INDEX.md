# Services Index - Complete Catalog
*Last Updated: 2025-08-19*

## Core Services

### Backend API
- **Port**: 10010
- **Container**: sutazai-backend
- **Status**: ✅ HEALTHY
- **Endpoints**: `/health`, `/api/v1/*`, `/docs`
- **Technology**: FastAPI, Python 3.11
- **Authentication**: JWT with refresh tokens

### Frontend UI
- **Port**: 10011  
- **Container**: sutazai-frontend
- **Status**: ✅ HEALTHY
- **Technology**: Streamlit, TornadoServer 6.5.2
- **Access**: http://localhost:10011

### Task Assignment Coordinator
- **Port**: 8551 (external) → 4000 (internal)
- **Container**: sutazai-task-assignment-coordinator
- **Status**: ✅ HEALTHY (after port fix)
- **Technology**: sutazai-mcp-unified-dev

## Database Services

### PostgreSQL
- **Port**: 10000
- **Container**: sutazai-postgres
- **Status**: ✅ RUNNING
- **Credentials**: sutazai/[ENV_PASSWORD]
- **Database**: sutazai_db

### Redis
- **Port**: 10001
- **Container**: sutazai-redis
- **Status**: ✅ RUNNING
- **Purpose**: Caching, session storage

### Neo4j
- **Ports**: 10002 (bolt), 10003 (http)
- **Container**: sutazai-neo4j
- **Status**: ✅ RUNNING
- **Purpose**: Graph database

### ChromaDB
- **Port**: 10100
- **Container**: sutazai-chromadb
- **Status**: ✅ RUNNING
- **Purpose**: Vector database

### Qdrant
- **Ports**: 10101 (http), 10102 (grpc)
- **Container**: sutazai-qdrant
- **Status**: ✅ RUNNING
- **Purpose**: Vector search

## AI/ML Services

### Ollama
- **Port**: 10104
- **Container**: sutazai-ollama
- **Status**: ✅ RUNNING
- **Model**: tinyllama (loaded)

### FAISS
- **Port**: 10103
- **Container**: sutazai-faiss
- **Status**: ✅ RUNNING
- **Purpose**: Vector similarity search

## Monitoring Services

### Prometheus
- **Port**: 10200
- **Container**: sutazai-prometheus
- **Status**: ✅ RUNNING
- **Retention**: 7 days

### Grafana
- **Port**: 10201
- **Container**: sutazai-grafana
- **Status**: ✅ RUNNING
- **Access**: admin/admin

### Consul
- **Port**: 10006
- **Container**: sutazai-consul
- **Status**: ✅ RUNNING
- **Purpose**: Service discovery

### Node Exporter
- **Port**: 10205
- **Container**: sutazai-node-exporter
- **Status**: ✅ RUNNING

### Redis Exporter
- **Port**: 10208
- **Container**: sutazai-redis-exporter
- **Status**: 🔄 RUNNING

## Infrastructure Services

### Kong Gateway
- **Port**: 10005
- **Container**: sutazai-kong
- **Status**: ❌ FAILED (404 responses)
- **Issue**: Configuration problem

### Portainer
- **Port**: 10314
- **Container**: sutazai-portainer
- **Status**: ✅ RUNNING (undocumented)

### MCP Orchestrator
- **Ports**: 
  - 12375: Docker API
  - 12376: Docker TLS
  - 18080: Management UI
  - 18081: Manager
  - 19090: Monitoring
- **Container**: mcp-orchestrator
- **Status**: ✅ RUNNING

## MCP Services (Real Implementation)

### MCP Server
- **Type**: STDIO Protocol
- **Location**: /opt/sutazaiapp/docker/mcp-services/real-mcp-server
- **Technology**: Node.js 20, @modelcontextprotocol/sdk
- **Features**: Tools, Resources, STDIO communication

## Agent Services

### Total Agents: 54
- See 05_AGENTS_CATALOG.md for complete list

## Network Configuration

### Primary Network
- **Name**: sutazai-network
- **Type**: Bridge
- **Subnet**: 172.18.0.0/16

### MCP Network
- **Name**: mcp-bridge
- **Type**: Bridge
- **Purpose**: MCP service isolation