# API Endpoints Index
*Last Updated: 2025-08-19*

## Backend API (Port 10010)

### Health & Status
- `GET /health` - Health check endpoint
- `GET /api/v1/status` - System status

### Authentication
- `POST /api/v1/auth/login` - User login (JWT)
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/logout` - User logout

### Security
- `POST /api/v1/security/encrypt` - Encrypt data
- `POST /api/v1/security/decrypt` - Decrypt data
- `GET /api/v1/security/report` - Security report

### Mesh Services
- `GET /api/v1/mesh/status` - Service mesh status
- `POST /api/v1/mesh/register` - Register service
- `DELETE /api/v1/mesh/deregister/{id}` - Deregister service
- `GET /api/v1/mesh/discover/{service}` - Discover services

### Agent Management
- `GET /api/v1/agents` - List all agents
- `POST /api/v1/agents/spawn` - Spawn new agent
- `GET /api/v1/agents/{id}/status` - Agent status
- `DELETE /api/v1/agents/{id}` - Remove agent

### Task Management
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks/{id}` - Get task status
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Cancel task

### Documentation
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc UI
- `GET /openapi.json` - OpenAPI specification

## MCP Server Endpoints (STDIO Protocol)

### Tools
- `get_status` - Get server status
- `execute_task` - Execute a task

### Resources
- `sutazai://config` - Server configuration
- `sutazai://logs` - Server logs

## Monitoring Endpoints

### Prometheus (Port 10200)
- `GET /metrics` - Prometheus metrics
- `GET /-/healthy` - Health check
- `GET /-/ready` - Readiness check

### Grafana (Port 10201)
- `GET /api/health` - Health check
- `GET /api/dashboards` - List dashboards
- `GET /api/datasources` - List data sources

### Consul (Port 10006)
- `GET /v1/health/node/{node}` - Node health
- `GET /v1/catalog/services` - List services
- `GET /v1/agent/services` - Agent services

## Database Endpoints

### PostgreSQL (Port 10000)
- Connection: `postgresql://sutazai:password@localhost:10000/sutazai_db`

### Redis (Port 10001)
- Connection: `redis://localhost:10001/0`

### Neo4j (Port 10002/10003)
- Bolt: `bolt://localhost:10002`
- HTTP: `http://localhost:10003`

### ChromaDB (Port 10100)
- API: `http://localhost:10100/api/v1`

### Qdrant (Port 10101/10102)
- HTTP: `http://localhost:10101`
- gRPC: `localhost:10102`