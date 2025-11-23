# SutazAI Platform - API Documentation

**Version**: 1.0.0  
**Last Updated**: 2025-11-15 20:35:00 UTC  
**Base URL**: `http://localhost:10200`  
**Protocol**: HTTP/1.1, WebSocket

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health & Monitoring](#health--monitoring)
3. [Vector Operations](#vector-operations)
4. [Agent Management](#agent-management)
5. [MCP Bridge API](#mcp-bridge-api)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [WebSocket API](#websocket-api)

---

## Authentication

### Overview

The SutazAI Platform uses JWT (JSON Web Token) authentication with HS256 algorithm.

**Token Types**:
- **Access Token**: 30 minutes expiry, used for API requests
- **Refresh Token**: 7 days expiry, used to obtain new access tokens

### Register User

**Endpoint**: `POST /api/v1/auth/register`

**Request**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

**Response** (201 Created):
```json
{
  "user_id": "uuid-here",
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2025-11-15T20:35:00Z"
}
```

**Error Responses**:
- `400`: Invalid email format or weak password
- `409`: Email already registered

### Login

**Endpoint**: `POST /api/v1/auth/login`

**Request**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "Bearer",
  "expires_in": 1800
}
```

**Error Responses**:
- `401`: Invalid credentials
- `429`: Too many login attempts

### Refresh Token

**Endpoint**: `POST /api/v1/auth/refresh`

**Headers**:
```
Authorization: Bearer <refresh_token>
```

**Response** (200 OK):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "Bearer",
  "expires_in": 1800
}
```

**Error Responses**:
- `401`: Invalid or expired refresh token
- `403`: Token revoked

### Logout

**Endpoint**: `POST /api/v1/auth/logout`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Successfully logged out"
}
```

### Get Current User

**Endpoint**: `GET /api/v1/auth/me`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "user_id": "uuid-here",
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2025-11-15T20:35:00Z",
  "last_login": "2025-11-15T20:40:00Z"
}
```

### Password Reset

**Endpoint**: `POST /api/v1/auth/reset-password`

**Request**:
```json
{
  "email": "user@example.com"
}
```

**Response** (200 OK):
```json
{
  "message": "Password reset email sent"
}
```

### Email Verification

**Endpoint**: `POST /api/v1/auth/verify-email`

**Request**:
```json
{
  "token": "verification-token-here"
}
```

**Response** (200 OK):
```json
{
  "message": "Email verified successfully"
}
```

---

## Health & Monitoring

### Simple Health Check

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "app": "SutazAI Platform API",
  "timestamp": "2025-11-15T20:35:00Z"
}
```

**No Authentication Required**

### Detailed Health Check

**Endpoint**: `GET /api/v1/health/detailed`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-11-15T20:35:00Z",
  "services": {
    "postgresql": {
      "status": "connected",
      "response_time_ms": 5.2
    },
    "redis": {
      "status": "connected",
      "response_time_ms": 1.8
    },
    "neo4j": {
      "status": "connected",
      "response_time_ms": 12.4
    },
    "rabbitmq": {
      "status": "connected",
      "response_time_ms": 3.6
    },
    "consul": {
      "status": "connected",
      "response_time_ms": 8.1
    },
    "kong": {
      "status": "connected",
      "response_time_ms": 15.3
    },
    "chromadb": {
      "status": "connected",
      "response_time_ms": 45.7
    },
    "qdrant": {
      "status": "connected",
      "response_time_ms": 22.9
    },
    "faiss": {
      "status": "connected",
      "response_time_ms": 18.4
    }
  },
  "summary": {
    "total_services": 9,
    "connected": 9,
    "disconnected": 0,
    "health_percentage": 100.0
  }
}
```

### Prometheus Metrics

**Endpoint**: `GET /metrics`

**Response** (200 OK, text/plain):
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status="200"} 1523

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1",endpoint="/health"} 1450
http_request_duration_seconds_bucket{le="0.5",endpoint="/health"} 1520
http_request_duration_seconds_sum{endpoint="/health"} 45.2
http_request_duration_seconds_count{endpoint="/health"} 1523

# HELP db_connections_active Active database connections
# TYPE db_connections_active gauge
db_connections_active{database="postgresql"} 8
db_connections_active{database="redis"} 12
db_connections_active{database="neo4j"} 3

# ... more metrics ...
```

**No Authentication Required**

---

## Vector Operations

### Create ChromaDB Collection

**Endpoint**: `POST /api/v1/vectors/chromadb/collections`

**Headers**:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request**:
```json
{
  "name": "documents",
  "metadata": {
    "description": "Document embeddings",
    "dimension": 768
  }
}
```

**Response** (201 Created):
```json
{
  "collection_id": "uuid-here",
  "name": "documents",
  "created_at": "2025-11-15T20:35:00Z",
  "metadata": {
    "description": "Document embeddings",
    "dimension": 768
  }
}
```

### Add Vectors to ChromaDB

**Endpoint**: `POST /api/v1/vectors/chromadb/add`

**Request**:
```json
{
  "collection_name": "documents",
  "embeddings": [[0.1, 0.2, ...]], // 768-dim vectors
  "documents": ["Document text here"],
  "metadatas": [{"source": "file.pdf", "page": 1}],
  "ids": ["doc1"]
}
```

**Response** (200 OK):
```json
{
  "added": 1,
  "collection": "documents",
  "ids": ["doc1"]
}
```

### Search ChromaDB

**Endpoint**: `POST /api/v1/vectors/chromadb/search`

**Request**:
```json
{
  "collection_name": "documents",
  "query_embeddings": [[0.1, 0.2, ...]], // 768-dim
  "n_results": 10,
  "where": {"source": "file.pdf"}
}
```

**Response** (200 OK):
```json
{
  "ids": [["doc1", "doc5", "doc3"]],
  "distances": [[0.12, 0.34, 0.45]],
  "documents": [["Document text 1", "Document text 5", ...]],
  "metadatas": [[{"source": "file.pdf", "page": 1}, ...]]
}
```

### Create Qdrant Collection

**Endpoint**: `POST /api/v1/vectors/qdrant/collections`

**Request**:
```json
{
  "collection_name": "embeddings",
  "vector_size": 768,
  "distance": "Cosine"
}
```

**Response** (200 OK):
```json
{
  "result": true,
  "status": "ok",
  "time": 0.0023
}
```

### Upsert Vectors to Qdrant

**Endpoint**: `POST /api/v1/vectors/qdrant/points`

**Request**:
```json
{
  "collection_name": "embeddings",
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, ...],
      "payload": {"text": "Document text", "category": "research"}
    }
  ]
}
```

**Response** (200 OK):
```json
{
  "result": {
    "operation_id": 123,
    "status": "completed"
  },
  "status": "ok",
  "time": 0.0045
}
```

### Search Qdrant

**Endpoint**: `POST /api/v1/vectors/qdrant/search`

**Request**:
```json
{
  "collection_name": "embeddings",
  "vector": [0.1, 0.2, ...],
  "limit": 10,
  "filter": {
    "must": [
      {"key": "category", "match": {"value": "research"}}
    ]
  }
}
```

**Response** (200 OK):
```json
{
  "result": [
    {
      "id": 1,
      "version": 0,
      "score": 0.95,
      "payload": {"text": "Document text", "category": "research"},
      "vector": null
    }
  ],
  "status": "ok",
  "time": 0.0028
}
```

### Create FAISS Index

**Endpoint**: `POST /api/v1/vectors/faiss/index`

**Request**:
```json
{
  "index_name": "documents",
  "dimension": 768,
  "index_type": "FlatL2"
}
```

**Response** (201 Created):
```json
{
  "index_name": "documents",
  "dimension": 768,
  "index_type": "FlatL2",
  "total_vectors": 0
}
```

### Add Vectors to FAISS

**Endpoint**: `POST /api/v1/vectors/faiss/add`

**Request**:
```json
{
  "index_name": "documents",
  "vectors": [[0.1, 0.2, ...]], // Shape: (N, 768)
  "ids": [1, 2, 3]
}
```

**Response** (200 OK):
```json
{
  "added": 3,
  "index_name": "documents",
  "total_vectors": 3
}
```

### Search FAISS

**Endpoint**: `POST /api/v1/vectors/faiss/search`

**Request**:
```json
{
  "index_name": "documents",
  "query_vectors": [[0.1, 0.2, ...]], // Shape: (1, 768)
  "k": 10
}
```

**Response** (200 OK):
```json
{
  "distances": [[0.12, 0.34, 0.45, ...]],
  "indices": [[1, 5, 3, ...]],
  "k": 10
}
```

### Multi-Database Search

**Endpoint**: `POST /api/v1/vectors/search`

**Request**:
```json
{
  "query": "Find documents about machine learning",
  "databases": ["chromadb", "qdrant", "faiss"],
  "limit": 10,
  "collection_mappings": {
    "chromadb": "documents",
    "qdrant": "embeddings",
    "faiss": "documents"
  }
}
```

**Response** (200 OK):
```json
{
  "results": {
    "chromadb": [...],
    "qdrant": [...],
    "faiss": [...]
  },
  "aggregated": [...], // Merged and ranked results
  "total_results": 30,
  "query_time_ms": 145.3
}
```

---

## Agent Management

### List All Agents

**Endpoint**: `GET /api/v1/agents`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "agents": [
    {
      "id": "letta",
      "name": "Letta",
      "status": "healthy",
      "capabilities": ["memory", "task-automation"],
      "endpoint": "http://sutazai-letta:8000"
    },
    {
      "id": "crewai",
      "name": "CrewAI",
      "status": "healthy",
      "capabilities": ["multi-agent", "orchestration"],
      "endpoint": "http://sutazai-crewai:8000"
    },
    {
      "id": "aider",
      "name": "Aider",
      "status": "healthy",
      "capabilities": ["code-editing", "pair-programming"],
      "endpoint": "http://sutazai-aider:8000"
    }
    // ... more agents
  ],
  "total": 8,
  "healthy": 8,
  "unhealthy": 0
}
```

### Get Agent Status

**Endpoint**: `GET /api/v1/agents/{agent_id}/status`

**Parameters**:
- `agent_id`: Agent identifier (e.g., "letta", "crewai")

**Response** (200 OK):
```json
{
  "agent_id": "letta",
  "status": "healthy",
  "last_health_check": "2025-11-15T20:35:00Z",
  "uptime_seconds": 75600,
  "metrics": {
    "requests_total": 1523,
    "requests_per_minute": 12.5,
    "average_response_time_ms": 245.7,
    "error_rate_percentage": 0.3
  },
  "capabilities": ["memory", "task-automation"],
  "llm_backend": {
    "type": "ollama",
    "model": "tinyllama",
    "endpoint": "http://host.docker.internal:11435"
  }
}
```

### Execute Agent Task

**Endpoint**: `POST /api/v1/agents/execute`

**Request**:
```json
{
  "agent_id": "letta",
  "task": {
    "type": "generate_text",
    "prompt": "Explain quantum computing in simple terms",
    "parameters": {
      "max_tokens": 500,
      "temperature": 0.7
    }
  }
}
```

**Response** (200 OK):
```json
{
  "task_id": "uuid-here",
  "agent_id": "letta",
  "status": "completed",
  "result": {
    "text": "Quantum computing is...",
    "tokens_used": 342,
    "execution_time_ms": 1250
  },
  "created_at": "2025-11-15T20:35:00Z",
  "completed_at": "2025-11-15T20:35:01Z"
}
```

**Async Execution**:
For long-running tasks, use `async: true`:

```json
{
  "agent_id": "letta",
  "task": {...},
  "async": true
}
```

**Response** (202 Accepted):
```json
{
  "task_id": "uuid-here",
  "status": "queued",
  "status_url": "/api/v1/agents/tasks/uuid-here"
}
```

### Get Task Status

**Endpoint**: `GET /api/v1/agents/tasks/{task_id}`

**Response** (200 OK):
```json
{
  "task_id": "uuid-here",
  "status": "running",
  "progress_percentage": 45,
  "estimated_completion_seconds": 30,
  "created_at": "2025-11-15T20:35:00Z"
}
```

**Status Values**:
- `queued`: Task in queue
- `running`: Task executing
- `completed`: Task finished successfully
- `failed`: Task failed with error
- `cancelled`: Task cancelled by user

---

## MCP Bridge API

**Base URL**: `http://localhost:11100`

### Health Check

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "services_registered": 16,
  "agents_registered": 12,
  "websocket_connections": 5,
  "uptime_seconds": 75600
}
```

**Performance**: ~20ms response time

### Service Status

**Endpoint**: `GET /status`

**Response** (200 OK):
```json
{
  "status": "operational",
  "version": "1.0.0",
  "services": {
    "total": 16,
    "healthy": 15,
    "degraded": 1,
    "down": 0
  },
  "agents": {
    "total": 12,
    "active": 12,
    "idle": 0,
    "error": 0
  }
}
```

### List Services

**Endpoint**: `GET /services`

**Response** (200 OK):
```json
{
  "services": [
    {
      "name": "postgres",
      "type": "database",
      "endpoint": "postgresql://sutazai-postgres:5432",
      "status": "healthy"
    },
    {
      "name": "redis",
      "type": "cache",
      "endpoint": "redis://sutazai-redis:6379",
      "status": "healthy"
    }
    // ... more services
  ],
  "count": 16
}
```

### List Agents

**Endpoint**: `GET /agents`

**Response** (200 OK):
```json
{
  "agents": [
    {
      "id": "letta",
      "capabilities": ["memory", "conversation", "task-automation"],
      "status": "active",
      "load": 0.25
    },
    {
      "id": "crewai",
      "capabilities": ["multi-agent", "orchestration", "autonomous"],
      "status": "active",
      "load": 0.15
    }
    // ... more agents
  ],
  "count": 12
}
```

### Route Message

**Endpoint**: `POST /route`

**Request**:
```json
{
  "message": "Generate Python code for quicksort",
  "context": {
    "user_id": "uuid-here",
    "session_id": "session-123"
  },
  "preferences": {
    "agent_type": "code-generation",
    "timeout_seconds": 30
  }
}
```

**Response** (200 OK):
```json
{
  "route_id": "uuid-here",
  "selected_agent": "aider",
  "reason": "Best match for code-generation capability",
  "confidence": 0.95,
  "estimated_time_seconds": 5,
  "message_forwarded": true
}
```

### Submit Task

**Endpoint**: `POST /tasks/submit`

**Request**:
```json
{
  "task_type": "code_generation",
  "description": "Create a REST API with FastAPI",
  "requirements": {
    "language": "python",
    "framework": "fastapi",
    "features": ["authentication", "database"]
  },
  "priority": "high"
}
```

**Response** (202 Accepted):
```json
{
  "task_id": "uuid-here",
  "status": "queued",
  "assigned_agents": ["aider", "gpt-engineer"],
  "orchestrator": "crewai",
  "estimated_completion_minutes": 15,
  "status_url": "/tasks/uuid-here/status"
}
```

### Get Metrics (Prometheus Format)

**Endpoint**: `GET /metrics`

**Response** (200 OK, text/plain):
```
# HELP mcp_requests_total Total requests to MCP Bridge
# TYPE mcp_requests_total counter
mcp_requests_total{endpoint="/route",status="200"} 1234

# HELP mcp_agent_tasks_active Active tasks per agent
# TYPE mcp_agent_tasks_active gauge
mcp_agent_tasks_active{agent="letta"} 3
mcp_agent_tasks_active{agent="crewai"} 5

# ... more metrics
```

### Get Metrics (JSON Format)

**Endpoint**: `GET /metrics/json`

**Response** (200 OK):
```json
{
  "requests": {
    "total": 1234,
    "per_minute": 12.5,
    "by_endpoint": {
      "/route": 856,
      "/tasks/submit": 234,
      "/health": 144
    }
  },
  "agents": {
    "active_tasks": {
      "letta": 3,
      "crewai": 5,
      "aider": 2
    },
    "completed_tasks": {
      "letta": 456,
      "crewai": 789
    }
  },
  "performance": {
    "average_response_time_ms": 25.7,
    "p95_response_time_ms": 45.2,
    "p99_response_time_ms": 67.8
  }
}
```

---

## WebSocket API

### Connect to Backend WebSocket

**Endpoint**: `WS ws://localhost:10200/ws`

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:10200/ws');

ws.onopen = () => {
  console.log('Connected to backend');
  
  // Send authentication
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

**Message Types**:

1. **Authentication**:
```json
{
  "type": "auth",
  "token": "jwt-token-here"
}
```

2. **Agent Status Update**:
```json
{
  "type": "agent_status",
  "agent_id": "letta",
  "status": "busy",
  "task_id": "uuid-here"
}
```

3. **Task Progress**:
```json
{
  "type": "task_progress",
  "task_id": "uuid-here",
  "progress": 65,
  "message": "Generating code..."
}
```

4. **Chat Message**:
```json
{
  "type": "chat",
  "message": "Hello from AI",
  "sender": "letta",
  "timestamp": "2025-11-15T20:35:00Z"
}
```

### Connect to MCP Bridge WebSocket

**Endpoint**: `WS ws://localhost:11100/ws/{client_id}`

**Connection**:
```javascript
const clientId = 'client-' + Math.random().toString(36).substr(2, 9);
const ws = new WebSocket(`ws://localhost:11100/ws/${clientId}`);

ws.onopen = () => {
  console.log('Connected to MCP Bridge');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'broadcast') {
    console.log('Broadcast:', data.message);
  } else if (data.type === 'direct') {
    console.log('Direct message:', data.message);
  }
};
```

**Message Types**:

1. **Broadcast** (sent to all connected clients):
```json
{
  "type": "broadcast",
  "message": "System maintenance in 5 minutes",
  "sender": "system",
  "timestamp": "2025-11-15T20:35:00Z"
}
```

2. **Direct Message** (sent to specific client):
```json
{
  "type": "direct",
  "to": "client-abc123",
  "message": "Your task is complete",
  "data": {...}
}
```

**Performance**:
- Average latency: 0.035ms
- Max concurrent connections: 100+
- Auto-reconnect on disconnect

---

## Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "timestamp": "2025-11-15T20:35:00Z",
    "request_id": "uuid-here"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `202 Accepted`: Request accepted for async processing
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Codes

- `AUTH_FAILED`: Authentication failed
- `TOKEN_EXPIRED`: JWT token expired
- `VALIDATION_ERROR`: Request validation error
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `SERVICE_UNAVAILABLE`: Backend service unavailable
- `DATABASE_ERROR`: Database operation failed
- `AGENT_UNAVAILABLE`: Requested agent unavailable

---

## Rate Limiting

### Limits

**Per API Key**:
- `100 requests/minute` for authenticated endpoints
- `20 requests/minute` for authentication endpoints
- `500 requests/minute` for health checks

**Per IP (unauthenticated)**:
- `10 requests/minute` for public endpoints

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1700000000
```

### Rate Limit Exceeded Response

**Status**: `429 Too Many Requests`

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds.",
    "retry_after": 45,
    "limit": 100,
    "period": "minute"
  }
}
```

---

## Appendix

### Authentication Example (Python)

```python
import requests

BASE_URL = "http://localhost:10200"

# Login
response = requests.post(
    f"{BASE_URL}/api/v1/auth/login",
    json={"email": "user@example.com", "password": "SecurePassword123!"}
)
tokens = response.json()
access_token = tokens["access_token"]

# Make authenticated request
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{BASE_URL}/api/v1/health/detailed", headers=headers)
print(response.json())
```

### WebSocket Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:10200/ws');

ws.addEventListener('open', () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
});

ws.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
});
```

### Vector Search Example (Python)

```python
import numpy as np

# Generate embedding (example)
query_embedding = np.random.rand(768).tolist()

# Search across all vector databases
response = requests.post(
    f"{BASE_URL}/api/v1/vectors/search",
    headers=headers,
    json={
        "query": "machine learning",
        "databases": ["chromadb", "qdrant", "faiss"],
        "limit": 10
    }
)

results = response.json()
print(f"Found {results['total_results']} results in {results['query_time_ms']}ms")
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-11-15 20:35:00 UTC  
**For support**: <support@sutazai.com>

