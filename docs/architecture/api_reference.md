# API Reference Documentation

**Last Updated**: 2025-01-03  
**Version**: 1.0.0  
**Maintainer**: API Team

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
4. [User Management](#user-management)
5. [Agent Operations](#agent-operations)
6. [Vector Operations](#vector-operations)
7. [WebSocket API](#websocket-api)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [API Versioning](#api-versioning)

## API Overview

### Base URLs

```yaml
Production: https://api.sutazai.com
Staging: https://staging-api.sutazai.com
Development: http://localhost:10200
WebSocket: ws://localhost:10200/ws
```

### Request Format

```http
Content-Type: application/json
Accept: application/json
X-Request-ID: {uuid}
Authorization: Bearer {jwt_token}
```

### Response Format

```json
{
  "success": true,
  "data": {...},
  "message": "Operation successful",
  "timestamp": "2025-01-03T10:30:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Authentication

### POST /api/v1/auth/login

**Login with email and password**

Request:
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "email": "user@example.com",
      "username": "johndoe",
      "is_active": true,
      "is_superuser": false
    }
  }
}
```

### POST /api/v1/auth/refresh

**Refresh access token**

Request:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

Response:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 3600
  }
}
```

### POST /api/v1/auth/logout

**Logout and invalidate tokens**

Headers:
```http
Authorization: Bearer {access_token}
```

Response:
```json
{
  "success": true,
  "message": "Successfully logged out"
}
```

### POST /api/v1/auth/register

**Register new user**

Request:
```json
{
  "email": "newuser@example.com",
  "username": "newuser",
  "password": "secure_password",
  "confirm_password": "secure_password"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "650e8400-e29b-41d4-a716-446655440001",
    "email": "newuser@example.com",
    "username": "newuser",
    "created_at": "2025-01-03T10:30:00Z"
  }
}
```

## Core Endpoints

### GET /api/v1/health

**Health check endpoint**

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-03T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "neo4j": "healthy",
    "rabbitmq": "healthy",
    "vector_dbs": {
      "chromadb": "healthy",
      "qdrant": "healthy",
      "faiss": "healthy"
    }
  }
}
```

### GET /api/v1/info

**System information**

Response:
```json
{
  "name": "SutazAI API",
  "version": "1.0.0",
  "environment": "production",
  "features": [
    "authentication",
    "agent_orchestration",
    "vector_search",
    "websocket"
  ],
  "limits": {
    "max_request_size": "10MB",
    "rate_limit": "1000 requests/hour",
    "websocket_connections": 100
  }
}
```

## User Management

### GET /api/v1/users/me

**Get current user profile**

Headers:
```http
Authorization: Bearer {access_token}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "username": "johndoe",
    "full_name": "John Doe",
    "is_active": true,
    "is_superuser": false,
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-03T10:30:00Z",
    "preferences": {
      "theme": "dark",
      "language": "en",
      "notifications": true
    }
  }
}
```

### PUT /api/v1/users/me

**Update current user profile**

Request:
```json
{
  "full_name": "John Smith",
  "preferences": {
    "theme": "light",
    "language": "en",
    "notifications": false
  }
}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "full_name": "John Smith",
    "updated_at": "2025-01-03T10:35:00Z"
  }
}
```

### POST /api/v1/users/change-password

**Change user password**

Request:
```json
{
  "current_password": "old_password",
  "new_password": "new_secure_password",
  "confirm_password": "new_secure_password"
}
```

Response:
```json
{
  "success": true,
  "message": "Password successfully changed"
}
```

### GET /api/v1/users

**List all users (admin only)**

Query Parameters:
```
?page=1&limit=20&search=john&is_active=true
```

Response:
```json
{
  "success": true,
  "data": {
    "users": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "user@example.com",
        "username": "johndoe",
        "is_active": true,
        "created_at": "2025-01-01T00:00:00Z"
      }
    ],
    "total": 150,
    "page": 1,
    "limit": 20,
    "pages": 8
  }
}
```

## Agent Operations

### GET /api/v1/agents

**List available agents**

Response:
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "id": "filesystem",
        "name": "File System Agent",
        "status": "online",
        "capabilities": ["read", "write", "list"],
        "port": 11401,
        "health": {
          "cpu_usage": 15.5,
          "memory_usage": 256,
          "active_tasks": 2
        }
      },
      {
        "id": "memory",
        "name": "Memory Agent",
        "status": "online",
        "capabilities": ["store", "retrieve", "search"],
        "port": 11402
      }
    ],
    "total": 18,
    "online": 16,
    "offline": 2
  }
}
```

### POST /api/v1/agents/execute

**Execute agent task**

Request:
```json
{
  "agent_id": "filesystem",
  "task": "list_files",
  "parameters": {
    "path": "/opt/sutazaiapp",
    "pattern": "*.py",
    "recursive": true
  },
  "timeout": 30000,
  "priority": "normal"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "task_id": "task_750e8400_e29b_41d4",
    "status": "completed",
    "result": {
      "files": [
        "/opt/sutazaiapp/backend/app/main.py",
        "/opt/sutazaiapp/backend/app/api/router.py"
      ],
      "count": 45
    },
    "execution_time": 1250,
    "agent_id": "filesystem"
  }
}
```

### GET /api/v1/agents/{agent_id}/status

**Get agent status**

Response:
```json
{
  "success": true,
  "data": {
    "id": "filesystem",
    "status": "online",
    "uptime": 86400,
    "metrics": {
      "total_tasks": 1250,
      "successful_tasks": 1200,
      "failed_tasks": 50,
      "average_execution_time": 850,
      "current_load": 0.35
    },
    "last_health_check": "2025-01-03T10:30:00Z"
  }
}
```

### POST /api/v1/agents/orchestrate

**Orchestrate multi-agent task**

Request:
```json
{
  "workflow": "code_analysis",
  "steps": [
    {
      "agent": "filesystem",
      "action": "list_files",
      "parameters": {"pattern": "*.py"}
    },
    {
      "agent": "code-index",
      "action": "analyze",
      "parameters": {"files": "${step1.result}"}
    },
    {
      "agent": "memory",
      "action": "store",
      "parameters": {"key": "analysis", "value": "${step2.result}"}
    }
  ],
  "parallel": false
}
```

Response:
```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_850e8400_e29b_41d4",
    "status": "running",
    "steps_completed": 1,
    "steps_total": 3,
    "current_step": "code-index.analyze"
  }
}
```

## Vector Operations

### POST /api/v1/vectors/embed

**Create embeddings**

Request:
```json
{
  "texts": [
    "This is the first document",
    "This is the second document"
  ],
  "model": "text-embedding-ada-002",
  "collection": "documents"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "embeddings": [
      {
        "id": "emb_001",
        "vector": [0.023, -0.045, 0.128, ...],
        "dimension": 1536
      },
      {
        "id": "emb_002",
        "vector": [0.012, -0.098, 0.234, ...],
        "dimension": 1536
      }
    ],
    "collection": "documents",
    "model": "text-embedding-ada-002"
  }
}
```

### POST /api/v1/vectors/search

**Semantic search**

Request:
```json
{
  "query": "How to configure authentication?",
  "collection": "documents",
  "top_k": 5,
  "threshold": 0.75,
  "filters": {
    "category": "authentication",
    "date_range": {
      "start": "2025-01-01",
      "end": "2025-01-31"
    }
  }
}
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "doc_001",
        "score": 0.92,
        "text": "Authentication configuration guide...",
        "metadata": {
          "category": "authentication",
          "created_at": "2025-01-02T10:00:00Z"
        }
      }
    ],
    "query_embedding_time": 125,
    "search_time": 45,
    "total_results": 5
  }
}
```

### POST /api/v1/vectors/upsert

**Upsert vectors**

Request:
```json
{
  "collection": "documents",
  "vectors": [
    {
      "id": "doc_001",
      "vector": [0.023, -0.045, 0.128, ...],
      "metadata": {
        "title": "Authentication Guide",
        "category": "security"
      }
    }
  ]
}
```

Response:
```json
{
  "success": true,
  "data": {
    "upserted": 1,
    "collection": "documents"
  }
}
```

### DELETE /api/v1/vectors/{collection}/{vector_id}

**Delete vector**

Response:
```json
{
  "success": true,
  "message": "Vector deleted successfully"
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:10200/ws');

ws.onopen = () => {
  // Send authentication
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer eyJhbGciOiJIUzI1NiIs...'
  }));
};
```

### Message Types

#### Subscribe to Events

Request:
```json
{
  "type": "subscribe",
  "channels": ["agent_updates", "task_status", "notifications"]
}
```

Response:
```json
{
  "type": "subscribed",
  "channels": ["agent_updates", "task_status", "notifications"]
}
```

#### Agent Status Update

Server Message:
```json
{
  "type": "agent_update",
  "data": {
    "agent_id": "filesystem",
    "status": "busy",
    "current_task": "task_123",
    "timestamp": "2025-01-03T10:30:00Z"
  }
}
```

#### Task Progress

Server Message:
```json
{
  "type": "task_progress",
  "data": {
    "task_id": "task_750e8400_e29b_41d4",
    "progress": 75,
    "status": "running",
    "message": "Processing file 15 of 20"
  }
}
```

#### Execute Command

Request:
```json
{
  "type": "execute",
  "command": "agent.status",
  "parameters": {
    "agent_id": "filesystem"
  },
  "request_id": "req_123"
}
```

Response:
```json
{
  "type": "command_result",
  "request_id": "req_123",
  "success": true,
  "data": {
    "status": "online",
    "tasks_queued": 3
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    }
  },
  "timestamp": "2025-01-03T10:30:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 400 | Invalid request parameters |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |
| TIMEOUT | 504 | Request timeout |

### Common Error Scenarios

#### Authentication Error
```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "details": {
      "token_expired": true,
      "expired_at": "2025-01-03T09:30:00Z"
    }
  }
}
```

#### Validation Error
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "errors": [
        {
          "field": "email",
          "reason": "Invalid email format"
        },
        {
          "field": "password",
          "reason": "Password must be at least 8 characters"
        }
      ]
    }
  }
}
```

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1704282600
X-RateLimit-Reset-After: 3600
```

### Rate Limit Tiers

| Tier | Requests/Hour | Burst | WebSocket Connections |
|------|---------------|-------|----------------------|
| Free | 100 | 10 | 1 |
| Basic | 1,000 | 50 | 5 |
| Pro | 10,000 | 200 | 20 |
| Enterprise | Unlimited | 1000 | 100 |

### Rate Limit Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "remaining": 0,
      "reset_at": "2025-01-03T11:00:00Z",
      "retry_after": 1800
    }
  }
}
```

## API Versioning

### Version Strategy

- URL path versioning: `/api/v1/`, `/api/v2/`
- Header versioning: `X-API-Version: 1.0`
- Default to latest stable version
- Deprecation notices in headers

### Version Headers

```http
X-API-Version: 1.0
X-API-Deprecation: true
X-API-Deprecation-Date: 2025-06-01
X-API-Sunset-Date: 2025-12-01
```

### Migration Guide

```json
{
  "deprecated_endpoints": [
    {
      "old": "/api/v1/users/profile",
      "new": "/api/v2/users/me",
      "deprecation_date": "2025-03-01",
      "sunset_date": "2025-06-01"
    }
  ],
  "breaking_changes": [
    {
      "endpoint": "/api/v2/agents/execute",
      "change": "Response structure modified",
      "migration": "Update response parsing to handle nested data structure"
    }
  ]
}
```

## SDK Examples

### Python SDK

```python
from sutazai import SutazaiClient

# Initialize client
client = SutazaiClient(
    api_key="your_api_key",
    base_url="http://localhost:10200"
)

# Authentication
auth = client.auth.login(
    email="user@example.com",
    password="secure_password"
)

# Execute agent task
result = client.agents.execute(
    agent_id="filesystem",
    task="list_files",
    parameters={"path": "/opt/sutazaiapp"}
)

# Vector search
results = client.vectors.search(
    query="authentication setup",
    collection="documents",
    top_k=5
)
```

### JavaScript SDK

```javascript
import { SutazaiClient } from '@sutazai/sdk';

// Initialize client
const client = new SutazaiClient({
  apiKey: 'your_api_key',
  baseUrl: 'http://localhost:10200'
});

// Authentication
const auth = await client.auth.login({
  email: 'user@example.com',
  password: 'secure_password'
});

// Execute agent task
const result = await client.agents.execute({
  agentId: 'filesystem',
  task: 'list_files',
  parameters: { path: '/opt/sutazaiapp' }
});

// WebSocket connection
const ws = client.websocket.connect();
ws.on('agent_update', (data) => {
  console.log('Agent status:', data);
});
```

## Related Documentation

- [System Architecture](./system_design.md)
- [Authentication Guide](./security_model.md)
- [WebSocket Protocol](./data_flow.md)
- [Error Handling Guide](../development/debugging_guide.md)
- [SDK Documentation](https://github.com/sutazai/sdk)

## API Testing

### Postman Collection

Available at: `https://api.sutazai.com/postman-collection.json`

### OpenAPI Specification

Available at: `https://api.sutazai.com/openapi.json`

### Interactive Documentation

- Swagger UI: `https://api.sutazai.com/docs`
- ReDoc: `https://api.sutazai.com/redoc`