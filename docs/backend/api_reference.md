# API Reference

## Overview

The SutazAI API provides comprehensive access to the multi-agent task automation system. This RESTful API follows OpenAPI 3.0 specifications and supports both synchronous and asynchronous operations.

## Base URL and Versioning

```
Base URL: http://localhost:8000
API Version: v1
Versioned Endpoints: /api/v1/*
```

## Authentication

### Bearer Token Authentication
```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication
```http
X-API-Key: sai_<api_key>
```

## Response Format

### Standard Response Structure
```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-string"
}
```

### Error Response Structure
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {},
    "field_errors": []
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-string"
}
```

## Core Endpoints

### System Health and Status

#### Get System Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ollama": "healthy",
    "agents": "healthy"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Get Detailed System Status
```http
GET /api/v1/system/status
```

**Response:**
```json
{
  "system": {
    "name": "SutazAI",
    "version": "1.0.0",
    "environment": "production",
    "uptime": 86400
  },
  "resources": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1,
    "active_connections": 42
  },
  "services": {
    "database": {
      "status": "healthy",
      "connections": 15,
      "response_time": 2.3
    },
    "redis": {
      "status": "healthy",
      "memory_usage": 45.6,
      "connected_clients": 8
    }
  }
}
```

## Agent Management API

### List All Agents
```http
GET /api/v1/agents
```

**Query Parameters:**
- `skip` (int): Number of records to skip (default: 0)
- `limit` (int): Maximum number of records to return (default: 100)
- `type` (string): Filter by agent type
- `status` (string): Filter by agent status (active, inactive, error)
- `search` (string): Search agents by name or description

**Response:**
```json
{
  "agents": [
    {
      "id": "agent-uuid",
      "name": "code-generation-improver",
      "type": "code_generation",
      "status": "active",
      "description": "Improves code quality and style",
      "version": "1.0.0",
      "capabilities": ["code_review", "refactoring", "optimization"],
      "config": {
        "model": "tinyllama:latest",
        "temperature": 0.7
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 34,
  "page": 1,
  "pages": 1
}
```

### Get Agent Details
```http
GET /api/v1/agents/{agent_id}
```

**Response:**
```json
{
  "id": "agent-uuid",
  "name": "code-generation-improver",
  "type": "code_generation",
  "status": "active",
  "description": "Improves code quality and style",
  "version": "1.0.0",
  "capabilities": ["code_review", "refactoring", "optimization"],
  "config": {
    "model": "tinyllama:latest",
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "metrics": {
    "total_tasks": 156,
    "successful_tasks": 148,
    "failed_tasks": 8,
    "average_response_time": 2.4,
    "last_active": "2024-01-01T00:00:00Z"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Create Agent Instance
```http
POST /api/v1/agents
```

**Request Body:**
```json
{
  "name": "custom-agent-instance",
  "type": "code_generation",
  "config": {
    "model": "tinyllama:latest",
    "temperature": 0.8,
    "max_tokens": 1024
  },
  "description": "Custom agent for specific task"
}
```

### Update Agent Configuration
```http
PUT /api/v1/agents/{agent_id}
```

**Request Body:**
```json
{
  "config": {
    "temperature": 0.5,
    "max_tokens": 2048
  },
  "description": "Updated description"
}
```

### Delete Agent
```http
DELETE /api/v1/agents/{agent_id}
```

## Task Management API

### Create Task
```http
POST /api/v1/tasks
```

**Request Body:**
```json
{
  "type": "code_review",
  "agent_id": "agent-uuid",
  "input": {
    "code": "def hello_world():\n    print('Hello, World!')",
    "language": "python",
    "review_type": "style_and_performance"
  },
  "config": {
    "timeout": 300,
    "priority": "normal"
  }
}
```

**Response:**
```json
{
  "id": "task-uuid",
  "type": "code_review",
  "status": "pending",
  "agent_id": "agent-uuid",
  "input": {},
  "output": null,
  "progress": 0,
  "created_at": "2024-01-01T00:00:00Z",
  "estimated_completion": "2024-01-01T00:05:00Z"
}
```

### Get Task Status
```http
GET /api/v1/tasks/{task_id}
```

**Response:**
```json
{
  "id": "task-uuid",
  "type": "code_review",
  "status": "completed",
  "agent_id": "agent-uuid",
  "input": {
    "code": "def hello_world():\n    print('Hello, World!')",
    "language": "python"
  },
  "output": {
    "review": "Code looks good with minor style improvements needed",
    "suggestions": [
      {
        "line": 2,
        "type": "style",
        "message": "Consider using f-strings for better readability"
      }
    ],
    "score": 85
  },
  "progress": 100,
  "duration": 2.4,
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:02:24Z"
}
```

### List Tasks
```http
GET /api/v1/tasks
```

**Query Parameters:**
- `skip` (int): Number of records to skip
- `limit` (int): Maximum number of records to return
- `status` (string): Filter by task status
- `type` (string): Filter by task type
- `agent_id` (string): Filter by agent ID
- `created_after` (datetime): Filter tasks created after date
- `created_before` (datetime): Filter tasks created before date

### Cancel Task
```http
POST /api/v1/tasks/{task_id}/cancel
```

**Response:**
```json
{
  "id": "task-uuid",
  "status": "cancelled",
  "message": "Task cancelled successfully"
}
```

## Real-time Communication

### WebSocket Task Updates
```javascript
// Connect to task updates
const ws = new WebSocket('ws://localhost:8000/ws/tasks/{task_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Task update:', update);
};

// Message format
{
  "type": "progress_update",
  "task_id": "task-uuid",
  "progress": 45,
  "status": "processing",
  "message": "Analyzing code structure...",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### WebSocket System Events
```javascript
// Connect to system events
const ws = new WebSocket('ws://localhost:8000/ws/system');

// Event types: agent_status, system_health, task_completed
{
  "type": "agent_status",
  "agent_id": "agent-uuid",
  "status": "active",
  "message": "Agent is now ready for tasks",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Model Management API

### List Available Models
```http
GET /api/v1/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "tinyllama:latest",
      "type": "ollama",
      "size": "637MB",
      "status": "loaded",
      "capabilities": ["text_generation", "code_completion"],
      "config": {
        "context_length": 2048,
        "temperature_range": [0.0, 2.0]
      }
    }
  ]
}
```

### Load Model
```http
POST /api/v1/models/{model_name}/load
```

### Unload Model
```http
POST /api/v1/models/{model_name}/unload
```

## Configuration API

### Get System Configuration
```http
GET /api/v1/config
```

**Response:**
```json
{
  "system": {
    "max_concurrent_tasks": 10,
    "default_timeout": 300,
    "log_level": "INFO"
  },
  "agents": {
    "auto_scale": true,
    "max_instances_per_type": 3,
    "health_check_interval": 30
  },
  "models": {
    "default_model": "tinyllama:latest",
    "auto_load": true,
    "memory_threshold": 0.8
  }
}
```

### Update Configuration
```http
PUT /api/v1/config
```

**Request Body:**
```json
{
  "system": {
    "max_concurrent_tasks": 15,
    "default_timeout": 600
  }
}
```

## Metrics and Analytics API

### Get System Metrics
```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "system": {
    "uptime": 86400,
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1
  },
  "tasks": {
    "total_completed": 1542,
    "success_rate": 94.3,
    "average_duration": 4.2,
    "tasks_per_hour": 42
  },
  "agents": {
    "active_count": 12,
    "total_count": 34,
    "utilization_rate": 67.8
  }
}
```

### Get Agent Analytics
```http
GET /api/v1/analytics/agents/{agent_id}
```

### Get Task Analytics
```http
GET /api/v1/analytics/tasks
```

**Query Parameters:**
- `start_date` (datetime): Start date for analytics
- `end_date` (datetime): End date for analytics
- `granularity` (string): hour, day, week, month

## File Upload API

### Upload File for Processing
```http
POST /api/v1/files/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: File to upload
- `type`: File type (document, code, data)
- `metadata`: Additional metadata (JSON)

**Response:**
```json
{
  "file_id": "file-uuid",
  "filename": "document.pdf",
  "size": 1024000,
  "type": "document",
  "upload_url": "/files/file-uuid",
  "processing_status": "pending"
}
```

### Get File Status
```http
GET /api/v1/files/{file_id}
```

## Batch Operations API

### Submit Batch Tasks
```http
POST /api/v1/batch/tasks
```

**Request Body:**
```json
{
  "tasks": [
    {
      "type": "code_review",
      "input": {"code": "..."},
      "agent_id": "agent-uuid-1"
    },
    {
      "type": "security_scan",
      "input": {"code": "..."},
      "agent_id": "agent-uuid-2"
    }
  ],
  "config": {
    "parallel": true,
    "timeout": 600
  }
}
```

**Response:**
```json
{
  "batch_id": "batch-uuid",
  "task_ids": ["task-uuid-1", "task-uuid-2"],
  "status": "submitted",
  "progress": 0
}
```

### Get Batch Status
```http
GET /api/v1/batch/{batch_id}
```

## Error Codes

### HTTP Status Codes
- **200**: OK - Request successful
- **201**: Created - Resource created successfully
- **400**: Bad Request - Invalid request parameters
- **401**: Unauthorized - Authentication required
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource not found
- **422**: Unprocessable Entity - Validation error
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server error
- **503**: Service Unavailable - Service temporarily unavailable

### Custom Error Codes
```json
{
  "AGENT_NOT_FOUND": "Agent with specified ID not found",
  "AGENT_UNAVAILABLE": "Agent is currently unavailable",
  "TASK_TIMEOUT": "Task execution exceeded timeout limit",
  "MODEL_NOT_LOADED": "Required model is not loaded",
  "INVALID_INPUT": "Input validation failed",
  "RESOURCE_LIMIT": "System resource limit exceeded",
  "PERMISSION_DENIED": "User lacks required permissions"
}
```

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

### Rate Limits by Endpoint
- **Authentication**: 10 requests per minute
- **Agent Operations**: 100 requests per hour
- **Task Creation**: 50 requests per minute
- **File Upload**: 10 requests per minute
- **Analytics**: 200 requests per hour

## SDK Examples

### Python SDK
```python
from sutazai_client import SutazAIClient

client = SutazAIClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create task
task = client.tasks.create(
    type="code_review",
    agent_id="agent-uuid",
    input={"code": "def hello(): pass"}
)

# Wait for completion
result = client.tasks.wait_for_completion(task.id)
print(result.output)
```

### JavaScript SDK
```javascript
import { SutazAIClient } from '@sutazai/client';

const client = new SutazAIClient({
    baseURL: 'http://localhost:8000',
    apiKey: 'your-api-key'
});

// Create task
const task = await client.tasks.create({
    type: 'code_review',
    agentId: 'agent-uuid',
    input: { code: 'def hello(): pass' }
});

// Get result
const result = await client.tasks.get(task.id);
```

## Postman Collection

The complete API can be tested using our Postman collection available at:
```
/docs/api/postman/sutazai-api-collection.json
```

Import this collection to get pre-configured requests for all endpoints with example data and authentication setup.