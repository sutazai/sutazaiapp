# SutazAI API Documentation

## Overview

The SutazAI Task Automation Platform provides a comprehensive REST API for managing agents, workflows, and automation tasks. All endpoints return JSON responses and accept JSON request bodies.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API operates in local mode without authentication. For production deployments, implement JWT or API key authentication.

## Core Endpoints

### System Health

#### GET /health
Check system health status

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "postgres": "healthy",
    "redis": "healthy",
    "ollama": "healthy"
  }
}
```

### Agent Management

#### GET /agents
List all available agents

**Response:**
```json
{
  "agents": [
    {
      "name": "senior-backend-developer",
      "status": "ready",
      "capabilities": ["code_generation", "testing", "deployment"],
      "description": "Backend development specialist"
    },
    {
      "name": "testing-qa-validator",
      "status": "ready",
      "capabilities": ["testing", "validation", "monitoring"],
      "description": "Quality assurance specialist"
    }
  ],
  "total": 40
}
```

#### GET /agents/{agent_name}
Get detailed information about a specific agent

**Parameters:**
- `agent_name` (path): Name of the agent

**Response:**
```json
{
  "name": "senior-backend-developer",
  "version": "1.0.0",
  "status": "ready",
  "capabilities": ["code_generation", "testing", "deployment"],
  "configuration": {
    "model": "tinyllama:latest",
    "max_tokens": 2048,
    "temperature": 0.7
  },
  "metrics": {
    "tasks_completed": 156,
    "average_response_time": 2.3,
    "success_rate": 0.98
  }
}
```

#### POST /agents/{agent_name}/execute
Execute a task with a specific agent

**Parameters:**
- `agent_name` (path): Name of the agent

**Request Body:**
```json
{
  "task": "analyze_code",
  "params": {
    "path": "/opt/sutazaiapp/backend",
    "language": "python",
    "checks": ["security", "performance", "style"]
  },
  "timeout": 300
}
```

**Response:**
```json
{
  "task_id": "task_123456",
  "agent": "senior-backend-developer",
  "status": "completed",
  "result": {
    "issues_found": 12,
    "suggestions": [
      {
        "file": "app.py",
        "line": 45,
        "severity": "medium",
        "message": "Consider using connection pooling",
        "fix": "db = create_pool(max_connections=10)"
      }
    ]
  },
  "duration": 2.3
}
```

### Workflow Management

#### GET /workflows
List available workflows

**Response:**
```json
{
  "workflows": [
    {
      "name": "code-improvement",
      "description": "Analyze and improve code quality",
      "required_agents": ["code-generation-improver", "testing-qa-validator"],
      "average_duration": 120
    },
    {
      "name": "deployment-pipeline",
      "description": "Complete deployment workflow",
      "required_agents": ["testing-qa-validator", "deployment-automation-master"],
      "average_duration": 300
    }
  ]
}
```

#### POST /workflows/{workflow_name}/start
Start a workflow execution

**Parameters:**
- `workflow_name` (path): Name of the workflow

**Request Body:**
```json
{
  "params": {
    "repository": "/opt/sutazaiapp",
    "branch": "main",
    "environment": "staging"
  },
  "agents": ["custom-agent-1", "custom-agent-2"]
}
```

**Response:**
```json
{
  "workflow_id": "wf_789012",
  "name": "code-improvement",
  "status": "running",
  "started_at": "2024-01-15T10:30:00Z",
  "steps": [
    {
      "step": 1,
      "agent": "code-generation-improver",
      "status": "completed",
      "duration": 45
    },
    {
      "step": 2,
      "agent": "testing-qa-validator",
      "status": "running",
      "progress": 0.6
    }
  ]
}
```

#### GET /workflows/{workflow_id}/status
Get workflow execution status

**Parameters:**
- `workflow_id` (path): Workflow execution ID

**Response:**
```json
{
  "workflow_id": "wf_789012",
  "status": "completed",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "results": {
    "improvements_made": 8,
    "tests_added": 12,
    "coverage_increase": 15
  }
}
```

### Task Coordination

#### POST /coordinator/assign
Automatically assign a task to the best agent

**Request Body:**
```json
{
  "task_description": "Analyze Python code for security vulnerabilities",
  "requirements": {
    "language": "python",
    "focus": "security",
    "urgency": "high"
  }
}
```

**Response:**
```json
{
  "assigned_agent": "semgrep-security-analyzer",
  "confidence": 0.95,
  "alternatives": [
    {
      "agent": "kali-security-specialist",
      "confidence": 0.82
    }
  ],
  "reasoning": "Best match for Python security analysis with Semgrep specialization"
}
```

### Model Management

#### GET /models
List available Ollama models

**Response:**
```json
{
  "models": [
    {
      "name": "tinyllama:latest",
      "size": "637MB",
      "status": "ready",
      "capabilities": ["text-generation", "code-completion"]
    },
    {
      "name": "codellama:7b",
      "size": "3.8GB",
      "status": "downloading",
      "progress": 0.45
    }
  ]
}
```

#### POST /models/load
Load a model into memory

**Request Body:**
```json
{
  "model": "codellama:7b",
  "options": {
    "gpu_layers": 32,
    "context_size": 4096
  }
}
```

### Monitoring & Metrics

#### GET /metrics
Get system metrics

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 7.5,
    "disk_usage": 32.1
  },
  "agents": {
    "active": 5,
    "idle": 35,
    "tasks_in_queue": 3
  },
  "performance": {
    "average_task_duration": 2.3,
    "tasks_per_minute": 12.5,
    "success_rate": 0.98
  }
}
```

## WebSocket Endpoints

### WS /ws/agents
Real-time agent status updates

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agents');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent update:', data);
};
```

**Message Format:**
```json
{
  "type": "agent_status",
  "agent": "senior-backend-developer",
  "status": "busy",
  "current_task": "analyzing_code",
  "progress": 0.75
}
```

### WS /ws/workflows/{workflow_id}
Real-time workflow progress

**Message Format:**
```json
{
  "type": "workflow_progress",
  "workflow_id": "wf_789012",
  "current_step": 3,
  "total_steps": 5,
  "status": "running",
  "message": "Running security analysis..."
}
```

## Error Responses

All endpoints use standard HTTP status codes and return error details in JSON format:

```json
{
  "error": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent 'unknown-agent' not found",
    "details": {
      "available_agents": ["senior-backend-developer", "testing-qa-validator"]
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}
```

### Common Error Codes

- `400` - Bad Request (invalid parameters)
- `404` - Not Found (agent/workflow not found)
- `409` - Conflict (resource busy)
- `500` - Internal Server Error
- `503` - Service Unavailable (agent not ready)

## Rate Limiting

Local deployments have no rate limiting. For production:
- Default: 100 requests per minute per IP
- Workflows: 10 concurrent workflows
- WebSocket: 5 connections per client

## Example Client Code

### Python
```python
import httpx

async def analyze_code():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents/code-generation-improver/execute",
            json={
                "task": "analyze_code",
                "params": {"path": "/app/src"}
            }
        )
        return response.json()
```

### JavaScript
```javascript
async function startWorkflow() {
  const response = await fetch('http://localhost:8000/api/v1/workflows/code-improvement/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      params: {repository: '/opt/sutazaiapp'}
    })
  });
  return response.json();
}
```

### cURL
```bash
# List agents
curl http://localhost:8000/api/v1/agents

# Execute task
curl -X POST http://localhost:8000/api/v1/agents/testing-qa-validator/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "run_tests", "params": {"path": "/app"}}'

# Start workflow
curl -X POST http://localhost:8000/api/v1/workflows/deployment-pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"params": {"service": "backend-api", "environment": "staging"}}'
```

## SDK Support

Official SDKs are planned for:
- Python (`pip install sutazai-sdk`)
- JavaScript/TypeScript (`npm install @sutazai/sdk`)
- Go (`go get github.com/sutazai/sdk-go`)

## API Versioning

The API uses URL versioning. Current version: `v1`

Future versions will maintain backward compatibility for at least 6 months after deprecation notice.