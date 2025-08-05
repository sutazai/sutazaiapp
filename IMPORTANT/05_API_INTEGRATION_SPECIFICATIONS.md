# SutazAI API & Integration Specifications
## Complete REST API and Service Integration Documentation

**Version:** 1.0  
**Date:** August 5, 2025  
**API Base URL:** `http://localhost:10010/api/v1`  
**Status:** SPECIFICATION COMPLETE

---

## TABLE OF CONTENTS

1. [API Overview](#api-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Core API Endpoints](#core-api-endpoints)
4. [Agent Communication Protocol](#agent-communication-protocol)
5. [WebSocket Events](#websocket-events)
6. [Integration Patterns](#integration-patterns)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [API Examples](#api-examples)

---

## API OVERVIEW

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  API Gateway │────▶│   Backend   │
│ Application │     │    (Kong)    │     │  (FastAPI)  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                    ┌───────▼────────┐
                    │ Load Balancer  │
                    └───────┬────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
    ┌───▼────┐  ┌─────────┐  ┌──────────┐  ┌──▼────┐
    │ Agents │  │ Database │  │  Cache   │  │  LLM  │
    └────────┘  └─────────┘  └──────────┘  └───────┘
```

### Base Configuration

```yaml
API Configuration:
  Base URL: http://localhost:10010
  Version: v1
  Format: JSON
  Authentication: JWT Bearer Token
  Rate Limit: 100 requests/minute
  Timeout: 30 seconds
```

---

## AUTHENTICATION & AUTHORIZATION

### JWT Token Structure

```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_id",
    "username": "john_doe",
    "roles": ["user", "developer"],
    "permissions": ["read", "write", "execute"],
    "iat": 1704067200,
    "exp": 1704070800
  }
}
```

### Authentication Flow

```python
# 1. Login Request
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "john_doe",
  "password": "secure_password_123"
}

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}

# 2. Using Token
GET /api/v1/protected/resource
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

# 3. Refresh Token
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

### Role-Based Access Control (RBAC)

```python
# Role Definitions
roles = {
    "admin": {
        "permissions": ["*"],
        "resources": ["*"]
    },
    "developer": {
        "permissions": ["read", "write", "execute"],
        "resources": ["agents", "tasks", "workflows"]
    },
    "user": {
        "permissions": ["read", "execute"],
        "resources": ["tasks", "results"]
    },
    "viewer": {
        "permissions": ["read"],
        "resources": ["tasks", "results", "metrics"]
    }
}

# Permission Check Decorator
from functools import wraps

def require_permission(permission: str, resource: str):
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not has_permission(user, permission, resource):
                raise HTTPException(403, "Insufficient permissions")
            return await f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.get("/api/v1/agents")
@require_permission("read", "agents")
async def list_agents():
    return {"agents": [...]}
```

---

## CORE API ENDPOINTS

### 1. Task Management

#### Submit Task
```http
POST /api/v1/tasks
Authorization: Bearer {token}
Content-Type: application/json

{
  "type": "code_review",
  "description": "Review the authentication module for security issues",
  "priority": 5,
  "metadata": {
    "file_path": "/src/auth/module.py",
    "language": "python",
    "rules": ["security", "performance"]
  }
}

Response: 201 Created
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_completion": "2025-08-05T10:30:00Z",
  "assigned_agent": null,
  "created_at": "2025-08-05T10:00:00Z"
}
```

#### Get Task Status
```http
GET /api/v1/tasks/{task_id}
Authorization: Bearer {token}

Response: 200 OK
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "code_review",
  "status": "processing",
  "progress": 65,
  "assigned_agent": "code-reviewer-01",
  "started_at": "2025-08-05T10:05:00Z",
  "metadata": {
    "lines_processed": 450,
    "issues_found": 3
  }
}
```

#### List Tasks
```http
GET /api/v1/tasks?status=pending&limit=10&offset=0
Authorization: Bearer {token}

Response: 200 OK
{
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "type": "code_review",
      "status": "pending",
      "priority": 5,
      "created_at": "2025-08-05T10:00:00Z"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

#### Cancel Task
```http
DELETE /api/v1/tasks/{task_id}
Authorization: Bearer {token}

Response: 200 OK
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "cancelled_at": "2025-08-05T10:15:00Z"
}
```

### 2. Agent Management

#### List Agents
```http
GET /api/v1/agents
Authorization: Bearer {token}

Response: 200 OK
{
  "agents": [
    {
      "agent_id": "master-coordinator",
      "name": "Master Coordinator",
      "type": "orchestrator",
      "status": "active",
      "capabilities": ["routing", "orchestration"],
      "current_load": 3,
      "max_capacity": 10,
      "health": {
        "status": "healthy",
        "last_heartbeat": "2025-08-05T10:00:00Z",
        "uptime_seconds": 3600
      }
    },
    {
      "agent_id": "code-reviewer-01",
      "name": "Code Reviewer",
      "type": "specialist",
      "status": "busy",
      "capabilities": ["code_review", "security_analysis"],
      "current_load": 5,
      "max_capacity": 5
    }
  ],
  "total": 13,
  "active": 10,
  "idle": 2,
  "busy": 8
}
```

#### Get Agent Details
```http
GET /api/v1/agents/{agent_id}
Authorization: Bearer {token}

Response: 200 OK
{
  "agent_id": "code-reviewer-01",
  "name": "Code Reviewer",
  "type": "specialist",
  "status": "active",
  "configuration": {
    "model": "mistral:7b-instruct-q4_K_M",
    "max_iterations": 3,
    "memory_enabled": true,
    "temperature": 0.7
  },
  "metrics": {
    "tasks_completed": 156,
    "average_duration": 4.5,
    "success_rate": 0.95,
    "error_rate": 0.05
  },
  "resources": {
    "cpu_usage": 45.2,
    "memory_usage": 1024,
    "memory_limit": 2048
  }
}
```

#### Update Agent Configuration
```http
PATCH /api/v1/agents/{agent_id}/config
Authorization: Bearer {token}
Content-Type: application/json

{
  "max_iterations": 5,
  "temperature": 0.8,
  "memory_enabled": false
}

Response: 200 OK
{
  "agent_id": "code-reviewer-01",
  "updated_fields": ["max_iterations", "temperature", "memory_enabled"],
  "configuration": {
    "model": "mistral:7b-instruct-q4_K_M",
    "max_iterations": 5,
    "memory_enabled": false,
    "temperature": 0.8
  }
}
```

### 3. Workflow Management

#### Create Workflow
```http
POST /api/v1/workflows
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Code Review Pipeline",
  "description": "Automated code review with security and performance checks",
  "steps": [
    {
      "step_id": "1",
      "type": "code_review",
      "agent": "code-reviewer-01",
      "config": {
        "rules": ["security", "performance"]
      }
    },
    {
      "step_id": "2",
      "type": "test_generation",
      "agent": "test-generator-01",
      "depends_on": ["1"]
    },
    {
      "step_id": "3",
      "type": "documentation",
      "agent": "doc-generator-01",
      "depends_on": ["1", "2"]
    }
  ]
}

Response: 201 Created
{
  "workflow_id": "wf-123456",
  "name": "Code Review Pipeline",
  "status": "created",
  "created_at": "2025-08-05T10:00:00Z"
}
```

#### Execute Workflow
```http
POST /api/v1/workflows/{workflow_id}/execute
Authorization: Bearer {token}
Content-Type: application/json

{
  "input": {
    "repository": "https://github.com/user/repo",
    "branch": "feature/authentication",
    "files": ["src/auth/*.py"]
  }
}

Response: 202 Accepted
{
  "execution_id": "exec-789012",
  "workflow_id": "wf-123456",
  "status": "running",
  "started_at": "2025-08-05T10:00:00Z",
  "steps_completed": 0,
  "steps_total": 3
}
```

### 4. Results & Outputs

#### Get Task Result
```http
GET /api/v1/tasks/{task_id}/result
Authorization: Bearer {token}

Response: 200 OK
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "code_review",
  "status": "completed",
  "result": {
    "summary": "Found 3 security issues and 2 performance improvements",
    "issues": [
      {
        "severity": "high",
        "type": "security",
        "file": "/src/auth/module.py",
        "line": 45,
        "description": "SQL injection vulnerability",
        "suggestion": "Use parameterized queries"
      }
    ],
    "metrics": {
      "lines_analyzed": 500,
      "time_taken": 4.5,
      "issues_found": 5
    }
  },
  "completed_at": "2025-08-05T10:10:00Z"
}
```

### 5. Monitoring & Metrics

#### System Health
```http
GET /api/v1/health
# No authentication required

Response: 200 OK
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ollama": "healthy",
    "agents": "degraded"
  },
  "timestamp": "2025-08-05T10:00:00Z"
}
```

#### Metrics
```http
GET /api/v1/metrics
Authorization: Bearer {token}

Response: 200 OK
{
  "system": {
    "cpu_usage": 65.4,
    "memory_usage": 18432,
    "memory_total": 29696,
    "disk_usage": 45.2
  },
  "agents": {
    "total": 13,
    "active": 10,
    "tasks_queued": 25,
    "tasks_processing": 8,
    "tasks_completed_today": 156
  },
  "performance": {
    "average_task_duration": 4.5,
    "p95_task_duration": 8.2,
    "cache_hit_rate": 0.35,
    "error_rate": 0.02
  }
}
```

---

## AGENT COMMUNICATION PROTOCOL

### Inter-Agent Messaging

```python
# Message Format
class AgentMessage:
    message_id: str
    from_agent: str
    to_agent: str
    type: str  # "request", "response", "event"
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str]

# Example: Task Delegation
message = {
    "message_id": "msg-123456",
    "from_agent": "master-coordinator",
    "to_agent": "code-reviewer-01",
    "type": "request",
    "payload": {
        "action": "review_code",
        "task_id": "task-789",
        "code": "def authenticate(user, password):...",
        "rules": ["security", "performance"]
    },
    "timestamp": "2025-08-05T10:00:00Z",
    "correlation_id": "corr-456"
}

# Publishing via Redis
redis_client.publish(f"agent:{to_agent}", json.dumps(message))

# Subscribing to messages
pubsub = redis_client.pubsub()
pubsub.subscribe(f"agent:{agent_id}")

for message in pubsub.listen():
    if message['type'] == 'message':
        handle_agent_message(json.loads(message['data']))
```

### Event-Driven Communication

```python
# Event Types
events = {
    "task.created": "New task created",
    "task.assigned": "Task assigned to agent",
    "task.started": "Task processing started",
    "task.completed": "Task completed successfully",
    "task.failed": "Task failed with error",
    "agent.online": "Agent came online",
    "agent.offline": "Agent went offline",
    "agent.overloaded": "Agent at capacity"
}

# Event Publication
async def publish_event(event_type: str, data: Dict):
    event = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    await redis_client.publish("events", json.dumps(event))

# Event Subscription
async def subscribe_to_events():
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("events")
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            event = json.loads(message['data'])
            await handle_event(event)
```

---

## WEBSOCKET EVENTS

### WebSocket Connection

```javascript
// Client-side connection
const ws = new WebSocket('ws://localhost:10010/ws');

ws.onopen = () => {
    console.log('Connected to SutazAI WebSocket');
    
    // Authenticate
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'Bearer eyJhbGciOiJIUzI1NiIs...'
    }));
    
    // Subscribe to events
    ws.send(JSON.stringify({
        type: 'subscribe',
        events: ['task.*', 'agent.*']
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    switch(data.type) {
        case 'task.completed':
            handleTaskCompleted(data.payload);
            break;
        case 'agent.status':
            updateAgentStatus(data.payload);
            break;
    }
};
```

### Server-Side WebSocket Handler

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
    
    async def broadcast(self, event_type: str, data: dict):
        message = json.dumps({
            "type": event_type,
            "payload": data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        for connection in self.active_connections:
            subscriptions = self.subscriptions.get(connection, [])
            if any(pattern_matches(pattern, event_type) for pattern in subscriptions):
                await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'auth':
                # Validate token
                if not validate_token(data['token']):
                    await websocket.close(code=1008)
                    return
            
            elif data['type'] == 'subscribe':
                manager.subscriptions[websocket] = data['events']
            
            elif data['type'] == 'ping':
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## INTEGRATION PATTERNS

### 1. Synchronous Request-Response

```python
# Client code
import requests

def execute_task_sync(task_data):
    response = requests.post(
        "http://localhost:10010/api/v1/tasks",
        json=task_data,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    
    if response.status_code == 201:
        task = response.json()
        
        # Poll for completion
        while task['status'] not in ['completed', 'failed']:
            time.sleep(2)
            response = requests.get(
                f"http://localhost:10010/api/v1/tasks/{task['task_id']}",
                headers={"Authorization": f"Bearer {token}"}
            )
            task = response.json()
        
        # Get result
        if task['status'] == 'completed':
            result = requests.get(
                f"http://localhost:10010/api/v1/tasks/{task['task_id']}/result",
                headers={"Authorization": f"Bearer {token}"}
            )
            return result.json()
    
    raise Exception(f"Task execution failed: {response.text}")
```

### 2. Asynchronous with Callbacks

```python
# Server-side callback handler
@app.post("/api/v1/tasks")
async def create_task_with_callback(
    task: TaskRequest,
    callback_url: Optional[str] = None
):
    task_id = await queue_task(task)
    
    if callback_url:
        # Store callback URL for later
        await redis_client.set(
            f"callback:{task_id}",
            callback_url,
            ex=3600
        )
    
    return {"task_id": task_id}

# Task completion handler
async def on_task_complete(task_id: str, result: dict):
    callback_url = await redis_client.get(f"callback:{task_id}")
    
    if callback_url:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json={
                    "task_id": task_id,
                    "status": "completed",
                    "result": result
                }
            )
```

### 3. Event Streaming (SSE)

```python
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

@app.get("/api/v1/events/stream")
async def event_stream(request: Request):
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("events")
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    yield {
                        "event": "message",
                        "data": message['data'].decode()
                    }
                
                # Send heartbeat
                yield {"event": "ping", "data": ""}
                
        finally:
            await pubsub.unsubscribe("events")
    
    return EventSourceResponse(event_generator())
```

### 4. GraphQL Integration

```python
# GraphQL Schema
from strawberry import Schema, type, field

@type
class Task:
    task_id: str
    type: str
    status: str
    description: str
    created_at: datetime

@type
class Agent:
    agent_id: str
    name: str
    status: str
    capabilities: List[str]

@type
class Query:
    @field
    async def task(self, task_id: str) -> Task:
        return await get_task(task_id)
    
    @field
    async def agents(self) -> List[Agent]:
        return await list_agents()

@type
class Mutation:
    @field
    async def create_task(self, type: str, description: str) -> Task:
        return await create_task(type, description)

schema = Schema(query=Query, mutation=Mutation)
```

---

## ERROR HANDLING

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "priority",
        "message": "Priority must be between 1 and 10"
      }
    ],
    "timestamp": "2025-08-05T10:00:00Z",
    "request_id": "req-123456"
  }
}
```

### Error Codes

```python
class ErrorCode(Enum):
    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    AGENT_OFFLINE = "AGENT_OFFLINE"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    DATABASE_ERROR = "DATABASE_ERROR"

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": ErrorCode.BAD_REQUEST.value,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request.headers.get("X-Request-ID")
            }
        }
    )
```

---

## RATE LIMITING

### Configuration

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# Create limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri="redis://localhost:6379"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.post("/api/v1/tasks")
@limiter.limit("10 per minute")
async def create_task(task: TaskRequest):
    return await process_task(task)

# Custom limits for different user tiers
def get_rate_limit(request: Request):
    user = get_current_user(request)
    if user.tier == "premium":
        return "1000 per minute"
    elif user.tier == "standard":
        return "100 per minute"
    else:
        return "10 per minute"

@app.post("/api/v1/expensive-operation")
@limiter.limit(get_rate_limit)
async def expensive_operation():
    pass
```

---

## API EXAMPLES

### Python Client Library

```python
# sutazai_client.py
import requests
from typing import Dict, Optional

class SutazAIClient:
    def __init__(self, base_url: str = "http://localhost:10010", token: Optional[str] = None):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
    
    def login(self, username: str, password: str) -> str:
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        self.session.headers["Authorization"] = f"Bearer {self.token}"
        return self.token
    
    def create_task(self, task_type: str, description: str, **kwargs) -> Dict:
        response = self.session.post(
            f"{self.base_url}/api/v1/tasks",
            json={
                "type": task_type,
                "description": description,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_task(self, task_id: str) -> Dict:
        response = self.session.get(f"{self.base_url}/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(self, task_id: str, timeout: int = 300) -> Dict:
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            task = self.get_task(task_id)
            if task["status"] in ["completed", "failed"]:
                return task
            time.sleep(2)
        
        raise TimeoutError(f"Task {task_id} did not complete in {timeout} seconds")

# Usage example
client = SutazAIClient()
client.login("john_doe", "password123")

# Create and wait for task
task = client.create_task(
    "code_review",
    "Review authentication module",
    priority=8,
    metadata={"file": "/src/auth.py"}
)

result = client.wait_for_task(task["task_id"])
print(f"Task completed: {result}")
```

### JavaScript/TypeScript Client

```typescript
// sutazai-client.ts
interface Task {
    task_id: string;
    type: string;
    status: string;
    description: string;
}

class SutazAIClient {
    private baseUrl: string;
    private token?: string;
    
    constructor(baseUrl: string = "http://localhost:10010") {
        this.baseUrl = baseUrl;
    }
    
    async login(username: string, password: string): Promise<void> {
        const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password})
        });
        
        const data = await response.json();
        this.token = data.access_token;
    }
    
    async createTask(type: string, description: string): Promise<Task> {
        const response = await fetch(`${this.baseUrl}/api/v1/tasks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({type, description})
        });
        
        return response.json();
    }
    
    async getTask(taskId: string): Promise<Task> {
        const response = await fetch(`${this.baseUrl}/api/v1/tasks/${taskId}`, {
            headers: {'Authorization': `Bearer ${this.token}`}
        });
        
        return response.json();
    }
}

// Usage
const client = new SutazAIClient();
await client.login("john_doe", "password123");

const task = await client.createTask("code_review", "Review auth module");
console.log(`Created task: ${task.task_id}`);
```

### cURL Examples

```bash
# Login
curl -X POST http://localhost:10010/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"john_doe","password":"password123"}'

# Create task
curl -X POST http://localhost:10010/api/v1/tasks \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{
    "type": "code_review",
    "description": "Review authentication module",
    "priority": 8
  }'

# Get task status
curl http://localhost:10010/api/v1/tasks/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."

# Stream events (SSE)
curl -N http://localhost:10010/api/v1/events/stream \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Accept: text/event-stream"
```

---

## API VERSIONING

### Version Strategy

```python
# URL Path Versioning
/api/v1/tasks  # Version 1
/api/v2/tasks  # Version 2

# Header Versioning (alternative)
GET /api/tasks
X-API-Version: 1.0

# Response includes version
{
  "api_version": "1.0",
  "data": {...}
}

# Deprecation notice
{
  "api_version": "1.0",
  "deprecation_warning": "Version 1.0 will be deprecated on 2025-12-31",
  "migration_guide": "https://docs.sutazai.com/migration/v2",
  "data": {...}
}
```

---

## OPENAPI SPECIFICATION

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: SutazAI API
  version: 1.0.0
  description: AI Task Automation API
servers:
  - url: http://localhost:10010/api/v1
paths:
  /tasks:
    post:
      summary: Create new task
      operationId: createTask
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TaskRequest'
      responses:
        '201':
          description: Task created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Task'
components:
  schemas:
    TaskRequest:
      type: object
      required:
        - type
        - description
      properties:
        type:
          type: string
        description:
          type: string
        priority:
          type: integer
          minimum: 1
          maximum: 10
    Task:
      type: object
      properties:
        task_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, queued, processing, completed, failed]
```

---

## CONCLUSION

This API specification provides a complete interface for integrating with the SutazAI system. Key features:

1. **RESTful Design**: Standard HTTP methods and status codes
2. **Authentication**: JWT-based with RBAC
3. **Real-time Updates**: WebSocket and SSE support
4. **Rate Limiting**: Configurable per-tier limits
5. **Comprehensive Error Handling**: Structured error responses
6. **Multiple Integration Patterns**: Sync, async, streaming

**Next Steps:**
1. Implement authentication middleware
2. Set up API gateway (Kong)
3. Deploy API documentation (Swagger)
4. Create client SDKs
5. Implement monitoring and analytics

---

**Document Status:** SPECIFICATION COMPLETE  
**API Version:** 1.0.0  
**OpenAPI Spec:** Available at `/api/v1/openapi.json`