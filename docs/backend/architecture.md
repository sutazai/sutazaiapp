# Backend Architecture

## System Overview

The SutazAI backend follows a modular microservices architecture designed for scalability, maintainability, and extensibility. The system orchestrates multiple AI agents and provides a unified API for task automation.

## Core Architecture Components

### 1. API Gateway Layer
- **FastAPI Application**: Main API server with async support
- **Request Routing**: Intelligent routing to appropriate services
- **Authentication & Authorization**: JWT-based security
- **Rate Limiting**: Request throttling and quota management
- **CORS Management**: Cross-origin resource sharing configuration

### 2. Service Orchestration
- **Agent Registry**: Central registry for all AI agents
- **Task Coordinator**: Intelligent task distribution and scheduling
- **Message Bus**: Inter-service communication using Redis
- **Load Balancer**: Request distribution across agent instances
- **Health Monitor**: Service health checking and recovery

### 3. Agent Framework
- **Agent Base Classes**: Common interfaces for all agents
- **Agent Lifecycle Management**: Creation, monitoring, and cleanup
- **Resource Management**: CPU, memory, and GPU allocation
- **Communication Protocol**: Standardized agent messaging
- **Agent Discovery**: Dynamic service discovery and registration

### 4. Data Layer
- **PostgreSQL**: Primary data store for system metadata
- **Redis**: Caching and message queue
- **Vector Databases**: ChromaDB and Qdrant for embeddings
- **Neo4j**: Graph database for relationship mapping
- **File Storage**: Local and distributed file handling

## System Flow Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Frontend UI   │
│    (nginx)      │◄──►│   (FastAPI)     │◄──►│  (Streamlit)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Task Coordinator│◄──►│ Agent Registry  │◄──►│ Service Monitor │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Message Bus     │    │ Agent Instances │    │ Health Checker  │
│    (Redis)      │◄──►│   (Containers)  │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Storage    │    │ Model Serving   │    │ Monitoring      │
│ (PostgreSQL)    │    │   (Ollama)      │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Service Architecture

### Main Application (`main.py`)
```python
# Core application structure
app = FastAPI(
    title="SutazAI Task Automation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware stack
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(AuthenticationMiddleware, ...)
app.add_middleware(RateLimitMiddleware, ...)

# Route registration
app.include_router(agent_router, prefix="/api/v1/agents")
app.include_router(task_router, prefix="/api/v1/tasks")
app.include_router(health_router, prefix="/health")
```

### Agent Management System
```python
class AgentManager:
    """Central agent lifecycle management"""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.coordinator = TaskCoordinator()
        self.monitor = HealthMonitor()
    
    async def create_agent(self, agent_type: str, config: dict):
        """Create and register new agent instance"""
        
    async def route_task(self, task: Task, preferences: dict):
        """Route task to appropriate agent"""
        
    async def monitor_agents(self):
        """Monitor agent health and performance"""
```

### Task Coordination
```python
class TaskCoordinator:
    """Intelligent task distribution and execution"""
    
    async def submit_task(self, task: Task) -> TaskResult:
        """Submit task for execution"""
        
    async def schedule_task(self, task: Task, schedule: Schedule):
        """Schedule task for future execution"""
        
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current task execution status"""
```

## Database Schema

### Core Tables
```sql
-- Agents registry
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status agent_status DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tasks tracking
CREATE TABLE tasks (
    id UUID PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    status task_status DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    agent_id UUID REFERENCES agents(id),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- System metrics
CREATE TABLE metrics (
    id UUID PRIMARY KEY,
    metric_type VARCHAR(100) NOT NULL,
    value NUMERIC,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### Indexing Strategy
```sql
-- Performance indexes
CREATE INDEX idx_agents_type_status ON agents(type, status);
CREATE INDEX idx_tasks_status_created ON tasks(status, created_at);
CREATE INDEX idx_metrics_type_timestamp ON metrics(metric_type, timestamp);

-- Full-text search
CREATE INDEX idx_tasks_search ON tasks USING gin(to_tsvector('english', input_data));
```

## API Design Patterns

### RESTful Endpoints
```python
@router.get("/agents", response_model=List[Agent])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    type_filter: Optional[str] = None,
    status_filter: Optional[AgentStatus] = None
):
    """List all registered agents with filtering"""

@router.post("/agents/{agent_id}/tasks", response_model=TaskResponse)
async def create_task(
    agent_id: str,
    task_request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """Create and execute task on specific agent"""

@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str):
    """Get detailed task information and results"""
```

### WebSocket Support
```python
@app.websocket("/ws/tasks/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: str):
    """Real-time task progress updates"""
    await websocket.accept()
    
    async for update in task_stream(task_id):
        await websocket.send_json(update)
```

## Configuration Management

### Environment-based Configuration
```python
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Database
    database_url: str
    redis_url: str
    
    # AI Models
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    
    # Security
    jwt_secret: str
    cors_origins: List[str] = ["*"]
    
    # Performance
    max_workers: int = 4
    task_timeout: int = 300
    
    class Config:
        env_file = ".env"
```

### Dynamic Configuration
```python
class ConfigManager:
    """Runtime configuration management"""
    
    def __init__(self):
        self.config_store = Redis()
    
    async def get_config(self, key: str, default=None):
        """Get configuration value with caching"""
        
    async def update_config(self, key: str, value: Any):
        """Update configuration with validation"""
```

## Security Architecture

### Authentication Flow
1. **JWT Token Generation**: User authentication with secure tokens
2. **Token Validation**: Middleware validates tokens on each request
3. **Role-Based Access**: Fine-grained permissions for different operations
4. **API Key Management**: External service authentication

### Security Middleware
```python
class SecurityMiddleware:
    """Comprehensive security middleware"""
    
    async def __call__(self, request: Request, call_next):
        # Rate limiting
        await self.check_rate_limit(request)
        
        # Authentication
        await self.validate_token(request)
        
        # Input sanitization
        await self.sanitize_input(request)
        
        # CORS validation
        await self.validate_cors(request)
        
        response = await call_next(request)
        
        # Security headers
        self.add_security_headers(response)
        
        return response
```

## Error Handling

### Global Exception Handler
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handling with logging"""
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request.headers.get("X-Request-ID"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### Custom Exception Types
```python
class AgentNotFoundError(HTTPException):
    def __init__(self, agent_id: str):
        super().__init__(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )

class TaskExecutionError(HTTPException):
    def __init__(self, task_id: str, error: str):
        super().__init__(
            status_code=422,
            detail=f"Task {task_id} execution failed: {error}"
        )
```

## Performance Optimization

### Async Processing
- **Async/Await**: Non-blocking I/O operations
- **Connection Pooling**: Database and Redis connection management
- **Background Tasks**: Long-running task execution
- **Caching Strategy**: Multi-level caching for performance

### Resource Management
```python
class ResourceManager:
    """System resource monitoring and allocation"""
    
    def __init__(self):
        self.cpu_monitor = CPUMonitor()
        self.memory_monitor = MemoryMonitor()
        self.gpu_monitor = GPUMonitor()
    
    async def allocate_resources(self, agent_type: str) -> ResourceAllocation:
        """Allocate resources based on current system load"""
        
    async def release_resources(self, allocation: ResourceAllocation):
        """Release allocated resources"""
```

## Monitoring and Observability

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Performance metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_AGENTS = Gauge('active_agents_total', 'Number of active agents')

# Business metrics
TASKS_PROCESSED = Counter('tasks_processed_total', 'Total tasks processed', ['type', 'status'])
AGENT_UTILIZATION = Histogram('agent_utilization_ratio', 'Agent utilization ratio')
```

### Health Check System
```python
class HealthChecker:
    """Comprehensive system health monitoring"""
    
    async def check_system_health(self) -> HealthStatus:
        """Perform comprehensive health check"""
        
        checks = {
            'database': await self.check_database(),
            'redis': await self.check_redis(),
            'ollama': await self.check_ollama(),
            'agents': await self.check_agents(),
            'storage': await self.check_storage()
        }
        
        return HealthStatus(
            status='healthy' if all(checks.values()) else 'degraded',
            checks=checks,
            timestamp=datetime.utcnow()
        )
```

## Deployment Architecture

### Containerization
- **Multi-stage Builds**: Optimized Docker images
- **Resource Limits**: CPU and memory constraints
- **Health Checks**: Container health monitoring
- **Volume Management**: Persistent data storage

### Scaling Strategy
- **Horizontal Scaling**: Multiple instance deployment
- **Load Balancing**: Request distribution
- **Auto-scaling**: Dynamic resource allocation
- **Circuit Breakers**: Fault tolerance patterns

## Development Guidelines

### Code Organization
```
backend/
├── app/
│   ├── api/           # API routes and handlers
│   ├── core/          # Core business logic
│   ├── models/        # Data models and schemas
│   ├── services/      # Business services
│   ├── utils/         # Utility functions
│   └── main.py        # Application entry point
├── tests/             # Test suites
├── scripts/           # Deployment and utility scripts
└── requirements.txt   # Python dependencies
```

### Best Practices
- **Type Hints**: Use type annotations throughout
- **Async/Await**: Prefer async operations for I/O
- **Dependency Injection**: Use FastAPI's dependency system
- **Error Handling**: Comprehensive exception handling
- **Testing**: Unit and integration test coverage
- **Documentation**: API documentation with OpenAPI