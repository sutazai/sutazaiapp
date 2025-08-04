# Agent Communication Data Flows

This document details the data flow patterns and communication protocols used in the SutazAI agent ecosystem.

## Communication Architecture Overview

```mermaid
graph TD
    A[Task Coordinator] --> B[Agent Registry]
    B --> C[Load Balancer]
    C --> D[Agent Pool]
    D --> E[Individual Agents]
    
    E --> F[Ollama Pool]
    E --> G[HTTP Client Pool]
    E --> H[Backend Services]
    
    F --> I[LLM Processing]
    G --> J[External APIs]
    H --> K[Database]
    H --> L[Message Queue]
    
    I --> M[Response Processing]
    J --> M
    K --> M
    L --> M
    
    M --> N[Result Aggregation]
    N --> O[Client Response]
    
    style A fill:#e1f5fe
    style O fill:#e8f5e8
    style C fill:#fff3e0
    style N fill:#fff3e0
```

## Core Communication Patterns

### 1. Request-Response Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant TC as Task Coordinator
    participant AR as Agent Registry
    participant A as Agent
    participant O as Ollama
    participant B as Backend
    
    C->>TC: Submit Task
    TC->>AR: Find Suitable Agent
    AR->>TC: Return Agent Info
    TC->>A: Assign Task
    
    activate A
    A->>B: Register Task Start
    A->>O: Query LLM
    O->>A: LLM Response
    A->>B: Update Progress
    A->>TC: Task Complete
    deactivate A
    
    TC->>C: Return Result
    
    Note over A,O: Connection pooling and<br/>circuit breaker protection
```

### 2. Multi-Agent Collaboration Flow

```mermaid
sequenceDiagram
    participant TC as Task Coordinator
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant A3 as Agent 3
    participant AGG as Aggregator
    participant C as Client
    
    C->>TC: Complex Task
    TC->>A1: Subtask 1
    TC->>A2: Subtask 2
    TC->>A3: Subtask 3
    
    par Parallel Processing
        A1->>AGG: Result 1
    and
        A2->>AGG: Result 2
    and
        A3->>AGG: Result 3
    end
    
    AGG->>AGG: Combine Results
    AGG->>TC: Final Result
    TC->>C: Complete Response
```

### 3. Streaming Data Flow

```mermaid
graph TD
    A[Streaming Request] --> B[WebSocket Connection]
    B --> C[Task Coordinator]
    C --> D[Agent Selection]
    D --> E[Agent Processing]
    
    E --> F{Processing Type}
    F -->|LLM Generation| G[Ollama Streaming]
    F -->|File Processing| H[Chunk Processing]
    F -->|Real-time Analysis| I[Live Analysis]
    
    G --> J[Stream Chunks]
    H --> J
    I --> J
    
    J --> K[WebSocket Broadcast]
    K --> L[Client Real-time Updates]
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style F fill:#fff3e0
```

## Data Flow Protocols

### HTTP API Communication

```json
{
  "request_pattern": {
    "method": "POST",
    "endpoint": "/api/v1/agents/execute",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer <token>",
      "X-Request-ID": "<uuid>",
      "X-Agent-Preference": "<agent_name>"
    },
    "body": {
      "task_type": "string",
      "content": "object",
      "priority": "enum[low,medium,high,critical]",
      "timeout": "integer",
      "metadata": {
        "source": "string",
        "user_id": "string",
        "session_id": "string"
      }
    }
  },
  "response_pattern": {
    "status": "integer",
    "data": {
      "task_id": "string",
      "result": "object",
      "agent_info": {
        "name": "string",
        "version": "string",
        "processing_time": "float"
      },
      "metadata": {
        "timestamp": "iso8601",
        "model_used": "string",
        "tokens_used": "integer"
      }
    },
    "error": "object|null"
  }
}
```

### WebSocket Communication

```json
{
  "connection_init": {
    "type": "connection_init",
    "payload": {
      "token": "jwt_token",
      "client_id": "unique_id"
    }
  },
  "task_stream": {
    "type": "start",
    "id": "operation_id",
    "payload": {
      "task_type": "streaming_analysis",
      "input": "data_to_process"
    }
  },
  "progress_update": {
    "type": "data",
    "id": "operation_id", 
    "payload": {
      "progress": 0.45,
      "current_step": "processing_chunk_3",
      "partial_result": "intermediate_data",
      "agent": "processing_agent_name"
    }
  },
  "completion": {
    "type": "complete",
    "id": "operation_id",
    "payload": {
      "final_result": "complete_output",
      "summary": "processing_summary"
    }
  }
}
```

### Agent-to-Agent Communication

```mermaid
graph TD
    A[Agent A] --> B[Message Bus]
    B --> C[Agent B]
    B --> D[Agent C]
    
    C --> E[Response Queue]
    D --> E
    E --> F[Result Aggregator]
    F --> G[Agent A]
    
    H[Event Store] --> I[Event Replay]
    I --> J[Failed Agent Recovery]
    
    B --> H
    E --> H
    
    style A fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style B fill:#fff3e0
    style F fill:#fff3e0
```

## Data Transformation Patterns

### 1. Input Preprocessing

```mermaid
graph TD
    A[Raw Input] --> B[Input Validator]
    B --> C{Input Type}
    
    C -->|Text| D[Text Preprocessor]
    C -->|File| E[File Parser]
    C -->|Structured Data| F[Schema Validator]
    C -->|Binary| G[Binary Processor]
    
    D --> H[Standardized Format]
    E --> H
    F --> H
    G --> H
    
    H --> I[Agent Input]
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style C fill:#fff3e0
```

### 2. Result Aggregation

```mermaid
graph TD
    A[Agent Results] --> B[Result Collector]
    B --> C[Data Harmonization]
    C --> D[Conflict Resolution]
    D --> E[Quality Assessment]
    E --> F[Format Standardization]
    F --> G[Final Output]
    
    H[Metadata Enrichment] --> I[Context Addition]
    I --> J[Provenance Tracking]
    J --> F
    
    B --> H
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fff3e0
```

### 3. Error Propagation

```mermaid
graph TD
    A[Agent Error] --> B[Error Classification]
    B --> C{Error Type}
    
    C -->|Recoverable| D[Retry Logic]
    C -->|Non-Recoverable| E[Fallback Agent]
    C -->|System Error| F[Circuit Breaker]
    C -->|Data Error| G[Data Validation]
    
    D --> H[Exponential Backoff]
    H --> I[Retry Attempt]
    
    E --> J[Alternative Processing]
    F --> K[Service Degradation]
    G --> L[Error Response]
    
    I --> M{Retry Success?}
    M -->|Yes| N[Continue Processing]
    M -->|No| O[Escalate Error]
    
    J --> N
    K --> L
    O --> L
    
    style A fill:#ffebee
    style N fill:#e8f5e8
    style L fill:#ffcdd2
    style C fill:#fff3e0
    style M fill:#fff3e0
```

## Connection Pooling and Resource Management

### Ollama Connection Pool

```python
class OllamaConnectionPool:
    """
    Manages Ollama connections with pooling and circuit breaker
    """
    def __init__(self, base_url: str, max_connections: int = 10):
        self.base_url = base_url
        self.max_connections = max_connections
        self.available_connections = asyncio.Queue(max_connections)
        self.active_connections = set()
        self.circuit_breaker = CircuitBreaker()
        
    async def acquire_connection(self):
        """Acquire a connection from the pool"""
        try:
            connection = await asyncio.wait_for(
                self.available_connections.get(), 
                timeout=30.0
            )
            self.active_connections.add(connection)
            return connection
        except asyncio.TimeoutError:
            raise ConnectionPoolExhausted()
    
    async def release_connection(self, connection):
        """Return connection to the pool"""
        if connection in self.active_connections:
            self.active_connections.remove(connection)
            await self.available_connections.put(connection)
```

### HTTP Client Pool

```python
class HTTPClientPool:
    """
    Manages HTTP clients with proper resource cleanup
    """
    def __init__(self, max_clients: int = 5):
        self.max_clients = max_clients
        self.clients = []
        self.semaphore = asyncio.Semaphore(max_clients)
        
    async def get_client(self) -> httpx.AsyncClient:
        """Get an HTTP client from the pool"""
        async with self.semaphore:
            if not self.clients:
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    limits=httpx.Limits(
                        max_connections=10,
                        max_keepalive_connections=5
                    )
                )
                return client
            return self.clients.pop()
    
    async def return_client(self, client: httpx.AsyncClient):
        """Return client to pool"""
        if len(self.clients) < self.max_clients:
            self.clients.append(client)
        else:
            await client.aclose()
```

## Message Queue Integration

### Task Queue Pattern

```mermaid
graph TD
    A[Task Producer] --> B[Priority Queue]
    B --> C[Task Router]
    C --> D[Agent Queue 1]
    C --> E[Agent Queue 2]
    C --> F[Agent Queue N]
    
    G[Agent 1] --> D
    H[Agent 2] --> E
    I[Agent N] --> F
    
    G --> J[Result Queue]
    H --> J
    I --> J
    
    J --> K[Result Processor]
    K --> L[Client Notification]
    
    M[Dead Letter Queue] --> N[Error Handler]
    B --> M
    D --> M
    E --> M
    F --> M
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style M fill:#ffcdd2
    style B fill:#fff3e0
    style C fill:#fff3e0
```

### Event-Driven Architecture

```python
class EventBus:
    """
    Event-driven communication between agents
    """
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_store = []
        
    async def publish(self, event_type: str, data: dict):
        """Publish event to all subscribers"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow(),
            "id": str(uuid.uuid4())
        }
        
        self.event_store.append(event)
        
        # Notify subscribers
        for callback in self.subscribers[event_type]:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type"""
        self.subscribers[event_type].append(callback)
```

## Performance Monitoring

### Metrics Collection

```mermaid
graph TD
    A[Agent Metrics] --> B[Metrics Collector]
    B --> C[Time Series DB]
    C --> D[Grafana Dashboard]
    
    E[Request Logs] --> F[Log Aggregator]
    F --> G[Search Engine]
    G --> H[Log Dashboard]
    
    I[Trace Data] --> J[Tracing System]
    J --> K[Service Map]
    K --> L[Performance Analysis]
    
    B --> M[Alert Manager]
    F --> M
    J --> M
    M --> N[Notifications]
    
    style A fill:#e1f5fe
    style D fill:#e8f5e8
    style H fill:#e8f5e8
    style L fill:#e8f5e8
    style N fill:#fff3e0
```

### Data Flow Metrics

```json
{
  "agent_metrics": {
    "request_rate": "requests/second",
    "response_time_p95": "milliseconds",
    "error_rate": "percentage",
    "active_connections": "count",
    "queue_depth": "count"
  },
  "system_metrics": {
    "cpu_usage": "percentage",
    "memory_usage": "bytes",
    "network_io": "bytes/second",
    "disk_io": "bytes/second"
  },
  "business_metrics": {
    "tasks_completed": "count",
    "user_satisfaction": "score",
    "cost_per_task": "currency",
    "uptime": "percentage"
  }
}
```

## Security Considerations

### Secure Communication

```mermaid
graph TD
    A[Client Request] --> B[TLS Termination]
    B --> C[Authentication]
    C --> D[Authorization]
    D --> E[Rate Limiting]
    E --> F[Input Validation]
    F --> G[Agent Processing]
    
    G --> H[Output Sanitization]
    H --> I[Response Encryption]
    I --> J[Client Response]
    
    K[Security Events] --> L[SIEM System]
    L --> M[Security Monitoring]
    
    C --> K
    D --> K
    E --> K
    
    style A fill:#e1f5fe
    style J fill:#e8f5e8
    style C fill:#fff8e1
    style D fill:#fff8e1
    style M fill:#ffecb3
```

This communication architecture ensures reliable, scalable, and secure data flow throughout the SutazAI agent ecosystem while maintaining high performance and fault tolerance.