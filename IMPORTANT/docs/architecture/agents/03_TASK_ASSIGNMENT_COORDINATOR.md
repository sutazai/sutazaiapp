# Task Assignment Coordinator - Technical Specification

**Service Name:** Task Assignment Coordinator  
**Container:** `sutazai-task-assignment-coordinator`  
**Port:** 8551  
**Version:** As deployed  
**Status:** Basic Structure (35% complete)

## Overview

The Task Assignment Coordinator is designed to distribute tasks across available agents based on capabilities, load, and priorities. Currently implements a task queue and agent registry but lacks the actual assignment logic and agent communication mechanisms.

## Technical Architecture

### Technology Stack
- **Framework:** FastAPI with async support
- **Queue Management:** In-memory Python list (not a proper queue)
- **State Storage:** Redis (optional, continues without)
- **HTTP Client:** httpx for agent communication (unused)
- **Runtime:** Python asyncio with uvicorn

### Core Components
```python
class TaskAssignmentCoordinator:
    task_queue = []              # Simple list, not thread-safe
    agent_registry = {}          # In-memory agent storage
    active_assignments = {}      # Current task assignments
    completed_assignments = []   # Historical data (unbounded)
    failed_assignments = []      # Failed task records
```

## Data Models

### Task Model
```python
class Task(BaseModel):
    task_id: str
    task_type: str
    description: str
    required_capabilities: List[str]
    priority: int = Field(default=5, ge=1, le=10)
    estimated_duration: int = Field(default=300)  # seconds
    max_retries: int = Field(default=3)
    data: Dict[str, Any] = Field(default_factory=dict)
    deadline: Optional[datetime] = None
```

### Agent Capability Model
```python
class AgentCapability(BaseModel):
    agent_id: str
    name: str
    capabilities: List[str]
    current_load: float = 0.0
    max_concurrent_tasks: int = 5
    performance_score: float = 1.0
    availability: str = "available"  # available, busy, offline
    last_seen: datetime
```

### Task Assignment Model
```python
class TaskAssignment(BaseModel):
    assignment_id: str
    task_id: str
    agent_id: str
    assigned_at: datetime
    status: str = "assigned"  # assigned, in_progress, completed, failed
    estimated_completion: datetime
    actual_completion: Optional[datetime] = None
    retry_count: int = 0
```

## API Endpoints

### Health & Status

#### GET /health
Health check with queue statistics.

**Response Example:**
```json
{
  "status": "healthy",
  "agent": "task-assignment-coordinator",
  "timestamp": "2025-08-07T00:43:52.878817",
  "queued_tasks": 0,
  "active_assignments": 0,
  "registered_agents": 0
}
```

#### GET /statistics
Comprehensive coordinator statistics.

**Response Example:**
```json
{
  "coordinator_stats": {
    "total_tasks_submitted": 0,
    "tasks_assigned": 0,
    "tasks_completed": 0,
    "tasks_failed": 0,
    "tasks_queued": 0,
    "average_completion_time": null,
    "agent_count": 0,
    "available_agents": 0
  },
  "queue_metrics": {
    "queue_size": 0,
    "max_queue_size": 1000,
    "oldest_task_age": null,
    "average_wait_time": null
  },
  "agent_metrics": {
    "total_agents": 0,
    "active_agents": 0,
    "average_load": 0.0,
    "average_performance": 0.0
  }
}
```

### Task Management

#### POST /submit_task
Submit a new task for assignment.

**Request Body:**
```json
{
  "task_id": "task-001",
  "task_type": "data_processing",
  "description": "Process customer data batch",
  "required_capabilities": ["data_processing", "python"],
  "priority": 7,
  "estimated_duration": 600,
  "data": {
    "batch_id": "batch-123",
    "input_path": "/data/input.csv"
  },
  "deadline": "2025-08-07T12:00:00Z"
}
```

**Response:**
```json
{
  "message": "Task submitted successfully",
  "task_id": "task-001",
  "queue_position": 1,
  "estimated_assignment": "immediate"
}
```

**Reality:** Task is added to queue but never assigned to any agent.

#### GET /queue
List all queued tasks awaiting assignment.

**Response Example:**
```json
{
  "queued_tasks": [],
  "count": 0,
  "oldest_task": null,
  "highest_priority": null
}
```

### Agent Management

#### POST /register_agent
Register an agent with the coordinator.

**Request Body:**
```json
{
  "agent_id": "processor-001",
  "name": "Data Processor Agent",
  "capabilities": ["data_processing", "python", "pandas"],
  "endpoint": "http://processor-001:8000",
  "max_concurrent_tasks": 3
}
```

**Response:**
```json
{
  "message": "Agent registered successfully",
  "agent_id": "processor-001",
  "assigned_tasks": []
}
```

#### GET /agents
List all registered agents.

**Response Example:**
```json
{
  "agents": [],
  "summary": {
    "total": 0,
    "available": 0,
    "busy": 0,
    "offline": 0
  }
}
```

### Assignment Management

#### GET /assignments/active
List all active task assignments.

**Response Example:**
```json
{
  "active_assignments": [],
  "count": 0,
  "by_agent": {},
  "by_status": {
    "assigned": 0,
    "in_progress": 0
  }
}
```

#### POST /assignment_callback
Callback endpoint for agents to report task completion.

**Request Body:**
```json
{
  "assignment_id": "assign-001",
  "task_id": "task-001",
  "agent_id": "processor-001",
  "status": "completed",
  "result": {
    "rows_processed": 10000,
    "errors": 0
  },
  "completion_time": "2025-08-07T01:15:30Z"
}
```

**Response:**
```json
{
  "message": "Assignment updated",
  "assignment_id": "assign-001",
  "new_status": "completed"
}
```

## Configuration

### Environment Variables
```bash
REDIS_URL=redis://redis:6379/0              # Redis connection
LOAD_BALANCING_ALGORITHM=weighted_round_robin  # Assignment algorithm
TASK_PRIORITY_ENABLED=true                  # Enable priority queue
AGENT_CAPABILITY_MATCHING=true              # Match capabilities
PERFORMANCE_BASED_ASSIGNMENT=true           # Consider performance
TASK_QUEUE_SIZE=1000                        # Maximum queue size
```

### Assignment Algorithms (Configured but Not Implemented)
1. **Round Robin:** Equal distribution (not implemented)
2. **Weighted Round Robin:** Based on agent capacity (not implemented)
3. **Least Loaded:** Assign to least busy agent (not implemented)
4. **Capability Match:** Best capability fit (not implemented)
5. **Performance Based:** Historical performance (not implemented)

## Background Tasks

### Task Assignment Processor
```python
async def task_assignment_processor(self):
    while True:
        try:
            # Should implement assignment logic here
            # Currently just sleeps
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Assignment processor error: {e}")
```

### Agent Health Monitor
```python
async def agent_health_monitor(self):
    while True:
        try:
            # Should ping agents and update status
            # Currently does nothing
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
```

### Assignment Timeout Handler
```python
async def assignment_timeout_handler(self):
    while True:
        try:
            # Should check for timed-out assignments
            # Currently not implemented
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Timeout handler error: {e}")
```

## Implementation Status

### What's Implemented
1. **API Structure:** All endpoints defined and responding
2. **Data Models:** Complete Pydantic models
3. **Task Queue:** Basic in-memory storage
4. **Agent Registry:** Simple dictionary storage
5. **Assignment Tracking:** Basic structure for tracking

### What's Missing
1. **Assignment Logic:** No actual task-to-agent matching
2. **Agent Communication:** Cannot call agent endpoints
3. **Load Balancing:** Algorithms defined but not implemented
4. **Capability Matching:** No matching logic
5. **Priority Processing:** Priority field ignored
6. **Retry Mechanism:** No retry on failure
7. **Deadline Enforcement:** Deadlines not checked
8. **Persistence:** All data lost on restart
9. **Health Monitoring:** No agent health checks
10. **Performance Tracking:** No metrics collection

## Architectural Issues

### Design Flaws
1. **Not Thread-Safe:** Using list instead of queue.Queue
2. **Unbounded Growth:** Historical data never cleaned
3. **No Pagination:** Endpoints return all data
4. **No Validation:** Agent endpoints not validated
5. **No Transactions:** Operations not atomic

### Scalability Problems
1. **Single Instance:** No clustering support
2. **In-Memory Only:** Cannot scale beyond single node
3. **No Load Limits:** Could accept unlimited tasks
4. **No Backpressure:** No flow control mechanisms

## Testing Strategy

### Unit Tests (Needed)
```python
# Test task queue operations
def test_task_submission():
    task = Task(task_id="test-001", ...)
    coordinator.submit_task(task)
    assert task in coordinator.task_queue

# Test agent registration
def test_agent_registration():
    agent = AgentCapability(agent_id="test-agent", ...)
    coordinator.register_agent(agent)
    assert agent.agent_id in coordinator.agent_registry

# Test assignment creation
def test_assignment_creation():
    # Test assignment logic when implemented
    pass
```

### Integration Tests (Needed)
1. End-to-end task flow
2. Multi-agent scenarios
3. Failure handling
4. Load testing
5. Persistence testing

### Manual Testing
```bash
# Submit a task
curl -X POST http://localhost:8551/submit_task \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-task-001",
    "task_type": "processing",
    "description": "Test task",
    "required_capabilities": ["python"],
    "priority": 8
  }'

# Register an agent
curl -X POST http://localhost:8551/register_agent \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test-agent-001",
    "name": "Test Agent",
    "capabilities": ["python", "data_processing"],
    "endpoint": "http://test-agent:8000"
  }'

# Check statistics
curl http://localhost:8551/statistics | jq

# View queue
curl http://localhost:8551/queue | jq
```

## Security Considerations

### Current State
- **Authentication:** None
- **Authorization:** None
- **Input Validation:** Basic Pydantic validation
- **Rate Limiting:** None
- **Audit Logging:** None

### Vulnerabilities
1. **DoS Attack:** Unlimited task submission
2. **Data Injection:** No sanitization of task data
3. **Information Disclosure:** All data publicly accessible
4. **Resource Exhaustion:** Unbounded queue growth

## Performance Metrics

### Current Performance
- **Task Submission:** < 10ms
- **Agent Registration:** < 5ms
- **Queue Query:** O(n) with queue size
- **Memory Usage:** Grows unbounded with tasks

### Bottlenecks
1. Linear search through task queue
2. No indexing on assignments
3. Full data serialization on every request
4. No caching mechanisms

## Development Roadmap

### Phase 1: Core Functionality (Priority)
- ⬜ Implement basic round-robin assignment
- ⬜ Add agent health checking
- ⬜ Implement task retry logic
- ⬜ Add Redis persistence
- ⬜ Create proper thread-safe queue

### Phase 2: Advanced Features
- ⬜ Implement capability matching
- ⬜ Add load-based assignment
- ⬜ Implement priority processing
- ⬜ Add deadline enforcement
- ⬜ Create performance tracking

### Phase 3: Production Features
- ⬜ Add authentication/authorization
- ⬜ Implement rate limiting
- ⬜ Add monitoring/metrics
- ⬜ Create admin UI
- ⬜ Add clustering support

### Phase 4: Optimization
- ⬜ Optimize assignment algorithms
- ⬜ Add caching layer
- ⬜ Implement connection pooling
- ⬜ Add batch operations
- ⬜ Create performance benchmarks

## Recommended Implementation

### Immediate Fix for Basic Functionality
```python
async def assign_tasks(self):
    """Basic round-robin assignment implementation"""
    while True:
        if self.task_queue and self.available_agents:
            task = self.task_queue.pop(0)
            agent = self.get_next_available_agent()
            
            assignment = TaskAssignment(
                assignment_id=generate_id(),
                task_id=task.task_id,
                agent_id=agent.agent_id,
                assigned_at=datetime.utcnow(),
                status="assigned"
            )
            
            # Send task to agent
            await self.send_task_to_agent(agent, task)
            
            # Track assignment
            self.active_assignments[assignment.assignment_id] = assignment
            
        await asyncio.sleep(1)
```

## Conclusion

The Task Assignment Coordinator provides a **well-structured API** for task distribution but lacks the core assignment logic. It's essentially a task queue and agent registry without the coordination logic that would make it functional.

**Current State:** API skeleton with data models
**Missing Core Feature:** Actual task assignment logic
**Effort to Complete:** 2-3 weeks for basic functionality
**Production Readiness:** Not suitable for any production use

The service needs significant development to fulfill its intended purpose of coordinating task assignments across agents.