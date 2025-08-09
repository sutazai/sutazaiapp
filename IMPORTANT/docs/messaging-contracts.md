# RabbitMQ Messaging Contracts Documentation

## Overview
This document defines the complete messaging contracts for inter-agent communication in the SUTAZAI system using RabbitMQ.

## Architecture

### Exchanges
| Exchange | Type | Purpose |
|----------|------|---------|
| `sutazai.main` | topic | Main message routing |
| `sutazai.agents` | topic | Agent-specific messages |
| `sutazai.tasks` | topic | Task distribution |
| `sutazai.resources` | topic | Resource management |
| `sutazai.system` | topic | System-wide messages |
| `sutazai.dlx` | fanout | Dead letter exchange |

### Queue Configuration
All queues are configured with:
- **TTL**: Message time-to-live (default 3600 seconds)
- **DLX**: Dead letter exchange for failed messages
- **Max Length**: 10,000 messages per queue
- **Durability**: Persistent storage for critical messages

## Message Schemas

### Base Message Structure
Every message inherits from `BaseMessage`:
```python
{
    "message_id": "uuid",
    "message_type": "MessageType enum",
    "source_agent": "agent_id",
    "target_agent": "agent_id or null",
    "timestamp": "ISO 8601 datetime",
    "correlation_id": "uuid or null",
    "priority": 0-3,
    "ttl": "seconds or null",
    "metadata": {}
}
```

## Agent Messages

### Agent Registration
**Type**: `agent.registration`  
**Queue**: `sutazai.agents` exchange  
**Purpose**: Register new agent with system

```json
{
    "message_type": "agent.registration",
    "agent_id": "hardware_resource_optimizer",
    "agent_type": "optimizer",
    "capabilities": ["resource_monitoring", "memory_optimization"],
    "version": "2.0.0",
    "host": "container_hostname",
    "port": 8116,
    "max_concurrent_tasks": 10,
    "supported_message_types": ["task.request", "resource.request"]
}
```

### Agent Heartbeat
**Type**: `agent.heartbeat`  
**Queue**: `sutazai.agents` exchange  
**Frequency**: Every 30 seconds

```json
{
    "message_type": "agent.heartbeat",
    "agent_id": "hardware_resource_optimizer",
    "status": "ready",
    "current_load": 0.3,
    "active_tasks": 3,
    "available_capacity": 7,
    "cpu_usage": 25.5,
    "memory_usage": 45.2,
    "uptime_seconds": 3600.0,
    "error_count": 0
}
```

## Task Messages

### Task Request
**Type**: `task.request`  
**Queue**: `tasks.{priority}` (high/normal/low)  
**Purpose**: Request task execution

```json
{
    "message_type": "task.request",
    "task_id": "task_12345",
    "task_type": "optimize_memory",
    "payload": {
        "target_threshold": 80,
        "aggressive": false
    },
    "requirements": {
        "capabilities": ["memory_optimization"],
        "min_memory_gb": 2
    },
    "priority": 1,
    "timeout_seconds": 300,
    "retry_count": 0,
    "max_retries": 3
}
```

### Task Status Update
**Type**: `task.status`  
**Queue**: `tasks.results`  
**Purpose**: Progress updates

```json
{
    "message_type": "task.status",
    "task_id": "task_12345",
    "status": "in_progress",
    "progress": 0.5,
    "message": "Analyzing memory usage",
    "current_step": "analysis",
    "total_steps": 3,
    "elapsed_seconds": 15.2
}
```

### Task Completion
**Type**: `task.completion`  
**Queue**: `tasks.results`  
**Purpose**: Final task result

```json
{
    "message_type": "task.completion",
    "task_id": "task_12345",
    "status": "completed",
    "result": {
        "memory_freed_gb": 2.5,
        "optimization_applied": ["cache_clear", "gc_collect"]
    },
    "execution_time_seconds": 45.3,
    "resource_usage": {
        "cpu": 15.2,
        "memory": 25.5
    }
}
```

## Resource Messages

### Resource Request
**Type**: `resource.request`  
**Queue**: `resources.requests`  
**Purpose**: Request resource allocation

```json
{
    "message_type": "resource.request",
    "request_id": "req_67890",
    "requesting_agent": "ai_agent_orchestrator",
    "task_id": "task_12345",
    "resources": {
        "cpu": 2.0,
        "memory": 4.0,
        "gpu": 0
    },
    "duration_seconds": 60,
    "exclusive": false,
    "preemptible": true
}
```

### Resource Allocation
**Type**: `resource.allocation`  
**Queue**: `resources.allocations`  
**Purpose**: Resource allocation response

```json
{
    "message_type": "resource.allocation",
    "request_id": "req_67890",
    "allocation_id": "alloc_11111",
    "allocated": true,
    "allocated_resources": {
        "cpu": 2.0,
        "memory": 4.0
    },
    "expires_at": "2025-08-07T12:00:00Z",
    "partial_allocation": false
}
```

## System Messages

### System Health
**Type**: `system.health`  
**Queue**: `system.health`  
**Purpose**: System-wide health status

```json
{
    "message_type": "system.health",
    "healthy": true,
    "components": {
        "rabbitmq": {"status": "healthy", "queue_depth": 42},
        "redis": {"status": "healthy", "memory_usage_mb": 256}
    },
    "active_agents": 5,
    "total_tasks_processed": 1542,
    "error_rate": 0.02
}
```

### System Alert
**Type**: `system.alert`  
**Queue**: `system.alerts`  
**Purpose**: System-wide alerts

```json
{
    "message_type": "system.alert",
    "alert_id": "alert_99999",
    "severity": "warning",
    "category": "resource",
    "title": "High Memory Usage",
    "description": "System memory usage above 90%",
    "affected_components": ["hardware_resource_optimizer"],
    "recommended_action": "Trigger memory optimization"
}
```

### Error Message
**Type**: `system.error`  
**Queue**: `system.errors`  
**Purpose**: Error notifications

```json
{
    "message_type": "system.error",
    "error_id": "err_55555",
    "error_code": "TASK_TIMEOUT",
    "error_message": "Task exceeded timeout of 300 seconds",
    "affected_task_id": "task_12345",
    "affected_agent_id": "task_assignment_coordinator",
    "retry_possible": true
}
```

## Routing Patterns

### Topic Exchange Routing Keys
- **Agent Messages**: `agent.{agent_id}.{message_type}`
  - Example: `agent.orchestrator.heartbeat`
- **Task Messages**: `task.{priority}.{task_type}`
  - Example: `task.high.optimize_memory`
- **Resource Messages**: `resource.{action}.{resource_type}`
  - Example: `resource.request.cpu`
- **System Messages**: `system.{category}.{severity}`
  - Example: `system.alert.critical`

## Agent Queue Mappings

### AI Agent Orchestrator
**Subscribes to**:
- `tasks.results` - Task completion notifications
- `system.health` - System health updates
- `agent.orchestrator` - Direct messages

**Publishes to**:
- `tasks.high_priority` - High priority tasks
- `tasks.normal_priority` - Normal priority tasks
- `system.alerts` - System alerts

### Task Assignment Coordinator
**Subscribes to**:
- `tasks.high_priority` - High priority task queue
- `tasks.normal_priority` - Normal priority task queue
- `tasks.low_priority` - Low priority task queue
- `agent.coordinator` - Direct messages

**Publishes to**:
- `tasks.results` - Task results
- `resources.requests` - Resource requests

### Resource Arbitration Agent
**Subscribes to**:
- `resources.requests` - Resource allocation requests
- `agent.arbitrator` - Direct messages

**Publishes to**:
- `resources.allocations` - Allocation responses
- `system.alerts` - Resource alerts

### Hardware Resource Optimizer
**Subscribes to**:
- `resources.requests` - Resource optimization requests
- `system.health` - Health check requests
- `agent.hardware_optimizer` - Direct messages

**Publishes to**:
- `resources.allocations` - Resource status updates
- `system.health` - System health metrics

## Testing Examples

### Send Test Task Request
```python
from schemas.task_messages import TaskRequestMessage
from agents.core.rabbitmq_client import RabbitMQClient

async def send_test_task():
    client = RabbitMQClient("test_client", "tester")
    await client.connect()
    
    task = TaskRequestMessage(
        source_agent="test_client",
        task_id="test_001",
        task_type="optimize_memory",
        payload={"threshold": 80},
        priority=1,
        timeout_seconds=60
    )
    
    await client.publish(task)
    await client.close()
```

### Consume Messages
```python
async def message_handler(message_data, raw_message):
    print(f"Received: {message_data['message_type']}")
    # Process message
    
async def start_consumer():
    client = RabbitMQClient("consumer", "worker")
    await client.connect()
    await client.consume("tasks.normal_priority", message_handler)
```

## Error Handling

### Retry Policy
- **Max Retries**: 3 attempts
- **Backoff**: Exponential (1s, 2s, 4s)
- **Dead Letter**: After max retries, messages go to DLQ

### Connection Resilience
- **Auto-reconnect**: Enabled with 5 second interval
- **Heartbeat**: 60 second timeout
- **Prefetch**: 10 messages per consumer

## Monitoring

### Key Metrics
- **Queue Depth**: Monitor via RabbitMQ Management UI (port 15672)
- **Message Rate**: Track publish/consume rates
- **Error Rate**: Monitor DLQ growth
- **Consumer Lag**: Track processing delays

### Health Checks
```bash
# Check RabbitMQ status
curl http://localhost:15672/api/overview

# Check queue depths
docker exec sutazai-rabbitmq rabbitmqctl list_queues

# Check connections
docker exec sutazai-rabbitmq rabbitmqctl list_connections
```

## Implementation Status

### Completed
- ✅ Centralized message schemas in `/schemas/`
- ✅ Queue configuration in `/schemas/queue_config.py`
- ✅ Base messaging client in `/agents/core/rabbitmq_client.py`
- ✅ Messaging agent base class
- ✅ Hardware Resource Optimizer integration

### Pending Integration
- ⏳ AI Agent Orchestrator (existing implementation needs messaging layer)
- ⏳ Task Assignment Coordinator (existing implementation needs messaging layer)
- ⏳ Resource Arbitration Agent (existing implementation needs messaging layer)
- ⏳ Multi-Agent Coordinator (container needs update)

## Migration Guide

To add messaging to existing agents:

1. **Inherit from MessagingAgent**:
```python
from agents.core.messaging_agent_base import MessagingAgent

class YourAgent(MessagingAgent):
    def __init__(self):
        super().__init__(
            agent_id="your_agent",
            agent_type="your_type",
            capabilities=["capability1", "capability2"]
        )
```

2. **Register Message Handlers**:
```python
async def _register_default_handlers(self):
    await self.register_handler("task.request", self.handle_task)
    await self.register_handler("resource.request", self.handle_resource)
```

3. **Publish Messages**:
```python
from schemas.task_messages import TaskCompletionMessage

completion = TaskCompletionMessage(
    source_agent=self.agent_id,
    task_id=task_id,
    status=TaskStatus.COMPLETED,
    result=result_data
)
await self.rabbitmq.publish(completion)
```

4. **Run with Messaging**:
```python
async def main():
    agent = YourAgent()
    await agent.run()
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check RabbitMQ is running: `docker ps | grep rabbitmq`
   - Verify network: `docker network ls`

2. **Queue Not Found**
   - Queues are created on first use
   - Check queue exists: `docker exec sutazai-rabbitmq rabbitmqctl list_queues`

3. **Message Not Received**
   - Check routing key matches binding
   - Verify exchange exists
   - Check consumer is connected

4. **High Memory Usage**
   - Set queue length limits
   - Enable message TTL
   - Monitor queue depths

## References
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [aio-pika Documentation](https://aio-pika.readthedocs.io/)
- [AMQP 0-9-1 Protocol](https://www.rabbitmq.com/amqp-0-9-1-reference.html)