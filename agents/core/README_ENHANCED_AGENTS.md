# Enhanced SutazAI Agent System v2.0

## Overview

The Enhanced SutazAI Agent System provides a production-ready, async-first architecture for all 131 agents with optimized Ollama integration, connection pooling, circuit breaker patterns, and request queue management.

### Key Features

- **Async-First Architecture**: Full async/await support with no threading bottlenecks
- **Connection Pooling**: Efficient Ollama connection reuse for limited hardware
- **Circuit Breaker Pattern**: Automatic failure detection and recovery
- **Request Queue Management**: Priority-based queuing with concurrency limits
- **Backward Compatibility**: Seamless migration from existing agents
- **Resource Optimization**: Designed for CPU-only hardware with limited resources
- **Comprehensive Monitoring**: Detailed metrics and health checks

## Architecture Components

### 1. BaseAgentV2 (`base_agent_v2.py`)

Enhanced base class for all agents with:

```python
from core.base_agent_v2 import BaseAgentV2

class MyAgent(BaseAgentV2):
    async def process_task(self, task):
        # Use async Ollama integration
        response = await self.query_ollama("Your prompt here")
        
        return TaskResult(
            task_id=task["id"],
            status="completed",
            result={"response": response},
            processing_time=0.5
        )
```

**Key Improvements:**
- Async Ollama queries with connection pooling
- Circuit breaker protection for resilience
- Comprehensive metrics and health monitoring
- Graceful shutdown and resource cleanup
- Backward compatibility with existing agent patterns

### 2. Connection Pool (`ollama_pool.py`)

Efficient connection management for Ollama:

```python
from core.ollama_pool import OllamaConnectionPool

async with OllamaConnectionPool(max_connections=2) as pool:
    response = await pool.generate("Your prompt")
    chat_response = await pool.chat(messages)
    embeddings = await pool.embeddings("Text to embed")
```

**Features:**
- Connection reuse and pooling
- Model warming and caching
- Automatic model pulling
- Health monitoring
- Resource-efficient for limited hardware

### 3. Circuit Breaker (`circuit_breaker.py`)

Fault tolerance and resilience:

```python
from core.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

# Protect any async function
result = await breaker.call(risky_function, arg1, arg2)

# Or use as decorator
@breaker
async def protected_function():
    # Function code here
    pass
```

**States:**
- **CLOSED**: Normal operation
- **OPEN**: Failing, reject requests
- **HALF_OPEN**: Testing recovery

### 4. Request Queue (`request_queue.py`)

Priority-based request management:

```python
from core.request_queue import RequestQueue, RequestPriority

queue = RequestQueue(max_concurrent=3)

# Submit with priority
request_id = await queue.submit(
    function_to_execute,
    arg1, arg2,
    priority=RequestPriority.HIGH
)

# Get result
result = await queue.get_result(request_id)
```

**Features:**
- Priority-based scheduling
- Configurable concurrency limits
- Request timeout and retry logic
- Comprehensive metrics

### 5. Migration Helper (`migration_helper.py`)

Backward compatibility and migration tools:

```python
from core.migration_helper import LegacyAgentWrapper

# Wrap existing v1 agents
wrapper = LegacyAgentWrapper(OldAgentClass)

# Or use factory function
agent = create_agent_factory("agent-name")
```

## Configuration

### Model Assignment

Each agent is automatically assigned an appropriate model based on complexity:

```python
from core.ollama_integration import OllamaConfig

# Get model for specific agent
model = OllamaConfig.get_model_for_agent("ai-system-architect")
# Returns: "tinyllama" (Opus model for complex reasoning)

config = OllamaConfig.get_model_config("simple-task-agent") 
# Returns: Full config with tinyllama for simple tasks
```

**Model Tiers:**
- **Opus Models** (`tinyllama`): Complex reasoning agents (31 agents)
- **Sonnet Models** (`tinyllama2.5-coder:7b`): Balanced performance (59 agents)
- **Default Model** (`tinyllama`): Simple tasks and monitoring (41 agents)

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_URL=http://ollama:10104
OLLAMA_DEFAULT_MODEL=tinyllama

# Agent Configuration  
AGENT_NAME=my-agent
AGENT_TYPE=specialized
BACKEND_URL=http://backend:8000

# Resource Limits
MAX_CONCURRENT_TASKS=3
MAX_OLLAMA_CONNECTIONS=2
LOG_LEVEL=INFO
```

## Migration Guide

### For New Agents

```python
from core.base_agent_v2 import BaseAgentV2

class NewAgent(BaseAgentV2):
    async def process_task(self, task):
        # Use async methods
        response = await self.query_ollama(task["prompt"])
        
        return TaskResult(
            task_id=task["id"],
            status="completed", 
            result={"response": response},
            processing_time=1.0
        )

if __name__ == "__main__":
    agent = NewAgent()
    agent.run()
```

### For Existing V1 Agents

#### Option 1: Legacy Wrapper (Immediate)
```python
from core.migration_helper import LegacyAgentWrapper
from old_agent import OldAgent

# Instant compatibility
wrapper = LegacyAgentWrapper(OldAgent)
wrapper.run()
```

#### Option 2: Full Migration (Recommended)
```python
# Update imports
# OLD: from agent_base import BaseAgent
from core.base_agent_v2 import BaseAgentV2

# Update class definition  
# OLD: class MyAgent(BaseAgent):
class MyAgent(BaseAgentV2):

    # Update methods to async
    async def process_task(self, task):
        # Update Ollama calls
        # OLD: response = self.query_ollama(prompt)
        response = await self.query_ollama(prompt)
        
        # Return TaskResult format
        return TaskResult(...)
```

### Migration Validation

```bash
# Run migration validator
cd /opt/sutazaiapp/agents/core
python migration_helper.py

# This will show:
# - Total agents: 131
# - V2 ready: X
# - Need attention: Y  
# - Migration plan
```

## Testing

### Quick Validation
```bash
cd /opt/sutazaiapp/agents/core
python test_enhanced_agent.py --quick
```

### Full Integration Tests (requires Ollama)
```bash
python test_enhanced_agent.py
```

### Test Individual Components
```python
import asyncio
from core.ollama_pool import OllamaConnectionPool

async def test():
    async with OllamaConnectionPool() as pool:
        response = await pool.generate("Hello!")
        print(response)

asyncio.run(test())
```

## Performance Optimization

### Hardware Constraints
Optimized for limited hardware environments:

- **Conservative Connection Limits**: Max 2 Ollama connections by default
- **Memory Efficient**: Connection pooling and model caching
- **CPU Optimized**: Async operations avoid thread overhead
- **Resource Monitoring**: Automatic cleanup and garbage collection

### Configuration Tuning

```python
# For very limited resources
agent = BaseAgentV2(
    max_concurrent_tasks=1,      # Process one task at a time
    max_ollama_connections=1,    # Single connection
    health_check_interval=60     # Less frequent health checks
)

# For better performance (if resources allow)
agent = BaseAgentV2(
    max_concurrent_tasks=5,      # More parallel processing
    max_ollama_connections=3,    # More connections
    health_check_interval=30     # More frequent monitoring
)
```

## Monitoring and Metrics

### Agent Health
```python
health = await agent.health_check()
print(f"Agent healthy: {health['healthy']}")
print(f"Tasks processed: {health['tasks_processed']}")
print(f"Avg processing time: {health['avg_processing_time']}")
```

### Connection Pool Stats
```python
stats = pool.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']}")
print(f"Hit rate: {stats['hit_rate']}")
```

### Circuit Breaker Status
```python
stats = breaker.get_stats()
print(f"State: {stats['state']}")
print(f"Trip count: {stats['trip_count']}")
print(f"Success rate: {stats['success_rate']}")
```

## Deployment

### Docker Integration
```dockerfile
FROM python:3.11-slim

# Copy enhanced agent system
COPY agents/core/ /app/agents/core/
COPY agents/my-agent/ /app/agents/my-agent/

# Install dependencies
RUN pip install httpx asyncio

# Run agent
CMD ["python", "/app/agents/my-agent/app.py"]
```

### Environment Setup
```bash
# Ensure Ollama is running
curl http://localhost:10104/api/version

# Pull required models
ollama pull tinyllama
ollama pull tinyllama2.5-coder:7b  
ollama pull tinyllama

# Start agent
python -m agents.my-agent.app
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check Ollama connectivity
   curl http://ollama:10104/api/version
   
   # Verify model availability
   curl http://ollama:10104/api/tags
   ```

2. **Memory Issues**
   ```python
   # Reduce resource usage
   agent = BaseAgentV2(
       max_concurrent_tasks=1,
       max_ollama_connections=1
   )
   ```

3. **Model Not Found**
   ```bash
   # Pull missing model
   ollama pull tinyllama
   ```

4. **Legacy Agent Issues**
   ```python
   # Use migration helper
   from core.migration_helper import run_migration_report
   run_migration_report()
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed connection pool, circuit breaker,
# and request queue operations
```

## Contributing

### Adding New Features
1. Follow async-first patterns
2. Include comprehensive error handling
3. Add metrics and monitoring
4. Provide backward compatibility
5. Include tests

### Code Style
- Use type hints
- Follow PEP 8
- Include docstrings
- Add comprehensive logging
- Handle all exceptions gracefully

## License

This enhanced agent system is part of the SutazAI project and follows the same licensing terms.

---

**For technical support or questions, please refer to the main SutazAI documentation or create an issue in the project repository.**