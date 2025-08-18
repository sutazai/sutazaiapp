# Agent Implementation Guide - Reality-Based Transformation

**Created:** August 8, 2025  
**Last Updated:** August 8, 2025  
**Status:** Definitive reality-based guide based on CLAUDE.md  

## Executive Summary

This guide provides a practical roadmap for transforming the current 7 Flask stub agents into functional, production-ready AI agents. Based on the verified system state documented in CLAUDE.md, this guide focuses on **realistic implementation** rather than conceptual features.

## Current State Assessment

### What Actually Exists (Verified)

#### 7 Running Agent Services - Flask Stubs
| Agent | Port | Current Functionality | Reality |
|-------|------|----------------------|---------|
| AI Agent Orchestrator | 8589 | Returns `{"status": "healthy"}` | Flask stub with `/health` endpoint only |
| Multi-Agent Coordinator | 8587 | Basic coordination stub | Flask stub with `/health` endpoint only |
| Resource Arbitration Agent | 8588 | Resource allocation stub | Flask stub with `/health` endpoint only |
| Task Assignment Coordinator | 8551 | Task routing stub | Flask stub with `/health` endpoint only |
| Hardware Resource Optimizer | 8002 | Hardware monitoring stub | Flask stub with `/health` endpoint only |
| Ollama Integration Specialist | 11015 | Ollama wrapper (may work) | Flask stub with `/health` endpoint only |
| AI Metrics Exporter | 11063 | Metrics collection (UNHEALTHY) | Flask stub - currently failing |

#### Supporting Infrastructure (Working)
- **PostgreSQL** (10000): Empty database, no schema created
- **Redis** (10001): Working cache layer
- **Neo4j** (10002/10003): Graph database available
- **Ollama** (10104): TinyLlama 637MB model loaded (NOT gpt-oss)
- **RabbitMQ** (10007/10008): Message queue available but not integrated
- **Monitoring Stack**: Prometheus, Grafana, Loki all operational

### What's conceptual (Doesn't Work)

The existing `/opt/sutazaiapp/docs/AGENT_IMPLEMENTATION_GUIDE.md` describes a complex implementation with:
- ❌ RabbitMQ message processing (agents don't consume messages)
- ❌ Redis state persistence (agents don't read/write Redis)
- ❌ Inter-agent communication (agents don't talk to each other)
- ❌ Task assignment and tracking (no actual task processing)
- ❌ Resource arbitration logic (no resource monitoring)

**Reality**: All agents have only `/health` endpoints that return hardcoded JSON regardless of input.

## Agent Architecture Patterns

### Base Agent Design (Recommended)

```python
#!/usr/bin/env python3
"""
Base Agent Pattern - Reality-focused implementation
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

class BaseAgent:
    """Real base agent with actual functionality"""
    
    def __init__(self, agent_id: str, port: int):
        self.agent_id = agent_id
        self.port = port
        self.redis_client = None
        self.ollama_url = "http://ollama:10104"
        self.status = "starting"
        
    async def initialize(self):
        """Initialize agent connections"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Test Ollama connection
            await self.test_ollama()
            
            self.status = "healthy"
            logging.info(f"{self.agent_id} initialized successfully")
            
        except Exception as e:
            self.status = "degraded"
            logging.error(f"{self.agent_id} initialization failed: {e}")
    
    async def test_ollama(self) -> bool:
        """Test connection to Ollama with actual model"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "tinyllama",  # Use actual model
                        "prompt": "Hello",
                        "stream": False
                    },
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            logging.warning(f"Ollama test failed: {e}")
            return False
    
    async def process_with_ollama(self, prompt: str) -> str:
        """Actually process with Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "tinyllama",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response')
                else:
                    return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def shutdown(self):
        """Cleanup connections"""
        if self.redis_client:
            await self.redis_client.close()
```

### Common Interface Contract

All agents should implement these endpoints:

```python
@app.get("/health")
async def health_check():
    """Standard health check"""
    return {
        "status": agent.status,
        "agent": agent.agent_id,
        "timestamp": datetime.utcnow().isoformat(),
        "port": agent.port
    }

@app.post("/process")
async def process_request(request: Dict[str, Any]):
    """Main processing endpoint - implement actual logic here"""
    return await agent.process_request(request)

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    return agent.get_metrics()
```

## Implementation Roadmap

### Phase 1: Convert One Agent (Week 1)

**Target**: Hardware Resource Optimizer (Port 8002)  
**Rationale**: Easiest to implement, clear purpose

**Steps**:
1. Replace Flask stub with FastAPI + BaseAgent
2. Implement actual system monitoring using `psutil`
3. Store metrics in Redis
4. Add Prometheus metrics endpoint

**Expected Outcome**: One fully functional agent that actually monitors system resources

### Phase 2: Core Orchestration (Week 2-3)

**Targets**: AI Agent Orchestrator (8589) + Task Assignment Coordinator (8551)

**AI Agent Orchestrator**:
- Convert to FastAPI
- Implement agent registry in Redis
- Add basic task routing (round-robin initially)
- Integrate with Ollama for task planning

**Task Assignment Coordinator**:
- Convert to FastAPI  
- Implement priority queue in Redis
- Add task timeout handling
- Basic load balancing

### Phase 3: Resource Management (Week 4-5)

**Target**: Resource Arbitration Agent (8588)

**Implementation**:
- Real resource monitoring and allocation
- Integration with Hardware Optimizer
- Conflict detection and resolution
- Resource reservation system

### Phase 4: Integration Agents (Week 6)

**Targets**: Remaining agents (8587, 11015, 11063)

**Multi-Agent Coordinator**:
- Message routing between agents
- Health monitoring of agent cluster
- Failure detection and recovery

**Ollama Integration Specialist**:
- Model management and switching
- Context optimization
- Batch processing for efficiency

**AI Metrics Exporter** (Fix current failure):
- Collect metrics from all agents
- Export to Prometheus
- Dashboard integration

## Implementation Examples

### Convert Hardware Resource Optimizer

```python
#!/usr/bin/env python3
"""
Hardware Resource Optimizer - Real Implementation
Port: 8002
"""
import psutil
import asyncio
from datetime import datetime
from typing import Dict, Any

class HardwareResourceOptimizer(BaseAgent):
    def __init__(self):
        super().__init__("hardware-resource-optimizer", 8002)
        self.metrics = {}
        
    async def collect_metrics(self):
        """Collect real system metrics"""
        try:
            self.metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": {
                    partition.device: {
                        "total": psutil.disk_usage(partition.mountpoint).total,
                        "used": psutil.disk_usage(partition.mountpoint).used,
                        "free": psutil.disk_usage(partition.mountpoint).free
                    }
                    for partition in psutil.disk_partitions()
                },
                "network_io": psutil.net_io_counters()._asdict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            await self.redis_client.hset(
                "system:metrics",
                "current",
                json.dumps(self.metrics)
            )
            
        except Exception as e:
            logging.error(f"Metrics collection failed: {e}")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process optimization requests"""
        action = request.get("action", "get_metrics")
        
        if action == "get_metrics":
            await self.collect_metrics()
            return {
                "status": "success",
                "metrics": self.metrics
            }
            
        elif action == "optimize_memory":
            # Implement memory optimization logic
            optimization_prompt = f"""
            System memory usage is at {self.metrics.get('memory_percent', 0)}%.
            Available memory: {self.metrics.get('memory_available', 0)} bytes.
            Suggest specific optimization actions.
            """
            
            suggestions = await self.process_with_ollama(optimization_prompt)
            
            return {
                "status": "success",
                "action": "memory_optimization",
                "suggestions": suggestions,
                "current_usage": self.metrics.get('memory_percent', 0)
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}"
            }

# FastAPI app setup
agent = HardwareResourceOptimizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()
    # Start background metrics collection
    asyncio.create_task(periodic_metrics_collection())
    yield
    await agent.shutdown()

async def periodic_metrics_collection():
    """Collect metrics every 30 seconds"""
    while True:
        await agent.collect_metrics()
        await asyncio.sleep(30)

app = FastAPI(
    title="Hardware Resource Optimizer",
    description="Real system resource monitoring and optimization",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": agent.status,
        "agent": agent.agent_id,
        "timestamp": datetime.utcnow().isoformat(),
        "last_metrics": agent.metrics.get("timestamp", "never")
    }

@app.post("/process")
async def process_request(request: Dict[str, Any]):
    return await agent.process_request(request)

@app.get("/metrics")
async def get_current_metrics():
    """Get current system metrics"""
    await agent.collect_metrics()
    return agent.metrics

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

## Integration Patterns

### Agent-to-Agent Communication

```python
class AgentClient:
    """Helper class for agent communication"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to another agent"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{endpoint}",
                json=data,
                timeout=30.0
            )
            return response.json()
    
    async def health_check(self) -> bool:
        """Check if agent is healthy"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
        except:
            return False

# Usage in agents
hardware_optimizer = AgentClient("http://hardware-resource-optimizer:8002")
task_coordinator = AgentClient("http://task-assignment-coordinator:8551")
```

### Redis Message Patterns

```python
class RedisMessaging:
    """Redis-based messaging for agent coordination"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def publish_event(self, channel: str, data: Dict[str, Any]):
        """Publish event to Redis channel"""
        await self.redis.publish(
            channel,
            json.dumps(data)
        )
    
    async def subscribe_to_events(self, channels: List[str], handler):
        """Subscribe to Redis channels"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(*channels)
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    await handler(message['channel'], data)
                except Exception as e:
                    logging.error(f"Event handling failed: {e}")
```

## Testing Strategies

### Unit Testing Agent Logic

```python
import pytest
from unittest.Mock import AsyncMock, MagicMock

class TestHardwareResourceOptimizer:
    
    @pytest.fixture
    async def agent(self):
        agent = HardwareResourceOptimizer()
        agent.redis_client = AsyncMock()
        agent.status = "healthy"
        return agent
    
    async def test_collect_metrics(self, agent):
        """Test metrics collection"""
        await agent.collect_metrics()
        
        assert "cpu_percent" in agent.metrics
        assert "memory_percent" in agent.metrics
        assert agent.metrics["cpu_percent"] >= 0
    
    async def test_process_get_metrics(self, agent):
        """Test metrics request processing"""
        result = await agent.process_request({"action": "get_metrics"})
        
        assert result["status"] == "success"
        assert "metrics" in result

# Run with: pytest tests/test_hardware_optimizer.py
```

### Integration Testing

```python
async def test_agent_communication():
    """Test communication between agents"""
    
    # Test orchestrator -> hardware optimizer
    orchestrator_client = AgentClient("http://localhost:8589")
    optimizer_client = AgentClient("http://localhost:8002")
    
    # Check both agents are healthy
    assert await orchestrator_client.health_check()
    assert await optimizer_client.health_check()
    
    # Request metrics from orchestrator
    response = await orchestrator_client.send_request(
        "process",
        {
            "action": "get_system_metrics",
            "target_agent": "hardware-resource-optimizer"
        }
    )
    
    assert response["status"] == "success"
```

### Load Testing

```bash
#!/bin/bash
# Simple load test for agent endpoints

echo "Testing Hardware Resource Optimizer..."
for i in {1..100}; do
    curl -s -X POST http://localhost:8002/process \
         -H "Content-Type: application/json" \
         -d '{"action": "get_metrics"}' > /dev/null &
done
wait

echo "Load test completed"
```

## Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

class AgentMetrics:
    """Prometheus metrics for agents"""
    
    def __init__(self, agent_id: str):
        self.requests_total = Counter(
            'agent_requests_total',
            'Total agent requests',
            ['agent_id', 'endpoint']
        )
        
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Agent request duration',
            ['agent_id', 'endpoint']
        )
        
        self.agent_health = Gauge(
            'agent_health_status',
            'Agent health status (1=healthy, 0=degraded)',
            ['agent_id']
        )
        
        self.agent_id = agent_id
    
    def record_request(self, endpoint: str):
        """Record a request"""
        self.requests_total.labels(
            agent_id=self.agent_id,
            endpoint=endpoint
        ).inc()
    
    def record_duration(self, endpoint: str, duration: float):
        """Record request duration"""
        self.request_duration.labels(
            agent_id=self.agent_id,
            endpoint=endpoint
        ).observe(duration)
    
    def set_health(self, healthy: bool):
        """Set health status"""
        self.agent_health.labels(agent_id=self.agent_id).set(1 if healthy else 0)

# Add to FastAPI app
@app.get("/metrics")
async def prometheus_metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

### Logging Standards

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in agents
logger.info("Processing request", 
           agent_id=self.agent_id,
           request_id=request_id,
           action=action)
```

## Best Practices

### Idempotency

```python
async def idempotent_process(self, request_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure requests are idempotent"""
    
    # Check if already processed
    cache_key = f"request:{request_id}"
    cached_result = await self.redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    # Process request
    result = await self._actual_process(data)
    
    # Cache result
    await self.redis_client.setex(
        cache_key,
        3600,  # 1 hour TTL
        json.dumps(result)
    )
    
    return result
```

### Retry Logic

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def reliable_agent_call(agent_client: AgentClient, endpoint: str, data: Dict) -> Dict:
    """Reliable agent call with retries"""
    return await agent_client.send_request(endpoint, data)
```

### Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker for agent calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        
        if self.state == "open":
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is open")
            else:
                self.state = "half-open"
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            self.state = "closed"
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e
```

### Graceful Degradation

```python
async def process_with_fallback(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """Process with graceful degradation"""
    
    try:
        # Try AI processing first
        if await self.test_ollama():
            return await self.process_with_ai(request)
    except Exception as e:
        logger.warning(f"AI processing failed: {e}")
    
    # Fallback to rule-based processing
    logger.info("Falling back to rule-based processing")
    return await self.process_with_rules(request)
```

## Migration Plan

### Week 1: Foundation
- [ ] Convert Hardware Resource Optimizer to FastAPI + BaseAgent
- [ ] Implement real system monitoring with psutil
- [ ] Add Redis integration for metrics storage
- [ ] Create Prometheus metrics endpoint

### Week 2-3: Core Orchestration  
- [ ] Convert AI Agent Orchestrator to FastAPI
- [ ] Implement agent registry in Redis
- [ ] Add task routing with Ollama integration
- [ ] Convert Task Assignment Coordinator
- [ ] Implement Redis-based task queue

### Week 4-5: Resource Management
- [ ] Convert Resource Arbitration Agent
- [ ] Implement resource monitoring and allocation
- [ ] Add conflict detection and resolution
- [ ] Integration with Hardware Optimizer

### Week 6: Remaining Agents
- [ ] Fix AI Metrics Exporter (currently failing)
- [ ] Convert Multi-Agent Coordinator
- [ ] Convert Ollama Integration Specialist
- [ ] Implement health monitoring system

### Week 7-8: Integration & Testing
- [ ] End-to-end integration testing
- [ ] Load testing and performance optimization
- [ ] Documentation updates
- [ ] Deployment verification

## Current System Constraints

### Model Reality
- **Available**: TinyLlama 637MB (loaded and working)
- **Expected by code**: gpt-oss (not present)
- **Action**: Update all agent code to use `tinyllama` model

### Database State
- **PostgreSQL**: Empty, no schema created
- **Action**: Create agent tables for state persistence
- **Redis**: Working, ready for agent data

### Message Queue
- **RabbitMQ**: Running but not integrated
- **Action**: Implement actual message consumption in agents

### Resource Allocation
- **Current**: No resource monitoring
- **Realistic**: Basic CPU, memory, disk monitoring with psutil
- **conceptual**: Complex distributed resource allocation

## Deployment Commands

### Building Updated Agent Images

```bash
# Build individual agent
docker build -f agents/hardware-resource-optimizer/Dockerfile \
             -t sutazai/hardware-optimizer:2.0 \
             agents/hardware-resource-optimizer/

# Update docker-compose for new version
docker-compose up -d hardware-resource-optimizer
```

### Verification Commands

```bash
# Test converted agent
curl http://localhost:8002/health
curl -X POST http://localhost:8002/process \
     -H "Content-Type: application/json" \
     -d '{"action": "get_metrics"}'

# Check Prometheus metrics
curl http://localhost:8002/metrics
```

### Health Monitoring

```bash
# Monitor all agent health
for port in 8002 8551 8587 8588 8589 11015 11063; do
    echo "Testing port $port..."
    curl -s http://localhost:$port/health | jq '.status'
done
```

## Future Enhancements

### Near-term (Next 3 months)
1. **Model Management**: Support multiple Ollama models
2. **Advanced Routing**: Machine learning-based agent selection
3. **Batch Processing**: Efficient handling of multiple requests
4. **Caching Layer**: Redis-based response caching

### Long-term (6+ months)
1. **Multi-cluster Support**: Agent deployment across multiple nodes
2. **GraphQL APIs**: Alternative to REST for complex queries
3. **Advanced Analytics**: Performance prediction and optimization
4. **Self-healing**: Automatic agent recovery and scaling

## Conclusion

This implementation guide provides a **realistic, step-by-step approach** to transform the current Flask stub agents into functional services. By focusing on:

1. **One agent at a time** (incremental approach)
2. **Real functionality** (actual system monitoring, task processing)
3. **Existing infrastructure** (Redis, Ollama, PostgreSQL)
4. **Practical patterns** (FastAPI, async/await, proper error handling)

We can build a working agent system that provides genuine value rather than maintaining elaborate stubs.

The key is to **start small**, **test frequently**, and **build incrementally** on the solid Docker Compose foundation that already exists.

---

**Next Actions:**
1. Begin with Hardware Resource Optimizer (Week 1)
2. Document any issues encountered during conversion
3. Update this guide based on actual implementation experience
4. Share learnings with the team for subsequent agent conversions