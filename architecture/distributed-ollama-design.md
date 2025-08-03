# Distributed Ollama Architecture Design for 131 AI Agents

## Executive Summary

This document presents a comprehensive distributed system architecture for managing 131 AI agents (36 Opus, 95 Sonnet) with limited Ollama capacity (OLLAMA_NUM_PARALLEL=2) on a WSL2 environment with 48GB RAM and 4GB GPU. The design emphasizes fault tolerance, efficient resource utilization, and graceful degradation under load.

## System Constraints

### Hardware Limitations
- **Total Memory**: 48GB RAM (WSL2)
- **GPU Memory**: 4GB VRAM
- **Ollama Parallelism**: 2 concurrent requests max
- **Network**: Docker bridge network (172.20.0.0/16)

### Agent Distribution
- **Total Agents**: 131 containers
  - Opus Agents: 36 (higher compute requirements)
  - Sonnet Agents: 95 (standard compute requirements)
- **Models**: tinyllama, qwen2.5-coder:7b, deepseek-r1:8b
- **Peak Load**: ~131 concurrent requests possible

## Architectural Patterns

### 1. Service Mesh Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Consul Service Registry                  │
│  ┌─────────────────┬─────────────────┬─────────────────┐  │
│  │ Agent Discovery │ Health Checking │ Load Balancing  │  │
│  └─────────────────┴─────────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│ Sidecar Proxy  │   │ Sidecar Proxy   │   │ Sidecar Proxy  │
│   (Envoy)      │   │   (Envoy)       │   │   (Envoy)      │
├────────────────┤   ├─────────────────┤   ├────────────────┤
│ Agent Instance │   │ Agent Instance  │   │ Agent Instance │
│    (Opus)      │   │   (Sonnet)      │   │   (Sonnet)     │
└────────────────┘   └─────────────────┘   └────────────────┘
```

### 2. Request Queue Management

```yaml
Queue Architecture:
  Primary Queue:
    - Type: Redis Streams
    - Partitions: 10 (based on agent hash)
    - Consumer Groups: 3 (high/medium/low priority)
    - Max Length: 10,000 messages
    - TTL: 5 minutes
    
  Overflow Queue:
    - Type: Redis List
    - Capacity: 50,000 messages
    - Spillover threshold: 80% primary queue
    
  Dead Letter Queue:
    - Failed requests after 3 retries
    - Manual intervention required
```

### 3. Ollama Gateway Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    Ollama Gateway Service                    │
├─────────────────────────────────────────────────────────────┤
│  Request Router │ Rate Limiter │ Circuit Breaker │ Cache   │
├─────────────────┴────────────────┴──────────────────┴───────┤
│                     Connection Pool (size: 10)               │
└──────────────────────────┬──────────────────────────────────┘
                          │
                ┌─────────▼─────────┐
                │  Ollama Service   │
                │ (PARALLEL_LIMIT=2)│
                └───────────────────┘
```

## Component Design

### 1. Distributed Request Coordinator

```python
class DistributedRequestCoordinator:
    """
    Manages request distribution across 131 agents with Ollama constraints
    """
    def __init__(self):
        self.redis_client = Redis(connection_pool=...)
        self.consul_client = Consul()
        self.circuit_breakers = {}
        self.request_queues = self._initialize_queues()
        
    async def submit_request(self, agent_id: str, request: OllamaRequest):
        # Partition based on agent_id hash
        partition = hash(agent_id) % 10
        queue_key = f"ollama:queue:{partition}"
        
        # Check circuit breaker state
        if self._is_circuit_open(agent_id):
            return await self._handle_circuit_open(request)
            
        # Add to appropriate queue with priority
        priority = self._calculate_priority(agent_id, request)
        await self.redis_client.xadd(
            queue_key,
            {
                'agent_id': agent_id,
                'request': request.json(),
                'priority': priority,
                'timestamp': time.time()
            }
        )
```

### 2. Load Balancing Strategy

```yaml
Load Balancing Algorithm:
  Primary Strategy: Weighted Round Robin
    - Opus agents: weight = 2
    - Sonnet agents: weight = 1
    
  Secondary Strategy: Least Connections
    - Track active connections per Ollama instance
    - Route to instance with fewest connections
    
  Fallback Strategy: Random Selection
    - Used when primary strategies fail
    - Prevents hotspots during failures
```

### 3. Caching Layer

```python
class DistributedOllamaCache:
    """
    Multi-tier caching for Ollama responses
    """
    def __init__(self):
        # L1: In-memory cache per agent (10MB limit)
        self.l1_cache = LRUCache(max_size_mb=10)
        
        # L2: Redis cache shared across agents (1GB limit)
        self.l2_cache = RedisCache(
            max_memory="1gb",
            eviction_policy="allkeys-lru"
        )
        
        # L3: Disk cache for large responses (10GB limit)
        self.l3_cache = DiskCache(
            directory="/var/cache/ollama",
            size_limit_gb=10
        )
        
    async def get_or_compute(self, key: str, compute_fn):
        # Check caches in order
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            value = await cache.get(key)
            if value:
                return value
                
        # Compute and store in all tiers
        value = await compute_fn()
        await self._store_in_tiers(key, value)
        return value
```

## Failure Handling

### 1. Circuit Breaker Configuration

```yaml
Circuit Breaker Settings:
  Failure Threshold: 5 failures in 60 seconds
  Recovery Timeout: 60 seconds
  Half-Open Requests: 1
  
  States:
    - CLOSED: Normal operation
    - OPEN: All requests fail fast
    - HALF_OPEN: Limited requests for testing
```

### 2. Bulkhead Pattern

```python
class OllamaBulkhead:
    """
    Isolates failures to prevent cascade
    """
    def __init__(self):
        self.bulkheads = {
            'opus': asyncio.Semaphore(10),      # Max 10 concurrent Opus
            'sonnet': asyncio.Semaphore(20),    # Max 20 concurrent Sonnet
            'system': asyncio.Semaphore(5)      # Max 5 system requests
        }
        
    async def execute(self, agent_type: str, fn):
        bulkhead = self.bulkheads.get(agent_type, self.bulkheads['system'])
        async with bulkhead:
            return await fn()
```

### 3. Graceful Degradation

```yaml
Degradation Levels:
  Level 0 (Normal):
    - All features enabled
    - Full model selection
    
  Level 1 (Minor Load):
    - Reduce max tokens to 1024
    - Increase cache TTL by 2x
    
  Level 2 (Major Load):
    - Switch all to tinyllama model
    - Enable aggressive caching
    - Disable non-critical agents
    
  Level 3 (Critical):
    - Queue all non-essential requests
    - Process only high-priority
    - Return cached/default responses
```

## Network Optimization

### 1. Connection Pooling

```python
class OllamaConnectionPool:
    """
    Manages HTTP connections efficiently
    """
    def __init__(self):
        self.pools = {
            'ollama-primary': aiohttp.TCPConnector(
                limit=10,
                limit_per_host=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            ),
            'ollama-fallback': aiohttp.TCPConnector(
                limit=5,
                limit_per_host=5
            )
        }
```

### 2. Request Batching

```python
class RequestBatcher:
    """
    Batches multiple small requests
    """
    def __init__(self):
        self.batch_size = 5
        self.batch_timeout = 100  # ms
        self.pending_batches = defaultdict(list)
        
    async def add_request(self, request):
        batch_key = self._get_batch_key(request)
        batch = self.pending_batches[batch_key]
        batch.append(request)
        
        if len(batch) >= self.batch_size:
            return await self._process_batch(batch_key)
        else:
            # Wait for more requests or timeout
            asyncio.create_task(
                self._timeout_batch(batch_key, self.batch_timeout)
            )
```

## Resource Management

### 1. Memory Management

```yaml
Memory Allocation:
  System Reserved: 3GB
  Ollama Service: 8GB
  Redis Cache: 2GB
  Agent Containers: 35GB total
    - Opus agents: 36 * 300MB = 10.8GB
    - Sonnet agents: 95 * 256MB = 24.3GB
```

### 2. GPU Utilization

```yaml
GPU Strategy:
  Model Loading:
    - Keep 2 models max in VRAM
    - LRU eviction for model swapping
    
  Scheduling:
    - Priority queue for GPU requests
    - Time-slicing for fairness
    
  Monitoring:
    - Track VRAM usage
    - Automatic model unloading at 90% usage
```

## Monitoring and Observability

### 1. Key Metrics

```yaml
System Metrics:
  - ollama_request_duration_seconds
  - ollama_queue_depth
  - ollama_circuit_breaker_state
  - agent_memory_usage_bytes
  - cache_hit_ratio
  - model_load_time_seconds
  
Business Metrics:
  - requests_per_agent_type
  - successful_completions
  - timeout_rate
  - error_rate_by_model
```

### 2. Distributed Tracing

```python
class DistributedTracer:
    """
    Traces requests across the system
    """
    def __init__(self):
        self.tracer = opentelemetry.trace.get_tracer(__name__)
        
    async def trace_ollama_request(self, agent_id, request):
        with self.tracer.start_as_current_span(
            "ollama_request",
            attributes={
                "agent.id": agent_id,
                "model.name": request.model,
                "request.tokens": len(request.prompt)
            }
        ) as span:
            # Trace through the entire pipeline
            yield span
```

## Deployment Considerations

### 1. Rolling Updates

```yaml
Update Strategy:
  - Update 10% of agents at a time
  - Wait for health checks (5 minutes)
  - Monitor error rates
  - Automatic rollback on failure
```

### 2. Configuration Management

```yaml
Dynamic Configuration:
  - Store in Consul KV
  - Watch for changes
  - Apply without restart
  
  Configurable Parameters:
    - Queue sizes
    - Timeout values
    - Circuit breaker thresholds
    - Cache policies
```

## Security Considerations

### 1. Network Isolation

```yaml
Network Policies:
  - Agents can only access Ollama through gateway
  - No direct agent-to-agent communication
  - Encrypted Redis connections
  - mTLS for service mesh
```

### 2. Rate Limiting

```yaml
Rate Limits:
  Per Agent:
    - Opus: 10 requests/minute
    - Sonnet: 20 requests/minute
    
  Global:
    - 120 requests/minute total
    - Burst allowance: 150%
```

## Conclusion

This distributed architecture provides a robust foundation for managing 131 AI agents with limited Ollama resources. The design emphasizes:

1. **Fault Tolerance**: Multiple failure handling mechanisms
2. **Efficiency**: Optimal resource utilization through caching and pooling
3. **Scalability**: Ability to handle load spikes gracefully
4. **Observability**: Comprehensive monitoring and tracing
5. **Flexibility**: Dynamic configuration and degradation strategies

The architecture can handle all agents making concurrent requests while respecting the OLLAMA_NUM_PARALLEL=2 constraint through intelligent queuing, caching, and load distribution.