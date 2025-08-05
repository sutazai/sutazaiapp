# Distributed AI System Architecture Rules

## Version 1.0 - Production Architecture Standards

These rules are mandatory for all components in the SutazAI distributed AI system. They address resource management, stability, performance, and operational excellence.

---

## 🏗️ Rule 1: Resource Management and Allocation

### 1.1 Memory Limits are Mandatory
✅ **EVERY container MUST have explicit memory limits defined**
- No container may run without memory limits
- Limits must be set based on actual usage patterns + 20% buffer
- Memory requests must be 50-75% of limits to allow burst capacity

```yaml
# REQUIRED format for all containers
resources:
  limits:
    memory: "2Gi"    # Hard limit
    cpu: "1.0"       # CPU limit
  requests:
    memory: "1Gi"    # Guaranteed allocation
    cpu: "0.5"       # CPU request
```

### 1.2 CPU Affinity Strategy
✅ **Services MUST be assigned to CPU cores based on their type**
- Infrastructure services: Cores 0-3 (PostgreSQL, Redis, etc.)
- Critical agents: Cores 4-7
- Standard agents: Cores 8-11
- Monitoring/auxiliary: Any available core

### 1.3 Memory Pool Architecture
✅ **Memory MUST be allocated from defined pools**
- Infrastructure Pool: 12GB (databases, message queues)
- Agent Critical Pool: 8GB (orchestration, core AI)
- Agent Standard Pool: 6GB (specialized agents)
- Monitoring Pool: 2GB
- Emergency Reserve: 1.38GB (NEVER allocate)

### 1.4 Resource Monitoring Requirements
✅ **All services MUST expose resource metrics**
- Memory usage via /metrics endpoint
- CPU usage percentage
- Disk I/O metrics
- Network bandwidth usage

---

## 🚀 Rule 2: Container Orchestration and Deployment

### 2.1 Phased Deployment Strategy
✅ **Agents MUST be deployed in priority phases**
- Phase 1: Critical infrastructure and core agents (max 15)
- Phase 2: Performance enhancement agents (max 25)
- Phase 3: Specialized function agents (max 70)
- Wait for health confirmation between phases

### 2.2 Health Check Requirements
✅ **Every container MUST implement health checks**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:${PORT}/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 2.3 Restart Policy Guidelines
✅ **Restart policies MUST prevent restart loops**
```yaml
restart: unless-stopped
deploy:
  restart_policy:
    condition: on-failure
    delay: 30s
    max_attempts: 3
    window: 300s
```

### 2.4 Container Dependencies
✅ **Dependencies MUST be explicitly declared**
```yaml
depends_on:
  redis:
    condition: service_healthy
  postgres:
    condition: service_healthy
```

---

## 🌐 Rule 3: Service Mesh Communication

### 3.1 Service Discovery Requirements
✅ **All services MUST register with Consul**
- Health endpoint mandatory
- Service tags for categorization
- Automatic deregistration on failure

### 3.2 API Gateway Routing
✅ **All external traffic MUST go through Kong API Gateway**
- No direct port exposure except gateway
- Rate limiting per service
- Request/response logging

### 3.3 Message Queue Patterns
✅ **Asynchronous operations MUST use RabbitMQ**
- Priority queues for critical operations
- Dead letter queues for failed messages
- Message TTL configuration

### 3.4 Circuit Breaker Implementation
✅ **Inter-service calls MUST implement circuit breakers**
- Open circuit after 5 consecutive failures
- Half-open retry after 30 seconds
- Fallback responses required

---

## 💾 Rule 4: Memory and CPU Optimization

### 4.1 Model Loading Strategy
✅ **AI models MUST use lazy loading with pooling**
```python
# Required pattern for model loading
class ModelManager:
    def __init__(self, max_memory_gb=8):
        self.memory_limit = max_memory_gb * 1024 * 1024 * 1024
        self.models = OrderedDict()  # LRU cache
        
    def get_model(self, name):
        if name in self.models:
            # Move to end (most recently used)
            self.models.move_to_end(name)
            return self.models[name]
        
        # Evict if necessary
        while self._get_memory_usage() + self._get_model_size(name) > self.memory_limit:
            self._evict_lru()
            
        return self._load_model(name)
```

### 4.2 Shared Library Volume
✅ **Common dependencies MUST be shared via volumes**
- Python packages in /opt/shared/python
- Model files in /opt/shared/models
- Configuration in /opt/shared/config

### 4.3 Memory-Mapped Operations
✅ **Large data operations MUST use memory mapping**
- Model weights via mmap
- Large datasets via numpy memmap
- Shared memory for inter-process communication

---

## 🔌 Rule 5: Port Management

### 5.1 Port Allocation Strategy
✅ **Ports MUST be allocated from designated ranges**
- 10000-10099: Infrastructure services
- 10100-10199: Core AI agents
- 10200-10399: Standard agents
- 10400-10499: Monitoring/metrics
- 10500-10599: Development/debug

### 5.2 Port Conflict Prevention
✅ **No service may use common ports**
- Forbidden: 80, 443, 8080, 8000, 3000, 5000
- Exception: Gateway services only

### 5.3 Dynamic Port Assignment
✅ **Services MUST support PORT environment variable**
```python
port = int(os.getenv('PORT', '10100'))
```

---

## 🔄 Rule 6: Agent Lifecycle Management

### 6.1 Graceful Shutdown
✅ **All agents MUST implement graceful shutdown**
```python
def signal_handler(signum, frame):
    logger.info("Shutdown signal received")
    # Save state
    save_current_state()
    # Close connections
    close_all_connections()
    # Exit cleanly
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### 6.2 State Persistence
✅ **Agent state MUST be persisted**
- State saved every 5 minutes
- State saved on shutdown
- State restored on startup

### 6.3 Idle Timeout Management
✅ **Agents MUST implement idle timeout**
- Hibernate after 5 minutes of inactivity
- Release resources when hibernating
- Quick wake-up on demand (<30s)

### 6.4 Resource Release Protocol
✅ **Resources MUST be released properly**
- Close database connections
- Release memory allocations
- Cleanup temporary files
- Unregister from service mesh

---

## 📊 Rule 7: Monitoring and Observability

### 7.1 Metrics Exposure
✅ **All services MUST expose Prometheus metrics**
```python
# Required metrics
- process_cpu_seconds_total
- process_resident_memory_bytes
- http_requests_total
- http_request_duration_seconds
- agent_tasks_completed_total
- agent_tasks_failed_total
```

### 7.2 Structured Logging
✅ **Logs MUST use structured JSON format**
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "INFO",
  "service": "agent-name",
  "message": "Task completed",
  "task_id": "123",
  "duration_ms": 1500,
  "memory_used_mb": 256
}
```

### 7.3 Distributed Tracing
✅ **Requests MUST include trace headers**
- X-Trace-ID for request correlation
- X-Span-ID for operation tracking
- X-Parent-Span-ID for call hierarchy

### 7.4 Health Dashboard Requirements
✅ **System MUST maintain real-time health dashboard**
- Service status visualization
- Resource usage graphs
- Alert status display
- Performance metrics

---

## ⚡ Rule 8: Performance and Scaling

### 8.1 Response Time SLAs
✅ **Services MUST meet response time SLAs**
- Health checks: <1 second
- API endpoints: <5 seconds
- Async operations: Acknowledge <1 second

### 8.2 Concurrent Request Limits
✅ **Services MUST implement request limits**
```python
# Required implementation
from asyncio import Semaphore

class ServiceHandler:
    def __init__(self, max_concurrent=10):
        self.semaphore = Semaphore(max_concurrent)
        
    async def handle_request(self, request):
        async with self.semaphore:
            return await self._process(request)
```

### 8.3 Caching Strategy
✅ **Responses MUST be cached where appropriate**
- Redis for shared cache
- Local cache for frequently accessed data
- Cache TTL based on data volatility

### 8.4 Load Balancing Rules
✅ **Load MUST be distributed evenly**
- Round-robin for stateless services
- Least-connections for stateful services
- Health-based routing

---

## 🔒 Rule 9: Security and Isolation

### 9.1 Network Segmentation
✅ **Services MUST use network isolation**
```yaml
networks:
  infrastructure:
    internal: true
  agents:
    internal: true
  monitoring:
    internal: true
  gateway:
    external: true
```

### 9.2 Secret Management
✅ **Secrets MUST never be hardcoded**
- Use environment variables
- Mount from secret volumes
- Rotate credentials regularly

### 9.3 Container Security
✅ **Containers MUST run as non-root**
```dockerfile
USER 1000:1000
```

---

## 🚨 Rule 10: Error Handling and Recovery

### 10.1 Comprehensive Error Handling
✅ **All operations MUST have error handling**
```python
try:
    result = perform_operation()
except OperationError as e:
    logger.error(f"Operation failed: {e}")
    metrics.increment('operation_failures')
    return fallback_response()
finally:
    cleanup_resources()
```

### 10.2 Retry Strategy
✅ **Failed operations MUST implement exponential backoff**
- Initial retry: 1 second
- Max retries: 3
- Backoff multiplier: 2

### 10.3 Failure Isolation
✅ **Failures MUST not cascade**
- Timeouts on all external calls
- Circuit breakers for dependencies
- Graceful degradation

---

## 📋 Rule 11: Operational Excellence

### 11.1 Documentation Requirements
✅ **Every service MUST have**
- README with setup instructions
- API documentation
- Configuration reference
- Troubleshooting guide

### 11.2 Automated Testing
✅ **Services MUST have test coverage**
- Unit tests: >80% coverage
- Integration tests for APIs
- Load tests for performance

### 11.3 Deployment Automation
✅ **Deployments MUST be automated**
- Infrastructure as Code
- Automated rollback capability
- Blue-green deployment support

### 11.4 Incident Response
✅ **Services MUST support debugging**
- Debug endpoints (protected)
- Heap dump capability
- Request tracing

---

## 🎯 Rule 12: Specific Issue Resolutions

### 12.1 Ollama CPU Usage
✅ **Ollama MUST be configured with limits**
```yaml
ollama:
  environment:
    OLLAMA_NUM_PARALLEL: 2
    OLLAMA_MAX_LOADED_MODELS: 1
    OLLAMA_CPU_LIMIT: 4
```

### 12.2 Port 8080 Conflicts
✅ **Port 8080 is RESERVED for gateway only**
- No service may bind to 8080
- Use allocated port ranges

### 12.3 Claude Instance Management
✅ **Multiple Claude instances MUST be pooled**
- Single shared instance preferred
- Connection pooling if multiple needed
- Memory limit per instance: 800MB

---

## 📐 Implementation Checklist

Before deploying any service, verify:

- [ ] Memory limits configured
- [ ] CPU limits set appropriately  
- [ ] Health check implemented
- [ ] Metrics endpoint exposed
- [ ] Service mesh registration
- [ ] Port from allocated range
- [ ] Graceful shutdown handler
- [ ] Error handling comprehensive
- [ ] Logs structured JSON
- [ ] Documentation complete
- [ ] Tests passing
- [ ] Security scan passed

---

## 🔄 Version History

- v1.0 (2025-01-15): Initial comprehensive rules based on system analysis
  - Addressed Ollama CPU usage
  - Fixed port conflicts
  - Mandated memory limits
  - Established service mesh patterns
  - Defined monitoring requirements

---

## 📞 Enforcement

These rules are enforced through:
1. Pre-deployment validation scripts
2. Continuous monitoring alerts
3. Automated compliance reports
4. Pull request checks

Non-compliance will result in:
1. Deployment prevention
2. Automated rollback
3. Alert escalation
4. Required remediation

---

Remember: **These rules ensure system stability, performance, and maintainability. They are not optional.**