# 🚀 SutazAI Distributed System Rules v2.0
## Comprehensive Engineering Standards for 69-Agent AI Platform

**System Context**: 69 AI agents, 12 CPU cores, 29GB RAM, Distributed Architecture  
**Last Updated**: August 5, 2025  
**Compliance Level**: MANDATORY

---

## 🏗️ FOUNDATION RULES (Original + Enhanced)

### 📌 Rule 1: No Fantasy Elements - Production Reality Only
✨ **Original**: Only real, production-ready implementations  
✨ **Enhanced for Distributed System**:
- All agent communications must use actual service discovery (Consul), not hardcoded IPs
- Resource allocations must respect actual system limits (12 cores, 29GB RAM)
- Use real monitoring endpoints (/health, /metrics), not placeholder implementations
- Implement actual circuit breakers and retry logic, not "TODO: add resilience"

**Enforcement**: Pre-commit hooks scan for banned terms (magic, wizard, placeholder, TODO)

---

### 📌 Rule 2: Zero Regression Tolerance - Distributed Stability
✨ **Original**: Do not break existing functionality  
✨ **Enhanced for 69 Agents**:
- Changes must be tested across all three agent phases (Critical/Performance/Specialized)
- Rolling deployments with automatic rollback on failure detection
- Canary testing: 10% of agents before full rollout
- Version compatibility matrix for inter-agent dependencies

**Enforcement**: Automated integration tests across agent clusters

---

### 📌 Rule 3: Distributed System Analysis Protocol
✨ **Original**: Analyze everything every time  
✨ **Enhanced for Complexity**:
```yaml
analysis_checklist:
  - All 69 agent health statuses
  - Service mesh topology and communication patterns
  - Resource allocation across phases (Critical: 8 CPU/16GB, Performance: 6 CPU/12GB, Specialized: 4 CPU/8GB)
  - Port usage within 10000-10599 range
  - Distributed state consistency
  - Network partition risks
```

**Tool**: `./scripts/distributed-analysis.py --comprehensive`

---

## 🌐 DISTRIBUTED COMPUTING RULES

### 📌 Rule 17: Container Lifecycle Management
✨ **Mandatory Standards**:
```yaml
container_requirements:
  startup:
    - Dependency health checks before ready
    - Graceful startup with exponential backoff
    - Service registration with Consul
  runtime:
    - Health endpoint: /health (JSON response)
    - Metrics endpoint: /metrics (Prometheus format)
    - Structured logging to stdout (JSON)
  shutdown:
    - SIGTERM handling with 30s grace period
    - Deregister from service discovery
    - Drain in-flight requests
```

---

### 📌 Rule 18: Service Mesh Communication
✨ **Inter-Agent Protocol**:
```python
# Mandatory communication pattern
async def call_agent(target_agent: str, payload: dict) -> dict:
    try:
        # Service discovery
        endpoint = await consul.health.service(target_agent)[0]
        
        # Circuit breaker
        if circuit_breaker.is_open(target_agent):
            return await fallback_response()
        
        # Timeout and retry
        response = await http_client.post(
            endpoint,
            json=payload,
            timeout=30,
            retry=ExponentialBackoff(max_attempts=3)
        )
        
        # Distributed tracing
        span.set_tag("target.agent", target_agent)
        span.set_tag("response.status", response.status)
        
        return response.json()
    except Exception as e:
        circuit_breaker.record_failure(target_agent)
        raise
```

---

### 📌 Rule 19: Distributed State Management
✨ **State Consistency Requirements**:
- **Strong Consistency**: Financial data, authentication → PostgreSQL with transactions
- **Eventual Consistency**: Analytics, metrics → Redis with TTL
- **Graph Relationships**: Agent dependencies → Neo4j
- **Vector Embeddings**: ChromaDB/Qdrant/FAISS with versioning

**No container-local state except explicit volume mounts**

---

## 💾 RESOURCE MANAGEMENT RULES

### 📌 Rule 20: Phase-Based Resource Allocation
✨ **Mandatory Limits**:
```yaml
resource_phases:
  critical_agents:  # Ports 10300-10319
    cpu_limit: "2"
    memory_limit: "4Gi"
    restart_policy: "always"
    priority_class: "critical"
    
  performance_agents:  # Ports 10320-10419
    cpu_limit: "1"
    memory_limit: "2Gi"
    restart_policy: "on-failure"
    priority_class: "high"
    
  specialized_agents:  # Ports 10420-10599
    cpu_limit: "0.5"
    memory_limit: "1Gi"
    restart_policy: "unless-stopped"
    priority_class: "standard"
```

---

### 📌 Rule 21: Memory Pool Architecture
✨ **Dynamic Allocation**:
```python
MEMORY_POOLS = {
    "infrastructure": {"size": "12Gi", "services": ["postgres", "neo4j", "ollama"]},
    "agent_critical": {"size": "8Gi", "max_per_agent": "512Mi", "dynamic": True},
    "agent_standard": {"size": "6Gi", "max_per_agent": "256Mi", "shared": True},
    "monitoring": {"size": "2Gi", "services": ["prometheus", "grafana", "loki"]},
    "emergency": {"size": "1.38Gi", "purpose": "OOM prevention"}
}
```

---

## 🤖 AI SYSTEM RULES

### 📌 Rule 22: Ollama Optimization (Critical Fix)
✨ **CPU Overload Prevention**:
```yaml
ollama_config:
  OLLAMA_NUM_PARALLEL: 1      # Reduce from default
  OLLAMA_NUM_THREADS: 4       # Match CPU affinity
  OLLAMA_MAX_LOADED_MODELS: 1 # Single model in memory
  OLLAMA_KEEP_ALIVE: 30s      # Aggressive unloading
  
  # Connection pooling
  connection_pool:
    max_size: 10
    timeout: 30s
    queue_timeout: 60s
```

---

### 📌 Rule 23: Agent Intelligence Standards
✨ **Capability Declaration**:
```json
{
  "agent_id": "senior-ai-engineer",
  "capabilities": {
    "cognitive": ["reasoning", "planning", "code_generation"],
    "performance_tier": "critical",
    "resource_requirements": {
      "ollama_priority": "high",
      "max_context_size": 4096,
      "timeout": 120
    }
  }
}
```

---

### 📌 Rule 24: Prompt Engineering Standards
✨ **Optimization for Shared Ollama**:
```python
PROMPT_TEMPLATE = {
    "system": "concise_role",  # < 100 tokens
    "context": "essential_only",  # < 500 tokens
    "task": "specific_measurable",  # < 200 tokens
    "constraints": {
        "max_tokens": 512,
        "temperature": 0.7,
        "stop_sequences": ["</response>", "\n\n"]
    }
}
```

---

## 🔧 INFRASTRUCTURE RULES

### 📌 Rule 25: Deployment Automation
✨ **Single Command Deployment**:
```bash
# Enhanced deploy.sh with distributed system support
./deploy.sh --env production --phase all

# Phased deployment for safety
./deploy.sh --phase infrastructure  # Core services first
./deploy.sh --phase critical       # Critical agents
./deploy.sh --phase performance    # Performance agents
./deploy.sh --phase specialized    # Remaining agents
```

---

### 📌 Rule 26: Container Security
✨ **Mandatory Hardening**:
```dockerfile
# Required in all Dockerfiles
USER nonroot:nonroot
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1
  
# Security options in docker-compose.yml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
```

---

## 📊 MONITORING & OBSERVABILITY RULES

### 📌 Rule 27: Comprehensive Metrics
✨ **Every Agent Must Export**:
```python
# Mandatory metrics
REQUIRED_METRICS = [
    "agent_requests_total",
    "agent_request_duration_seconds",
    "agent_errors_total",
    "agent_resource_usage_bytes",
    "agent_ollama_queue_depth",
    "agent_circuit_breaker_state"
]
```

---

### 📌 Rule 28: Distributed Tracing
✨ **Trace Every Cross-Agent Call**:
```python
# OpenTelemetry integration
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent_operation") as span:
    span.set_attribute("agent.id", AGENT_ID)
    span.set_attribute("agent.phase", AGENT_PHASE)
    span.set_attribute("operation.type", operation_type)
```

---

## 🚨 PRODUCTION OPERATIONAL RULES

### 📌 Rule 29: Incident Response
✨ **Automated Remediation**:
```yaml
incident_response:
  cpu_overload:
    - Suspend Specialized agents
    - Reduce Ollama threads
    - Enable request queuing
    
  memory_pressure:
    - Hibernate idle agents
    - Clear caches
    - Trigger GC
    
  network_partition:
    - Activate local fallbacks
    - Queue writes
    - Alert operators
```

---

### 📌 Rule 30: Compliance Validation
✨ **Continuous Enforcement**:
```python
# Run every deployment
compliance_checks = [
    "resource_limits_enforced",
    "health_endpoints_responding",
    "metrics_exported",
    "logging_structured",
    "security_headers_present",
    "network_policies_applied"
]

# Automated remediation
for check in compliance_checks:
    if not validate(check):
        auto_remediate(check)
        alert_team(check)
```

---

## 📋 IMPLEMENTATION PRIORITIES

### 🔥 Immediate (24 hours)
1. Fix Ollama CPU usage (Rule 22)
2. Enforce memory limits on all 34 containers (Rule 20)
3. Implement health checks (Rule 17)
4. Resolve port conflicts (Rule 18)

### ⚡ Short-term (1 week)
1. Deploy service mesh (Rules 18-19)
2. Implement monitoring (Rules 27-28)
3. Set up phased deployment (Rule 25)
4. Add circuit breakers (Rule 18)

### 🎯 Medium-term (1 month)
1. Complete distributed tracing (Rule 28)
2. Implement auto-scaling (Rule 20)
3. Add predictive monitoring (Rule 27)
4. Deploy compliance automation (Rule 30)

---

## ✅ VALIDATION SCRIPTS

```bash
# Validate entire system compliance
./scripts/validate-all-rules.sh

# Check specific rule compliance
./scripts/check-rule.py --rule 22  # Ollama optimization
./scripts/check-rule.py --rule 20  # Resource limits

# Continuous compliance monitoring
./scripts/compliance-monitor.py --continuous --alert-channel ops
```

---

## 📈 SUCCESS METRICS

- **System Stability**: 99.9% uptime, zero unplanned restarts
- **Resource Efficiency**: <80% CPU, <85% memory utilization
- **Response Times**: P95 < 2s, P99 < 5s
- **Error Rates**: <0.1% of requests
- **Deployment Success**: 100% successful rollouts
- **Compliance Score**: >95% rule adherence

---

## 🔐 ENFORCEMENT

1. **Pre-commit Hooks**: Validate rules before code commit
2. **CI/CD Gates**: Block deployments that violate rules  
3. **Runtime Monitoring**: Continuous compliance checking
4. **Automated Remediation**: Self-healing for common violations
5. **Team Reviews**: Weekly rule compliance reports

---

**Remember**: These rules ensure your 69-agent AI system operates reliably, efficiently, and safely at scale. Every rule has been crafted based on your specific architecture and constraints. Treat them as your engineering constitution.

*"With great agent power comes great operational responsibility"* 🚀