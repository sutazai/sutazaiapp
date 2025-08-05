# SutazAI Resource Scheduling Policy

## Policy Overview

This document defines the comprehensive resource scheduling policy for the SutazAI distributed AI system, designed to optimize computational resource allocation across 69 AI agents while maintaining system stability and maximizing throughput.

## System Specifications

- **Hardware**: 12 CPU cores, 29.38GB RAM
- **Current Utilization**: 28% memory, 2% CPU (severely under-utilized)
- **Target Utilization**: 40-60% CPU, 80% memory
- **Container Count**: 69 AI agents (currently 7 active)

## Scheduling Principles

### 1. Priority-Based Resource Allocation

**Priority Levels:**
- **P1 - Critical (1000)**: Infrastructure services that must never fail
- **P2 - High (800)**: Active AI agents and monitoring systems  
- **P3 - Medium (600)**: On-demand ML/AI compute workloads
- **P4 - Low (400)**: Development tools and code generators
- **P5 - Background (200)**: Batch jobs and background processing

### 2. Fair Share Scheduling

Each priority tier gets guaranteed minimum resources with ability to burst:
- P1 Critical: 50% CPU, 48% Memory (guaranteed)
- P2 High: 33% CPU, 27% Memory (guaranteed)
- P3 Medium: 17% CPU, 17% Memory (burstable)
- P4 Low: 8% CPU, 7% Memory (best effort)
- P5 Background: 8% CPU, 3% Memory (preemptible)

### 3. Resource Pool Isolation

**Pool 1 - Infrastructure**: Backend, Frontend, Databases
**Pool 2 - Active Agents**: Running AI agents, monitoring
**Pool 3 - ML Compute**: PyTorch, TensorFlow, JAX workloads
**Pool 4 - Development**: Code tools, IDE agents
**Pool 5 - Background**: Batch processing, maintenance

## CPU Core Assignment Strategy

```
Cores 0-1:   System Reserved (OS, kernel)
Cores 2-3:   Backend API, Frontend UI
Cores 4-5:   Database services (PostgreSQL, Redis, Neo4j)
Cores 6-9:   Ollama LLM service (4 cores dedicated)
Cores 10-11: Active AI agents, monitoring, development
```

### CPU Affinity Rules

1. **Critical Services**: Fixed core assignment to prevent migration
2. **AI Agents**: Shared pool with load balancing
3. **ML Workloads**: Burst to available cores when needed
4. **Background Jobs**: Use only idle cores

## Memory Management Policy

### Memory Pool Allocation

```yaml
Infrastructure Pool: 14GB
  - Backend API: 2GB
  - Frontend: 1GB  
  - PostgreSQL: 2GB
  - Redis: 512MB
  - Ollama: 8GB
  - Neo4j: 2GB

Active Agents Pool: 8GB
  - Per Agent: 1GB average
  - Monitoring Stack: 2GB
  - Buffer: 1GB

ML Compute Pool: 5GB
  - Burst Capacity: 8GB
  - Per Workload: 4GB max

Development Pool: 2GB
  - Per Tool: 512MB
  - Concurrent Limit: 4 tools

System Reserve: 2GB
  - OS Buffer: 1.5GB
  - Emergency: 512MB
```

### Memory Reclamation

1. **Automatic**: OOM killer targets lowest priority containers first
2. **Proactive**: Scale down idle containers before memory pressure
3. **Emergency**: Kill background jobs to protect critical services

## Container Scheduling Algorithms

### 1. Bin Packing Algorithm

**First Fit Decreasing (FFD)** for resource allocation:
1. Sort containers by resource requirements (descending)
2. Assign to first node with sufficient resources
3. Optimize for CPU and memory utilization simultaneously

### 2. Gang Scheduling

For multi-container applications requiring coordination:
- Schedule all related containers simultaneously
- Ensure inter-container communication latency < 10ms
- Apply to: CrewAI multi-agent systems, distributed ML training

### 3. Work-Conserving Scheduler

- No CPU cycles idle if work pending
- Dynamic priority adjustment based on SLA requirements
- Preemption allowed for lower priority tasks

## Auto-Scaling Policies

### Horizontal Scaling Rules

```yaml
Scale Up Triggers:
  - CPU utilization > 70% for 60 seconds
  - Memory utilization > 80% for 30 seconds
  - Queue depth > 50 pending tasks
  - Response time > 2x baseline

Scale Down Triggers:
  - CPU utilization < 30% for 300 seconds
  - Memory utilization < 40% for 300 seconds
  - Queue depth < 10 pending tasks
  - Zero activity for 600 seconds (development tools)

Scaling Limits:
  - Infrastructure: Fixed (no scaling)
  - Active Agents: 1-3 replicas
  - ML Compute: 0-2 replicas  
  - Development: 0-2 replicas
  - Background: 0-1 replicas
```

### Vertical Scaling (Resource Adjustments)

```yaml
CPU Scaling:
  - Minimum: 0.1 cores
  - Maximum: 4 cores (infrastructure), 2 cores (others)
  - Step size: 0.5 cores
  - Cooldown: 60 seconds

Memory Scaling:
  - Minimum: 128MB
  - Maximum: 8GB (ML workloads), 2GB (others)
  - Step size: 256MB
  - Cooldown: 30 seconds
```

## Quality of Service (QoS) Classes

### Guaranteed QoS
- **Services**: Infrastructure pool services
- **Resource Limits**: requests = limits
- **Preemption**: Never preempted
- **Node Selection**: Dedicated nodes preferred

### Burstable QoS  
- **Services**: Active agents, ML compute
- **Resource Limits**: requests < limits
- **Preemption**: Only by Guaranteed class
- **Node Selection**: Shared nodes allowed

### Best Effort QoS
- **Services**: Development tools, background jobs
- **Resource Limits**: No requests specified
- **Preemption**: Preempted by higher classes
- **Node Selection**: Any available resources

## Placement Constraints and Affinity

### Anti-Affinity Rules

1. **Database Services**: Never co-locate on same core
   ```yaml
   postgres ≠ neo4j ≠ redis (different CPU cores)
   ```

2. **AI Agents**: Distribute across available cores
   ```yaml
   maxSkew: 1 (even distribution)
   ```

3. **Compute Workloads**: Avoid resource contention
   ```yaml
   pytorch ≠ tensorflow ≠ jax (different time slots)
   ```

### Preferred Affinity

1. **Related Services**: Co-locate when beneficial
   ```yaml
   backend + chromadb (data locality)
   monitoring-stack (observability services)
   ```

2. **Cache Locality**: Keep frequently communicating services close
   ```yaml
   backend + redis (sub-millisecond latency)
   ```

## Failure Handling and Recovery

### Container Restart Policy

```yaml
Infrastructure Services:
  restart_policy: always
  max_attempts: unlimited
  backoff_delay: exponential (5s, 10s, 20s, 60s)

Active Agents:
  restart_policy: on-failure  
  max_attempts: 5
  backoff_delay: linear (10s, 20s, 30s)

On-Demand Services:
  restart_policy: on-failure
  max_attempts: 3
  backoff_delay: exponential (30s, 60s, 120s)

Background Services:
  restart_policy: no
  max_attempts: 1
  backoff_delay: none
```

### Circuit Breaker Pattern

- **Open Circuit**: After 5 consecutive failures
- **Half-Open**: Test with single request after 60s
- **Closed Circuit**: Resume normal operation after success

### Graceful Degradation

1. **Service Mesh**: Route traffic away from unhealthy instances
2. **Load Shedding**: Drop non-critical requests under pressure
3. **Fallback**: Switch to cached responses or simplified logic

## Resource Monitoring and Observability

### Key Performance Indicators

```yaml
System Level:
  - overall_cpu_utilization: target 40-60%
  - overall_memory_utilization: target 75-80%
  - container_density: containers per GB RAM
  - resource_efficiency: useful work / total consumption

Application Level:
  - agent_task_completion_rate: tasks/minute
  - average_response_time: milliseconds
  - queue_depth: pending tasks
  - error_rate: failed requests/total requests

Infrastructure Level:
  - node_availability: uptime percentage
  - network_latency: inter-service communication
  - storage_iops: disk operations per second
  - memory_swap_rate: pages swapped per second
```

### Alerting Thresholds

```yaml
Critical Alerts (Page Immediately):
  - CPU utilization > 90% for 2 minutes
  - Memory utilization > 95% for 1 minute
  - Container restart rate > 5/hour
  - Queue depth > 1000 pending tasks

Warning Alerts (Notify):
  - CPU utilization > 80% for 5 minutes
  - Memory utilization > 85% for 3 minutes
  - Response time > 5x baseline
  - Disk space < 10% free

Info Alerts (Log):
  - Scaling events (up/down)
  - Configuration changes
  - Planned maintenance windows
```

## Implementation Schedule

### Phase 1: Foundation (Week 1)
- Deploy resource pool configuration
- Implement CPU core affinity
- Update container resource limits
- Deploy monitoring dashboards

### Phase 2: Scheduling (Week 2)
- Implement priority-based scheduling
- Deploy auto-scaling policies
- Configure placement constraints
- Test failure scenarios

### Phase 3: Optimization (Week 3)
- Fine-tune resource allocation
- Implement predictive scaling
- Deploy advanced monitoring
- Performance benchmarking

### Phase 4: Production (Week 4)
- Full production deployment
- 24/7 monitoring enablement
- Documentation and training
- Performance validation

## Configuration Files

### Primary Configuration
- `/opt/sutazaiapp/docker-compose.resource-optimized.yml`
- `/opt/sutazaiapp/configs/resource-pools.yaml`

### Monitoring Configuration
- `/opt/sutazaiapp/monitoring/prometheus/rules.yml`
- `/opt/sutazaiapp/monitoring/grafana/dashboards/`

### Scaling Configuration
- `/opt/sutazaiapp/configs/autoscaling.yaml`
- `/opt/sutazaiapp/configs/placement-policies.yaml`

## Compliance and Governance

### Resource Governance
- No service can exceed allocated resource limits
- All resource changes require approval
- Monthly resource utilization reviews
- Capacity planning for 6-month horizon

### Security Considerations
- Container isolation with user namespaces
- Resource limits prevent DoS attacks
- Network policies restrict inter-service communication
- Audit logging for all resource changes

### Performance SLAs
- Critical services: 99.9% uptime
- AI agents: 95% availability
- Response time: P95 < 500ms
- Task completion: 99% success rate

This comprehensive scheduling policy ensures optimal resource utilization while maintaining system stability and meeting performance requirements for the SutazAI distributed AI platform.