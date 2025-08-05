# 🚀 SutazAI Rules Quick Reference Card
## 69-Agent System Engineering Standards

---

## 🔥 CRITICAL FIXES (Do First!)

```yaml
# Fix Ollama CPU (Rule 22)
OLLAMA_NUM_PARALLEL: 1
OLLAMA_NUM_THREADS: 4
OLLAMA_MAX_LOADED_MODELS: 1

# Container Resources (Rule 20)
Critical Agents: 2 CPU, 4GB RAM
Performance Agents: 1 CPU, 2GB RAM  
Specialized Agents: 0.5 CPU, 1GB RAM

# Required Health Check (Rule 17)
HEALTHCHECK CMD curl -f http://localhost:8080/health
```

---

## 📋 DAILY CHECKLIST

### Before Any Code Change:
- [ ] Run distributed analysis: `./scripts/distributed-analysis.py`
- [ ] Check resource usage: `docker stats`
- [ ] Verify no port conflicts in 10000-10599 range

### Before Deployment:
- [ ] All containers have memory limits
- [ ] Health endpoints return JSON
- [ ] Circuit breakers configured
- [ ] Phased deployment ready

### After Deployment:
- [ ] Check Prometheus metrics: http://localhost:10200
- [ ] Verify all agents healthy: `./scripts/check-agents.sh`
- [ ] Monitor Ollama CPU: <50%

---

## 🏗️ ARCHITECTURE RULES

### Container Requirements
```yaml
Every container needs:
- Memory limit
- CPU limit  
- Health endpoint (/health)
- Metrics endpoint (/metrics)
- Structured JSON logging
- Non-root user
```

### Service Communication
```python
# Always use service discovery
endpoint = consul.discover("agent-name")

# Always add circuit breaker
if circuit_breaker.is_open(target):
    return fallback()

# Always set timeout
response = await call(timeout=30)
```

### Port Allocation
```
10000-10199: Infrastructure
10200-10299: Monitoring  
10300-10599: Agents (by phase)
```

---

## 🤖 AI AGENT RULES

### Agent Phases
1. **Critical (10300-10319)**: Never suspend, high priority
2. **Performance (10320-10419)**: Suspend under pressure
3. **Specialized (10420-10599)**: First to hibernate

### Ollama Usage
```python
# Queue all requests
await ollama_queue.put(request)

# Limit context size
max_tokens = 512  # Not 4096!

# Cache responses
if cache.has(prompt_hash):
    return cache.get(prompt_hash)
```

---

## 🚨 EMERGENCY PROCEDURES

### CPU Overload
```bash
# 1. Suspend specialized agents
docker stop $(docker ps -q --filter "label=phase=specialized")

# 2. Reduce Ollama threads
docker exec ollama set OLLAMA_NUM_THREADS=2

# 3. Enable emergency queuing
./scripts/enable-emergency-mode.sh
```

### Memory Pressure
```bash
# 1. Check memory usage
./scripts/memory-analysis.py

# 2. Hibernate idle agents
./scripts/hibernate-idle-agents.sh

# 3. Clear caches
redis-cli FLUSHDB
```

### Service Mesh Failure
```bash
# 1. Verify Consul
consul members

# 2. Restart affected services
docker-compose restart consul kong rabbitmq

# 3. Re-register agents
./scripts/reregister-agents.sh
```

---

## 📊 MONITORING COMMANDS

```bash
# System Overview
curl http://localhost:10200/api/v1/query?query=up

# Agent Health
for p in {10300..10599}; do
  curl -s localhost:$p/health | jq .status
done

# Resource Usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Ollama Status
curl http://localhost:10104/api/tags
```

---

## ✅ VALIDATION SCRIPTS

```bash
# Before commit
./scripts/pre-commit-validation.sh

# After deployment  
./scripts/post-deploy-validation.sh

# Daily compliance
./scripts/compliance-check.py --all

# Rule-specific
./scripts/check-rule.py --rule 22  # Ollama
./scripts/check-rule.py --rule 20  # Resources
```

---

## 📞 ESCALATION

### Severity Levels
- **P1**: System down, multiple agents failing → Page on-call
- **P2**: Single agent failing, high CPU → Slack #alerts  
- **P3**: Degraded performance → Ticket in Jira
- **P4**: Compliance warnings → Weekly review

### Key Metrics to Watch
- Ollama CPU: Must be <50%
- Memory usage: Must be <85%
- Agent health: Must be >95%
- Response time P95: Must be <2s

---

## 🎯 GOLDEN RULES

1. **Never deploy without memory limits**
2. **Always use service discovery, not IPs**
3. **Every call needs timeout and retry**
4. **Phase deployment: Infrastructure → Critical → Others**
5. **Monitor first, optimize second**

---

**Remember**: These rules keep 69 agents running smoothly on 12 CPUs! 🚀