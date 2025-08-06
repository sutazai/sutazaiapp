# EMERGENCY DEPLOYMENT PLAN - REAL SYSTEM STATUS

> **📋 Complete Technology Stack**: See `TECHNOLOGY_STACK_REPOSITORY_INDEX.md` for comprehensive technology inventory and verified deployment components.

## User Request: "clean it all up!!!!" - Deploy Top 20 AI Agents

### VERIFIED SYSTEM STATUS (Current Reality Check)
- **Total Containers Running**: 26 services (verified via docker-compose ps)
- **Infrastructure Status**: 
  - ✅ PostgreSQL (10000 - no tables yet), Redis (10001), Neo4j (10002-10003) - all VERIFIED HEALTHY
  - ✅ Ollama (10104) with TinyLlama model currently loaded - VERIFIED HEALTHY
  - ✅ ChromaDB (10100), Qdrant (10101-10102), FAISS (10103) - vector stores operational
  - ✅ Kong Gateway (10005), Consul (10006), RabbitMQ (10007-10008) - VERIFIED WORKING service mesh
- **Monitoring**: ✅ Prometheus (10200), Grafana (10201), Loki (10202) - fully operational
- **Backend/Frontend**: ✅ FastAPI (10010) Version 17.0.0 with 70+ endpoints HEALTHY, Streamlit (10011) WORKING

### CRITICAL REALITY CHECK - What Actually Works
- **Infrastructure**: Solid foundation with databases, queues, service discovery
- **AI Capability**: Only Ollama (TinyLlama currently loaded) provides real AI - everything else returns stubs
- **Agent Status**: 5 agents running with health endpoints (out of 44 defined), but mostly return placeholder responses
- **Service Mesh**: Kong/Consul/RabbitMQ operational for real service communication

### TOP 20 AI AGENTS - ACTUAL DEPLOYMENT PRIORITY

#### CURRENTLY RUNNING AGENTS (5 verified)
1. **ai-agent-orchestrator** (Port 8589) - ✅ HEALTHY - Needs real orchestration logic
2. **multi-agent-coordinator** (Port 8587) - ✅ HEALTHY - Needs actual coordination
3. **hardware-resource-optimizer** (Port 8002) - ✅ HEALTHY - Basic resource monitoring
4. **resource-arbitration-agent** (Port 8588) - ✅ HEALTHY - Needs arbitration logic  
5. **task-assignment-coordinator** (Port 8551) - ✅ HEALTHY - Needs task distribution

#### PHASE 1: ENHANCE EXISTING AGENTS (Week 1)
**Goal**: Make the 5 running agents actually functional instead of stubs

**Priority Actions**:
1. **Integrate Ollama with agents**: All agents need to call TinyLlama model for actual AI
2. **Implement real orchestration**: Use RabbitMQ for task distribution
3. **Connect to databases**: Use PostgreSQL for state, Redis for caching
4. **Add vector search**: Integrate with ChromaDB/Qdrant for knowledge

**Implementation Commands**:
```bash
# Test current agent functionality
curl -X POST http://localhost:8589/process \
  -H "Content-Type: application/json" \
  -d '{"task": "analyze_system", "priority": "high"}'

# Verify Ollama integration
curl -X POST http://localhost:10104/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "System analysis request"}'

# Check RabbitMQ for task queues
curl -u admin:admin http://localhost:10008/api/queues
```

#### PHASE 2: DEPLOY NEXT 15 AGENTS (Week 2-3)
**Based on docker-compose.yml services available**:

6. **AgentGPT** (Port 11066) - General purpose agent
7. **AgentZero** (Port 11067) - Coordination and communication  
8. **Aider** (Port TBD) - Code assistance and development
9. **AI Metrics Exporter** (Port 11068) - Currently unhealthy, needs fixing
10. **Backend Developer Agent** - API development and database management
11. **Frontend Developer Agent** - UI/UX development with Streamlit
12. **Database Administrator Agent** - PostgreSQL/Redis/Neo4j management
13. **Security Analysis Agent** - System security and vulnerability assessment
14. **Performance Monitor Agent** - System optimization and monitoring
15. **Documentation Agent** - Technical writing and knowledge management
16. **QA Testing Agent** - Quality assurance and automated testing
17. **DevOps Agent** - Deployment and infrastructure management  
18. **Data Analysis Agent** - Analytics using vector databases
19. **Integration Agent** - Service-to-service communication
20. **Health Monitor Agent** - System health and alerting

#### PHASE 3: SYSTEM INTEGRATION (Week 4)
**Real Integration Points**:
- **Kong Gateway**: Route agent requests through API gateway
- **Consul Discovery**: Register all agents with service discovery
- **RabbitMQ Queues**: Implement async task distribution
- **Vector Databases**: Knowledge search across ChromaDB/Qdrant
- **Monitoring**: Full observability via Prometheus/Grafana

### DEPLOYMENT STRATEGY (Realistic Implementation)

#### Step 1: Fix Current Unhealthy Services
```bash
# Fix AI metrics exporter
docker-compose restart ai-metrics-exporter
docker logs sutazai-ai-metrics --tail=50

# Ensure ChromaDB is fully started
docker-compose restart chromadb
curl -f http://localhost:10100/api/v1/heartbeat
```

#### Step 2: Enhance Agent Logic (Replace Stubs)
**Template for Real Agent Implementation**:
```python
# agents/[agent-name]/app.py - REAL implementation
import requests
import json
from flask import Flask, jsonify, request

app = Flask(__name__)
OLLAMA_URL = "http://ollama:11434"

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    
    # REAL AI processing via Ollama
    response = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "tinyllama",
        "prompt": f"Task: {data.get('task', '')}\nInput: {json.dumps(data)}",
        "stream": False
    })
    
    if response.ok:
        result = response.json()
        return jsonify({
            "status": "success",
            "result": result.get('response', ''),
            "agent": "real_implementation",
            "input": data
        })
    else:
        return jsonify({"status": "error", "message": "Ollama unavailable"}), 500
```

#### Step 3: Service Mesh Integration
```bash
# Register agents with Consul
curl -X PUT http://localhost:10006/v1/agent/service/register \
  -d '{
    "ID": "ai-agent-orchestrator",
    "Name": "orchestrator",
    "Address": "ai-agent-orchestrator", 
    "Port": 8589,
    "Check": {
      "HTTP": "http://ai-agent-orchestrator:8589/health",
      "Interval": "30s"
    }
  }'

# Configure Kong routes for agents
curl -X POST http://localhost:8001/services \
  -d "name=agent-orchestrator" \
  -d "url=http://ai-agent-orchestrator:8589"

curl -X POST http://localhost:8001/services/agent-orchestrator/routes \
  -d "paths[]=/api/v1/orchestrate"
```

### SUCCESS METRICS (Realistic Targets)

#### Week 1 Goals:
- [ ] 5 existing agents return real AI responses via Ollama
- [ ] Task distribution via RabbitMQ working  
- [ ] Database integration for state persistence
- [ ] Basic error handling and logging

#### Week 2-3 Goals:
- [ ] 15 additional agents deployed and registered
- [ ] Kong gateway routing all agent traffic
- [ ] Consul service discovery fully operational
- [ ] Vector search integration working

#### Week 4 Goals:
- [ ] End-to-end task workflows functional
- [ ] Full monitoring and alerting
- [ ] Performance optimization
- [ ] Documentation complete

### EMERGENCY FIXES NEEDED IMMEDIATELY

1. **Fix AI Metrics Exporter** (currently unhealthy):
```bash
docker-compose logs sutazai-ai-metrics
docker-compose restart ai-metrics-exporter
```

2. **Ensure ChromaDB Full Startup**:
```bash
docker-compose restart chromadb
# Wait for full initialization
sleep 60
curl -f http://localhost:10100/api/v1/heartbeat
```

3. **Verify Backend/Frontend Health**:
```bash
# Backend API should respond
curl -f http://localhost:10010/health
# Frontend should load
curl -f http://localhost:10011
```

### REALISTIC RESOURCE REQUIREMENTS

**Current System Load**: 26 containers running
**Memory Usage**: ~8-12GB estimated  
**CPU Usage**: Moderate with spikes during AI inference
**Disk Usage**: Models + databases + logs

**Scaling Considerations**:
- Ollama is the bottleneck - single model server
- Consider model caching and request queuing
- Monitor resource usage via cAdvisor/Prometheus

This deployment plan is based on ACTUAL system capabilities and verified infrastructure, not theoretical implementations.