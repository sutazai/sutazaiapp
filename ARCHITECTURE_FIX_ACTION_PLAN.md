# ARCHITECTURE FIX ACTION PLAN - SutazAI System
**Created:** 2025-08-07  
**Priority:** CRITICAL  
**Timeline:** Immediate to 30 days

## IMMEDIATE ACTIONS (TODAY)

### 1. Stop the Bleeding
```bash
# Fix MCP restart loops
docker stop mcp-proxy mcp-registry github-mcp-server filesystem-mcp-server
docker rm mcp-proxy mcp-registry github-mcp-server filesystem-mcp-server

# Document actual running services
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}" > actual_services.txt
```

### 2. Backup Current State
```bash
# Backup configurations
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)
cp -r config/ config.backup.$(date +%Y%m%d_%H%M%S)/

# Export database if needed
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup_$(date +%Y%m%d).sql
```

### 3. Switch to Minimal Configuration
```bash
# Stop everything
docker-compose down

# Start minimal tier (5 containers only)
docker-compose -f docker-compose.minimal.yml up -d

# Verify core services
curl http://localhost:10010/health  # Backend
curl http://localhost:10104/api/tags # Ollama
```

## SHORT-TERM FIXES (Week 1)

### 1. Clean Docker Compose
Create new streamlined docker-compose.yml:

```yaml
# docker-compose.clean.yml
version: '3.8'

networks:
  sutazai-network:
    external: true

services:
  # Only include ACTUALLY RUNNING services
  postgres:
    # ... minimal config
  
  redis:
    # ... minimal config
    
  ollama:
    # ... minimal config
    
  backend:
    # ... minimal config
    
  frontend:
    # ... minimal config
```

### 2. Fix Configuration Files

#### A. Update port-registry-actual.yaml
```yaml
# ACTUAL running services only
core_services:
  backend: 10010
  postgres: 10000
  redis: 10001
  ollama: 10104
  ollama_integration: 8090

# REMOVE all non-running services
```

#### B. Remove service-mesh.json
Since Kong/Consul/RabbitMQ are NOT running:
```bash
mv config/service-mesh.json config/archive/service-mesh.json.defunct
```

### 3. Update Documentation
```bash
# Update CLAUDE.md with reality
- Change "28 containers running" to "11 containers running"
- Remove service mesh claims
- Mark all agents as "NOT IMPLEMENTED"
- Update port registry
```

## MEDIUM-TERM FIXES (Weeks 2-4)

### 1. Implement Tiered Architecture

#### Phase 1: Validate Minimal Tier
- Run for 3 days with just 5 containers
- Monitor resource usage
- Test all core functionality
- Document what works/doesn't work

#### Phase 2: Add Monitoring (if needed)
```bash
# Add Standard tier components gradually
docker-compose -f docker-compose.minimal.yml \
              -f docker-compose.standard.yml up -d prometheus

# Test for 24 hours, then add:
docker-compose -f docker-compose.minimal.yml \
              -f docker-compose.standard.yml up -d grafana
```

#### Phase 3: Add Vector DB (if needed)
```bash
# Only if vector search is actually used
docker-compose -f docker-compose.minimal.yml \
              -f docker-compose.standard.yml up -d qdrant
```

### 2. Remove Phantom Services

#### Step 1: Identify unused services
```python
# script: identify_phantom_services.py
import yaml
import subprocess

# Load docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    compose = yaml.safe_load(f)

# Get running containers
result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                       capture_output=True, text=True)
running = set(result.stdout.strip().split('\n'))

# Find phantoms
defined_services = set(compose.get('services', {}).keys())
phantom_services = defined_services - running

print(f"Phantom services to remove ({len(phantom_services)}):")
for service in sorted(phantom_services):
    print(f"  - {service}")
```

#### Step 2: Create clean configuration
```bash
# Remove all phantom services from docker-compose.yml
# Keep ONLY services that are actually running
```

### 3. Fix Service Mesh (Decision Required)

#### Option A: Remove Completely
```bash
# If not needed (recommended)
# Remove from docker-compose.yml:
- kong
- consul  
- rabbitmq

# Remove configuration files:
rm config/consul-services.json
rm config/service-mesh.json
```

#### Option B: Implement Properly
```bash
# If actually needed (justify first!)
# 1. Deploy Kong
docker-compose up -d kong

# 2. Configure routes
curl -X POST http://localhost:8001/services \
  -d name=backend \
  -d url=http://backend:8000

# 3. Add service discovery
docker-compose up -d consul

# 4. Register services
# ... implement registration logic
```

## LONG-TERM FIXES (Month 1)

### 1. Implement Real Agent (if needed)

#### Replace ONE stub with real implementation:
```python
# agents/example_real_agent/app.py
from fastapi import FastAPI
from typing import Dict, Any
import ollama

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "type": "real_agent"}

@app.post("/process")
async def process(data: Dict[str, Any]):
    # ACTUAL LOGIC HERE
    prompt = data.get("prompt", "")
    
    # Use Ollama for processing
    response = ollama.chat(
        model='tinyllama',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return {
        "status": "completed",
        "result": response['message']['content']
    }
```

### 2. Database Schema Creation
```sql
-- Create proper tables for PostgreSQL
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    last_heartbeat TIMESTAMP,
    configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    type VARCHAR(100),
    payload JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Add indexes
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
```

### 3. Monitoring Implementation
```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

## SUCCESS METRICS

### Week 1 Goals:
- [ ] Zero restart loops
- [ ] docker-compose.yml < 500 lines
- [ ] All defined services running
- [ ] Documentation reflects reality

### Week 2 Goals:
- [ ] Monitoring operational (if needed)
- [ ] Vector DB integrated (if needed)
- [ ] One real agent implemented

### Month 1 Goals:
- [ ] Service mesh decision finalized
- [ ] Database schema implemented
- [ ] 100% service utilization
- [ ] Clean, maintainable architecture

## VALIDATION CHECKLIST

### After Each Change:
```bash
# 1. Check all services are healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# 2. Test core functionality
curl http://localhost:10010/health
curl http://localhost:10104/api/tags

# 3. Check logs for errors
docker-compose logs --tail=50 | grep -i error

# 4. Monitor resource usage
docker stats --no-stream

# 5. Validate configuration
python scripts/validate-containers.py
```

## ROLLBACK PLAN

If issues occur:
```bash
# 1. Stop current deployment
docker-compose down

# 2. Restore backup
cp docker-compose.yml.backup.20250807_* docker-compose.yml

# 3. Restart original configuration
docker-compose up -d

# 4. Document what failed
echo "Rollback reason: [describe issue]" >> rollback.log
```

---

**Remember:** Every change must follow the 19 CLAUDE.md rules. No fantasy features. No unnecessary complexity. Build what works, document honestly, maintain ruthlessly.