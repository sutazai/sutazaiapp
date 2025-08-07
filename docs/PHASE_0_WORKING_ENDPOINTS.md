# SUTAZAI System - Working Endpoints Documentation
**Generated**: August 7, 2025  
**Phase**: 0 - Critical Fixes  
**Status**: VERIFIED WORKING

## ✅ Core Infrastructure Endpoints

### 1. Backend API (Port 10010)
```bash
# Health Check
curl http://127.0.0.1:10010/health
```
**Response**: Full system health with service statuses
- Status: ✅ HEALTHY
- Ollama: Connected
- Database: Connected
- Redis: Connected
- Qdrant: Connected
- ChromaDB: Disconnected (known issue)

### 2. Ollama LLM Server (Port 10104)
```bash
# List Models
curl http://127.0.0.1:10104/api/tags

# Generate Text
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello world"}'
```
**Status**: ✅ WORKING
- Model: tinyllama:latest (637MB)

### 3. AI Agent Orchestrator (Port 8589)
```bash
curl http://127.0.0.1:8589/health
```
**Status**: ✅ HEALTHY (stub implementation)

### 4. Ollama Integration Specialist (Port 11015)
```bash
curl http://127.0.0.1:11015/health
```
**Status**: ✅ HEALTHY

## ⚠️ Database Endpoints

### PostgreSQL (Port 10000)
```bash
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -c "\dt"
```
**Tables**: users, agents, tasks, models, agent_interactions, system_state, etc.

### Redis (Port 10001)
```bash
docker exec -it sutazai-redis redis-cli ping
```
**Response**: PONG

### Neo4j (Port 10002/10003)
- Browser: http://localhost:10002
- Bolt: bolt://localhost:10003

## 📊 Monitoring Stack

### Prometheus (Port 10200)
```bash
curl http://127.0.0.1:10200/-/healthy
```

### Grafana (Port 10201)
- URL: http://localhost:10201
- Login: admin/admin

### Loki (Port 10202)
```bash
curl http://127.0.0.1:10202/ready
```

## 🔧 Service Mesh

### Kong API Gateway (Port 10005)
```bash
curl http://127.0.0.1:10005/status
```

### Consul (Port 10006)
```bash
curl http://127.0.0.1:10006/v1/status/leader
```

### RabbitMQ (Port 10007/10008)
- AMQP: amqp://localhost:10007
- Management: http://localhost:10008

## 📝 Test Commands Summary

```bash
# Quick system health check
curl -s http://127.0.0.1:10010/health | python3 -m json.tool

# Test Ollama text generation
curl -X POST http://127.0.0.1:10104/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "What is Docker?", "stream": false}'

# Check all running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep healthy
```