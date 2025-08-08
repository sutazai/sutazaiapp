# SutazAI System API Reference

**Version:** 17.0.0  
**Base URL:** http://localhost:10010  
**Last Updated:** 2025-08-08  
**System Status:** Degraded (Model mismatch - TinyLlama loaded, backend expects gpt-oss)

## Table of Contents
- [API Overview](#api-overview)
- [Authentication & Authorization](#authentication--authorization)
- [Core Endpoints](#core-endpoints)
- [Agent Management](#agent-management)
- [Task Management](#task-management)
- [Model Management](#model-management)
- [Data Management](#data-management)
- [System Management](#system-management)
- [WebSocket Endpoints](#websocket-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)
- [API Examples](#api-examples)
- [OpenAPI Specification](#openapi-specification)
- [Current Limitations](#current-limitations)

---

## API Overview

### Current Reality Check ⚠️
- **Service Status:** Backend running on port 10010 (degraded due to model mismatch)
- **Model Reality:** TinyLlama (637MB) loaded, NOT gpt-oss as expected by backend
- **Agent Status:** 7 Flask stub services running (return hardcoded responses)
- **Enterprise Features:** Available but many components are stubs
- **Database:** PostgreSQL empty (no tables created yet)
- **Authentication:** JWT planned but not fully implemented (currently allows anonymous access)

### Base Configuration
```yaml
Base URL: http://localhost:10010
API Version: v1
Protocol: HTTP/HTTPS
Content-Type: application/json
```

### Feature Flags
- `SUTAZAI_ENTERPRISE_FEATURES=1` (enabled)
- `SUTAZAI_ENABLE_KNOWLEDGE_GRAPH=1` (enabled)
- `SUTAZAI_ENABLE_COGNITIVE=1` (enabled)

### Service Dependencies
| Service | Port | Status | Notes |
|---------|------|--------|-------|
| PostgreSQL | 10000 | ✅ Healthy | Database empty - needs schema |
| Redis | 10001 | ✅ Healthy | Cache layer working |
| Neo4j | 10002/10003 | ✅ Healthy | Graph database available |
| Ollama | 10104 | ✅ Healthy | TinyLlama loaded (NOT gpt-oss) |
| Backend API | 10010 | ⚠️ Degraded | Model mismatch causing issues |

---

## Authentication & Authorization

### Current State
- **JWT Implementation:** Planned but incomplete
- **Default Mode:** Anonymous access allowed for basic endpoints
- **Enterprise Mode:** Bearer token expected but falls back to anonymous
- **Admin Endpoints:** Some require authentication (partially implemented)

### Authentication Header
```http
Authorization: Bearer <jwt_token>
```

### User Roles
- `anonymous` - Basic access to public endpoints
- `user` - Standard user access
- `admin` - Full system access

### Security Note
> ⚠️ Current authentication is mostly stub implementation. Production deployment requires full JWT implementation.

---

## Core Endpoints

### Health Check
```http
GET /health
```

**Description:** Comprehensive system health check with service status  
**Authentication:** None required  
**Response:**
```json
{
  "status": "degraded",
  "service": "sutazai-backend",
  "version": "17.0.0",
  "enterprise_features": true,
  "timestamp": "2025-08-08T10:30:00Z",
  "services": {
    "ollama": "connected",
    "chromadb": "disconnected",
    "qdrant": "connected",
    "database": "connected",
    "redis": "connected",
    "models": {
      "status": "available",
      "loaded_count": 1
    }
  },
  "system": {
    "cpu_percent": 23.5,
    "memory_percent": 67.2,
    "memory_used_gb": 5.4,
    "memory_total_gb": 8.0,
    "gpu_available": false
  }
}
```

### System Information
```http
GET /
```

**Description:** Root endpoint with comprehensive system information  
**Authentication:** None required

### System Status (Enterprise)
```http
GET /api/v1/system/status
```

**Description:** Enterprise system status with component health  
**Authentication:** Optional (falls back to anonymous)

---

## Agent Management

### List Agents
```http
GET /agents
```

**Description:** Get list of available AI agents with health status  
**Authentication:** None required  
**Response:**
```json
{
  "agents": [
    {
      "id": "task_coordinator",
      "name": "AGI Coordinator",
      "status": "active",
      "type": "reasoning",
      "description": "Central AGI reasoning system",
      "capabilities": ["reasoning", "learning", "consciousness"],
      "health": "healthy"
    },
    {
      "id": "autogpt",
      "name": "AutoGPT Agent",
      "status": "inactive",
      "type": "autonomous",
      "description": "Autonomous task execution agent",
      "capabilities": ["planning", "execution", "web_browsing"],
      "health": "degraded"
    }
  ]
}
```

### Agent Consensus
```http
POST /api/v1/agents/consensus
```

**Description:** Multi-agent consensus decision making  
**Authentication:** Optional  
**Request:**
```json
{
  "prompt": "Should we implement feature X?",
  "agents": ["agent1", "agent2", "agent3"],
  "consensus_type": "majority"
}
```

**Response:**
```json
{
  "analysis": "Consensus analysis for: Should we implement feature X?",
  "agents_consulted": ["agent1", "agent2", "agent3"],
  "consensus_reached": true,
  "consensus_type": "majority",
  "confidence": 0.85,
  "agent_votes": {
    "agent1": "agree",
    "agent2": "agree",
    "agent3": "disagree"
  },
  "timestamp": "2025-08-08T10:30:00Z"
}
```

---

## Task Management

### Chat Interface
```http
POST /chat
```

**Description:** Chat with AI models with enhanced processing  
**Authentication:** None required  
**Request:**
```json
{
  "message": "Explain quantum computing",
  "model": "tinyllama",
  "agent": "task_coordinator",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Quantum computing harnesses quantum mechanical phenomena...",
  "model": "tinyllama",
  "agent": "task_coordinator",
  "consciousness_enhancement": true,
  "reasoning_pathways": ["analytical", "creative"],
  "consciousness_level": 0.8,
  "vector_context_used": false,
  "timestamp": "2025-08-08T10:30:00Z",
  "processing_time": "1.2s"
}
```

### Advanced Thinking
```http
POST /think
```

**Description:** AGI Coordinator deep thinking process  
**Authentication:** Optional  
**Request:**
```json
{
  "query": "How can we solve climate change?",
  "reasoning_type": "deductive"
}
```

### Public Thinking (No Auth)
```http
POST /public/think
```

**Description:** Public thinking endpoint without authentication  
**Authentication:** None required

### Task Execution
```http
POST /execute
```

**Description:** Execute tasks through appropriate agents  
**Authentication:** Optional  
**Request:**
```json
{
  "description": "Create a Python script to analyze data",
  "type": "coding"
}
```

**Response:**
```json
{
  "result": "Task execution completed...",
  "status": "completed",
  "task_id": "task_20250808_103000",
  "task_type": "coding",
  "execution_time": "3.4s",
  "success_probability": 0.92,
  "orchestrated": false,
  "resources_used": ["cognitive_processing", "knowledge_retrieval"],
  "timestamp": "2025-08-08T10:30:00Z"
}
```

### Reasoning
```http
POST /reason
```

**Description:** Apply advanced reasoning to complex problems  
**Request:**
```json
{
  "type": "deductive",
  "description": "Analyze the causes of software bugs"
}
```

### Learning
```http
POST /learn
```

**Description:** Learn and integrate new knowledge  
**Request:**
```json
{
  "content": "Machine learning is a subset of AI...",
  "type": "text"
}
```

---

## Model Management

### List Models
```http
GET /models
```

**Description:** Get available AI models from Ollama  
**Response:**
```json
{
  "models": [
    {
      "id": "tinyllama",
      "name": "tinyllama",
      "status": "loaded",
      "type": "language_model",
      "capabilities": ["text_generation", "reasoning"],
      "size": "637MB"
    }
  ],
  "total_models": 1,
  "default_model": "tinyllama",
  "recommended_models": ["tinyllama", "llama3:8b", "llama:7b"]
}
```

### Model Generation
```http
POST /api/v1/models/generate
```

**Description:** Generate text using specific models  
**Authentication:** Optional  
**Request:**
```json
{
  "prompt": "Write a haiku about programming",
  "model": "tinyllama",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Simple Chat
```http
POST /simple-chat
```

**Description:** Direct Ollama model interaction  
**Request:**
```json
{
  "message": "Hello, how are you?"
}
```

---

## Data Management

### System Metrics
```http
GET /metrics
```

**Description:** Comprehensive system metrics and analytics  
**Authentication:** Optional

### Public Metrics
```http
GET /public/metrics
```

**Description:** Public system metrics (no authentication)  
**Response:**
```json
{
  "timestamp": "2025-08-08T10:30:00Z",
  "system": {
    "cpu_percent": 23.5,
    "memory_percent": 67.2,
    "uptime": "6h 32m"
  },
  "services": {
    "ollama": "healthy",
    "chromadb": "unhealthy",
    "qdrant": "healthy"
  },
  "performance": {
    "avg_response_time_ms": 245,
    "success_rate": 98.5,
    "requests_per_minute": 45
  }
}
```

### Prometheus Metrics
```http
GET /prometheus-metrics
```

**Description:** Metrics in Prometheus format for monitoring  
**Content-Type:** text/plain

---

## System Management

### Self-Improvement
```http
POST /improve
```

**Description:** Trigger system self-improvement analysis  
**Authentication:** Optional

### Enterprise Improvement Analysis
```http
POST /api/v1/improvement/analyze
```

**Description:** Comprehensive system analysis for improvement  
**Authentication:** Optional

### Apply Improvements
```http
POST /api/v1/improvement/apply
```

**Description:** Apply selected system improvements  
**Authentication:** Optional  
**Request:**
```json
{
  "improvement_ids": ["opt_001", "opt_002"]
}
```

---

## Orchestration Endpoints

### Create Agent
```http
POST /api/v1/orchestration/agents
```

**Description:** Create new agent through orchestration system  
**Authentication:** Optional  
**Request:**
```json
{
  "agent_type": "research",
  "name": "research_agent_001",
  "config": {
    "specialization": "data_analysis",
    "max_tasks": 10
  }
}
```

### Create Workflow
```http
POST /api/v1/orchestration/workflows
```

**Description:** Create and execute workflows  
**Request:**
```json
{
  "name": "data_processing_workflow",
  "description": "Process customer data",
  "tasks": [
    {"type": "extract", "source": "database"},
    {"type": "transform", "rules": "clean_data"},
    {"type": "load", "target": "warehouse"}
  ],
  "agents": ["data_agent", "transform_agent"]
}
```

### Orchestration Status
```http
GET /api/v1/orchestration/status
```

**Description:** Get orchestration system status  
**Authentication:** Optional

---

## Consciousness Processing Endpoints

### Consciousness Process
```http
POST /api/v1/consciousness/process
```

**Description:** Process data through consciousness reasoning engine  
**Request:**
```json
{
  "input_data": "Analyze market trends",
  "processing_type": "analytical",
  "use_consciousness": true,
  "reasoning_depth": 3
}
```

### Creative Synthesis
```http
POST /api/v1/consciousness/creative
```

**Description:** Creative synthesis through consciousness engine  
**Request:**
```json
{
  "prompt": "Design a sustainable city",
  "synthesis_mode": "cross_domain",
  "reasoning_depth": 3,
  "use_consciousness": true
}
```

### Consciousness State
```http
GET /api/v1/consciousness/consciousness
```

**Description:** Get current consciousness state  
**Response:**
```json
{
  "consciousness_active": true,
  "awareness_level": 0.8,
  "cognitive_load": 0.6,
  "active_processes": ["reasoning", "learning"],
  "consciousness_activity": {
    "perception": 0.7,
    "cognition": 0.8,
    "metacognition": 0.6
  }
}
```

---

## WebSocket Endpoints

### Real-time Updates
```
ws://localhost:10010/ws
```

**Description:** WebSocket connection for real-time system updates  
**Authentication:** None required  
**Messages:**
```json
{
  "type": "metrics_update",
  "data": {
    "cpu_percent": 25.3,
    "memory_percent": 68.1,
    "timestamp": "2025-08-08T10:30:00Z"
  }
}
```

---

## Request/Response Formats

### Standard Response Envelope
```json
{
  "status": "success|error|degraded",
  "data": {},
  "message": "Human readable message",
  "timestamp": "2025-08-08T10:30:00Z",
  "processing_time": "1.2s",
  "metadata": {
    "version": "17.0.0",
    "enterprise_features": true
  }
}
```

### Error Response Format
```json
{
  "status": "error",
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "No language models are currently available",
    "details": "Please ensure Ollama is running with models installed"
  },
  "timestamp": "2025-08-08T10:30:00Z"
}
```

### Pagination Format
```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "pages": 5
  }
}
```

---

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Code Reference
| Code | Description | Resolution |
|------|-------------|------------|
| `MODEL_UNAVAILABLE` | No AI models loaded | Install models via Ollama |
| `AGENT_OFFLINE` | Agent service unreachable | Check agent service status |
| `SERVICE_DEGRADED` | Service partially functional | Check service dependencies |
| `AUTH_REQUIRED` | Authentication needed | Provide valid JWT token |
| `INVALID_MODEL` | Requested model not found | Use available model from `/models` |

### Retry Strategies
- **Exponential Backoff:** For temporary failures
- **Circuit Breaker:** For service unavailability
- **Model Fallback:** Automatically use available models

---

## API Examples

### cURL Examples

#### Basic Health Check
```bash
curl -X GET http://localhost:10010/health
```

#### Chat with AI
```bash
curl -X POST http://localhost:10010/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain machine learning",
    "model": "tinyllama",
    "agent": "task_coordinator"
  }'
```

#### Get Available Models
```bash
curl -X GET http://localhost:10010/models
```

#### Execute Task
```bash
curl -X POST http://localhost:10010/execute \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a Python function to sort a list",
    "type": "coding"
  }'
```

### Python Client Example
```python
import requests
import json

class SutazAIClient:
    def __init__(self, base_url="http://localhost:10010"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def chat(self, message, model="tinyllama", agent="task_coordinator"):
        data = {
            "message": message,
            "model": model,
            "agent": agent
        }
        response = self.session.post(f"{self.base_url}/chat", json=data)
        return response.json()
    
    def execute_task(self, description, task_type="general"):
        data = {
            "description": description,
            "type": task_type
        }
        response = self.session.post(f"{self.base_url}/execute", json=data)
        return response.json()

# Usage
client = SutazAIClient()
health = client.health_check()
result = client.chat("What is artificial intelligence?")
```

### JavaScript/TypeScript Example
```typescript
class SutazAIClient {
  private baseURL: string;

  constructor(baseURL: string = "http://localhost:10010") {
    this.baseURL = baseURL;
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }

  async chat(message: string, model: string = "tinyllama"): Promise<any> {
    const response = await fetch(`${this.baseURL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message,
        model,
        agent: "task_coordinator"
      })
    });
    return response.json();
  }

  async executeTask(description: string, type: string = "general"): Promise<any> {
    const response = await fetch(`${this.baseURL}/execute`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        description,
        type
      })
    });
    return response.json();
  }
}

// Usage
const client = new SutazAIClient();
const health = await client.healthCheck();
const result = await client.chat("Explain quantum physics");
```

---

## OpenAPI Specification

### Swagger UI
- **URL:** http://localhost:10010/docs
- **ReDoc:** http://localhost:10010/redoc

### OpenAPI JSON
```bash
curl http://localhost:10010/openapi.json
```

### Key Schema Definitions
```json
{
  "ChatRequest": {
    "type": "object",
    "properties": {
      "message": {"type": "string"},
      "model": {"type": "string", "default": "task_coordinator"},
      "agent": {"type": "string"},
      "temperature": {"type": "number", "default": 0.7}
    },
    "required": ["message"]
  },
  "TaskRequest": {
    "type": "object",
    "properties": {
      "description": {"type": "string"},
      "type": {"type": "string", "default": "general"}
    },
    "required": ["description"]
  }
}
```

---

## Current Limitations

### Known Issues ⚠️
1. **Model Mismatch:** Backend expects `gpt-oss` but `tinyllama` is loaded
2. **Empty Database:** PostgreSQL has no tables - needs schema initialization
3. **Stub Agents:** 7 agent services return hardcoded responses, no real processing
4. **ChromaDB Issues:** Vector database has connection problems
5. **Incomplete Authentication:** JWT implementation is partial

### Temporary Workarounds
```bash
# Fix model mismatch (option 1: load expected model)
docker exec sutazai-ollama ollama pull gpt-oss

# Fix model mismatch (option 2: update backend config)
# Edit backend configuration to use "tinyllama" instead of "gpt-oss"

# Check actual service status
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# Test direct Ollama connection
curl http://127.0.0.1:10104/api/tags
```

### Production Readiness Gaps
- [ ] Complete JWT authentication implementation
- [ ] Initialize database schema
- [ ] Replace agent stubs with real processing logic
- [ ] Fix ChromaDB connectivity issues
- [ ] Implement proper error handling and retry logic
- [ ] Add comprehensive API rate limiting
- [ ] Security vulnerability scanning
- [ ] Performance optimization for production load

---

## Rate Limiting

### Current State
- **Implementation:** Not fully implemented
- **Default Limits:** None enforced
- **Headers:** Not present in responses

### Planned Implementation
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641024000
```

---

## Monitoring & Observability

### Health Monitoring
```bash
# Check backend health
curl http://127.0.0.1:10010/health

# Prometheus metrics
curl http://127.0.0.1:10010/prometheus-metrics

# System metrics
curl http://127.0.0.1:10010/public/metrics
```

### Grafana Dashboards
- **URL:** http://localhost:10201
- **Credentials:** admin/admin
- **Dashboards:** Pre-configured system monitoring

### Log Aggregation
```bash
# View backend logs
docker-compose logs -f backend

# View Ollama logs
docker-compose logs -f ollama
```

---

## Migration & Upgrade Path

### Current to Production
1. **Fix Model Configuration**
   - Load gpt-oss model or update backend to use tinyllama
   
2. **Initialize Database Schema**
   - Create necessary PostgreSQL tables
   - Set up proper migrations
   
3. **Implement Real Agent Logic**
   - Replace Flask stubs with actual processing
   - Add inter-agent communication
   
4. **Complete Authentication**
   - Implement full JWT validation
   - Add proper user management
   
5. **Production Hardening**
   - Add rate limiting
   - Implement proper error handling
   - Security audit and fixes

---

## Contact & Support

### Development Team
- **Backend Developer:** Senior Backend Expert
- **API Documentation:** API Documentation Specialist
- **System Architecture:** Principal Systems Architect

### Resources
- **Source Code:** `/opt/sutazaiapp/backend/`
- **Documentation:** `/opt/sutazaiapp/docs/`
- **IMPORTANT Files:** `/opt/sutazaiapp/IMPORTANT/`
- **System Truth:** `/opt/sutazaiapp/CLAUDE.md`

---

*This documentation reflects the actual current state of the SutazAI system as of 2025-08-08. For the most up-to-date information, always refer to the system's actual endpoints and the CLAUDE.md file for ground truth.*