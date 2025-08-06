# API Specification - ACTUAL WORKING SYSTEM

> **📋 Complete Technology Stack**: See `TECHNOLOGY_STACK_REPOSITORY_INDEX.md` for comprehensive technology inventory including FastAPI backend infrastructure.

## Base URL
`http://localhost:10010`

## VERIFIED STATUS
System is LIVE with 70+ actual endpoints. NO authentication required - all endpoints are open.

## ACTUAL ENDPOINTS (VERIFIED WORKING)

### Health & System Status
```
GET /health
GET /api/v1/system/health
GET /api/v1/system/status
GET /api/v1/system/
```

### Coordinator & Orchestration
```
GET /api/v1/coordinator/status
POST /api/v1/coordinator/task
GET /api/v1/coordinator/tasks
GET /api/v1/coordinator/agents
POST /api/v1/coordinator/agents/discover
POST /api/v1/coordinator/agents/start-all
POST /api/v1/coordinator/agents/activate-agi
GET /api/v1/coordinator/agents/status
POST /api/v1/coordinator/agents/stop-all
GET /api/v1/coordinator/collective/status
POST /api/v1/coordinator/deploy/mass-activation
POST /api/v1/coordinator/deploy/activate-collective
GET /api/v1/coordinator/deploy/status
```

### Models & AI Processing
```
GET /api/v1/models/
POST /api/v1/models/pull
POST /api/v1/models/generate
POST /api/v1/models/chat
POST /api/v1/models/embed
GET /api/v1/models/status
POST /chat
POST /simple-chat
GET /models
```

### Vector Database Operations
```
POST /api/v1/vectors/initialize
POST /api/v1/vectors/add
POST /api/v1/vectors/search
GET /api/v1/vectors/stats
POST /api/v1/vectors/optimize
```

### Security & Authentication
```
POST /api/v1/security/login
POST /api/v1/security/refresh
POST /api/v1/security/logout
GET /api/v1/security/report
GET /api/v1/security/audit/events
POST /api/v1/security/encrypt
POST /api/v1/security/decrypt
GET /api/v1/security/compliance/status
POST /api/v1/security/compliance/gdpr/{action}
GET /api/v1/security/config
POST /api/v1/security/test/vulnerability-scan
```

### Cognitive Processing & AI Reasoning
```
POST /api/v1/processing/process
GET /api/v1/processing/system_state
POST /api/v1/processing/creative
POST /public/think
POST /think
POST /execute
POST /reason
POST /learn
POST /improve
```

### Document Management
```
GET /api/v1/documents/
```

### Orchestration & Workflows
```
POST /api/v1/orchestration/agents
POST /api/v1/orchestration/workflows
GET /api/v1/orchestration/status
```

### Self-Improvement System
```
POST /api/v1/improvement/analyze
POST /api/v1/improvement/apply
```

### Monitoring & Metrics
```
GET /metrics
GET /public/metrics
GET /prometheus-metrics
```

### Legacy Endpoints
```
GET /agents
POST /api/v1/agents/consensus
GET /api/v1/docs/endpoints
GET /
```

## SAMPLE RESPONSES (VERIFIED)

### Health Check Response
```json
{
  "status": "healthy",
  "service": "sutazai-backend", 
  "version": "17.0.0",
  "enterprise_features": false,
  "timestamp": "2025-08-06T08:15:51.956819",
  "gpu_available": false,
  "services": {
    "ollama": "connected",
    "chromadb": "disconnected", 
    "qdrant": "connected",
    "database": "connected (no tables created yet)",
    "redis": "connected",
    "models": {"status": "available", "loaded_count": 1, "model": "tinyllama"},
    "agents": {"status": "active", "active_count": 5, "total_defined": 44, "orchestration_active": true}
  },
  "system": {
    "cpu_percent": 18.9,
    "memory_percent": 46.9,
    "memory_used_gb": 13.38,
    "memory_total_gb": 29.38,
    "gpu_available": false
  }
}
```

## INTERACTIVE DOCUMENTATION
- Swagger UI: `http://localhost:10010/docs`
- OpenAPI Spec: `http://localhost:10010/openapi.json`

## BACKEND VERSION
- Current Version: 17.0.0
- Title: "SutazAI automation/advanced automation System"
- Description: "Autonomous General Intelligence Platform with Enterprise Features"

This specification reflects the ACTUAL working API as deployed and verified on August 6, 2025.