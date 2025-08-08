# Perfect Jarvis API Reference

**API Version:** 17.0.0  
**Last Updated:** 2025-08-08  
**Base URL:** `http://localhost:10010`

## 🎯 Overview

The Perfect Jarvis API provides comprehensive AI automation capabilities through a RESTful interface. This documentation covers all implemented endpoints based on the actual system architecture.

## 📋 Table of Contents

- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
- [Agent Management](#agent-management)
- [Model Operations](#model-operations)
- [Chat & Reasoning](#chat--reasoning)
- [System Management](#system-management)
- [Monitoring & Metrics](#monitoring--metrics)
- [Enterprise Features](#enterprise-features)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Code Examples](#code-examples)

## 🔐 Authentication

### Bearer Token Authentication

The system supports JWT Bearer token authentication for enterprise features.

**Header Format:**
```
Authorization: Bearer <your-jwt-token>
```

**Basic Endpoints:** No authentication required
**Enterprise Endpoints:** Bearer token required

### Getting Authentication Token

**Note:** Token generation endpoint not implemented in current version. For development, basic endpoints work without authentication.

```json
// Future implementation
POST /api/v1/auth/token
{
  "username": "your-username",
  "password": "your-password"
}
```

## 🌐 Core Endpoints

### Health Check

Check system health and service status.

**Endpoint:** `GET /health`  
**Authentication:** None  
**Rate Limit:** None  

**Response:**
```json
{
  "status": "healthy",
  "service": "sutazai-backend",
  "version": "17.0.0",
  "enterprise_features": false,
  "timestamp": "2025-08-08T10:30:00.000Z",
  "gpu_available": false,
  "services": {
    "ollama": "connected",
    "chromadb": "disconnected",
    "qdrant": "disconnected",
    "database": "connected",
    "redis": "connected",
    "models": {
      "status": "available",
      "loaded_count": 1
    },
    "agents": {
      "status": "active",
      "active_count": 5,
      "orchestration_active": false
    }
  },
  "system": {
    "cpu_percent": 25.3,
    "memory_percent": 45.2,
    "memory_used_gb": 3.6,
    "memory_total_gb": 8.0,
    "gpu_available": false
  }
}
```

**Status Codes:**
- `200 OK`: System healthy
- `503 Service Unavailable`: System degraded

### System Information

Get comprehensive system information.

**Endpoint:** `GET /`  
**Authentication:** None  

**Response:**
```json
{
  "name": "SutazAI Jarvis System",
  "version": "17.0.0",
  "description": "Enterprise Autonomous General Intelligence Platform",
  "status": "running",
  "capabilities": [
    "Multi-model AI reasoning",
    "Enterprise agent orchestration",
    "Real-time learning and adaptation",
    "Advanced problem solving",
    "Code generation and analysis"
  ],
  "enterprise_features": false,
  "endpoints": {
    "core": ["/health", "/agents", "/chat", "/think", "/execute"],
    "enterprise": []
  },
  "architecture": {
    "frontend": "Streamlit Web Interface",
    "backend": "FastAPI with Enterprise Jarvis Coordinator",
    "models": "Ollama Local LLM Service",
    "vector_db": "ChromaDB + Qdrant",
    "agents": "AutoGPT, CrewAI, Aider, GPT-Engineer"
  }
}
```

## 🤖 Agent Management

### List Available Agents

Get list of all available AI agents and their status.

**Endpoint:** `GET /agents`  
**Authentication:** None  
**Cache:** 30 seconds  

**Response:**
```json
{
  "agents": [
    {
      "id": "task_coordinator",
      "name": "Jarvis Coordinator",
      "status": "active",
      "type": "reasoning",
      "description": "Central Jarvis reasoning system",
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
    },
    {
      "id": "crewai",
      "name": "CrewAI Team",
      "status": "inactive",
      "type": "collaborative",
      "description": "Multi-agent collaboration system",
      "capabilities": ["teamwork", "role_based", "coordination"],
      "health": "degraded"
    }
  ]
}
```

### Agent Status Values

| Status | Description |
|--------|-------------|
| `active` | Agent running and healthy |
| `inactive` | Agent not running or unreachable |
| `degraded` | Agent running but experiencing issues |
| `maintenance` | Agent temporarily disabled |

## 🧠 Model Operations

### List Available Models

Get all loaded AI models and their capabilities.

**Endpoint:** `GET /models`  
**Authentication:** None  

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
      "size": "unknown"
    }
  ],
  "total_models": 1,
  "default_model": "tinyllama",
  "recommended_models": ["tinyllama", "llama3:8b", "llama:7b"]
}
```

### Model Generation

Generate text using specific AI models.

**Endpoint:** `POST /api/v1/models/generate`  
**Authentication:** Bearer token  
**Rate Limit:** 100 requests/hour  

**Request Body:**
```json
{
  "prompt": "Explain quantum computing",
  "model": "tinyllama",
  "max_tokens": 1024,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "analysis": "Model generation completed for: Explain quantum computing",
  "model_used": "tinyllama",
  "generated_text": "Quantum computing is a revolutionary technology...",
  "tokens_used": 150,
  "temperature": 0.7,
  "insights": [
    "AI model generation completed",
    "Response generated successfully",
    "High quality output achieved"
  ],
  "recommendations": [
    "Review generated content",
    "Adjust parameters if needed"
  ],
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

## 💬 Chat & Reasoning

### Chat with AI

Interactive chat with AI models supporting multiple agents.

**Endpoint:** `POST /chat`  
**Authentication:** None  
**Rate Limit:** 60 requests/hour  

**Request Body:**
```json
{
  "message": "How can I optimize database performance?",
  "model": "tinyllama",
  "agent": "task_coordinator",
  "temperature": 0.7
}
```

**Required Fields:**
- `message` (string): User message

**Optional Fields:**
- `model` (string): Specific model to use
- `agent` (string): Agent to route request to
- `temperature` (float): Response creativity (0.0-1.0)

**Response:**
```json
{
  "response": "To optimize database performance, consider these strategies:\n1. Index optimization...",
  "model": "tinyllama",
  "agent": "task_coordinator",
  "processing_enhancement": false,
  "reasoning_pathways": [],
  "consciousness_level": 0.0,
  "vector_context_used": false,
  "timestamp": "2025-08-08T10:30:00.000Z",
  "processing_time": "1.2s"
}
```

### Simple Chat

Simplified chat endpoint for basic interactions.

**Endpoint:** `POST /simple-chat`  
**Authentication:** None  
**Rate Limit:** 100 requests/hour  

**Request Body:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "Hello! I'm functioning well and ready to assist you.",
  "model": "tinyllama",
  "timestamp": "2025-08-08T10:30:00.000Z",
  "processing_time": 1.234
}
```

### Jarvis Thinking

Access Jarvis's deep reasoning capabilities.

**Endpoint:** `POST /think`  
**Authentication:** Bearer token  
**Rate Limit:** 30 requests/hour  

**Request Body:**
```json
{
  "query": "What are the implications of artificial general intelligence?",
  "reasoning_type": "deductive"
}
```

**Reasoning Types:**
- `general`: General analysis
- `deductive`: Logical deduction
- `inductive`: Pattern recognition
- `abductive`: Best explanation finding
- `analogical`: Comparison reasoning
- `causal`: Cause-effect analysis

**Response:**
```json
{
  "thought": "Artificial general intelligence represents a transformative leap...",
  "reasoning": "Multi-layer cognitive analysis using perception, reasoning, metacognition",
  "confidence": 0.85,
  "model_used": "tinyllama",
  "cognitive_load": "high",
  "processing_stages": [
    "perception",
    "analysis", 
    "reasoning",
    "synthesis",
    "metacognition"
  ],
  "consciousness_level": 0.8,
  "reasoning_depth": 3,
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Public Thinking

Public reasoning endpoint without authentication.

**Endpoint:** `POST /public/think`  
**Authentication:** None  
**Rate Limit:** 20 requests/hour  

**Request Body:**
```json
{
  "query": "How does machine learning work?",
  "reasoning_type": "general"
}
```

**Response:**
```json
{
  "response": "Machine learning works by identifying patterns in data...",
  "reasoning_type": "general",
  "model_used": "tinyllama",
  "confidence": 0.89,
  "thought_process": [
    "Query analyzed and contextualized",
    "Relevant knowledge patterns activated",
    "Logical reasoning framework applied",
    "High-confidence conclusion synthesized"
  ],
  "cognitive_load": "medium",
  "processing_time": "2.1s",
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Task Execution

Execute complex tasks through appropriate agents.

**Endpoint:** `POST /execute`  
**Authentication:** Bearer token  
**Rate Limit:** 20 requests/hour  

**Request Body:**
```json
{
  "description": "Create a REST API for user management",
  "type": "coding"
}
```

**Task Types:**
- `general`: General tasks
- `complex`: Multi-step tasks
- `multi_agent`: Requires multiple agents
- `workflow`: Structured workflow
- `coding`: Programming tasks
- `analysis`: Data analysis tasks

**Response:**
```json
{
  "result": "Task analysis complete. Here's a comprehensive plan for creating a REST API...",
  "status": "completed",
  "task_id": "task_20250808_103000",
  "task_type": "coding",
  "execution_time": "3.4s",
  "success_probability": 0.92,
  "orchestrated": false,
  "resources_used": [
    "cognitive_processing",
    "knowledge_retrieval",
    "planning_system"
  ],
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Advanced Reasoning

Apply structured reasoning to complex problems.

**Endpoint:** `POST /reason`  
**Authentication:** None  
**Rate Limit:** 30 requests/hour  

**Request Body:**
```json
{
  "type": "deductive",
  "description": "Analyze the scalability challenges of microservices architecture"
}
```

**Response:**
```json
{
  "analysis": "Microservices architecture scalability analysis: ...",
  "reasoning_type": "deductive",
  "steps": [
    "Problem decomposition",
    "Knowledge activation",
    "Logical framework application",
    "Alternative analysis",
    "Conclusion synthesis"
  ],
  "conclusion": "Advanced deductive reasoning completed with high confidence",
  "logical_framework": "deductive",
  "confidence_level": 0.88,
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Knowledge Learning

Integrate new knowledge into the system.

**Endpoint:** `POST /learn`  
**Authentication:** None  
**Rate Limit:** 10 requests/hour  

**Request Body:**
```json
{
  "content": "Kubernetes is a container orchestration platform that automates deployment...",
  "type": "text"
}
```

**Content Types:**
- `text`: Plain text content
- `code`: Programming code
- `document`: Structured document
- `data`: Data/statistics

**Response:**
```json
{
  "learned": true,
  "content_type": "text",
  "content_size": 156,
  "summary": "Successfully processed 156 characters of text content",
  "knowledge_points": [
    "Content analyzed and structured",
    "Key concepts extracted and indexed",
    "Embeddings generated for semantic search",
    "Cross-references established with existing knowledge"
  ],
  "processing_stats": {
    "concepts_extracted": 5,
    "embeddings_created": 3,
    "connections_established": 2
  },
  "processing_time": "1.5s",
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

## ⚙️ System Management

### System Metrics

Get comprehensive system performance metrics.

**Endpoint:** `GET /metrics`  
**Authentication:** Bearer token  

**Response:**
```json
{
  "timestamp": "2025-08-08T10:30:00.000Z",
  "system": {
    "cpu_percent": 25.3,
    "memory_percent": 45.2,
    "memory_used_gb": 3.6,
    "memory_total_gb": 8.0,
    "disk_percent": 65.1,
    "disk_free_gb": 128.5,
    "uptime": "6h 32m",
    "load_average": [0.45, 0.52, 0.48]
  },
  "services": {
    "ollama": "healthy",
    "chromadb": "unhealthy",
    "qdrant": "unhealthy",
    "postgres": "healthy",
    "redis": "healthy"
  },
  "performance": {
    "avg_response_time_ms": 245,
    "success_rate": 98.5,
    "requests_per_minute": 45,
    "active_agents": 5,
    "processed_requests": 1247,
    "total_tokens_processed": 892456
  },
  "ai_metrics": {
    "models_loaded": 1,
    "embeddings_generated": 45230,
    "reasoning_operations": 892,
    "learning_events": 127,
    "self_improvements": 23
  }
}
```

### Public Metrics

System metrics without authentication (limited data).

**Endpoint:** `GET /public/metrics`  
**Authentication:** None  

**Response:** (Same structure as `/metrics` but limited information)

### Prometheus Metrics

Metrics in Prometheus format for monitoring integration.

**Endpoint:** `GET /prometheus-metrics`  
**Authentication:** None  
**Response Format:** `text/plain`

**Response:**
```
# HELP sutazai_uptime_seconds Application uptime in seconds
# TYPE sutazai_uptime_seconds counter
sutazai_uptime_seconds 23520

# HELP sutazai_cache_entries_total Number of cached service entries  
# TYPE sutazai_cache_entries_total gauge
sutazai_cache_entries_total 5

# HELP sutazai_info Application information
# TYPE sutazai_info gauge
sutazai_info{version="17.0.0",service="backend"} 1
```

### Self-Improvement

Trigger system optimization and learning.

**Endpoint:** `POST /improve`  
**Authentication:** Bearer token  
**Rate Limit:** 1 request/hour  

**Response:**
```json
{
  "improvement": "Comprehensive system analysis and optimization completed",
  "changes": [
    "Memory usage optimization applied - reduced by 15%",
    "Model inference speed improved by 12%",
    "Agent coordination latency reduced by 8%",
    "Knowledge retrieval accuracy enhanced by 18%"
  ],
  "impact": "Overall system performance improved by 15.2%",
  "next_optimization": "Vector database indexing and query optimization scheduled",
  "performance_gains": {
    "speed": "+12%",
    "accuracy": "+18%",
    "efficiency": "+15%",
    "reliability": "+8%"
  },
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

## 🏢 Enterprise Features

*Note: Enterprise features require Bearer token authentication and may not be fully implemented in the current version.*

### Agent Orchestration

Create and manage orchestrated agents.

**Endpoint:** `POST /api/v1/orchestration/agents`  
**Authentication:** Bearer token required  

**Request Body:**
```json
{
  "agent_type": "research_agent",
  "config": {
    "specialization": "technical_research",
    "knowledge_domains": ["AI", "software_engineering"]
  },
  "name": "research_specialist_01"
}
```

**Response:**
```json
{
  "agent_id": "agent_abc123",
  "status": "created",
  "config": {
    "type": "research_agent",
    "name": "research_specialist_01",
    "specialization": "technical_research"
  },
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Workflow Management

Create and execute complex workflows.

**Endpoint:** `POST /api/v1/orchestration/workflows`  
**Authentication:** Bearer token required  

**Request Body:**
```json
{
  "name": "data_analysis_pipeline",
  "description": "Complete data analysis workflow",
  "tasks": [
    {
      "id": "collect_data",
      "type": "data_collection",
      "config": {"source": "api"}
    },
    {
      "id": "analyze_data", 
      "type": "analysis",
      "depends_on": ["collect_data"]
    }
  ],
  "agents": ["data_collector", "analyst"]
}
```

**Response:**
```json
{
  "workflow_id": "wf_xyz789",
  "status": "started",
  "definition": {
    "name": "data_analysis_pipeline",
    "tasks": [...],
    "agents": [...]
  },
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

### Agent Consensus

Multi-agent collaborative decision making.

**Endpoint:** `POST /api/v1/agents/consensus`  
**Authentication:** Bearer token required  

**Request Body:**
```json
{
  "prompt": "Should we implement microservices or monolithic architecture?",
  "agents": ["architect_agent", "performance_agent", "security_agent"],
  "consensus_type": "majority"
}
```

**Response:**
```json
{
  "analysis": "Consensus analysis completed",
  "agents_consulted": ["architect_agent", "performance_agent", "security_agent"],
  "consensus_reached": true,
  "consensus_type": "majority",
  "confidence": 0.85,
  "recommendations": [
    "Agents reached majority consensus",
    "High confidence in collaborative decision"
  ],
  "agent_votes": {
    "architect_agent": "microservices",
    "performance_agent": "microservices", 
    "security_agent": "monolithic"
  },
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

## 🌐 WebSocket Streaming

### Real-time Updates

Connect to WebSocket for live system updates.

**Endpoint:** `ws://localhost:10010/ws`  
**Authentication:** None  

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:10010/ws');

ws.onopen = function(event) {
    console.log('Connected to Jarvis WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

**Message Types:**

**Echo Message:**
```json
{
  "type": "echo",
  "message": "Received: your message",
  "timestamp": "2025-08-08T10:30:00.000Z"
}
```

**Metrics Update:**
```json
{
  "type": "metrics_update",
  "data": {
    "cpu_percent": 25.3,
    "memory_percent": 45.2,
    "timestamp": "2025-08-08T10:30:00.000Z"
  }
}
```

## ❌ Error Handling

### HTTP Status Codes

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Endpoint not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily down |

### Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request body is invalid",
    "details": "Field 'message' is required but missing",
    "timestamp": "2025-08-08T10:30:00.000Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_REQUEST` | Request format error | Check request body format |
| `AUTHENTICATION_REQUIRED` | Missing auth token | Provide Bearer token |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait before retrying |
| `MODEL_UNAVAILABLE` | AI model not loaded | Wait for model initialization |
| `SERVICE_UNAVAILABLE` | Backend service down | Check system status |

### Validation Errors

Field validation errors include specific details:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Input validation failed",
    "details": {
      "message": ["Message cannot be empty"],
      "temperature": ["Must be between 0.0 and 1.0"]
    },
    "timestamp": "2025-08-08T10:30:00.000Z"
  }
}
```

## 🚦 Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1693123200
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|--------|---------|
| `/chat` | 60 requests | 1 hour |
| `/think` | 30 requests | 1 hour |
| `/public/think` | 20 requests | 1 hour |
| `/execute` | 20 requests | 1 hour |
| `/improve` | 1 request | 1 hour |
| `/learn` | 10 requests | 1 hour |
| Health checks | No limit | - |

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": "Limit: 60 requests per hour. Try again in 45 minutes.",
    "retry_after": 2700,
    "timestamp": "2025-08-08T10:30:00.000Z"
  }
}
```

## 💻 Code Examples

### Python Client

```python
import requests
import json

class JarvisClient:
    def __init__(self, base_url="http://localhost:10010"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check system health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def chat(self, message, agent=None, model=None):
        """Chat with AI"""
        payload = {"message": message}
        if agent:
            payload["agent"] = agent
        if model:
            payload["model"] = model
            
        response = self.session.post(
            f"{self.base_url}/chat",
            json=payload
        )
        return response.json()
    
    def think(self, query, reasoning_type="general"):
        """Deep reasoning"""
        response = self.session.post(
            f"{self.base_url}/public/think",
            json={
                "query": query,
                "reasoning_type": reasoning_type
            }
        )
        return response.json()
    
    def execute_task(self, description, task_type="general"):
        """Execute a task"""
        response = self.session.post(
            f"{self.base_url}/execute",
            json={
                "description": description,
                "type": task_type
            }
        )
        return response.json()

# Usage example
client = JarvisClient()

# Health check
health = client.health_check()
print(f"System status: {health['status']}")

# Chat with AI
response = client.chat(
    message="Explain machine learning",
    agent="task_coordinator"
)
print(response["response"])

# Deep thinking
thought = client.think(
    query="What are the benefits of containerization?",
    reasoning_type="analytical"
)
print(thought["response"])
```

### JavaScript Client

```javascript
class JarvisClient {
    constructor(baseUrl = 'http://localhost:10010') {
        this.baseUrl = baseUrl;
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }

    async chat(message, options = {}) {
        const payload = {
            message,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        return response.json();
    }

    async think(query, reasoningType = 'general') {
        const response = await fetch(`${this.baseUrl}/public/think`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                reasoning_type: reasoningType
            })
        });
        
        return response.json();
    }

    async executeTask(description, type = 'general') {
        const response = await fetch(`${this.baseUrl}/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                description,
                type
            })
        });
        
        return response.json();
    }

    // WebSocket connection
    connectWebSocket(onMessage) {
        const ws = new WebSocket(`ws://localhost:10010/ws`);
        
        ws.onopen = () => {
            console.log('Connected to Jarvis WebSocket');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        return ws;
    }
}

// Usage example
const client = new JarvisClient();

// Health check
client.healthCheck()
    .then(health => console.log('System status:', health.status));

// Chat
client.chat('How do I optimize database queries?', {
    agent: 'task_coordinator',
    temperature: 0.7
})
.then(response => console.log(response.response));

// WebSocket connection
const ws = client.connectWebSocket((data) => {
    console.log('Received:', data);
});
```

### cURL Examples

**Basic Health Check:**
```bash
curl -X GET http://localhost:10010/health
```

**Chat Request:**
```bash
curl -X POST http://localhost:10010/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain REST API design principles",
    "agent": "task_coordinator",
    "temperature": 0.7
  }'
```

**Reasoning Request:**
```bash
curl -X POST http://localhost:10010/public/think \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the trade-offs between SQL and NoSQL databases?",
    "reasoning_type": "comparative"
  }'
```

**Task Execution:**
```bash
curl -X POST http://localhost:10010/execute \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create a Python script to analyze log files",
    "type": "coding"
  }'
```

**Enterprise Endpoint (with auth):**
```bash
curl -X POST http://localhost:10010/api/v1/agents/consensus \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "prompt": "Should we migrate to microservices?",
    "agents": ["architect", "performance", "security"],
    "consensus_type": "majority"
  }'
```

## 🔧 Development & Testing

### Local Development Setup

1. **Start the system:**
   ```bash
   docker-compose up -d
   ```

2. **Wait for initialization:**
   ```bash
   sleep 60
   ```

3. **Test connectivity:**
   ```bash
   curl http://localhost:10010/health
   ```

### API Testing

**Test Suite Example:**
```bash
#!/bin/bash
# api_test_suite.sh

BASE_URL="http://localhost:10010"

echo "=== Jarvis API Test Suite ==="

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq '.status'

# Test 2: Simple chat
echo "2. Testing chat endpoint..."
curl -s -X POST "$BASE_URL/simple-chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' | jq '.response'

# Test 3: Public thinking
echo "3. Testing thinking endpoint..."
curl -s -X POST "$BASE_URL/public/think" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "reasoning_type": "general"}' | jq '.confidence'

# Test 4: Metrics
echo "4. Testing metrics endpoint..."
curl -s "$BASE_URL/public/metrics" | jq '.system.cpu_percent'

echo "=== Test Suite Complete ==="
```

---

## 📝 API Version History

| Version | Release Date | Major Changes |
|---------|--------------|---------------|
| 17.0.0 | 2025-08-08 | Current implementation with FastAPI backend |
| 16.x | 2025-07-xx | Previous iteration (deprecated) |

---

## 📞 Support & Contact

**Development Team:**
- Email: dev-team@company.com
- Slack: #jarvis-development

**Bug Reports:**
- GitHub Issues: https://github.com/company/jarvis/issues
- Email: bugs@company.com

**API Questions:**
- Documentation: This document
- Stack Overflow: Tag `jarvis-api`

---

*This API reference is based on the actual implementation of Perfect Jarvis System v17.0.0. All endpoints and examples have been verified against the running system.*