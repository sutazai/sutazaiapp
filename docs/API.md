# SutazAI API Documentation

## Overview

SutazAI provides a comprehensive REST API for programmatic access to all system functionality. The API is built with FastAPI and provides automatic OpenAPI documentation.

## API Base URLs

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`
- **API Version**: `v1`

## Authentication

### API Key Authentication
```bash
# Include API key in headers
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/status
```

### Bearer Token Authentication
```bash
# Include bearer token
curl -H "Authorization: Bearer your-jwt-token" http://localhost:8000/api/v1/status
```

## Core Endpoints

### System Status

#### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "ai_models": "healthy",
    "cache": "healthy"
  }
}
```

#### GET /api/v1/status
Detailed system status.

```bash
curl http://localhost:8000/api/v1/status
```

Response:
```json
{
  "system": "operational",
  "ai_agents": {
    "total": 4,
    "active": 4,
    "idle": 0
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 68.1,
    "disk_usage": 23.4
  },
  "neural_network": {
    "total_nodes": 1000,
    "active_connections": 5000,
    "global_activity": 0.75
  }
}
```

## AI Generation Endpoints

### POST /api/v1/generate/code
Generate code using the Code Generation Module.

**Request Body:**
```json
{
  "prompt": "Create a Python function to calculate fibonacci numbers",
  "language": "python",
  "style": "functional",
  "complexity": "intermediate",
  "include_tests": true,
  "include_docs": true
}
```

**Response:**
```json
{
  "generated_code": "def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)",
  "tests": "def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5",
  "documentation": "## Fibonacci Function

Calculates the nth fibonacci number...",
  "quality_score": 0.92,
  "execution_time": 0.245,
  "model_used": "code-llama-7b"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/generate/code   -H "Content-Type: application/json"   -d '{
    "prompt": "Create a REST API endpoint",
    "language": "python",
    "include_tests": true
  }'
```

### POST /api/v1/generate/text
Generate text using language models.

**Request Body:**
```json
{
  "prompt": "Explain quantum computing",
  "max_length": 500,
  "temperature": 0.7,
  "style": "academic",
  "format": "markdown"
}
```

**Response:**
```json
{
  "generated_text": "# Quantum Computing

Quantum computing represents...",
  "word_count": 487,
  "confidence_score": 0.89,
  "execution_time": 1.23,
  "model_used": "mistral-7b"
}
```

## Knowledge Graph Endpoints

### GET /api/v1/knowledge/search
Search the knowledge graph.

**Query Parameters:**
- `q` (string): Search query
- `limit` (int): Maximum results (default: 10)
- `offset` (int): Pagination offset (default: 0)
- `type` (string): Entity type filter

```bash
curl "http://localhost:8000/api/v1/knowledge/search?q=machine%20learning&limit=5"
```

**Response:**
```json
{
  "results": [
    {
      "id": "ml_001",
      "title": "Introduction to Machine Learning",
      "content": "Machine learning is a subset of artificial intelligence...",
      "type": "concept",
      "confidence": 0.95,
      "relationships": [
        {"type": "related_to", "entity": "artificial_intelligence"},
        {"type": "has_subcategory", "entity": "deep_learning"}
      ]
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 5
}
```

### POST /api/v1/knowledge/add
Add knowledge to the graph.

**Request Body:**
```json
{
  "title": "Neural Networks",
  "content": "Neural networks are computing systems...",
  "type": "concept",
  "tags": ["ai", "machine_learning", "neural_networks"],
  "relationships": [
    {"type": "subcategory_of", "target": "machine_learning"}
  ]
}
```

**Response:**
```json
{
  "id": "nn_001",
  "status": "added",
  "relationships_created": 3,
  "indexed": true
}
```

## Neural Network Endpoints

### GET /api/v1/neural/status
Get neural network status.

**Response:**
```json
{
  "network_state": {
    "total_nodes": 1000,
    "active_nodes": 850,
    "total_connections": 5000,
    "active_connections": 4200,
    "global_activity": 0.75,
    "learning_rate": 0.01
  },
  "performance": {
    "average_response_time": 0.023,
    "throughput": 1250.5,
    "error_rate": 0.001
  },
  "recent_activity": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "event": "learning_update",
      "nodes_affected": 45
    }
  ]
}
```

### POST /api/v1/neural/stimulate
Stimulate neural network nodes.

**Request Body:**
```json
{
  "node_ids": ["node_001", "node_002"],
  "stimulus_strength": 0.8,
  "duration": 1000,
  "pattern": "sequential"
}
```

**Response:**
```json
{
  "stimulation_id": "stim_123",
  "nodes_stimulated": 2,
  "propagation_paths": 15,
  "response_time": 0.045,
  "network_changes": {
    "synaptic_weights_updated": 23,
    "new_connections": 2,
    "pruned_connections": 1
  }
}
```

## AI Agent Endpoints

### GET /api/v1/agents
List available AI agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "autogpt_001",
      "name": "AutoGPT Agent",
      "type": "autonomous",
      "status": "active",
      "capabilities": ["task_execution", "web_browsing", "file_operations"],
      "current_task": "code_optimization",
      "performance": {
        "tasks_completed": 157,
        "success_rate": 0.94,
        "average_execution_time": 45.2
      }
    },
    {
      "id": "localagi_001", 
      "name": "Local AGI",
      "type": "general",
      "status": "active",
      "capabilities": ["reasoning", "planning", "learning"],
      "current_task": null,
      "performance": {
        "tasks_completed": 89,
        "success_rate": 0.97,
        "average_execution_time": 12.8
      }
    }
  ]
}
```

### POST /api/v1/agents/{agent_id}/tasks
Assign task to an agent.

**Request Body:**
```json
{
  "task_type": "code_generation",
  "description": "Create a web scraper for news articles",
  "requirements": {
    "language": "python",
    "libraries": ["requests", "beautifulsoup4"],
    "output_format": "json"
  },
  "priority": "high",
  "deadline": "2024-01-02T00:00:00Z"
}
```

**Response:**
```json
{
  "task_id": "task_456",
  "assigned_to": "autogpt_001",
  "status": "accepted",
  "estimated_completion": "2024-01-01T14:30:00Z",
  "tracking_url": "/api/v1/tasks/task_456"
}
```

## Model Management Endpoints

### GET /api/v1/models
List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "code_llama_7b",
      "name": "Code Llama 7B",
      "type": "code_generation",
      "status": "loaded",
      "size": "3.8GB",
      "capabilities": ["code_completion", "code_generation", "code_explanation"],
      "performance": {
        "tokens_per_second": 45.2,
        "memory_usage": "6.2GB",
        "gpu_utilization": 78.5
      }
    }
  ]
}
```

### POST /api/v1/models/{model_id}/generate
Generate using specific model.

**Request Body:**
```json
{
  "prompt": "def calculate_prime_numbers(n):",
  "max_tokens": 200,
  "temperature": 0.3,
  "stop_sequences": ["

"]
}
```

## Data Management Endpoints

### POST /api/v1/documents/upload
Upload document for processing.

**Form Data:**
- `file`: Document file
- `category`: Document category
- `tags`: Comma-separated tags

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload   -F "file=@document.pdf"   -F "category=research"   -F "tags=ai,machine_learning"
```

**Response:**
```json
{
  "document_id": "doc_789",
  "filename": "document.pdf",
  "size": 2048576,
  "pages": 15,
  "processing_status": "queued",
  "extracted_entities": 23,
  "indexed": true
}
```

### GET /api/v1/documents/{document_id}
Get document information.

**Response:**
```json
{
  "id": "doc_789",
  "filename": "document.pdf",
  "upload_date": "2024-01-01T10:00:00Z",
  "size": 2048576,
  "pages": 15,
  "category": "research",
  "tags": ["ai", "machine_learning"],
  "processing_status": "completed",
  "content_summary": "This document discusses advanced machine learning techniques...",
  "extracted_entities": [
    {"text": "neural networks", "type": "concept", "confidence": 0.95},
    {"text": "deep learning", "type": "concept", "confidence": 0.92}
  ]
}
```

## Analytics Endpoints

### GET /api/v1/analytics/performance
Get system performance analytics.

**Query Parameters:**
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `metric`: Specific metric (optional)

**Response:**
```json
{
  "timeframe": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-01T23:59:59Z"
  },
  "metrics": {
    "cpu_usage": {
      "average": 45.2,
      "peak": 89.1,
      "minimum": 12.3
    },
    "memory_usage": {
      "average": 68.1,
      "peak": 92.4,
      "minimum": 34.7
    },
    "api_requests": {
      "total": 15420,
      "success_rate": 99.2,
      "average_response_time": 0.145
    }
  }
}
```

## WebSocket Endpoints

### WS /api/v1/ws/chat
Real-time chat interface.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Send message
ws.send(JSON.stringify({
  type: 'message',
  content: 'Hello, SutazAI!'
}));
```

**Message Format:**
```json
{
  "type": "message",
  "content": "Your message here",
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "prompt",
      "issue": "Field is required"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Default**: 100 requests per minute
- **Generation endpoints**: 10 requests per minute
- **Upload endpoints**: 5 requests per minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## SDKs and Libraries

### Python SDK
```python
from sutazai import SutazAIClient

client = SutazAIClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Generate code
result = client.generate_code("Create a REST API")
print(result.code)

# Search knowledge
results = client.search_knowledge("machine learning")
for result in results:
    print(result.title)
```

### JavaScript SDK
```javascript
import { SutazAIClient } from '@sutazai/sdk';

const client = new SutazAIClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Generate text
const result = await client.generateText({
  prompt: 'Explain quantum computing',
  maxLength: 500
});

console.log(result.text);
```

## Interactive Documentation

Visit the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces provide:
- Complete API reference
- Interactive request testing
- Request/response examples
- Authentication testing

---

For more information, see the [API Reference](https://docs.sutazai.com/api) or contact support@sutazai.com.
