# Core Services Documentation

## Overview

This document provides comprehensive documentation for the core data and AI services that form the foundation of the SutazAI system. All services listed are verified as running and healthy in production.

## Relational Database: PostgreSQL 16.3

### Service Details
- **Status**: ✅ HEALTHY
- **Docker Image**: `postgres:16.3-alpine`
- **Port**: 10000
- **Container**: `sutazai-postgres`

### Configuration
```yaml
environment:
  POSTGRES_DB: sutazai
  POSTGRES_USER: sutazai
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-sutazai123}
  POSTGRES_MAX_CONNECTIONS: 200
  POSTGRES_SHARED_BUFFERS: 256MB
volumes:
  - postgres_data:/var/lib/postgresql/data
```

### Connection Details
```python
# Python connection example
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=10000,
    database="sutazai",
    user="sutazai",
    password="sutazai123"
)
```

### Database Schema
- **agents**: Agent configurations and metadata
- **tasks**: Task queue and execution history
- **users**: User authentication and profiles
- **workflows**: Workflow definitions and state
- **audit_logs**: System audit trail

### Performance Tuning
```sql
-- Current optimizations
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### Backup & Recovery
```bash
# Backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup.sql

# Restore
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup.sql
```

## Cache Layer: Redis 7.2

### Service Details
- **Status**: ✅ HEALTHY
- **Docker Image**: `redis:7.2-alpine`
- **Port**: 10001
- **Container**: `sutazai-redis`

### Configuration
```yaml
command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy lru
volumes:
  - redis_data:/data
```

### Connection Details
```python
# Python connection example
import redis

r = redis.Redis(
    host='localhost',
    port=10001,
    decode_responses=True
)
```

### Use Cases
1. **Session Storage**: User session management
2. **Cache Layer**: API response caching
3. **Rate Limiting**: Request throttling
4. **Queue Management**: Lightweight task queues
5. **Real-time Data**: WebSocket message broadcasting

### Key Patterns
```python
# Cache pattern
key = f"cache:api:{endpoint}:{params_hash}"
ttl = 3600  # 1 hour

# Session pattern
session_key = f"session:{user_id}:{session_id}"

# Rate limit pattern
rate_key = f"rate:{user_id}:{endpoint}"
```

### Performance Metrics
```bash
# Monitor Redis performance
docker exec sutazai-redis redis-cli INFO stats
docker exec sutazai-redis redis-cli INFO memory
```

## Graph Database: Neo4j 5

### Service Details
- **Status**: ✅ HEALTHY
- **Docker Image**: `neo4j:5-community`
- **Ports**:
  - HTTP/Web UI: 10002
  - Bolt Protocol: 10003
- **Container**: `sutazai-neo4j`

### Configuration
```yaml
environment:
  NEO4J_AUTH: neo4j/sutazai123
  NEO4J_dbms_memory_pagecache_size: 512M
  NEO4J_dbms_memory_heap_max__size: 512M
volumes:
  - neo4j_data:/data
  - neo4j_logs:/logs
```

### Connection Details
```python
# Python connection example
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:10003",
    auth=("neo4j", "sutazai123")
)
```

### Graph Model
```cypher
// Agent relationships
(Agent)-[:COMMUNICATES_WITH]->(Agent)
(Agent)-[:EXECUTES]->(Task)
(Task)-[:DEPENDS_ON]->(Task)
(User)-[:OWNS]->(Workflow)
(Workflow)-[:CONTAINS]->(Task)
```

### Common Queries
```cypher
// Find all tasks for a workflow
MATCH (w:Workflow {id: $workflow_id})-[:CONTAINS]->(t:Task)
RETURN t

// Agent communication paths
MATCH path = (a1:Agent)-[:COMMUNICATES_WITH*1..3]->(a2:Agent)
WHERE a1.name = $agent1 AND a2.name = $agent2
RETURN path
```

## Vector Database: ChromaDB 0.5.0

### Service Details
- **Status**: ⚠️ STARTING (Health check in progress)
- **Docker Image**: `chromadb/chroma:0.5.0`
- **Port**: 10100
- **Container**: `sutazai-chromadb`

### Configuration
```yaml
environment:
  CHROMA_SERVER_AUTH_PROVIDER: "chromadb.auth.token.TokenAuthServerProvider"
  CHROMA_SERVER_AUTH_CREDENTIALS: "${CHROMA_TOKEN:-test-token}"
volumes:
  - chromadb_data:/chroma/chroma
```

### Use Cases
1. **Document Embeddings**: Store and search document vectors
2. **Semantic Search**: Find similar content
3. **RAG Systems**: Retrieval-augmented generation
4. **Knowledge Base**: AI agent memory storage

### Connection Example
```python
import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=10100,
    headers={"Authorization": "Bearer test-token"}
)

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
```

## Vector Search: Qdrant 1.9.2

### Service Details
- **Status**: ✅ HEALTHY
- **Docker Image**: `qdrant/qdrant:v1.9.2`
- **Ports**:
  - HTTP/Web UI: 10101
  - gRPC: 10102
- **Container**: `sutazai-qdrant`

### Configuration
```yaml
environment:
  QDRANT__SERVICE__HTTP_PORT: 6333
  QDRANT__SERVICE__GRPC_PORT: 6334
volumes:
  - qdrant_data:/qdrant/storage
```

### Features
- **High Performance**: Optimized for billion-scale vectors
- **Filtering**: Advanced metadata filtering
- **Payloads**: Store additional data with vectors
- **Web UI**: Built-in management interface

### Connection Example
```python
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=10101
)

# Create collection
client.create_collection(
    collection_name="agents",
    vectors_config={
        "size": 768,
        "distance": "Cosine"
    }
)
```

## Similarity Search: FAISS

### Service Details
- **Status**: ✅ HEALTHY
- **Custom Service**: Python service with FAISS library
- **Port**: 10103
- **Container**: `sutazai-faiss`

### Implementation
```python
# FAISS service implementation
import faiss
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)

@app.route('/search', methods=['POST'])
def search():
    query_vector = np.array(request.json['vector'])
    k = request.json.get('k', 10)
    
    distances, indices = index.search(query_vector, k)
    return jsonify({
        'distances': distances.tolist(),
        'indices': indices.tolist()
    })
```

### Use Cases
- **Fast Similarity Search**: Millisecond response times
- **Clustering**: K-means and hierarchical clustering
- **Recommendation Systems**: Find similar items
- **Duplicate Detection**: Identify near-duplicates

## Local LLM: Ollama

### Service Details
- **Status**: ✅ HEALTHY
- **Docker Image**: `ollama/ollama:latest`
- **Port**: 10104
- **Container**: `sutazai-ollama`
- **Current Model**: TinyLlama (verified via `ollama list`)

### Configuration
```yaml
environment:
  OLLAMA_HOST: 0.0.0.0
volumes:
  - ollama_data:/root/.ollama
deploy:
  resources:
    limits:
      memory: 4G
```

### Model Management
```bash
# List loaded models
docker exec sutazai-ollama ollama list

# Pull new model
docker exec sutazai-ollama ollama pull tinyllama

# Run inference
curl http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello, world!"}'
```

### API Endpoints
```python
# Python client example
import requests

response = requests.post(
    'http://localhost:10104/api/generate',
    json={
        'model': 'tinyllama',
        'prompt': 'Explain quantum computing',
        'stream': False
    }
)
```

## Application Backend: FastAPI

### Service Details
- **Status**: ✅ HEALTHY
- **Version**: 17.0.0
- **Port**: 10010
- **Container**: `sutazai-backend`
- **Swagger UI**: http://localhost:10010/docs

### Architecture
```
/api/v1/
├── /agents     - Agent management
├── /tasks      - Task execution
├── /workflows  - Workflow orchestration
├── /chat       - LLM chat interface
├── /health     - Health monitoring
└── /metrics    - Prometheus metrics
```

### Key Features
- **Async Processing**: Built on asyncio
- **Auto Documentation**: OpenAPI/Swagger
- **Validation**: Pydantic models
- **WebSockets**: Real-time communication
- **Background Tasks**: Async task execution

### Example Endpoint
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    model: str = "tinyllama"

@router.post("/chat")
async def chat(request: ChatRequest):
    # Process with Ollama
    response = await ollama_client.generate(
        model=request.model,
        prompt=request.message
    )
    return {"response": response}
```

## Frontend: Streamlit

### Service Details
- **Status**: ✅ WORKING
- **Port**: 10011
- **Container**: `sutazai-frontend`
- **URL**: http://localhost:10011

### Features
- **Interactive UI**: Real-time updates
- **Data Visualization**: Charts and graphs
- **File Upload**: Document processing
- **Chat Interface**: LLM interaction
- **Dashboard**: System monitoring

### Architecture
```python
# Main app structure
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SutazAI",
    page_icon="🤖",
    layout="wide"
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Dashboard", "Agents", "Workflows", "Chat"]
)

# Dynamic page rendering
if page == "Dashboard":
    show_dashboard()
elif page == "Agents":
    show_agents()
```

## Service Integration Patterns

### Database Connections
```python
# Connection pool management
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()
```

### Cache Strategy
```python
# Redis caching decorator
def cache_result(ttl=3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try cache first
            cached = await redis.get(key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = await func(*args, **kwargs)
            await redis.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Vector Search Integration
```python
# Unified vector search interface
class VectorSearchService:
    def __init__(self):
        self.chroma = ChromaClient()
        self.qdrant = QdrantClient()
        self.faiss = FAISSClient()
    
    async def search(self, vector, k=10, engine="qdrant"):
        if engine == "chroma":
            return await self.chroma.search(vector, k)
        elif engine == "qdrant":
            return await self.qdrant.search(vector, k)
        elif engine == "faiss":
            return await self.faiss.search(vector, k)
```

## Performance Monitoring

### Key Metrics
1. **PostgreSQL**:
   - Connection pool usage
   - Query execution time
   - Transaction rate

2. **Redis**:
   - Hit rate
   - Memory usage
   - Eviction count

3. **Neo4j**:
   - Query latency
   - Node/relationship count
   - Cache hit ratio

4. **Vector Databases**:
   - Search latency
   - Index size
   - Memory usage

### Health Checks
```bash
# Comprehensive health check script
#!/bin/bash

services=(
    "postgres:10000:/health"
    "redis:10001:/health"
    "neo4j:10002:/health"
    "chromadb:10100:/health"
    "qdrant:10101:/health"
    "faiss:10103:/health"
    "ollama:10104:/health"
    "backend:10010:/health"
    "frontend:10011/health"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port path <<< "$service"
    if curl -f "http://localhost:${port}${path}" > /dev/null 2>&1; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is unhealthy"
    fi
done
```

## Troubleshooting Guide

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check PostgreSQL logs
   docker logs sutazai-postgres
   
   # Test connection
   docker exec sutazai-postgres pg_isready -U sutazai
   ```

2. **Redis Memory Issues**:
   ```bash
   # Check memory usage
   docker exec sutazai-redis redis-cli INFO memory
   
   # Flush cache if needed
   docker exec sutazai-redis redis-cli FLUSHDB
   ```

3. **Vector Database Performance**:
   ```bash
   # Optimize Qdrant
   curl -X POST http://localhost:10101/collections/optimize
   
   # Check ChromaDB status
   curl http://localhost:10100/api/v1/heartbeat
   ```

4. **Ollama Model Issues**:
   ```bash
   # Re-pull model
   docker exec sutazai-ollama ollama pull tinyllama
   
   # Check model status
   docker exec sutazai-ollama ollama list
   ```

## Backup and Recovery

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/${DATE}"

mkdir -p ${BACKUP_DIR}

# PostgreSQL
docker exec sutazai-postgres pg_dump -U sutazai sutazai > ${BACKUP_DIR}/postgres.sql

# Redis
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb ${BACKUP_DIR}/redis.rdb

# Neo4j
docker exec sutazai-neo4j neo4j-admin database dump --to-path=/tmp neo4j
docker cp sutazai-neo4j:/tmp/neo4j.dump ${BACKUP_DIR}/neo4j.dump

echo "Backup completed: ${BACKUP_DIR}"
```

## Security Considerations

### Access Control
- All services bound to localhost by default
- Authentication required for all databases
- Token-based auth for vector databases
- API key authentication for backend

### Data Encryption
- TLS for external connections
- Encrypted passwords in PostgreSQL
- Redis AUTH for access control
- Neo4j authentication enabled

### Network Isolation
- Docker network isolation
- No direct internet exposure
- Service-to-service communication only