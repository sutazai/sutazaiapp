# Phase 4: FastAPI Backend Deployment Plan

## Overview
Deploy a comprehensive FastAPI backend that integrates with all deployed services for the SutazAI Platform.

## Architecture Design

### Core Components
1. **FastAPI Application** (Port 10200)
   - Async/await pattern for all database operations
   - Connection pooling for all services
   - Automatic OpenAPI documentation
   - Health check endpoints for each service

2. **Service Integrations**
   - PostgreSQL: SQLAlchemy with async support
   - Redis: Connection pooling for caching
   - RabbitMQ: Event-driven messaging with aio-pika
   - ChromaDB: Vector embeddings storage
   - Qdrant: Similarity search operations
   - FAISS: Fast approximate nearest neighbor search
   - Neo4j: Graph database for relationships
   - Consul: Service discovery and configuration
   - Kong: API gateway integration

## Implementation Steps

### Step 1: Create Backend Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       └── router.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── security.py
│   ├── models/
│   ├── schemas/
│   └── services/
├── requirements.txt
├── Dockerfile
└── docker-compose-backend.yml
```

### Step 2: Core Dependencies
```python
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
sqlalchemy==2.0.35
asyncpg==0.30.0
redis==5.2.0
aio-pika==9.5.0
httpx==0.28.0
pydantic==2.10.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.12
```

### Step 3: Database Connections
- Implement async connection pooling
- Create database models with SQLAlchemy
- Setup Redis caching layer
- Configure vector database clients

### Step 4: API Endpoints
```python
# Core endpoints needed:
POST   /api/v1/agents/create
GET    /api/v1/agents/list
POST   /api/v1/chat/message
GET    /api/v1/health/check
POST   /api/v1/vectors/store
POST   /api/v1/vectors/search
POST   /api/v1/tasks/submit
GET    /api/v1/tasks/status/{task_id}
```

### Step 5: Service Registration
- Register with Consul for service discovery
- Configure Kong routes for API gateway
- Setup health checks for monitoring

### Step 6: Security Implementation
- JWT authentication
- API key management
- Rate limiting with Redis
- CORS configuration

### Step 7: Testing
- Unit tests with pytest
- Integration tests for each service
- Load testing with locust
- API documentation verification

## Docker Configuration

```yaml
# docker-compose-backend.yml
services:
  backend:
    build: ./backend
    container_name: sutazai-backend
    ports:
      - "10200:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:password@sutazai-postgres:5432/sutazai
      REDIS_URL: redis://sutazai-redis:6379
      RABBITMQ_URL: amqp://guest:guest@sutazai-rabbitmq:5672
      CHROMADB_URL: http://sutazai-chromadb:8000
      QDRANT_URL: http://sutazai-qdrant:6334
      FAISS_URL: http://sutazai-faiss:8000
      NEO4J_URL: bolt://sutazai-neo4j:7687
      CONSUL_URL: http://sutazai-consul:8500
      KONG_ADMIN_URL: http://sutazai-kong:8001
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.30
    depends_on:
      - postgres
      - redis
      - rabbitmq
      - chromadb
      - qdrant
      - faiss
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Success Criteria
- [ ] All service connections established
- [ ] Health checks passing for all services
- [ ] API documentation accessible at /docs
- [ ] Authentication system functional
- [ ] Vector storage and retrieval working
- [ ] Message queue integration operational
- [ ] Service registered with Consul
- [ ] Routes configured in Kong

## Next Actions
1. Create backend directory structure
2. Install FastAPI and dependencies
3. Implement database connections
4. Create API endpoints
5. Test all integrations
6. Deploy with Docker Compose