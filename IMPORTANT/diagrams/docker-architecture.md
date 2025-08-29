# Docker Architecture Diagram

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit UI :11000]
    end
    
    subgraph "API Gateway Layer"
        KONG[Kong API Gateway :10008-10009]
    end
    
    subgraph "Backend Services"
        BACKEND[FastAPI Backend :10200]
        MCP[MCP Bridge :11100]
    end
    
    subgraph "Data Layer"
        POSTGRES[(PostgreSQL :10000)]
        REDIS[(Redis :10001)]
        NEO4J[(Neo4j :10002-10003)]
        RABBITMQ[RabbitMQ :10004-10005]
    end
    
    subgraph "Vector Databases"
        CHROMA[(ChromaDB :10100)]
        QDRANT[(Qdrant :10101-10102)]
        FAISS[(FAISS :10103)]
    end
    
    subgraph "AI Agents"
        LETTA[Letta Agent :11401]
        AUTOGPT[AutoGPT :11402]
        CREWAI[CrewAI :11403]
        AIDER[Aider :11404]
        LANGCHAIN[LangChain :11405]
        OLLAMA[Ollama :11435]
        SEMGREP[Semgrep :11801]
    end
    
    UI --> KONG
    KONG --> BACKEND
    BACKEND --> MCP
    BACKEND --> POSTGRES
    BACKEND --> REDIS
    BACKEND --> NEO4J
    BACKEND --> RABBITMQ
    MCP --> LETTA
    MCP --> AUTOGPT
    MCP --> CREWAI
    MCP --> AIDER
    MCP --> LANGCHAIN
    BACKEND --> CHROMA
    BACKEND --> QDRANT
    BACKEND --> FAISS
    LANGCHAIN --> OLLAMA
```

## Network Configuration

| Service | IP Address | Port Range | Status |
|---------|------------|------------|--------|
| PostgreSQL | 172.20.0.10 | 10000 | ✅ Healthy |
| Redis | 172.20.0.11 | 10001 | ✅ Healthy |
| Neo4j | 172.20.0.12 | 10002-10003 | ✅ Healthy |
| RabbitMQ | 172.20.0.13 | 10004-10005 | ✅ Healthy |
| Backend | 172.20.0.30 | 10200 | ✅ Healthy |
| Frontend | 172.20.0.31 | 11000 | ✅ Fixed IP Conflict |
| MCP Bridge | 172.20.0.100 | 11100 | ✅ Healthy |
| ChromaDB | 172.20.0.20 | 10100 | ✅ Healthy |
| Qdrant | 172.20.0.21 | 10101-10102 | ✅ Healthy |
| FAISS | 172.20.0.22 | 10103 | ✅ Healthy |
| AI Agents | 172.20.0.101-199 | 11401-11801 | Mixed |
| Ollama | Dynamic | 11435 | ⚠️ Unhealthy |
| Semgrep | Dynamic | 11801 | ⚠️ Unhealthy |

## Docker Compose Structure

```yaml
# Core Infrastructure
docker-compose-core.yml:
  - PostgreSQL (Primary Database)
  - Redis (Cache & PubSub)
  - Neo4j (Graph Database)
  - RabbitMQ (Message Queue)
  - Kong (API Gateway)

# Backend Services
docker-compose-backend.yml:
  - FastAPI Backend Service
  - Service Connections Manager

# Frontend
docker-compose-frontend.yml:
  - Streamlit UI (Fixed IP: 172.20.0.31)

# Vector Databases
docker-compose-vectors.yml:
  - ChromaDB
  - Qdrant
  - FAISS

# AI Agents
docker-compose-agents.yml:
  - MCP Bridge
  - Letta, AutoGPT, CrewAI
  - Aider, LangChain, Ollama
  - Semgrep, and more
```

## Service Dependencies

```mermaid
graph LR
    subgraph "Phase 1: Core"
        P1[PostgreSQL, Redis, Neo4j, RabbitMQ]
    end
    
    subgraph "Phase 2: Vectors"
        P2[ChromaDB, Qdrant, FAISS]
    end
    
    subgraph "Phase 3: Backend"
        P3[Backend API]
    end
    
    subgraph "Phase 4: Services"
        P4[MCP Bridge, Agents, Frontend]
    end
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
```

## Port Allocation Strategy

### Core Services (10000-10099)
- 10000: PostgreSQL
- 10001: Redis
- 10002-10003: Neo4j
- 10004-10005: RabbitMQ
- 10008-10009: Kong API Gateway

### Vector Databases (10100-10199)
- 10100: ChromaDB
- 10101-10102: Qdrant
- 10103: FAISS

### Backend Services (10200-10299)
- 10200: FastAPI Backend

### Frontend (11000-11099)
- 11000: Streamlit UI

### MCP & Agents (11100-11999)
- 11100: MCP Bridge
- 11401-11499: Core Agents
- 11500-11599: Extended Agents
- 11600-11699: Specialized Agents
- 11700-11799: Browser Agents
- 11800-11899: Security Agents

## Health Check Endpoints

| Service | Health Check URL | Expected Response |
|---------|-----------------|-------------------|
| Backend | `http://localhost:10200/health` | `{"status": "healthy"}` |
| Frontend | `http://localhost:11000/_stcore/health` | `{"status": "ok"}` |
| MCP Bridge | `http://localhost:11100/health` | `{"status": "healthy"}` |
| ChromaDB | `http://localhost:10100/api/v1/heartbeat` | `{"nanosecond heartbeat": ...}` |
| Qdrant | `http://localhost:10101/health` | `{"title": "qdrant"}` |
| Neo4j | `http://localhost:10002` | Browser UI |
| RabbitMQ | `http://localhost:10005` | Management UI |

## Critical Issues Fixed

1. ✅ **Network IP Conflict**: Frontend moved from 172.20.0.30 to 172.20.0.31
2. ⚠️ **Ollama Memory**: Needs resource adjustment (using 24MB of 23GB allocated)
3. ⚠️ **Semgrep Health**: Health check configuration needs update
4. ✅ **Documentation**: Architecture diagrams created
5. ✅ **Script Organization**: Scripts moved to categorical subdirectories

Generated: 2025-08-29