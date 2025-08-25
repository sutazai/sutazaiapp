# SutazAI System Architecture - Comprehensive Map

## System Overview
SutazAI is a comprehensive AI orchestration platform with multi-agent workflow system, service mesh architecture, and distributed processing capabilities.

## 🏗️ Core Architecture Components

### 1. **Application Layer**
```
┌─────────────────┐    ┌──────────────────────┐
│   Frontend      │    │      Backend         │
│   (Streamlit)   │◄──►│     (FastAPI)        │
│   Port: 10011   │    │    Port: 10010       │
└─────────────────┘    └──────────────────────┘
        │                         │
        │                         ▼
        │              ┌──────────────────────┐
        │              │  API Gateway (Kong)  │
        │              │    Port: 10005       │
        │              └──────────────────────┘
        │                         │
        ▼                         ▼
┌─────────────────┐    ┌──────────────────────┐
│  Agent Control  │    │   Service Mesh       │
│   Dashboard     │    │   (Consul Registry)  │
│                 │    │    Port: 10006       │
└─────────────────┘    └──────────────────────┘
```

### 2. **Database Layer**
```
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│   PostgreSQL     │  │     Redis       │  │     Neo4j        │
│  (Primary DB)    │  │   (Cache)       │  │  (Graph DB)      │
│  Port: 10000     │  │  Port: 10001    │  │  Port: 10002-3   │
└──────────────────┘  └─────────────────┘  └──────────────────┘
```

### 3. **AI & Vector Services**
```
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│    ChromaDB      │  │     Qdrant      │  │      FAISS       │
│  Port: 10100     │  │  Port: 10101-2  │  │  Port: 10103     │
└──────────────────┘  └─────────────────┘  └──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Ollama      │
                    │   (LLM Engine)  │
                    │  Port: 10104    │
                    └─────────────────┘
```

### 4. **Monitoring Stack**
```
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│   Prometheus     │  │    Grafana      │  │      Loki        │
│  Port: 10200     │  │  Port: 10201    │  │  Port: 10202     │
└──────────────────┘  └─────────────────┘  └──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Alertmanager   │
                    │  Port: 10203    │
                    └─────────────────┘
```

## 🔄 Service Dependencies & Relationships

### Primary Dependencies
```
Frontend → Backend → [Database Layer]
             ↓
        API Gateway (Kong) → Service Mesh (Consul)
             ↓
        MCP Orchestrator → [MCP Services]
```

### Database Relationships
```
Backend ──┬── PostgreSQL (User data, tasks, agents)
          ├── Redis (Caching, sessions)
          ├── Neo4j (Knowledge graph)
          ├── ChromaDB (Vector embeddings)
          ├── Qdrant (Vector search)
          └── FAISS (Vector similarity)
```

### Service Mesh Communication
```
Consul Registry ──┬── Kong API Gateway
                  ├── MCP Services (11103-11128)
                  ├── Agent Services (8588-8589)
                  └── Hardware Optimizer (11110)
```

## 🛠️ API Structure & Endpoints

### Core API Endpoints (Port 10010)
```
/api/v1/
├── health                    # System health check
├── status                    # System status
├── agents/                   # Agent management
│   ├── GET /                 # List all agents
│   └── GET /{agent_id}       # Get specific agent
├── tasks/                    # Task management
│   ├── POST /                # Create task
│   └── GET /{task_id}        # Get task status
├── chat/                     # Chat interface
│   ├── POST /                # Standard chat
│   └── POST /stream          # Streaming chat
├── mesh/                     # Service mesh
│   ├── GET /status           # Mesh status
│   ├── GET /v2/services      # Service discovery
│   └── POST /v2/enqueue      # Task enqueuing
├── cache/                    # Cache management
│   ├── POST /clear           # Clear cache
│   ├── POST /invalidate      # Invalidate by tags
│   └── GET /stats            # Cache statistics
├── hardware/                 # Hardware optimization
├── mcp/                      # MCP integration
├── auth/                     # Authentication
└── metrics                   # Prometheus metrics
```

### Frontend Pages Structure
```
Pages/
├── Dashboard                 # Main system overview
├── AI Chat                   # Chat interface
├── Agent Control             # Agent management
└── Hardware Optimization     # Resource optimization
```

## 🗄️ Database Schemas

### PostgreSQL Schema
```sql
users
├── id (SERIAL PRIMARY KEY)
├── username (VARCHAR UNIQUE)
├── email (VARCHAR UNIQUE)
├── password_hash (VARCHAR)
├── is_active (BOOLEAN)
└── timestamps

agents
├── id (SERIAL PRIMARY KEY)
├── name (VARCHAR UNIQUE)
├── type (VARCHAR)
├── description (TEXT)
├── endpoint (VARCHAR)
├── capabilities (JSONB)
└── timestamps

tasks
├── id (SERIAL PRIMARY KEY)
├── title (VARCHAR)
├── agent_id (FK → agents.id)
├── user_id (FK → users.id)
├── status (VARCHAR)
├── payload (JSONB)
├── result (JSONB)
└── timestamps

chat_history
├── id (SERIAL PRIMARY KEY)
├── user_id (FK → users.id)
├── message (TEXT)
├── response (TEXT)
├── agent_used (VARCHAR)
└── timestamps
```

### Agent Registry Structure
```yaml
UnifiedAgent:
  id: string
  name: string
  type: "claude" | "container" | "external"
  description: string
  capabilities: [string]
  priority: integer
  deployment_info:
    method: string
    config_path: string
    container_name: string
```

## 🚀 Deployment Structure

### Docker Compose Services (17 containers)
```yaml
Core Infrastructure:
├── postgres (Database)
├── redis (Cache)
├── neo4j (Graph DB)
├── rabbitmq (Message Queue)
├── kong (API Gateway)
└── consul (Service Registry)

Applications:
├── backend (FastAPI)
└── frontend (Streamlit)

AI Services:
├── chromadb (Vector DB)
├── qdrant (Vector Search)
├── faiss (Vector Similarity)
└── ollama (LLM Engine)

Monitoring:
├── prometheus (Metrics)
├── grafana (Dashboards)
├── loki (Logs)
├── alertmanager (Alerts)
└── node-exporter (System Metrics)

MCP Orchestration:
├── mcp-orchestrator (Docker-in-Docker)
└── mcp-manager (MCP Control)
```

### MCP Services Architecture
```yaml
MCP Services (Ports 11103-11128):
├── files (File operations)
├── language-server (Code analysis)
├── ultimatecoder (Code generation)
├── sequentialthinking (AI reasoning)
├── context7 (Documentation)
├── ddg (Web search)
├── extended-memory (Persistence)
├── memory-bank-mcp (Memory cache)
├── mcp_ssh (Remote access)
├── nx-mcp (Monorepo tools)
├── playwright-mcp (Browser automation)
├── knowledge-graph-mcp (Graph reasoning)
└── compass-mcp (Discovery)
```

## 🔧 Configuration Management

### Port Registry
```yaml
Core Services: 10000-10011
Monitoring: 10200-10207
Vector DBs: 10100-10104
Agent Services: 8588-8589, 11110
MCP Services: 11103-11128
Docker API: 12375-12376
MCP Management: 18080-18081
```

### System Configuration Files
```
config/
├── core/
│   ├── system.yaml          # System settings
│   ├── ports.yaml           # Port assignments
│   └── docker.yaml          # Docker configuration
├── agents/
│   └── unified_agent_registry.json
├── mcp/
│   └── mcp_mesh_registry.yaml
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   └── alertmanager.yml
└── security/
    └── semgrep_custom_rules.yaml
```

## 🔐 Security Architecture

### Authentication Flow
```
User Request → SecurityMiddleware → JWT Validation → User Context
                      ↓
API Key Middleware → Service Authentication → Rate Limiting
                      ↓
CORS Security → Origin Validation → Request Processing
```

### Security Features
- JWT-based authentication
- API key middleware for service-to-service communication
- Rate limiting and DDoS protection
- CORS security with explicit origin whitelisting
- Input validation and sanitization
- Circuit breaker patterns for resilience

## 🚦 Data Flow Architecture

### Request Flow
```
Frontend (Streamlit) 
    ↓ HTTP/WebSocket
Backend (FastAPI)
    ↓ Connection Pool
Database Layer (PostgreSQL/Redis/Neo4j)
    ↓ Service Mesh
MCP Services / Agent Services
    ↓ Message Queue
Task Processing & Results
```

### Agent Communication
```
Unified Agent Registry ←→ Service Discovery
         ↓
Agent Selection & Routing
         ↓
Task Orchestration (Service Mesh)
         ↓
Distributed Processing
         ↓
Result Aggregation & Caching
```

## 📊 Monitoring & Observability

### Metrics Collection
```
Application Metrics → Prometheus → Grafana Dashboards
System Metrics → Node Exporter → Prometheus
Logs → Loki → Grafana Log Viewer
Alerts → Alertmanager → Notification Channels
```

### Health Check Hierarchy
```
System Health → Service Health → Component Health
      ↓                ↓              ↓
Load Balancer     Circuit Breakers  Retry Logic
```

## 🔄 Scalability Patterns

### Horizontal Scaling
- MCP services with multiple instances
- Load balancing with Kong API Gateway
- Connection pooling for database efficiency
- Distributed caching with Redis

### Performance Optimizations
- Circuit breaker patterns for fault tolerance
- Connection pooling for database access
- Multi-level caching (Redis + application cache)
- Async processing with background tasks
- Resource optimization and monitoring

This comprehensive architecture map shows SutazAI as a sophisticated, production-ready AI orchestration platform with robust infrastructure, security, and scalability features.