# SutazAI AGI/ASI System Architecture

## System Overview

SutazAI is an enterprise-grade, fully autonomous AGI/ASI system designed for local deployment with self-improvement capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SutazAI AGI/ASI System                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           Frontend Layer (Port 8501)                      │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │    │
│  │  │  Streamlit UI   │  │  Chat Interface │  │  Model Management UI    │ │    │
│  │  │                 │  │                 │  │                         │ │    │
│  │  │  - Dashboard    │  │  - Multi-modal  │  │  - Model Download      │ │    │
│  │  │  - System Mon   │  │  - History      │  │  - Performance Stats   │ │    │
│  │  │  - Settings     │  │  - Voice Input  │  │  - Agent Control       │ │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │ HTTP/WebSocket                               │
│  ┌────────────────────────────────┴────────────────────────────────────────┐    │
│  │                          Backend API Layer (Port 8000)                   │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │    │
│  │  │   FastAPI Core  │  │ Model Manager   │  │  Vector DB Manager      │ │    │
│  │  │                 │  │                 │  │                         │ │    │
│  │  │  - REST APIs    │  │  - Ollama API   │  │  - ChromaDB API        │ │    │
│  │  │  - WebSockets   │  │  - Model Cache  │  │  - Qdrant API          │ │    │
│  │  │  - Auth/CORS    │  │  - Load Balance │  │  - FAISS Integration   │ │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│  ┌────────────────────────────────┴────────────────────────────────────────┐    │
│  │                          AI Agent Orchestration Layer                    │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │    │
│  │  │   AutoGPT       │  │    CrewAI       │  │     GPT-Engineer        │ │    │
│  │  │  - Autonomous   │  │  - Multi-agent  │  │  - Code generation     │ │    │
│  │  │  - Task chains  │  │  - Workflows    │  │  - Project management  │ │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │    │
│  │  │     Aider       │  │    AgentZero    │  │     LangChain          │ │    │
│  │  │  - Code editor  │  │  - Core agent   │  │  - Chain management    │ │    │
│  │  │  - Git integ    │  │  - Reasoning    │  │  - Tool integration    │ │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
│                                   │                                              │
│  ┌────────────────────────────────┴────────────────────────────────────────┐    │
│  │                            Model Serving Layer                           │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│    │
│  │  │                      Ollama (Port 11434)                             ││    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐││    │
│  │  │  │ DeepSeek-R1  │  │   Qwen 2.5   │  │   Llama 3.2  │  │  Nomic  │││    │
│  │  │  │     8B       │  │      3B      │  │      3B      │  │  Embed  │││    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘││    │
│  │  └─────────────────────────────────────────────────────────────────────┘│    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                          Data & Storage Layer                               │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │  │
│  │  │  PostgreSQL     │  │     Redis       │  │   Vector Databases      │   │  │
│  │  │  (Port 5432)    │  │  (Port 6379)    │  │                         │   │  │
│  │  │                 │  │                 │  │  ┌─────────┐ ┌────────┐│   │  │
│  │  │  - User data    │  │  - Cache        │  │  │ChromaDB │ │ Qdrant ││   │  │
│  │  │  - Configs      │  │  - Sessions     │  │  │Port 8001│ │Port6333││   │  │
│  │  │  - History      │  │  - Queue        │  │  └─────────┘ └────────┘│   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                         Monitoring & Observability                          │  │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐ │  │
│  │  │   Prometheus (9090)     │  │         Grafana (3000)                  │ │  │
│  │  │  - Metrics collection   │  │  - Dashboards                           │ │  │
│  │  │  - Alerts               │  │  - Visualization                        │ │  │
│  │  └─────────────────────────┘  └─────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Component Dependency Matrix

| Component | Depends On | Required By | Network | Ports |
|-----------|------------|--------------|---------|-------|
| **Frontend (Streamlit)** | Backend API | End Users | sutazai-network | 8501 |
| **Backend API (FastAPI)** | PostgreSQL, Redis, Ollama, ChromaDB, Qdrant | Frontend, AI Agents | sutazai-network | 8000 |
| **PostgreSQL** | - | Backend API, AI Agents | sutazai-network | 5432 |
| **Redis** | - | Backend API (cache/queue) | sutazai-network | 6379 |
| **Ollama** | - | Backend API, AI Agents | sutazai-network | 11434 |
| **ChromaDB** | - | Backend API, Vector Search | sutazai-network | 8001 |
| **Qdrant** | - | Backend API, Vector Search | sutazai-network | 6333-6334 |
| **Prometheus** | All services | Grafana | sutazai-network | 9090 |
| **Grafana** | Prometheus | Monitoring UI | sutazai-network | 3000 |
| **AutoGPT** | Backend API, Ollama | Agent Orchestrator | sutazai-network | Internal |
| **CrewAI** | Backend API, Ollama | Agent Orchestrator | sutazai-network | Internal |
| **GPT-Engineer** | Backend API, Ollama | Code Generation | sutazai-network | Internal |
| **Aider** | Backend API, Git | Code Editing | sutazai-network | Internal |

## Data Flow

1. **User Interaction Flow**:
   ```
   User → Streamlit UI → FastAPI Backend → AI Agents → Model Layer → Response
   ```

2. **Model Inference Flow**:
   ```
   Request → Backend API → Ollama → Model Selection → Inference → Response
   ```

3. **Vector Search Flow**:
   ```
   Query → Embedding Generation → Vector DB (ChromaDB/Qdrant/FAISS) → Results
   ```

4. **Agent Collaboration Flow**:
   ```
   Task → Agent Orchestrator → Agent Selection → Execution → Result Aggregation
   ```

## Security Architecture

- **Network Isolation**: All services communicate through internal Docker network
- **API Authentication**: JWT tokens for API access
- **Data Encryption**: TLS for external communication, encrypted storage
- **Access Control**: Role-based access control (RBAC) at API level
- **Audit Logging**: All actions logged to PostgreSQL

## Scalability Considerations

1. **Horizontal Scaling**:
   - Backend API: Can run multiple instances behind load balancer
   - Vector Databases: Qdrant supports clustering
   - Model Serving: Ollama can be replicated

2. **Vertical Scaling**:
   - GPU support for model inference
   - Memory optimization for large models
   - SSD storage for vector databases

## Self-Improvement Architecture

The system implements autonomous self-improvement through:

1. **Performance Monitoring**: Continuous metrics collection
2. **Issue Detection**: AI-driven analysis of logs and metrics
3. **Solution Generation**: Code generation for fixes and optimizations
4. **Human Approval**: Proposed changes presented for review
5. **Automated Deployment**: Approved changes deployed via CI/CD

## Deployment Architecture

- **Containerization**: All components run in Docker containers
- **Orchestration**: Docker Compose for single-node deployment
- **Configuration**: Environment variables and volume mounts
- **Persistence**: Data volumes for databases and models

## API Architecture

### REST Endpoints

- `/api/v1/chat` - Chat interface
- `/api/v1/models` - Model management
- `/api/v1/vectors` - Vector operations
- `/api/v1/agents` - Agent control
- `/api/v1/system` - System status

### WebSocket Endpoints

- `/ws/chat` - Real-time chat
- `/ws/agents` - Agent status updates

## Future Architecture Enhancements

1. **Kubernetes Support**: For multi-node deployments
2. **Message Queue**: RabbitMQ/Kafka for async processing
3. **Distributed Training**: FSDP for model fine-tuning
4. **Edge Deployment**: Lightweight inference at edge nodes
5. **Multi-tenancy**: Isolated environments per user/organization