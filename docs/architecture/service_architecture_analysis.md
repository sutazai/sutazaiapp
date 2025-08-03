=== Mermaid Graph ===
graph TD
    backend --> chromadb
    backend --> neo4j
    backend --> redis
    backend --> qdrant
    backend --> postgres
    backend --> ollama
    frontend --> backend
    mcp-server --> chromadb
    mcp-server --> backend
    mcp-server --> redis
    mcp-server --> qdrant
    mcp-server --> neo4j
    mcp-server --> postgres
    mcp-server --> ollama
    ai-metrics-exporter --> backend
    ai-metrics-exporter --> ollama
    ai-metrics-exporter --> postgres
    ai-metrics-exporter --> redis
    context-framework --> ollama
    context-framework --> chromadb
    context-framework --> qdrant
    context-framework --> neo4j
    llamaindex --> neo4j
    llamaindex --> chromadb
    llamaindex --> qdrant
    llamaindex --> ollama
    autogpt --> backend
    autogpt --> ollama
    crewai --> backend
    crewai --> ollama
    letta --> ollama
    letta --> redis
    letta --> postgres
    letta --> backend
    agentgpt --> redis
    agentgpt --> postgres
    agentgpt --> ollama
    dify --> ollama
    dify --> postgres
    dify --> redis
    langflow --> postgres
    langflow --> redis
    finrobot --> redis
    finrobot --> postgres
    finrobot --> ollama
    agentzero --> ollama
    agentzero --> postgres
    agentzero --> redis
    code-improver --> ollama
    code-improver --> postgres
    code-improver --> redis
    service-hub --> postgres
    service-hub --> redis
    health-monitor --> postgres
    grafana --> prometheus
    grafana --> loki
    promtail --> loki

    %% Node definitions
    prometheus["prometheus"]
    health-monitor["health-monitor"]
    context-framework["context-framework"]
    finrobot["finrobot"]
    postgres["postgres<br/>(14 deps)"]
    dify["dify"]
    chromadb["chromadb"]
    loki["loki"]
    crewai["crewai"]
    promtail["promtail"]
    langflow["langflow"]
    grafana["grafana"]
    ollama["ollama<br/>(24 deps)"]
    llamaindex["llamaindex"]
    mcp-server["mcp-server"]
    qdrant["qdrant<br/>(5 deps)"]
    code-improver["code-improver"]
    letta["letta"]
    ai-metrics-exporter["ai-metrics-exporter"]
    neo4j["neo4j<br/>(5 deps)"]
    agentzero["agentzero"]
    redis["redis<br/>(12 deps)"]
    autogpt["autogpt"]
    backend["backend<br/>(7 deps)"]
    agentgpt["agentgpt"]
    frontend["frontend"]
    service-hub["service-hub"]

    %% Apply styles
    style postgres fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff
    style redis fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff
    style neo4j fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff
    style backend fill:#4ECDC4,stroke:#216969,stroke-width:3px,color:#fff
    style frontend fill:#4ECDC4,stroke:#216969,stroke-width:3px,color:#fff
    style ollama fill:#4ECDC4,stroke:#216969,stroke-width:3px,color:#fff
    style chromadb fill:#45B7D1,stroke:#1864AB,stroke-width:2px,color:#fff
    style qdrant fill:#45B7D1,stroke:#1864AB,stroke-width:2px,color:#fff
    style prometheus fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#fff
    style grafana fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#fff
    style loki fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#fff
    style mcp-server fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style service-hub fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style ai-metrics-exporter fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#fff
    style context-framework fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style llamaindex fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff
    style autogpt fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff
    style crewai fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff
    style letta fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style agentgpt fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff
    style dify fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style langflow fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style finrobot fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style agentzero fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff
    style code-improver fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style health-monitor fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff
    style promtail fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff


=== Summary Report ===
# SutazAI Service Architecture Analysis

## Executive Summary

The SutazAI system consists of 46+ interconnected services organized into the following categories:
- **Core Services**: Backend API, Frontend UI, and Ollama LLM service
- **Data Layer**: PostgreSQL (primary DB), Redis (cache/messaging), Neo4j (graph DB)
- **Vector Stores**: ChromaDB, Qdrant, and FAISS for embeddings
- **AI Agents**: 19 specialized agents for various tasks
- **Monitoring**: Comprehensive Prometheus/Grafana stack
- **Infrastructure**: Service orchestration and management tools

## Critical Services (by dependency count)

| Service | Dependencies | Role |
|---------|--------------|------|
| ollama | 24 | LLM inference engine - central to all AI operations |
| postgres | 14 | Primary database - stores all persistent data |
| redis | 12 | Cache & message broker - enables async communication |
| backend | 7 | API gateway - coordinates all service interactions |
| neo4j | 5 | Graph database - stores relationships and knowledge graphs |
| qdrant | 5 | Vector database - handles embeddings and similarity search |

## Communication Patterns

### 1. Synchronous HTTP (REST APIs)
- Frontend → Backend
- AI Agents → Backend
- Monitoring services → Backend

### 2. Asynchronous Messaging (Redis Pub/Sub)
- 12 services use Redis for async communication
- Enables event-driven architecture
- Supports task queuing and result caching

### 3. Database Connections
- 14 services connect directly to PostgreSQL
- 5 services use Neo4j for graph operations
- Multiple services share vector stores

### 4. LLM Service Integration
- 24 services depend on Ollama
- Centralized LLM inference reduces resource usage
- Supports model switching without service changes
