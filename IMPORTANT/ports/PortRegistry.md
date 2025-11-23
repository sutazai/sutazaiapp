# SutazaiApp Port Registry

## Multi-Agent AI System Port Allocation

### Last Updated: 2025-01-13 (Verified against running deployment)

---

## Core Infrastructure (10000-10099)
- **10000**: PostgreSQL (Database)
- **10001**: Redis (Cache & Message Broker)
- **10002**: Neo4j HTTP
- **10003**: Neo4j Bolt
- **10006**: Consul (Service Discovery HTTP)
- **10007**: Consul DNS
- **10008**: Kong Proxy
- **10009**: Kong Admin API
- **10004**: RabbitMQ AMQP
- **10005**: RabbitMQ Management UI

## AI & Vector Services (10100-10199)
- **10100**: ChromaDB (Vector Database)
- **10101**: Qdrant HTTP
- **10102**: Qdrant gRPC
- **10103**: FAISS Service
- **10104-10199**: Reserved for additional AI services

## Application Services (10200-10299)
- **10200**: Backend API (FastAPI) - Main REST/WebSocket API
- **10201-10299**: Reserved for application layer

## Frontend Services (11000-11099)
- **11000**: Frontend UI (Streamlit/JARVIS)
- **11001-11099**: Reserved for UI components

## LLM Services (11400-11499)
- **11434**: Ollama (Local LLM Server) - CRITICAL PORT - **HOST SERVICE (Not Dockerized)**
- **11435-11499**: Reserved for additional LLM services

## Monitoring Stack (10300-10399) [PLANNED - NOT YET DEPLOYED]
- **10300**: Prometheus (Metrics Collection) - PLANNED
- **10301**: Grafana (Visualization) - PLANNED
- **10303**: AlertManager (Alerting) - PLANNED
- **10304**: Blackbox Exporter (Probing) - PLANNED
- **10305**: Node Exporter (System Metrics) - PLANNED
- **10310**: Loki (Logging) - PLANNED
- **10311**: Jaeger (Tracing) - PLANNED
- **10312-10399**: Reserved for additional monitoring services

## Agent Services (11000-11299)
### Core Agents (11300-11324)
- **11300**: Letta Agent (Task Automation)
- **11301**: AutoGPT Agent
- **11302**: LocalAGI Agent
- **11303**: Agent Zero (Coordinator)
- **11304**: LangChain Orchestrator
- **11305**: AutoGen Coordinator
- **11306**: CrewAI Manager
- **11307**: GPT Engineer
- **11308**: OpenDevin
- **11309**: Aider
- **11310**: Deep Researcher
- **11311**: FinRobot
- **11312**: Semgrep Security
- **11313**: Browser Use
- **11314**: Skyvern
- **11315**: BigAGI
- **11316**: AgentGPT
- **11317**: PrivateGPT
- **11318**: LlamaIndex
- **11319**: ShellGPT
- **11320**: PentestGPT
- **11321**: Jarvis Core
- **11322**: Langflow
- **11323**: Dify
- **11324**: Flowise

## MCP Bridge Services (11100-11199)
- **11100**: MCP PostgreSQL Bridge
- **11101**: MCP Files Bridge
- **11102**: MCP HTTP Bridge
- **11103**: MCP DDG Bridge
- **11104**: MCP GitHub Bridge
- **11105**: MCP Memory Bridge
- **11106-11199**: Reserved for additional MCP services

## External/Testing (12000+)
- **12375**: Docker-in-Docker (DinD)
- **9000**: Portainer HTTP
- **9443**: Portainer HTTPS

---

## Network Configuration
- **Primary Network**: 172.20.0.0/16 (sutazai-network)
- **Frontend Range**: 172.20.0.30-39
- **Backend Range**: 172.20.0.10-29
- **Monitoring Range**: 172.20.0.40-49
- **Agent Range**: 172.20.0.100-199

## Service IP Assignments (Current Deployment)

- 172.20.0.10: PostgreSQL
- 172.20.0.11: Redis
- 172.20.0.12: Neo4j
- 172.20.0.13: RabbitMQ
- 172.20.0.14: Consul
- 172.20.0.20: ChromaDB
- 172.20.0.21: Qdrant
- 172.20.0.22: FAISS Service
- 172.20.0.31: Frontend (Streamlit/JARVIS)
- 172.20.0.35: Kong Gateway
- 172.20.0.40: Backend API (FastAPI)
- 172.20.0.41-49: Reserved for Monitoring Stack (Prometheus, Grafana, etc.)
- 172.20.0.100+: Reserved for Agent Services

**Note**: Ollama (11434) runs on **host** at 172.17.0.1 (accessible via host.docker.internal)
