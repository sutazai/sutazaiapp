# SutazaiApp Port Registry
## Multi-Agent AI System Port Allocation
### Last Updated: 2025-08-27

---

## Core Infrastructure (10000-10099)
- **10000**: PostgreSQL (Database)
- **10001**: Redis (Cache & Message Broker)
- **10002**: Neo4j HTTP
- **10003**: Neo4j Bolt
- **10005**: Kong API Gateway (Proxy)
- **10006**: Consul (Service Discovery)
- **10007**: RabbitMQ AMQP
- **10008**: RabbitMQ Management UI
- **10010**: Backend API (FastAPI)
- **10011**: Frontend (Streamlit/Jarvis UI)
- **10015**: Kong Admin API

## AI & Vector Services (10100-10199)
- **10100**: ChromaDB (Vector Database)
- **10101**: Qdrant HTTP
- **10102**: Qdrant gRPC
- **10103**: FAISS Service (when activated)
- **10104**: Ollama (Model Server) - CRITICAL PORT
- **10105-10199**: Reserved for additional AI services

## Monitoring Stack (10200-10299)
- **10200**: Prometheus
- **10201**: Grafana
- **10203**: AlertManager
- **10204**: Blackbox Exporter
- **10205**: Node Exporter
- **10210**: Loki (Logging)
- **10211**: Jaeger (Tracing)

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

## Service IP Assignments
- 172.20.0.10: PostgreSQL
- 172.20.0.11: Redis
- 172.20.0.12: Neo4j
- 172.20.0.13: Kong
- 172.20.0.14: Consul
- 172.20.0.15: RabbitMQ
- 172.20.0.20: ChromaDB
- 172.20.0.21: Qdrant
- 172.20.0.22: Ollama
- 172.20.0.30: Backend API
- 172.20.0.31: Frontend
- 172.20.0.40: Prometheus
- 172.20.0.41: Grafana
- 172.20.0.100+: Agent Services