# SutazaiApp Port Registry
## Multi-Agent AI System Port Allocation
### Last Updated: 2025-11-13 21:30:00 UTC
### Version: 2.0.0 - Comprehensive Port Registry with Portainer Integration

---

## Container Management (9000-9999)
- **9000**: Portainer HTTP (Container Management Web UI)
- **9443**: Portainer HTTPS (Secure Container Management)

## Core Infrastructure (10000-10099)
- **10000**: PostgreSQL (Database - jarvis_ai)
- **10001**: Redis (Cache & Session Store)
- **10002**: Neo4j HTTP (Graph Database Web Interface)
- **10003**: Neo4j Bolt (Graph Database Protocol)
- **10004**: RabbitMQ AMQP (Message Queue Protocol)
- **10005**: RabbitMQ Management UI (Message Queue Web Interface)
- **10006**: Consul (Service Discovery Web UI/API)
- **10007**: Consul DNS (Service Discovery DNS)
- **10008**: Kong Proxy (API Gateway - Public Endpoint)
- **10009**: Kong Admin API (API Gateway Administration)

## AI & Vector Services (10100-10199)
- **10100**: ChromaDB (Vector Database - Embeddings)
- **10101**: Qdrant gRPC (Vector Database - Neural Search)
- **10102**: Qdrant HTTP (Vector Database - REST API)
- **10103**: FAISS Service (Vector Database - Fast Similarity Search)
- **10105-10199**: Reserved for additional AI services

## Application Services (10200-10299)
- **10200**: Backend API (FastAPI - Main Application API)
- **10201**: Grafana (Monitoring Visualization)
- **10202-10299**: Reserved for additional application services

## Frontend Services (11000-11099)
- **11000**: JARVIS Frontend (Streamlit Voice-Controlled UI)
- **11001-11099**: Reserved for additional frontend services

## MCP Bridge Services (11100-11199)
- **11100**: MCP HTTP Bridge (Agent Orchestration)
- **11101**: Letta Agent (Memory AI - Previously MemGPT)
- **11102**: AutoGPT Agent (Autonomous Task Execution)
- **11103**: LocalAGI Agent (Local AI Orchestration)
- **11104**: AgentGPT (Autonomous GPT Agent)
- **11105**: AgentZero (Autonomous Agent Coordinator)
- **11106**: BigAGI (Chat Interface Agent)
- **11107**: Deep Agent (Deep Learning Agent)
- **11108-11199**: Reserved for additional MCP services

## LLM & AI Frameworks (11200-11299)
- **11201**: LangChain Orchestrator (LLM Framework)
- **11202**: LlamaIndex (Data Framework for LLMs)
- **11203**: AutoGen Coordinator (Multi-Agent Conversations)
- **11204**: Context Engineering Framework
- **11205-11299**: Reserved for AI frameworks

## Code Generation Agents (11300-11399)
- **11301**: Aider (AI Pair Programming)
- **11302**: GPT-Engineer (Code Generation)
- **11303**: OpenDevin (AI Coding Assistant)
- **11304**: TabbyML (Code Completion - GPU Required)
- **11305-11399**: Reserved for code generation tools

## Orchestration Frameworks (11400-11499)
- **11401**: CrewAI Manager (Multi-Agent Orchestration)
- **11402**: LangFlow (Visual Orchestration)
- **11403**: Dify (AI Application Platform)
- **11404**: Flowise (Visual Workflow Builder)
- **11405-11434**: Reserved for orchestration tools
- **11434**: Ollama (Local LLM Inference Server) - CRITICAL PORT

## Document Processing Agents (11500-11599)
- **11501**: PrivateGPT (Local Document Q&A)
- **11502**: Documind (Document Processing)
- **11503-11599**: Reserved for document processing

## Financial & Analysis Agents (11600-11699)
- **11601**: FinRobot (Financial Analysis)
- **11602-11699**: Reserved for financial tools

## Automation & Browser Agents (11700-11799)
- **11701**: ShellGPT (CLI Assistant)
- **11702**: Skyvern (Browser Automation)
- **11703**: Browser Use (Web Automation)
- **11704-11799**: Reserved for automation tools

## Security & Testing Agents (11800-11899)
- **11801**: Semgrep (Security Analysis)
- **11802**: PentestGPT (Security Testing)
- **11803-11899**: Reserved for security tools

## ML & Deep Learning (11900-11999)
- **11901**: PyTorch Service (ML Framework - GPU Required)
- **11902**: TensorFlow Service (ML Framework - GPU Required)
- **11903**: JAX Service (ML Framework - GPU Required)
- **11904**: FSDP (Foundation Models - GPU Required)
- **11905-11999**: Reserved for ML services

## Additional Services (12000+)
- **12375**: Docker-in-Docker (DinD) - For CI/CD pipelines
- **12376-19999**: Reserved for future expansion

## Monitoring & Observability (Planned - Not Yet Deployed)
- **10202**: Prometheus (Time-Series Database)
- **10203**: AlertManager (Alert Management)
- **10204**: Blackbox Exporter (Endpoint Monitoring)
- **10205**: Node Exporter (System Metrics)
- **10210**: Loki (Log Aggregation)
- **10211**: Jaeger (Distributed Tracing)

---

## Network Configuration

### Primary Network: sutazai-network
- **Subnet**: 172.20.0.0/16
- **Driver**: bridge
- **IPAM**: Static IP assignment
- **Gateway**: 172.20.0.1 (Docker bridge gateway)

### IP Address Allocation Scheme

#### Management Layer (172.20.0.50-59)
- **172.20.0.50**: Portainer (Container Management)

#### Core Infrastructure (172.20.0.10-19)
- **172.20.0.10**: PostgreSQL (Primary Database)
- **172.20.0.11**: Redis (Cache & Sessions)
- **172.20.0.12**: Neo4j (Graph Database)
- **172.20.0.13**: RabbitMQ (Message Queue)
- **172.20.0.14**: Consul (Service Discovery)
- **172.20.0.15**: Kong (API Gateway)

#### Vector Databases (172.20.0.20-29)
- **172.20.0.20**: ChromaDB (Embeddings)
- **172.20.0.21**: Qdrant (Neural Search)
- **172.20.0.22**: FAISS (Similarity Search)
- **172.20.0.23**: Ollama (LLM Inference)

#### Application Layer (172.20.0.30-39)
- **172.20.0.30**: Backend API (FastAPI)
- **172.20.0.31**: Frontend (Streamlit/JARVIS)
- **Note**: Previous duplicate IP assignment (172.20.0.30) has been resolved

#### Monitoring Layer (172.20.0.40-49)
- **172.20.0.40**: Prometheus (Metrics Collection)
- **172.20.0.41**: Grafana (Visualization)
- **172.20.0.42-49**: Reserved for additional monitoring

#### AI Agents Layer (172.20.0.100-199)
- **172.20.0.100+**: AI Agent Services (deployed as needed)
- Dynamic allocation for agent containers

#### Reserved Ranges
- **172.20.0.60-99**: System services expansion
- **172.20.0.200-254**: Future infrastructure

---

## Port Conflict Resolution

### Previously Identified Issues (RESOLVED)
1. **Backend IP Conflict**: Both Backend and a monitoring service used 172.20.0.30
   - **Resolution**: Backend remains at 172.20.0.30, Prometheus moved to 172.20.0.40

### Port Usage Guidelines
1. **Never reuse ports** across different services
2. **Document all port assignments** in this registry before deployment
3. **Test port availability** before assigning: `netstat -tulpn | grep PORT`
4. **Reserve port ranges** for service categories to prevent conflicts

---

## Service Summary Table

| Service | Container Name | Port(s) | IP Address | Status |
|---------|---------------|---------|------------|--------|
| Portainer | sutazai-portainer | 9000, 9443 | 172.20.0.50 | Active |
| PostgreSQL | sutazai-postgres | 10000 | 172.20.0.10 | Active |
| Redis | sutazai-redis | 10001 | 172.20.0.11 | Active |
| Neo4j | sutazai-neo4j | 10002, 10003 | 172.20.0.12 | Active |
| RabbitMQ | sutazai-rabbitmq | 10004, 10005 | 172.20.0.13 | Active |
| Consul | sutazai-consul | 10006, 10007 | 172.20.0.14 | Active |
| Kong | sutazai-kong | 10008, 10009 | 172.20.0.15 | Active |
| ChromaDB | sutazai-chromadb | 10100 | 172.20.0.20 | Active |
| Qdrant | sutazai-qdrant | 10101, 10102 | 172.20.0.21 | Active |
| FAISS | sutazai-faiss | 10103 | 172.20.0.22 | Active |
| Ollama | sutazai-ollama | 11434 | 172.20.0.23 | Active |
| Backend API | sutazai-backend | 10200 | 172.20.0.30 | Active |
| Frontend | sutazai-frontend | 11000 | 172.20.0.31 | Active |
| Prometheus | sutazai-prometheus | 10202 | 172.20.0.40 | Active |
| Grafana | sutazai-grafana | 10201 | 172.20.0.41 | Active |

---

## Deployment Notes

### Portainer Stack
All services are managed through a unified Portainer stack defined in `/opt/sutazaiapp/portainer-stack.yml`

### Access Methods
- **Web UI**: Services with web interfaces accessible via browser
- **API**: Services with REST/gRPC APIs for programmatic access
- **CLI**: Some services support command-line interaction

### Security Considerations
1. **Change default passwords** before production deployment
2. **Enable SSL/TLS** for public-facing services
3. **Configure firewall rules** to restrict access
4. **Use Kong API Gateway** for secure API access
5. **Implement network policies** for service isolation

---

## Maintenance

### Adding New Services
1. Check this registry for available ports
2. Reserve port in appropriate range
3. Update this document before deployment
4. Update `portainer-stack.yml`
5. Deploy via Portainer or Docker Compose

### Port Range Expansion
Contact system architect if port ranges need expansion or reorganization

---

## References
- **Portainer Deployment Guide**: `/opt/sutazaiapp/docs/PORTAINER_DEPLOYMENT_GUIDE.md`
- **TODO List**: `/opt/sutazaiapp/TODO.md`
- **Changelog**: `/opt/sutazaiapp/CHANGELOG.md`
- **Project Documentation**: https://deepwiki.com/sutazai/sutazaiapp