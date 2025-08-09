---
title: SutazAI System Overview
version: 1.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Draft
next_review: 2025-08-15
related_docs: 
  - 02-component-architecture.md
  - 03-data-flow.md
  - /IMPORTANT/10_canonical/current_state/system_reality.md
  - /IMPORTANT/10_canonical/current_state/context.mmd
  - /IMPORTANT/10_canonical/current_state/containers.mmd
---

# SutazAI System Overview

## Overview

SutazAI is a Docker Compose-based AI agent orchestration platform that coordinates multiple specialized agents for task execution. The system leverages local LLM capabilities via Ollama and provides a web interface for user interaction. This document provides the authoritative high-level view using the C4 model.

## Prerequisites

- Docker and Docker Compose installed
- Minimum 8GB RAM, 4 CPU cores
- Understanding of containerized microservices
- Access to canonical diagrams under `/IMPORTANT/10_canonical/current_state/`

## Implementation Details

### System Context (C4 Level 1)

The system operates within these boundaries:

**Users:**
- End Users: Access via Streamlit UI (port 10011)
- System Administrators: Monitor via Grafana (port 10201)
- Developers: Direct API access (port 10010)

**External Dependencies:**
- None currently (all services self-contained)
- Future: External API integrations planned

**Core Capabilities:**
- Local LLM text generation (TinyLlama via Ollama)
- Multi-agent task orchestration (7 agents, currently stubs)
- Web-based user interface
- Comprehensive monitoring infrastructure

### Container Architecture (C4 Level 2)

**Verified Running Containers (28 active):**

| Container Group | Services | Status | Purpose |
|-----------------|----------|--------|---------|
| Core Application | Backend API, Frontend UI | ✅ Operational | User interaction and API |
| AI Services | Ollama, 7 Agent Services | ⚠️ Partial (stubs) | LLM and agent processing |
| Data Layer | PostgreSQL, Redis, Neo4j | ✅ Operational | Persistence and caching |
| Vector Stores | Qdrant, ChromaDB, FAISS | ⚠️ Not integrated | Similarity search (future) |
| Infrastructure | Kong, Consul, RabbitMQ | ⚠️ Unconfigured | Service mesh and messaging |
| Observability | Prometheus, Grafana, Loki, AlertManager | ✅ Running | Monitoring and logging |

### Current System State

Based on verified inspection (2025-08-08):
- **Reality Check**: 28 containers running (59 defined in docker-compose.yml)
- **Model**: TinyLlama 637MB loaded (not gpt-oss as documented)
- **Agents**: 7 Flask stubs returning hardcoded JSON
- **Database**: PostgreSQL has 14 tables but uses SERIAL PKs (needs UUID migration)
- **Authentication**: Not implemented (critical gap)

## Configuration

### Port Registry

```yaml
# Application Layer
10010: Backend FastAPI
10011: Frontend Streamlit

# Data Services
10000: PostgreSQL
10001: Redis
10002: Neo4j Browser
10003: Neo4j Bolt

# AI Services
10104: Ollama LLM Server
8002:  Hardware Resource Optimizer (stub)
8551:  Task Assignment Coordinator (functional)
8587:  Multi-Agent Coordinator (stub)
8588:  Resource Arbitration Agent (stub)
8589:  AI Agent Orchestrator (stub)
11015: Ollama Integration Specialist (stub)
11063: AI Metrics Exporter (unhealthy)

# Vector Databases
10100: ChromaDB (connection issues)
10101: Qdrant HTTP
10102: Qdrant gRPC
10103: FAISS

# Infrastructure
10005: Kong API Gateway
10006: Consul
10007: RabbitMQ AMQP
10008: RabbitMQ Management

# Observability
10200: Prometheus
10201: Grafana
10202: Loki
10203: AlertManager
10220: Node Exporter
10221: cAdvisor
```

### Environment Management

- **Development**: Docker Compose with hot-reload
- **Staging**: Not configured
- **Production**: Not configured

## Testing & Validation

### System Health Checks

```bash
# Verify core services
curl http://localhost:10010/health  # Backend API
curl http://localhost:10104/api/tags  # Ollama models

# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}"

# Test LLM generation
curl -X POST http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello"}'
```

### Current Test Coverage
- **Unit Tests**: 0% (ISSUE-0015)
- **Integration Tests**: None
- **E2E Tests**: None
- **Manual Testing**: Ad-hoc only

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Backend shows "degraded" | Ollama connection issue | Ensure tinyllama model loaded |
| Agents return static responses | Stub implementation | Awaiting real logic (ISSUE-0002) |
| ChromaDB restarting | Connection issues | Check logs, may need config fix |
| No authentication | Not implemented | Critical gap (ISSUE-0005) |
| Grafana empty | No dashboards | Import standard dashboards (ISSUE-0016) |

### Diagnostic Commands

```bash
# View logs
docker-compose logs -f [service-name]

# Restart service
docker-compose restart [service-name]

# Check resource usage
docker stats

# Database access
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
```

## References

### Authoritative Sources
- [System Reality Check](/opt/sutazaiapp/CLAUDE.md)
- [Canonical Context Diagram](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/context.mmd)
- [Canonical Container Diagram](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/containers.mmd)
- [Deployment Topology](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/deployment_topology.md)
- [Issue Tracking](/opt/sutazaiapp/IMPORTANT/02_issues/)

### Related Documentation
- [Component Architecture](02-component-architecture.md)
- [Data Flow](03-data-flow.md)
- [Technology Stack](04-technology-stack.md)
- [ADR-0001: UUID Primary Keys](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0001.md)
- [ADR-0002: Local LLM Strategy](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0002.md)

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-08-08 | 0.1.0 | Initial draft | Documentation Lead |
| 2025-08-08 | 1.0.0 | Comprehensive rewrite based on system verification | System Architect |