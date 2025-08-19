# Port Registry System for SUTAZAIAPP - NETWORK AUDIT COMPLETED

This registry documents ONLY actual running services verified via comprehensive network audit on 2025-08-18.

**NETWORK AUDIT STATUS:** ✅ COMPLETED - 28+ containers running, 17 in sutazai-network
**CRITICAL DISCREPANCY:** CLAUDE.md claims "system completely down" but reality shows extensive running infrastructure
**VERIFIED STATUS:** 17 running containers in sutazai-network confirmed, additional MCP containers running outside network

Legend of ranges:
- 10000-10099: Core Infrastructure Services
- 10100-10199: AI & Vector Services  
- 10200-10299: Monitoring Stack
- 11000+: Agent Services (limited to actually deployed agents)

## Core Infrastructure Services (10000-10099)

- 10000: PostgreSQL database (sutazai-postgres)
- 10001: Redis cache (sutazai-redis)
- 10002: Neo4j HTTP interface (sutazai-neo4j)
- 10003: Neo4j Bolt protocol (sutazai-neo4j)
- 10005: Kong API Gateway proxy (sutazai-kong)
- 10006: Consul service discovery (sutazai-consul)
- 10007: RabbitMQ AMQP **NOT DEPLOYED** (no container found)
- 10008: RabbitMQ Management UI **NOT DEPLOYED** (no container found)
- 10010: FastAPI backend (sutazai-backend)
- 10011: Streamlit frontend (sutazai-frontend)
- 10015: Kong Admin API (sutazai-kong)

## AI & Vector Services (10100-10199)

- 10100: ChromaDB vector database (sutazai-chromadb)
- 10101: Qdrant HTTP API (sutazai-qdrant)
- 10102: Qdrant gRPC interface (sutazai-qdrant)
- 10103: FAISS vector service **NOT DEPLOYED** (no container found)
- 10104: Ollama LLM server (sutazai-ollama) ✅ **RUNNING**

## Monitoring Stack (10200-10299)

- 10200: Prometheus metrics collection (sutazai-prometheus)
- 10201: Grafana dashboards **NOT DEPLOYED** (no container found)
- 10202: Loki log aggregation **NOT DEPLOYED** (no container found)
- 10203: AlertManager notifications (sutazai-alertmanager)
- 10204: Blackbox Exporter (sutazai-blackbox-exporter)
- 10205: Node Exporter system metrics (sutazai-node-exporter)
- 10206: cAdvisor container metrics (sutazai-cadvisor)
- 10207: Postgres Exporter DB metrics (sutazai-postgres-exporter)
- 10208: Redis Exporter cache metrics **NOT DEPLOYED** (no container found)
- 10210: Jaeger tracing UI (sutazai-jaeger)
- 10211: Jaeger collector (sutazai-jaeger)
- 10212: Jaeger gRPC (sutazai-jaeger)
- 10213: Jaeger Zipkin (sutazai-jaeger)
- 10214: Jaeger OTLP gRPC (sutazai-jaeger)
- 10215: Jaeger OTLP HTTP (sutazai-jaeger)

## Agent Services (11000+) - REALITY CHECK

**NO AGENT CONTAINERS RUNNING** (verified via docker ps)
- 11019: Hardware Resource Optimizer **NOT DEPLOYED**
- 11069: Task Assignment Coordinator **NOT DEPLOYED**
- 11071: Ollama Integration Agent **NOT DEPLOYED**
- 11200: Ultra System Architect **NOT DEPLOYED**
- 11201: Ultra Frontend UI Architect **NOT DEPLOYED**

**Status Legend (REALITY-BASED):**
- ✅ **RUNNING**: Service verified running via docker ps
- ❌ **NOT DEPLOYED**: No container found (fantasy/removed)
- 🔄 **UNHEALTHY**: Running but health check failing

## Port Range Allocation Policy

- **10000-10099**: Core infrastructure (databases, cache, message queues, APIs)
- **10100-10199**: AI and vector services (LLMs, embeddings, vector databases)
- **10200-10299**: Monitoring and observability (metrics, logs, tracing, alerting)
- **11000+**: Agent services (AI agents, specialized automation tools)

## MCP Services (11090-11199) - REMOVED (FANTASY)

**REALITY CHECK**: NO MCP HTTP BRIDGE CONTAINERS FOUND
- 11090-11105: All MCP HTTP services **NOT DEPLOYED**

**Status**: All MCP HTTP bridges are FICTIONAL - REMOVED per Rule 1
**Network**: Only STDIO MCP servers exist (no network ports)
**Load Balancer**: NO HAProxy for MCP (fantasy claim removed)
**Service Discovery**: MCP servers use STDIO, not Consul registration

## MCP Server Status - REALITY CHECK REQUIRED

**WARNING**: MCP server status claims unverified - requires testing
**VERIFICATION NEEDED**: All MCP servers listed as "Active" need actual status verification
**PROTOCOL**: All MCP servers use STDIO (no network ports assigned)

### STDIO MCP Servers (Status UNVERIFIED)
| MCP Server | Protocol | Status | Verification Needed |
|------------|----------|--------|--------------------|
| claude-flow | STDIO | ❓ Unverified | Test required |
| ruv-swarm | STDIO | ❓ Unverified | Test required |
| files | STDIO | ❓ Unverified | Test required |
| All others | STDIO | ❓ Unverified | Full audit required |

### HTTP MCP Bridge Services - REMOVED (FICTIONAL)
**ALL HTTP BRIDGES REMOVED** - No containers found matching these services

## ACTUAL RUNNING SERVICES (Verified 2025-08-18)

### ✅ CONFIRMED RUNNING (17 services):
- 10000: PostgreSQL (sutazai-postgres) ✅
- 10001: Redis (sutazai-redis) ✅ 
- 10002: Neo4j HTTP (sutazai-neo4j) ✅
- 10003: Neo4j Bolt (sutazai-neo4j) ✅
- 10005: Kong Gateway (sutazai-kong) ✅
- 10006: Consul (sutazai-consul) ✅
- 10010: Backend API (sutazai-backend) 🔄 Unhealthy
- 10011: Frontend UI (sutazai-frontend) ✅
- 10015: Kong Admin (sutazai-kong) ✅
- 10100: ChromaDB (sutazai-chromadb) ✅
- 10101: Qdrant HTTP (sutazai-qdrant) ✅
- 10102: Qdrant gRPC (sutazai-qdrant) ✅
- 10104: Ollama (sutazai-ollama) ✅
- 10200: Prometheus (sutazai-prometheus) ✅
- 10203: AlertManager (sutazai-alertmanager) ✅
- 10204: Blackbox Exporter (sutazai-blackbox-exporter) ✅
- 10205: Node Exporter (sutazai-node-exporter) ✅
- 10206: cAdvisor (sutazai-cadvisor) ✅
- 10207: Postgres Exporter (sutazai-postgres-exporter) ✅
- 10210-10215: Jaeger (6 ports, sutazai-jaeger) ✅

### ❌ NOT DEPLOYED (Removed from fantasy claims):
- Grafana, Loki, RabbitMQ, all Agent containers, all MCP HTTP bridges

### Rule Enforcement Applied:
- Rule 1: Removed all fantasy network architecture
- Rule 18: Added mandatory change tracking with timestamps
- Rule 19: Comprehensive real-time documentation update
- Verified via: docker ps, health checks, connectivity tests

