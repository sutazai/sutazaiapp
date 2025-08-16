# Port Registry System for SUTAZAIAPP

This registry documents actual port allocations for deployed services. Port assignments must match the reality of docker-compose.yml and running containers.

**CRITICAL:** This registry now reflects ACTUAL deployed services only. Fantasy/placeholder allocations have been removed.

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
- 10007: RabbitMQ AMQP (sutazai-rabbitmq)
- 10008: RabbitMQ Management UI (sutazai-rabbitmq)
- 10010: FastAPI backend (sutazai-backend)
- 10011: Streamlit frontend (sutazai-frontend)
- 10015: Kong Admin API (sutazai-kong)

## AI & Vector Services (10100-10199)

- 10100: ChromaDB vector database (sutazai-chromadb)
- 10101: Qdrant HTTP API (sutazai-qdrant)
- 10102: Qdrant gRPC interface (sutazai-qdrant)
- 10103: FAISS vector service (sutazai-faiss) **[DEFINED BUT NOT RUNNING]**
- 10104: Ollama LLM server (sutazai-ollama) **[RESERVED - CRITICAL]**

## Monitoring Stack (10200-10299)

- 10200: Prometheus metrics collection (sutazai-prometheus)
- 10201: Grafana dashboards (sutazai-grafana)
- 10202: Loki log aggregation (sutazai-loki)
- 10203: AlertManager notifications (sutazai-alertmanager)
- 10204: Blackbox Exporter (sutazai-blackbox-exporter)
- 10205: Node Exporter system metrics (sutazai-node-exporter)
- 10206: cAdvisor container metrics (sutazai-cadvisor)
- 10207: Postgres Exporter DB metrics (sutazai-postgres-exporter)
- 10208: Redis Exporter cache metrics (sutazai-redis-exporter) **[DEFINED BUT NOT RUNNING]**
- 10210: Jaeger tracing UI (sutazai-jaeger)
- 10211: Jaeger collector (sutazai-jaeger)
- 10212: Jaeger gRPC (sutazai-jaeger)
- 10213: Jaeger Zipkin (sutazai-jaeger)
- 10214: Jaeger OTLP gRPC (sutazai-jaeger)
- 10215: Jaeger OTLP HTTP (sutazai-jaeger)

## Agent Services (11000+)

**Currently Deployed Agents:**
- 11019: Hardware Resource Optimizer (sutazai-hardware-resource-optimizer) **[DEFINED BUT NOT RUNNING]**
- 11069: Task Assignment Coordinator (sutazai-task-assignment-coordinator) **[DEFINED BUT NOT RUNNING]**
- 11071: Ollama Integration Agent (sutazai-ollama-integration) **[DEFINED BUT NOT RUNNING]**
- 11200: Ultra System Architect (sutazai-ultra-system-architect) **[RUNNING]**
- 11201: Ultra Frontend UI Architect (sutazai-ultra-frontend-ui-architect) **[DEFINED BUT NOT RUNNING]**

**Status Legend:**
- **[RUNNING]**: Service is active and healthy
- **[DEFINED BUT NOT RUNNING]**: Service defined in docker-compose.yml but not currently active
- **[RESERVED - CRITICAL]**: Port must never be changed

## Port Range Allocation Policy

- **10000-10099**: Core infrastructure (databases, cache, message queues, APIs)
- **10100-10199**: AI and vector services (LLMs, embeddings, vector databases)
- **10200-10299**: Monitoring and observability (metrics, logs, tracing, alerting)
- **11000+**: Agent services (AI agents, specialized automation tools)

## Notes

- Port assignments must match docker-compose.yml external port mappings exactly
- Multi-port services (Neo4j, RabbitMQ, Jaeger) use consecutive port blocks
- No fictional or placeholder services are listed - this registry reflects deployment reality
- Services marked as **[DEFINED BUT NOT RUNNING]** may need investigation
- Ollama port 10104 is critical and reserved - never modify

