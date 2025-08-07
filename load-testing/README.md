# SutazAI Load Testing Framework

## Overview
Comprehensive load testing suite for SutazAI's distributed AI system with 69+ active agents, multiple databases, and service mesh architecture.

## Architecture Under Test
- **Agents**: 69+ AI agents across multiple categories
- **Services**: 84 microservices in docker-compose
- **Databases**: PostgreSQL, Redis, Neo4j, ChromaDB, Qdrant
- **AI Models**: Ollama integration with multiple LLMs
- **API Gateway**: Kong/HAProxy
- **Monitoring**: Prometheus, Grafana, Jaeger

## Test Categories

### 1. Individual Agent Tests
- Single agent endpoint performance
- Agent response time and throughput
- Agent resource utilization
- Agent failure scenarios

### 2. Concurrent User Tests
- Jarvis interface load testing
- Multi-user concurrent requests
- Session management under load
- User authentication scaling

### 3. Database Performance Tests
- PostgreSQL connection pooling
- Redis cache performance
- Neo4j graph query performance
- Vector database (ChromaDB/Qdrant) scaling

### 4. System-Wide Scenarios
- Full system integration load
- Cross-agent communication patterns
- Service mesh resilience
- End-to-end user journeys

### 5. Breaking Point Tests
- Progressive load increase
- Resource exhaustion scenarios
- Cascading failure detection
- Recovery time measurement

## Performance Baselines

### Agent Performance SLAs
- Response Time: P95 < 2s, P99 < 5s
- Throughput: Min 100 req/s per agent
- Error Rate: < 1% under normal load
- Availability: 99.9% uptime

### Database Performance SLAs
- PostgreSQL: Query time P95 < 100ms
- Redis: Operation time P95 < 10ms
- Neo4j: Graph query P95 < 500ms
- Vector DBs: Search P95 < 200ms

### System-Wide SLAs
- End-to-end latency: P95 < 3s
- Concurrent users: 1000+ simultaneous
- Agent orchestration: P95 < 1s
- Service mesh overhead: < 50ms

## Test Execution
```bash
# Install K6
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run individual tests
k6 run tests/agent-performance.js
k6 run tests/database-load.js
k6 run tests/system-integration.js

# Run full test suite
./run-load-tests.sh

# Generate reports
./generate-reports.sh
```

## Monitoring During Tests
- Real-time metrics in Grafana
- K6 performance dashboard
- Resource utilization tracking
- Error rate monitoring
- Response time distributions

## Test Data Management
- Synthetic test data generation
- Test data cleanup procedures
- Database state management
- Result archival strategy