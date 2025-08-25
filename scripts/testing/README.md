# SutazAI Comprehensive Testing Workflow

## Overview
This testing framework provides systematic validation of the SutazAI multi-agent orchestration platform, covering infrastructure, AI services, agent networks, service mesh, and frontend-backend integration.

## Architecture Testing Coverage

### System Components Validated
- **Infrastructure**: PostgreSQL, Redis, Neo4j, RabbitMQ, Consul, Kong
- **AI Services**: Ollama LLM, ChromaDB, Qdrant, FAISS vector databases
- **Agent Network**: 200+ AI agents, MCP orchestrator, Docker-in-Docker
- **Service Mesh**: Consul service discovery, Kong API gateway, load balancing
- **Integration**: Streamlit frontend, FastAPI backend, WebSocket communication

### Testing Philosophy
- **Real System Validation**: Tests actual services, not mocks
- **Dependency-Aware**: Respects service startup order
- **Performance-Focused**: Measures response times, throughput, resource usage
- **Comprehensive Reporting**: Detailed JSON reports with actionable insights

## Quick Start

### Prerequisites
```bash
# Ensure system is running
docker-compose up -d
docker ps | grep -c "Up"  # Should show 38+ running containers

# Install test dependencies
pip install pytest asyncio aiohttp redis psycopg2 neo4j chromadb qdrant-client
```

### Run Complete Test Suite
```bash
# Execute full testing workflow (recommended)
cd scripts/testing
python systematic_testing_workflow.py

# View results
cat reports/sutazai_test_report_*.json | jq .
```

### Run Individual Test Modules
```bash
# Infrastructure only
python infrastructure_validation_tests.py

# AI services only  
python ai_services_validation_tests.py

# Agent network only
python agent_network_mcp_tests.py

# Service mesh only
python service_mesh_api_gateway_tests.py

# Frontend-backend integration only
python frontend_backend_integration_tests.py
```

## Testing Phases & Dependencies

### Phase 1: Core Infrastructure
**Dependencies**: None (foundation layer)
**Services**: PostgreSQL, Redis, Neo4j, RabbitMQ
**Parallel Execution**: All tests can run concurrently
**Expected Duration**: 30-60 seconds

```bash
# Manual execution
python infrastructure_validation_tests.py
```

### Phase 2: AI Services
**Dependencies**: Phase 1 must pass (databases required)
**Services**: Ollama, ChromaDB, Qdrant, FAISS
**Parallel Execution**: Vector DB tests can run concurrently
**Expected Duration**: 60-120 seconds

```bash
# Manual execution (requires Phase 1 success)
python ai_services_validation_tests.py
```

### Phase 3: Agent Network & MCP
**Dependencies**: Phase 1-2 must pass (requires databases and AI services)
**Services**: MCP orchestrator, agent registry, Docker-in-Docker
**Parallel Execution**: Limited (Docker API constraints)
**Expected Duration**: 90-180 seconds

```bash
# Manual execution (requires Phase 1-2 success)
python agent_network_mcp_tests.py
```

### Phase 4: Service Mesh & API Gateway
**Dependencies**: Phase 1-3 must pass (requires full stack)
**Services**: Consul, Kong, load balancing, routing
**Parallel Execution**: Service discovery tests can run concurrently
**Expected Duration**: 45-90 seconds

```bash
# Manual execution (requires Phase 1-3 success)
python service_mesh_api_gateway_tests.py
```

### Phase 5: Frontend-Backend Integration
**Dependencies**: All previous phases must pass
**Services**: Streamlit UI, FastAPI backend, WebSocket communication
**Parallel Execution**: UI workflow tests run sequentially
**Expected Duration**: 120-240 seconds

```bash
# Manual execution (requires all phases success)
python frontend_backend_integration_tests.py
```

## Report Analysis

### Report Location
All reports are saved to `reports/` directory with timestamps:
```
reports/
├── sutazai_test_report_20240825_143022.json
├── infrastructure_report_20240825_143025.json
├── ai_services_report_20240825_143028.json
└── ...
```

### Report Structure
```json
{
  "test_run_id": "sutazai_20240825_143022",
  "timestamp": "2024-08-25T14:30:22",
  "overall_grade": "A-",
  "system_readiness": 0.89,
  "phases": {
    "infrastructure": {
      "passed": true,
      "grade": "A",
      "score": 0.95,
      "tests": {...}
    }
  },
  "recommendations": [...],
  "performance_metrics": {...}
}
```

### Grade Interpretation
- **A (90-100%)**: Excellent performance, production ready
- **B (80-89%)**: Good performance, minor optimization opportunities
- **C (70-79%)**: Acceptable performance, some issues need attention
- **D (60-69%)**: Poor performance, significant issues require fixes
- **F (<60%)**: Critical failures, system not operational

### Key Metrics
- **Response Times**: API endpoint latency (target: <100ms)
- **Throughput**: Requests per second (target: >100 RPS)
- **Resource Usage**: Memory and CPU utilization
- **Error Rates**: Failed requests percentage (target: <1%)
- **System Readiness**: Overall operational status (target: >0.85)

## Troubleshooting

### Common Issues

#### Infrastructure Phase Failures
```bash
# PostgreSQL connection issues
docker logs sutazai-postgres-1
netstat -an | grep 10000

# Redis connection issues  
docker logs sutazai-redis-1
redis-cli -p 10001 ping

# Neo4j authentication issues
docker logs sutazai-neo4j-1
curl -u neo4j:password http://localhost:10002
```

#### AI Services Phase Failures
```bash
# Ollama model availability
curl http://localhost:10020/api/tags

# ChromaDB v2 API issues
curl http://localhost:10030/api/v1/heartbeat

# Qdrant collection issues
curl http://localhost:10040/collections
```

#### Agent Network Phase Failures
```bash
# MCP orchestrator status
docker logs sutazai-mcp-orchestrator-1
curl http://localhost:10050/status

# Agent registry connectivity
curl http://localhost:10051/agents
```

#### Service Mesh Phase Failures
```bash
# Consul cluster status
curl http://localhost:10060/v1/status/leader
curl http://localhost:10060/v1/agent/members

# Kong gateway status
curl http://localhost:10070/status
```

#### Integration Phase Failures
```bash
# Frontend accessibility
curl http://localhost:10011

# Backend API health
curl http://localhost:10010/health

# WebSocket connectivity
wscat -c ws://localhost:10010/ws
```

### Performance Optimization

#### Database Optimization
```sql
-- PostgreSQL connection pooling
SHOW max_connections;
SHOW shared_buffers;

-- Redis memory usage
redis-cli info memory
redis-cli config get maxmemory
```

#### AI Services Optimization
```bash
# Ollama GPU utilization
nvidia-smi
docker stats sutazai-ollama-1

# Vector database indexing
curl http://localhost:10040/collections/sutazai/points/count
```

### Debugging Test Failures

#### Enable Verbose Logging
```bash
# Run with debug output
python systematic_testing_workflow.py --verbose

# Individual module debugging
python -m pytest infrastructure_validation_tests.py -v -s
```

#### Test-Specific Debugging
```python
# Add to test modules
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable async debugging
import asyncio
asyncio.get_event_loop().set_debug(True)
```

## CI/CD Integration

### Exit Codes
- **0**: All tests passed (system ready)
- **1**: Some tests failed (investigate issues)
- **2**: Critical failures (system not operational)

### Jenkins Integration
```groovy
pipeline {
    stages {
        stage('System Testing') {
            steps {
                sh 'cd scripts/testing && python systematic_testing_workflow.py'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'scripts/testing/reports',
                    reportFiles: '*.json',
                    reportName: 'SutazAI Test Report'
                ])
            }
        }
    }
}
```

### GitHub Actions Integration
```yaml
- name: Run SutazAI Tests
  run: |
    cd scripts/testing
    python systematic_testing_workflow.py
  env:
    PYTHONPATH: ${{ github.workspace }}

- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: scripts/testing/reports/
```

### Docker Health Integration
```bash
# Use test results for container health
docker-compose exec sutazai-backend python scripts/testing/systematic_testing_workflow.py --quiet --exit-on-fail
```

## Advanced Usage

### Custom Test Configuration
```python
# Create custom test config
config = {
    "timeout": 30,
    "parallel_tests": True,
    "performance_thresholds": {
        "response_time": 100,  # ms
        "throughput": 100,     # RPS
        "error_rate": 0.01     # 1%
    }
}

# Run with custom config
workflow = SutazAITestingWorkflow(config)
await workflow.run_complete_testing_suite()
```

### Selective Phase Testing
```python
# Skip certain phases
phases_to_run = [TestPhase.INFRASTRUCTURE, TestPhase.AI_SERVICES]
workflow = SutazAITestingWorkflow()
await workflow.run_selective_phases(phases_to_run)
```

### Performance Benchmarking
```python
# Run performance-focused tests
workflow = SutazAITestingWorkflow()
benchmark_results = await workflow.run_performance_benchmarks()
```

### Load Testing Integration
```bash
# Generate load during testing
artillery quick --count 10 --num 100 http://localhost:10010/health &
python systematic_testing_workflow.py
```

## Monitoring & Alerting

### Prometheus Integration
Tests expose metrics at `/metrics` endpoint for Prometheus scraping:
- `sutazai_test_duration_seconds`
- `sutazai_test_success_total`
- `sutazai_response_time_seconds`

### Grafana Dashboards
Import dashboard templates from `monitoring/grafana/dashboards/testing.json`

### Alert Rules
Configure alerts for:
- Test failure rate > 5%
- Response time > 200ms
- System readiness < 0.8

## Contributing

### Adding New Tests
1. Create test class inheriting from base validator
2. Implement async test methods with proper error handling
3. Add performance metrics collection
4. Update workflow orchestrator to include new tests
5. Document test purpose and expected outcomes

### Test Naming Conventions
- Test methods: `test_{component}_{aspect}()`
- Test classes: `{Component}Validator`
- Report files: `{component}_report_{timestamp}.json`

### Performance Standards
- All tests must complete within 5 minutes
- Individual test timeout: 30 seconds
- Memory usage: <100MB per test module
- Error handling: Graceful degradation for service unavailability

---

**Generated by SutazAI Testing Framework v1.0**  
**Last Updated**: 2024-08-25  
**Total Test Coverage**: 5 phases, 25+ services, 200+ agents