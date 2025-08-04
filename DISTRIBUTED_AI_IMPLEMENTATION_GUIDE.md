# Distributed AI Services Implementation Guide

## Overview

This implementation provides a scalable, resource-efficient distributed architecture for integrating 40+ AI services in a CPU-constrained environment. The system uses official Docker images, implements service mesh patterns, and provides dynamic scaling based on demand.

## Architecture Components

### 1. Core Infrastructure (Tier 1)
- **Kong API Gateway**: Unified API interface with rate limiting
- **Consul**: Service discovery and health checking
- **RabbitMQ**: Asynchronous task queue
- **Redis**: Distributed cache and session storage
- **Envoy**: Service mesh sidecar proxy

### 2. Persistent AI Services (Tier 2)
- **Ollama**: LLM inference with model management
- **ChromaDB**: Vector database for embeddings
- **Qdrant**: Advanced vector search
- **n8n**: Workflow automation

### 3. On-Demand Services (Tier 3)
- **LangChain**: Chain and agent execution
- **AutoGPT**: Autonomous task completion
- **Letta**: Memory-enhanced AI agents
- **TabbyML**: Code completion
- And 30+ more services configured for lazy loading

## Quick Start

### 1. Prerequisites
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Clone the repository
cd /opt/sutazaiapp
```

### 2. Deploy Infrastructure
```bash
# Deploy all services
./scripts/deploy-distributed-ai.sh --env dev --phase all

# Or deploy in phases
./scripts/deploy-distributed-ai.sh --env dev --phase infra
./scripts/deploy-distributed-ai.sh --env dev --phase ai
./scripts/deploy-distributed-ai.sh --env dev --phase monitor
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Gateway | http://localhost:8000 | - |
| Kong Admin | http://localhost:8001 | - |
| Consul UI | http://localhost:8500 | - |
| RabbitMQ | http://localhost:15672 | admin/admin |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| Jaeger | http://localhost:16686 | - |

## Using the Unified API Client

### Python Example
```python
from services.api_adapter.unified_ai_client import UnifiedAIClient

# Initialize client
client = UnifiedAIClient()

# Chat completion
response = client.chat_completion(
    prompt="Explain quantum computing",
    model="tinyllama"
)

# Execute LangChain
result = client.execute_chain(
    prompt="Analyze this text for sentiment",
    chain_type="simple"
)

# Vector search
results = client.vector_search(
    query="machine learning concepts",
    collection="knowledge_base",
    k=5
)
```

### REST API Examples
```bash
# Chat completion via Ollama
curl -X POST http://localhost:8000/api/ollama/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "prompt": "What is Docker?",
    "stream": false
  }'

# Execute LangChain
curl -X POST http://localhost:8000/api/langchain/execute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize this text",
    "chain_type": "simple"
  }'
```

## Resource Management

### Memory Limits
Services are configured with appropriate memory limits:
- Ollama: 4GB (2GB minimum)
- Vector DBs: 2GB each (1GB minimum)
- Agent Services: 2GB each (512MB minimum)
- Workflow Engines: 1GB (512MB minimum)

### CPU Allocation
- Core services: 0.5-1 CPU
- AI services: 0.25-2 CPUs
- Total system requirement: 4-8 CPUs

### Dynamic Scaling
The service scaler automatically:
- Starts services based on queue depth
- Stops idle services after timeout
- Manages memory pressure
- Balances resource allocation

## Adding New Services

### 1. Update Docker Compose
```yaml
# Add to docker-compose.distributed-ai.yml
new-ai-service:
  <<: *ai-service
  image: official/image:tag
  networks:
    - ai-mesh
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1'
    replicas: 0  # Start with 0 for on-demand
```

### 2. Register with Kong
```yaml
# Add to config/kong/kong.yml
- name: new-ai-service
  url: http://new-ai-service:8080
  routes:
    - name: new-ai-route
      paths:
        - /api/new-ai
```

### 3. Add to Service Scaler
```python
# Update services/scaler/service_scaler.py
self.service_configs['new-ai'] = {
    'image': 'official/image:tag',
    'min_replicas': 0,
    'max_replicas': 2,
    'scale_up_threshold': 3,
    'idle_timeout': 300,
    'memory_limit': '2g',
    'cpu_limit': 1.0
}
```

## Monitoring and Observability

### Grafana Dashboards
- AI Services Overview: System metrics
- Service Health: Up/down status
- Performance Metrics: Request rates, latency
- Resource Usage: CPU, memory, disk

### Prometheus Queries
```promql
# Service request rate
rate(http_requests_total{job="ai-services"}[5m])

# Memory usage by service
container_memory_usage_bytes{name=~".*ai.*"}

# Service availability
up{job=~"ai-services|ollama|chromadb"}
```

### Distributed Tracing
Access Jaeger UI at http://localhost:16686 to:
- Trace requests across services
- Identify performance bottlenecks
- Debug service communication

## Troubleshooting

### Check Service Health
```bash
# Via Consul
curl http://localhost:8500/v1/health/service/ollama

# Via direct health endpoint
curl http://localhost:8000/api/ollama/health

# Check container logs
docker logs sutazai-distributed_ollama_1
```

### Common Issues

1. **Service not responding**
   - Check if service is scaled down (replicas=0)
   - Verify health checks in Consul
   - Check resource limits

2. **High memory usage**
   - Review service scaler logs
   - Adjust memory limits
   - Enable more aggressive idle timeouts

3. **Slow response times**
   - Check queue depth in RabbitMQ
   - Review cache hit rates
   - Scale up services if needed

## Production Considerations

### Security
- Enable Kong authentication plugins
- Use TLS for all communications
- Implement network policies
- Secure RabbitMQ and Redis

### Persistence
- Mount volumes for data persistence
- Regular backups of vector databases
- Export Consul KV store
- Backup workflow definitions

### High Availability
- Deploy Consul in cluster mode
- Use Redis Sentinel
- RabbitMQ clustering
- Multi-region deployment option

## Next Steps

1. **Customize Service Configuration**
   - Adjust resource limits based on workload
   - Configure service-specific settings
   - Optimize model loading strategies

2. **Implement Additional Services**
   - Add remaining 30+ AI services
   - Configure specialized tools
   - Integrate custom models

3. **Production Hardening**
   - Enable security features
   - Setup monitoring alerts
   - Implement backup strategies
   - Configure auto-scaling policies

## Support

For issues or questions:
1. Check service logs: `docker logs <container>`
2. Review Consul UI for service health
3. Monitor Grafana dashboards
4. Check RabbitMQ queue status

The system is designed to be self-healing and will automatically recover from most transient failures.