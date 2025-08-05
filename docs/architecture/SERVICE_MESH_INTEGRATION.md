# SutazAI Service Mesh Integration

## Overview

This document describes the comprehensive service mesh integration for SutazAI, providing service discovery, API gateway routing, message queuing, and health monitoring across all services.

## Architecture Components

### 1. Consul (Service Discovery)
- **Purpose**: Service registration and discovery
- **Port**: 10006 (UI and HTTP API)
- **Configuration**: `/opt/sutazaiapp/config/consul/services.json`
- **Features**:
  - Automatic service registration with health checks
  - DNS-based service discovery
  - Key-value store for configuration
  - Multi-datacenter support

### 2. Kong (API Gateway)
- **Purpose**: API gateway with load balancing and routing
- **Ports**: 
  - 10005 (Proxy)
  - 10007 (Admin API)
- **Configuration**: `/opt/sutazaiapp/config/kong/kong.yml`
- **Features**:
  - Route-based request routing
  - Rate limiting and authentication
  - Plugin ecosystem (CORS, monitoring, etc.)
  - Load balancing across service instances

### 3. RabbitMQ (Message Queue)
- **Purpose**: Asynchronous messaging between agents and services
- **Ports**:
  - 10041 (AMQP)
  - 10042 (Management UI)
- **Configuration**: 
  - `/opt/sutazaiapp/config/rabbitmq/rabbitmq.conf`
  - `/opt/sutazaiapp/config/rabbitmq/definitions.json`
- **Features**:
  - Multiple virtual hosts for service isolation
  - Durable queues and exchanges
  - Agent-specific communication channels
  - High availability with clustering

### 4. Health Monitoring
- **Purpose**: Comprehensive health checking and monitoring
- **Port**: 10008 (Health Check API)
- **Features**:
  - Service health monitoring
  - Dependency tracking
  - Prometheus metrics export
  - Integration with Consul health checks

## Service Registration

All SutazAI services are automatically registered with Consul with the following information:

### Core Services
- **backend**: Main API service (port 8000)
- **frontend**: Streamlit UI (port 8501)
- **postgres**: Primary database (port 5432)
- **redis**: Cache and session store (port 6379)
- **neo4j**: Graph database (port 7687)

### AI Services
- **ollama**: LLM inference engine (port 11434)
- **chromadb**: Vector database (port 8000)
- **qdrant**: Vector database (port 6333)
- **faiss**: Vector search (port 8000)

### AI Agents
- **autogpt**: Autonomous task execution
- **crewai**: Collaborative multi-agent system
- **letta**: Memory-enabled conversational agent
- **aider**: Code assistance agent

### Workflow Engines
- **langflow**: Visual workflow builder (port 7860)
- **flowise**: Low-code AI workflows (port 3000)
- **dify**: AI application platform (port 5000)
- **n8n**: Workflow automation (port 5678)

### Monitoring
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Visualization dashboards (port 3000)

## API Routes

Kong provides unified API access through structured routes:

### Core API Routes
- `/api/v1/*` → Backend API
- `/` → Frontend application

### AI Service Routes
- `/api/ollama/*` → Ollama LLM service
- `/api/chromadb/*` → ChromaDB vector database
- `/api/qdrant/*` → Qdrant vector database
- `/api/faiss/*` → FAISS vector search

### Agent Routes
- `/api/agents/autogpt/*` → AutoGPT agent
- `/api/agents/crewai/*` → CrewAI multi-agent
- `/api/agents/letta/*` → Letta memory agent
- `/api/agents/aider/*` → Aider code assistant

### Workflow Routes
- `/api/workflows/langflow/*` → Langflow workflows
- `/api/workflows/flowise/*` → Flowise workflows
- `/api/workflows/n8n/*` → n8n automation
- `/api/apps/dify/*` → Dify applications

### Monitoring Routes
- `/api/metrics/prometheus/*` → Prometheus metrics
- `/api/dashboards/grafana/*` → Grafana dashboards
- `/api/health/*` → Health check API
- `/api/mcp/*` → MCP communication server

## Message Queues

RabbitMQ provides structured messaging with multiple virtual hosts:

### Virtual Hosts
- `/sutazai`: Main application messaging
- `/agents`: AI agent communication
- `/monitoring`: System monitoring and metrics

### Key Exchanges and Queues

#### Agent Communication
- **agents.topic**: Topic exchange for agent routing
- **agent.*.tasks**: Agent-specific task queues
- **agent.responses**: Agent response collection

#### Workflow Management
- **workflows.topic**: Workflow orchestration
- **workflow.*.tasks**: Service-specific workflow queues

#### System Monitoring
- **monitoring.topic**: System metrics and health
- **system.health.checks**: Health check notifications
- **system.metrics.collection**: Metrics aggregation

#### AI Service Coordination
- **ollama.requests**: LLM inference requests
- **vector.indexing.tasks**: Vector database operations

## Health Checks

Comprehensive health monitoring across all services:

### Health Check Types
1. **HTTP Checks**: For web services and APIs
2. **TCP Checks**: For database and cache services
3. **Dependency Checks**: Verifying service dependencies

### Health Check Endpoints
- **Global Health**: `GET /health` - Overall system health
- **Service Health**: `GET /health/{service}` - Individual service status
- **Deep Health**: `GET /health/deep/{service}` - Service + dependencies
- **Metrics**: `GET /metrics` - Prometheus-format metrics

### Health Status Levels
- **healthy**: Service is fully operational
- **warning**: Service operational but dependencies have issues
- **unhealthy**: Service is not responding or failing

## Service Discovery Integration

Services discover each other through multiple mechanisms:

### 1. DNS Resolution
- Consul provides DNS for service discovery
- Services accessible via `{service-name}.service.consul`

### 2. HTTP API
- Consul HTTP API for programmatic service discovery
- Real-time service health and metadata

### 3. Service Registration
- Automatic registration on service startup
- Health check registration with appropriate intervals
- Metadata and tags for service categorization

## Usage

### Starting the Service Mesh

```bash
# Start complete service mesh infrastructure
/opt/sutazaiapp/scripts/service-mesh/start-service-mesh.sh

# Start without validation
/opt/sutazaiapp/scripts/service-mesh/start-service-mesh.sh --no-validation

# Show current status
/opt/sutazaiapp/scripts/service-mesh/start-service-mesh.sh --status
```

### Validating Service Mesh

```bash
# Complete validation
python3 /opt/sutazaiapp/scripts/service-mesh/validate-service-mesh.py

# Detailed output
python3 /opt/sutazaiapp/scripts/service-mesh/validate-service-mesh.py --detailed

# Validate specific component
python3 /opt/sutazaiapp/scripts/service-mesh/validate-service-mesh.py --component consul
```

### Service Registration

Services can register with Consul programmatically:

```python
from scripts.service_mesh.service_discovery_client import ServiceDiscoveryClient, ServiceConfig

client = ServiceDiscoveryClient()
await client.initialize()

service_config = ServiceConfig(
    name="my-service",
    id="my-service-1",
    address="my-service",
    port=8080,
    tags=["api", "custom"],
    meta={"version": "1.0"},
    health_check={"http": "http://my-service:8080/health"}
)

await client.register_service(service_config)
```

### Health Monitoring

Query service health programmatically:

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    # Get overall system health
    async with session.get("http://health-check-server:8080/health") as response:
        health_data = await response.json()
    
    # Get specific service health
    async with session.get("http://health-check-server:8080/health/backend") as response:
        service_health = await response.json()
```

## Configuration Files

### Consul Services (`/opt/sutazaiapp/config/consul/services.json`)
Defines all services with health checks, metadata, and registration details.

### Kong Gateway (`/opt/sutazaiapp/config/kong/kong.yml`)
Configures API routes, rate limiting, CORS, and other gateway features.

### RabbitMQ Setup (`/opt/sutazaiapp/config/rabbitmq/`)
- `rabbitmq.conf`: Server configuration
- `definitions.json`: Exchanges, queues, and bindings

## Monitoring and Observability

### Prometheus Metrics
- Service health status
- Response times
- Request rates
- System resource usage

### Grafana Dashboards
- Service mesh overview
- Individual service details
- Alert management
- Performance analytics

### Logging
- Centralized logging via Loki
- Service mesh component logs
- Health check results
- Configuration changes

## Security Considerations

### Network Security
- Services communicate within isolated Docker networks
- TLS encryption for external communication
- Rate limiting on all API endpoints

### Authentication
- Kong plugin-based authentication
- Service-to-service authentication via tokens
- Admin API access controls

### Secrets Management
- Environment variable-based secrets
- No secrets in configuration files
- Secure credential storage

## Troubleshooting

### Common Issues

1. **Service Registration Failures**
   - Check Consul connectivity
   - Verify service health endpoints
   - Review service configuration

2. **Kong Route Issues**
   - Validate service URLs in Kong config
   - Check route path conflicts
   - Verify upstream service health

3. **RabbitMQ Connection Problems**
   - Check virtual host permissions
   - Verify queue and exchange configuration
   - Monitor connection limits

4. **Health Check Failures**
   - Verify health endpoints are responsive
   - Check network connectivity
   - Review health check timeouts

### Diagnostic Commands

```bash
# Check Consul service catalog
curl http://localhost:10006/v1/catalog/services

# Check Kong services
curl http://localhost:10007/services

# Check RabbitMQ overview
curl -u admin:adminpass http://localhost:10042/api/overview

# Check health monitoring
curl http://localhost:10008/health
```

## Performance Tuning

### Consul
- Adjust health check intervals based on service criticality
- Configure appropriate timeout values
- Use service tags for efficient filtering

### Kong
- Configure upstream health checks
- Set appropriate rate limits
- Enable response caching where applicable

### RabbitMQ
- Configure queue TTL based on message patterns
- Set appropriate prefetch values
- Monitor memory and disk usage

## High Availability

### Consul Clustering
- Multi-node Consul cluster for production
- Cross-datacenter replication
- Automated failover

### Kong Load Balancing
- Multiple Kong instances behind load balancer
- Database clustering for Kong
- Health check-based upstream selection

### RabbitMQ Clustering
- Multi-node RabbitMQ cluster
- Queue mirroring across nodes
- Automatic partition handling

## Future Enhancements

1. **Service Mesh Evolution**
   - Integration with Istio or Linkerd
   - Advanced traffic management
   - Circuit breaker patterns

2. **Enhanced Security**
   - Mutual TLS between services
   - Advanced authentication methods
   - Network policy enforcement

3. **Observability Improvements**
   - Distributed tracing
   - Advanced alerting rules
   - Performance analytics

4. **Automation**
   - GitOps-based configuration management
   - Automated scaling based on metrics
   - Self-healing capabilities

## Conclusion

The SutazAI service mesh provides a robust foundation for microservices communication, monitoring, and management. It enables reliable service discovery, intelligent routing, asynchronous messaging, and comprehensive health monitoring across the entire AI platform.

For questions or issues, refer to the troubleshooting section or check the service mesh component logs in `/opt/sutazaiapp/logs/`.