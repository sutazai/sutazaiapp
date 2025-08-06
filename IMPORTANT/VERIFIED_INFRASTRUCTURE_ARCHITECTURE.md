# Verified Infrastructure Architecture

## Executive Summary

This document describes the ACTUAL running infrastructure components of the SutazAI system as verified through health checks and container status. All components listed here are confirmed operational.

## Core Infrastructure Stack

### API Gateway & Service Mesh

#### Kong Gateway 3.5
- **Status**: ✅ RUNNING HEALTHY
- **Purpose**: API gateway with load balancing, rate limiting, authentication
- **Ports**: 
  - Gateway: 10005
  - Admin API: 8001
- **Docker Image**: `kong:3.5`
- **Configuration**:
  ```yaml
  environment:
    KONG_DATABASE: "off"
    KONG_DECLARATIVE_CONFIG: "/kong/kong.yml"
    KONG_PROXY_ACCESS_LOG: "/dev/stdout"
    KONG_ADMIN_ACCESS_LOG: "/dev/stdout"
    KONG_PROXY_ERROR_LOG: "/dev/stderr"
    KONG_ADMIN_ERROR_LOG: "/dev/stderr"
  ```
- **Health Check**: `curl -f http://localhost:10005/`
- **Key Features**:
  - Request routing and transformation
  - Authentication and authorization
  - Traffic control and analytics
  - Plugin ecosystem for extensibility

#### Consul Service Discovery
- **Status**: ✅ RUNNING HEALTHY
- **Purpose**: Service discovery, configuration management, health checking
- **Ports**:
  - HTTP API/UI: 10006
  - DNS: 8600
  - Serf LAN: 8301
  - Serf WAN: 8302
  - Server RPC: 8300
- **Docker Image**: `hashicorp/consul:latest`
- **Web UI**: http://localhost:10006/ui
- **Configuration**:
  ```yaml
  command: "agent -server -bootstrap-expect=1 -ui -client=0.0.0.0"
  ```
- **Health Check**: `curl -f http://localhost:10006/v1/status/leader`
- **Key Features**:
  - Service registration and discovery
  - Health checking and monitoring
  - Key/value store for configuration
  - Multi-datacenter support

### Message Queue Infrastructure

#### RabbitMQ 3.12
- **Status**: ✅ RUNNING HEALTHY
- **Purpose**: Message queuing, event streaming, async communication
- **Ports**:
  - AMQP: 10007 (5672)
  - Management UI: 10008 (15672)
- **Docker Image**: `rabbitmq:3.12-management`
- **Management UI**: http://localhost:10008
- **Default Credentials**: guest/guest (development only)
- **Configuration**:
  ```yaml
  environment:
    RABBITMQ_DEFAULT_USER: guest
    RABBITMQ_DEFAULT_PASS: guest
  ```
- **Key Features**:
  - Multiple messaging patterns (pub/sub, work queues, RPC)
  - Message persistence and durability
  - Clustering and high availability
  - Management UI for monitoring

## Network Architecture

### Docker Network Configuration
- **Network Name**: `sutazai-network`
- **Type**: Bridge network (external)
- **Purpose**: Isolated communication between all services
- **Configuration**:
  ```bash
  docker network create sutazai-network
  ```

### Port Allocation Strategy
```
10000-10099: Core databases (PostgreSQL, Redis, Neo4j)
10100-10199: Vector databases and AI storage
10200-10299: Monitoring and observability
10000-10999: General infrastructure services
11000-11999: AI agents and specialized services
8000-8999:   Internal service communication
```

## Service Communication Patterns

### Synchronous Communication
1. **Direct HTTP/REST**:
   - Frontend → Backend API (FastAPI)
   - Backend → AI Services
   - Services → Health endpoints

2. **Kong Gateway Routing**:
   - External requests → Kong → Backend services
   - API versioning and load balancing
   - Authentication and rate limiting

### Asynchronous Communication
1. **RabbitMQ Message Patterns**:
   - **Work Queues**: Task distribution to agents
   - **Publish/Subscribe**: Event broadcasting
   - **RPC**: Request-reply for long operations
   - **Topic Exchange**: Selective message routing

2. **Event Flow Example**:
   ```
   User Request → Backend → RabbitMQ → Agent Workers → Response Queue → Backend → User
   ```

### Service Discovery Pattern
1. **Consul Registration**:
   - Services register on startup
   - Health checks every 10 seconds
   - Automatic deregistration on failure

2. **Discovery Flow**:
   ```
   Service A → Consul DNS → Service B Location → Direct Connection
   ```

## High Availability Architecture

### Current Configuration
- **Single Node**: All services on one Docker host
- **Container Restart**: Automatic restart on failure
- **Health Monitoring**: Prometheus + Grafana dashboards

### Production Recommendations
1. **Multi-Node Deployment**:
   - Kong: Deploy multiple instances behind load balancer
   - Consul: 3-5 node cluster for consensus
   - RabbitMQ: Clustered with mirrored queues

2. **Data Persistence**:
   - Volume mounts for all stateful services
   - Regular backup schedules
   - Disaster recovery procedures

## Security Architecture

### Network Security
- **Isolated Network**: Docker bridge network isolation
- **Internal Communication**: Services use internal hostnames
- **External Access**: Only through defined ports

### Service Security
1. **Kong Gateway**:
   - API key authentication
   - JWT token validation
   - Rate limiting per consumer
   - IP restriction plugins

2. **Consul**:
   - ACL system for access control
   - TLS encryption for cluster communication
   - Intention-based service authorization

3. **RabbitMQ**:
   - User authentication
   - Virtual host isolation
   - SSL/TLS support
   - Fine-grained permissions

## Monitoring & Observability

### Health Monitoring Endpoints
```bash
# Infrastructure Health Checks
curl -f http://localhost:10005/               # Kong Gateway
curl -f http://localhost:10006/v1/status/leader # Consul
curl -f http://localhost:10008/api/health     # RabbitMQ Management

# Service Health Pattern
curl -f http://localhost:[PORT]/health
```

### Metrics Collection
- **Prometheus**: Scrapes metrics from all services
- **Service Exporters**: Each service exposes /metrics endpoint
- **Custom Metrics**: Business and AI-specific metrics

### Log Aggregation
- **Loki**: Centralized log storage
- **Promtail**: Log shipping from containers
- **Grafana**: Unified log and metric visualization

## Deployment Architecture

### Container Management
```yaml
# Docker Compose Service Pattern
service-name:
  image: image:version
  container_name: sutazai-service-name
  ports:
    - "external:internal"
  environment:
    - CONFIG_VAR=value
  networks:
    - sutazai-network
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:port/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  restart: unless-stopped
```

### Service Dependencies
```
1. Network Infrastructure (Docker network)
   ↓
2. Data Layer (PostgreSQL, Redis, Neo4j)
   ↓
3. Message Queue (RabbitMQ)
   ↓
4. Service Discovery (Consul)
   ↓
5. API Gateway (Kong)
   ↓
6. Application Services (Backend, Frontend)
   ↓
7. AI Services (Ollama, Agents)
   ↓
8. Monitoring Stack (Prometheus, Grafana)
```

## Infrastructure Scaling Patterns

### Horizontal Scaling
- **Stateless Services**: Add more container instances
- **Load Balancing**: Kong distributes traffic
- **Service Discovery**: Consul manages instance registry

### Vertical Scaling
- **Resource Limits**: Configure container CPU/memory limits
- **Host Resources**: Scale underlying Docker host

### Data Scaling
- **PostgreSQL**: Read replicas for query scaling
- **Redis**: Redis Cluster for distributed caching
- **RabbitMQ**: Clustered queues for message throughput

## Disaster Recovery

### Backup Strategy
1. **Data Backups**:
   - PostgreSQL: pg_dump scheduled backups
   - Redis: RDB snapshots
   - Neo4j: Online backup procedures

2. **Configuration Backups**:
   - Kong: Declarative config in version control
   - Consul: KV store export
   - RabbitMQ: Definition export

### Recovery Procedures
1. **Service Failure**: Automatic container restart
2. **Data Loss**: Restore from latest backup
3. **Complete Failure**: Rebuild from docker-compose.yml

## Performance Optimization

### Current Optimizations
- **Connection Pooling**: Database connection management
- **Caching Layer**: Redis for frequent queries
- **Async Processing**: RabbitMQ for background tasks
- **Load Distribution**: Kong load balancing

### Monitoring Performance
```bash
# Container resource usage
docker stats

# Service-specific metrics
curl http://localhost:10200/metrics  # Prometheus
curl http://localhost:10005/status   # Kong
```

## Troubleshooting Guide

### Common Issues

1. **Service Won't Start**:
   ```bash
   docker-compose logs [service-name]
   docker-compose restart [service-name]
   ```

2. **Port Conflicts**:
   ```bash
   lsof -i :[PORT]
   # Kill process or change port in docker-compose.yml
   ```

3. **Network Issues**:
   ```bash
   docker network inspect sutazai-network
   docker network prune
   docker network create sutazai-network
   ```

4. **Health Check Failures**:
   ```bash
   docker exec [container] curl localhost:[port]/health
   docker-compose ps
   ```

## Future Infrastructure Roadmap

### Phase 1: Stability (Current)
- ✅ Basic infrastructure running
- ✅ Health monitoring
- ✅ Service discovery

### Phase 2: Resilience
- [ ] Multi-node deployment
- [ ] Automated backups
- [ ] Circuit breakers

### Phase 3: Scale
- [ ] Kubernetes migration
- [ ] Auto-scaling policies
- [ ] Global load balancing

### Phase 4: Advanced Features
- [ ] Service mesh (Istio/Linkerd)
- [ ] Distributed tracing (Jaeger)
- [ ] Chaos engineering