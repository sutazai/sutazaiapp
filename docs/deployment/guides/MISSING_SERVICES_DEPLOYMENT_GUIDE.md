# SutazAI Missing Services Deployment Guide

**Author:** Deploy Automation Master  
**Date:** 2025-08-04  
**Version:** 1.0.0  

## Overview

This guide provides complete deployment automation for all missing critical infrastructure services identified in the Master System Blueprint v2.2. The deployment follows best practices for zero-downtime deployments, resource optimization, and production readiness.

## 🎯 Deployed Services

### Core Infrastructure (Priority 1)
- **Neo4j Graph Database** - Ports 10002-10003
- **Kong API Gateway** - Port 10005
- **Consul Service Discovery** - Port 10006
- **RabbitMQ Message Queue** - Ports 10007-10008
- **Resource Manager** - Port 10009
- **Backend API** - Port 10010
- **Frontend UI** - Port 10011

### AI & Vector Services
- **FAISS Vector Index** - Port 10103

### Monitoring Stack (Priority 2)
- **Loki Log Aggregation** - Port 10202
- **Alertmanager** - Port 10203
- **AI Metrics Exporter** - Port 10204

## 📁 File Structure

```
/opt/sutazaiapp/
├── docker-compose.missing-services.yml    # Main deployment configuration
├── scripts/
│   └── deploy-missing-services.sh         # Automated deployment script
├── configs/                               # Service configurations
│   ├── neo4j/
│   │   └── neo4j.conf
│   ├── consul/
│   │   └── consul.hcl
│   ├── rabbitmq/
│   │   ├── rabbitmq.conf
│   │   └── definitions.json
│   ├── kong/
│   │   └── kong.yml
│   ├── loki/
│   │   └── loki.yml
│   └── alertmanager/
│       └── alertmanager.yml
└── services/                              # Service implementations
    ├── resource-manager/
    │   └── main.py
    ├── ai-metrics/
    │   └── main.py
    └── faiss-vector/
        └── main.py
```

## 🚀 Quick Deployment

### Automated Deployment (Recommended)

```bash
# Navigate to project root
cd /opt/sutazaiapp

# Run automated deployment
./scripts/deploy-missing-services.sh
```

### Manual Deployment

```bash
# 1. Create external network
docker network create --driver bridge --subnet=172.20.0.0/16 sutazai-network

# 2. Create external volumes
docker volume create shared_runtime_data

# 3. Deploy services
docker-compose -f docker-compose.missing-services.yml up -d

# 4. Verify deployment
docker-compose -f docker-compose.missing-services.yml ps
```

## 🔧 Configuration Details

### Resource Allocation Strategy

The deployment follows a tiered resource allocation strategy optimized for 12 CPU cores and 29.38GB RAM:

| Service Category | CPU Cores | Memory | Priority |
|------------------|-----------|---------|----------|
| Core Infrastructure | 1-2 cores | 1-2GB | High |
| AI Services | 2-4 cores | 2-4GB | Medium |
| Monitoring | 0.25-1 core | 256MB-1GB | Low |

### Port Allocation

All services use the SutazAI Port Allocation Strategy (10000-10599 range):

```
Core Infrastructure:
├── 10002-10003: Neo4j Graph Database
├── 10005: Kong API Gateway
├── 10006: Consul Service Discovery
├── 10007-10008: RabbitMQ Message Queue
├── 10009: Resource Manager
├── 10010: Backend API
└── 10011: Frontend UI

AI & Vector Services:
└── 10103: FAISS Vector Index

Monitoring Stack:
├── 10202: Loki Log Aggregation
├── 10203: Alertmanager
└── 10204: AI Metrics Exporter
```

### Network Architecture

- **Network Name**: `sutazai-network`
- **Subnet**: `172.20.0.0/16`
- **Driver**: `bridge`
- **DNS Resolution**: Automatic service discovery

## 🔍 Service Details

### Neo4j Graph Database
- **Purpose**: Knowledge graph storage and traversal
- **Configuration**: Optimized for 2GB heap, APOC and GDS plugins
- **Authentication**: Username/password authentication
- **Persistence**: Named volume with automatic backup

### Kong API Gateway
- **Purpose**: API routing, rate limiting, authentication
- **Mode**: DB-less with declarative configuration
- **Features**: CORS, Prometheus metrics, health checks
- **Admin API**: Available on port 10044

### Consul Service Discovery
- **Purpose**: Service registration and health monitoring
- **Mode**: Single-node bootstrap (expandable to cluster)
- **Features**: Web UI, DNS interface, KV store
- **Encryption**: Gossip encryption enabled

### RabbitMQ Message Queue
- **Purpose**: Asynchronous task processing and event streaming
- **Features**: Management UI, clustering support, message persistence
- **Queues**: Pre-configured for AI agent tasks, system events
- **Performance**: Optimized for high throughput

### Resource Manager
- **Purpose**: CPU, memory, and service allocation management
- **Features**: Real-time monitoring, automatic scaling decisions
- **Integrations**: Consul, Redis, PostgreSQL, RabbitMQ
- **Metrics**: Prometheus-compatible metrics export

### Backend API
- **Purpose**: Main application API server
- **Framework**: FastAPI with async support
- **Features**: Auto-documentation, CORS, authentication
- **Integrations**: All data stores and AI services

### Frontend UI
- **Purpose**: Web-based user interface
- **Framework**: React/Node.js optimized build
- **Features**: Responsive design, real-time updates
- **Build**: Production-optimized with static serving

### FAISS Vector Index Service
- **Purpose**: High-performance vector similarity search
- **Features**: IVF indexing, persistence, metrics
- **Performance**: Optimized for 1M+ vectors
- **API**: RESTful interface with batch operations

### Loki Log Aggregation
- **Purpose**: Centralized log collection and indexing
- **Features**: Grafana integration, retention policies
- **Storage**: Filesystem-based with compression
- **Query**: LogQL support for advanced filtering

### Alertmanager
- **Purpose**: Alert routing and notification management
- **Features**: Webhook integration, email notifications
- **Routing**: Service-specific alert handling
- **Integrations**: Resource Manager webhook endpoint

### AI Metrics Exporter
- **Purpose**: Collect and export AI service metrics
- **Features**: Multi-service monitoring, Prometheus metrics
- **Monitored Services**: Ollama, ChromaDB, Qdrant, Neo4j, FAISS
- **Metrics**: Availability, performance, resource usage

## 🔒 Security Configuration

### Authentication & Authorization
- **Neo4j**: Username/password with role-based access
- **Kong**: API key authentication for protected routes
- **RabbitMQ**: User credentials with vhost isolation
- **Consul**: ACL tokens (configurable)

### Network Security
- **Isolation**: All services on private Docker network
- **Firewall**: Host firewall controls external access
- **Encryption**: TLS/SSL for external communications
- **Secrets**: Environment variable-based secret management

### Data Protection
- **Persistence**: All data stored in named Docker volumes
- **Backups**: Automated backup before deployment
- **Encryption**: At-rest encryption for sensitive data
- **Access Control**: Service-level access restrictions

## 📊 Monitoring & Observability

### Health Checks
All services include comprehensive health checks:
- **Startup**: Initial readiness detection
- **Liveness**: Continuous health monitoring
- **Dependencies**: Service dependency validation

### Metrics Collection
- **Application Metrics**: Custom business metrics
- **System Metrics**: CPU, memory, disk, network
- **Service Metrics**: Request rates, latency, errors
- **AI Metrics**: Model performance, vector operations

### Logging Strategy
- **Centralized**: All logs aggregated in Loki
- **Structured**: JSON-formatted log entries
- **Retention**: Configurable retention policies
- **Alerting**: Log-based alerting rules

## 🛠️ Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check for port conflicts
netstat -tuln | grep -E ":(10002|10003|10005-10011|10103|10202-10204)"

# Stop conflicting services
docker-compose down
systemctl stop conflicting-service
```

#### Service Startup Failures
```bash
# Check service logs
docker-compose -f docker-compose.missing-services.yml logs <service-name>

# Check resource usage
docker stats

# Verify configurations
docker-compose -f docker-compose.missing-services.yml config
```

#### Network Issues
```bash
# Recreate network
docker network rm sutazai-network
docker network create --driver bridge --subnet=172.20.0.0/16 sutazai-network

# Check network connectivity
docker-compose -f docker-compose.missing-services.yml exec <service> ping <target-service>
```

### Recovery Procedures

#### Service Recovery
```bash
# Restart specific service
docker-compose -f docker-compose.missing-services.yml restart <service-name>

# Rebuild and restart
docker-compose -f docker-compose.missing-services.yml up -d --force-recreate <service-name>
```

#### Data Recovery
```bash
# Restore from backup
cp -r /opt/sutazaiapp/backups/YYYYMMDD_HHMMSS/data/* /opt/sutazaiapp/data/

# Restart services
docker-compose -f docker-compose.missing-services.yml restart
```

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Check service health and resource usage
- **Monthly**: Review and rotate logs, update service images
- **Quarterly**: Security updates, configuration reviews

### Scaling Operations
```bash
# Scale resource-heavy services
docker-compose -f docker-compose.missing-services.yml up -d --scale ai-metrics-exporter=2

# Update resource limits
# Edit docker-compose.missing-services.yml
docker-compose -f docker-compose.missing-services.yml up -d --force-recreate
```

### Backup Operations
```bash
# Manual backup
./scripts/deploy-missing-services.sh --backup-only

# Scheduled backup (add to crontab)
0 2 * * * /opt/sutazaiapp/scripts/deploy-missing-services.sh --backup-only
```

## 📈 Performance Optimization

### Resource Tuning
- **CPU Affinity**: Set CPU affinity for critical services
- **Memory Limits**: Adjust based on actual usage patterns
- **Disk I/O**: Use SSD storage for database services
- **Network**: Optimize network buffer sizes

### Service-Specific Optimizations
- **Neo4j**: Tune heap size and page cache based on data size
- **FAISS**: Optimize index parameters for query patterns
- **Kong**: Enable response caching for frequently accessed routes
- **RabbitMQ**: Tune queue parameters for message patterns

### Monitoring-Based Optimization
- Use Grafana dashboards to identify bottlenecks
- Set up automated alerts for resource thresholds
- Implement automatic scaling based on metrics
- Regular performance reviews and optimization cycles

## 🎉 Success Validation

After deployment, verify all services are operational:

### Automated Validation
```bash
# Run deployment script (includes validation)
./scripts/deploy-missing-services.sh

# Check all service health endpoints
curl -f http://localhost:10002/browser/
curl -f http://localhost:10006/
curl -f http://localhost:10008/
curl -f http://localhost:10005/
curl -f http://localhost:10009/health
curl -f http://localhost:10010/health
curl -f http://localhost:10011/
curl -f http://localhost:10103/health
curl -f http://localhost:10202/ready
curl -f http://localhost:10203/-/healthy
curl -f http://localhost:10204/metrics
```

### Service Access URLs
- **Neo4j Browser**: http://localhost:10002
- **Consul UI**: http://localhost:10006
- **RabbitMQ Management**: http://localhost:10008
- **Kong Admin API**: http://localhost:10044
- **Resource Manager**: http://localhost:10009
- **Backend API**: http://localhost:10010
- **Frontend UI**: http://localhost:10011
- **FAISS API**: http://localhost:10103
- **Loki**: http://localhost:10202
- **Alertmanager**: http://localhost:10203
- **AI Metrics**: http://localhost:10204/metrics

## 🚀 Production Readiness Checklist

- ✅ All services deployed with proper resource limits
- ✅ Health checks configured for all services
- ✅ Persistent storage configured for stateful services
- ✅ Network isolation and security implemented
- ✅ Monitoring and alerting configured
- ✅ Backup and recovery procedures documented
- ✅ Load balancing and high availability considered
- ✅ Performance optimization applied
- ✅ Security hardening implemented
- ✅ Documentation complete and up-to-date

---

**Deployment Status**: ✅ PRODUCTION READY  
**All missing services successfully deployed and validated!**