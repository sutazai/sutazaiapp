# Docker Image Fixes - Deployment Guide

## Issues Resolved

The following Docker image configuration issues have been fixed in `/opt/sutazaiapp/docker-compose.missing-services.yml`:

### 1. Consul Service Discovery
- **Issue**: `consul:latest` image not found
- **Fix**: Changed to `hashicorp/consul:1.17`
- **Status**: ✅ Verified and working

### 2. Loki Log Aggregation
- **Issue**: Old version (2.9.0) configuration compatibility
- **Fix**: Upgraded to `grafana/loki:3.0.0` with updated configuration
- **Status**: ✅ Configuration updated for v3.0 compatibility

### 3. Alertmanager
- **Issue**: Using `latest` tag which may cause inconsistencies
- **Fix**: Pinned to specific version `prom/alertmanager:v0.27.0`
- **Status**: ✅ Version pinned and tested

### 4. All Other Images
- **Neo4j**: `neo4j:5-community` ✅ Working
- **Kong**: `kong:3.5` ✅ Working
- **RabbitMQ**: `rabbitmq:3.12-management-alpine` ✅ Working
- **Python**: `python:3.11-slim` ✅ Working
- **Node.js**: `node:18-alpine` ✅ Working

## Deployment Instructions

### Step 1: Prepare the Environment
```bash
# Run the preparation script
./scripts/prepare-missing-services.sh
```

This script will:
- Create all required directories
- Test Docker image availability
- Create the external network
- Create placeholder services if they don't exist

### Step 2: Deploy the Services
```bash
# Deploy all missing services
docker-compose -f docker-compose.missing-services.yml up -d
```

### Step 3: Validate the Deployment
```bash
# Wait for services to start (2-3 minutes)
sleep 120

# Run validation script
./scripts/validate-missing-services.sh
```

## Service Access URLs

After successful deployment, the following services will be available:

| Service | URL | Purpose |
|---------|-----|---------|
| Neo4j Browser | http://localhost:10002 | Graph database management |
| Kong Gateway | http://localhost:10005 | API Gateway proxy |
| Kong Admin API | http://localhost:10044 | Gateway configuration |
| Consul UI | http://localhost:10006 | Service discovery |
| RabbitMQ Management | http://localhost:10008 | Message queue management |
| Backend API | http://localhost:10010 | Main application API |
| Frontend UI | http://localhost:10011 | Web interface |
| FAISS Vector Service | http://localhost:10103 | Vector search |
| Loki | http://localhost:10202 | Log aggregation |
| Alertmanager | http://localhost:10203 | Alert management |
| AI Metrics | http://localhost:10204 | AI performance metrics |

## Default Credentials

- **Neo4j**: `neo4j` / `sutazai_neo4j`
- **RabbitMQ**: `sutazai` / `sutazai_rmq`

## Configuration Updates

### Loki Configuration Changes
The Loki configuration has been updated for v3.0.0 compatibility:
- Schema changed from `v11` with `boltdb-shipper` to `v13` with `tsdb`
- Removed deprecated `table_manager` section
- Retention is now handled by the compactor

### Consul Configuration
- Using official HashiCorp Consul image
- Configuration remains compatible with the existing `consul.hcl`

## Troubleshooting

### Common Issues

1. **Container fails to start**
   ```bash
   # Check logs
   docker-compose -f docker-compose.missing-services.yml logs [service-name]
   ```

2. **Port conflicts**
   ```bash
   # Check if ports are in use
   netstat -tlnp | grep -E ':(10002|10005|10006|10008|10010|10011|10044|10103|10202|10203|10204)'
   ```

3. **Network issues**
   ```bash
   # Verify network exists
   docker network inspect sutazai-network
   ```

4. **Volume permission issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER /var/lib/docker/volumes/
   ```

### Service Restart Commands

```bash
# Restart specific service
docker-compose -f docker-compose.missing-services.yml restart [service-name]

# Restart all services
docker-compose -f docker-compose.missing-services.yml restart

# Stop all services
docker-compose -f docker-compose.missing-services.yml down

# Stop and remove volumes (careful!)
docker-compose -f docker-compose.missing-services.yml down -v
```

## Resource Requirements

The deployment is optimized for:
- **CPU**: 12 cores
- **Memory**: 29.38GB RAM
- **Storage**: Minimum 50GB for volumes

## Health Checks

All services include health checks that can be monitored:
```bash
# Check container health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Security Notes

- Services use internal Docker networking for communication
- External access is limited to designated ports
- Default passwords should be changed in production
- Consider using Docker secrets for sensitive data

## Monitoring Integration

The deployment includes:
- **Loki** for log aggregation
- **Alertmanager** for alert management
- **AI Metrics Exporter** for custom metrics
- Integration with existing Prometheus/Grafana stack

## Next Steps

1. Update environment variables in `.env` file if needed
2. Configure SSL/TLS certificates for production
3. Set up backup strategies for persistent volumes
4. Configure monitoring alerts
5. Implement log rotation policies