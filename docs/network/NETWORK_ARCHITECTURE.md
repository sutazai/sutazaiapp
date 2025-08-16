# MCP Network Architecture and Operations Guide

## Overview

This document describes the comprehensive network architecture implemented to resolve MCP service isolation and connectivity issues. The solution provides proper network segmentation, service discovery, load balancing, and multi-client support.

## Architecture Components

### 1. Network Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    sutazai-network                          │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │   Backend    │  │   Consul    │  │   HAProxy LB     │    │
│  │   (10010)    │  │   (11090)   │  │   (11099)        │    │
│  └──────────────┘  └─────────────┘  └──────────────────┘    │
│                           │                  │               │
│                           └──────────────────┼───────────────┤
│                                              │               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                mcp-internal                             ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      ││
│  │  │ MCP-PG  │ │ MCP-Files│ │ MCP-HTTP│ │ MCP-DDG │ ...  ││
│  │  │ (11100) │ │ (11101) │ │ (11102) │ │ (11103) │      ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2. Network Segments

#### Primary Network: `sutazai-network`
- **Purpose**: Main infrastructure communication
- **Components**: Backend, Consul, HAProxy, existing services
- **Subnet**: Managed by Docker (default bridge)

#### Isolated Network: `mcp-internal`
- **Purpose**: MCP service isolation and communication
- **Subnet**: 172.20.0.0/24
- **Components**: All MCP services
- **Access**: Via HAProxy load balancer only

### 3. Port Allocation

#### Infrastructure Ports (11090-11099)
- **11090**: MCP Consul UI and API
- **11091**: Network monitoring dashboard
- **11099**: HAProxy statistics interface

#### MCP Service Ports (11100-11199)
- **11100**: PostgreSQL MCP Service
- **11101**: Files MCP Service  
- **11102**: HTTP Fetch MCP Service
- **11103**: DuckDuckGo Search MCP Service
- **11104**: GitHub MCP Service
- **11105**: Extended Memory MCP Service

## Service Discovery

### Consul Integration

**Service Registration**
- Automatic registration via HTTP API
- Health check integration
- Metadata and capability tagging
- Service deregistration on shutdown

**Discovery Endpoints**
```bash
# List all services
curl http://localhost:11090/v1/agent/services

# Get service health
curl http://localhost:11090/v1/health/service/mcp-postgres

# Query specific service
curl http://localhost:11090/v1/catalog/service/mcp-files
```

## Load Balancing

### HAProxy Configuration

**Features**
- Round-robin load balancing
- Health check integration
- Automatic failover
- Session affinity (when needed)
- Multi-client support

**Backends**
- Each MCP service has dedicated backend
- Health checks every 10 seconds
- Automatic server marking (UP/DOWN)
- Graceful degradation

**Access Points**
```bash
# HAProxy statistics
http://localhost:11099/stats

# Direct service access through load balancer
curl http://localhost:11100/health  # PostgreSQL
curl http://localhost:11101/health  # Files
curl http://localhost:11102/health  # HTTP
```

## Multi-Client Support

### Concurrent Access Handling

**Architecture Benefits**
- Load balancing prevents service overload
- Network isolation prevents interference
- Health monitoring ensures availability
- Circuit breaking protects from cascading failures

**Client Types Supported**
- Claude Code (primary)
- Codex (secondary)
- Direct API access
- Monitoring tools

## Health Monitoring

### Network Monitor Dashboard

**Access**: http://localhost:11091

**Features**
- Real-time service health monitoring
- Network connectivity testing
- Response time tracking
- Historical health trends
- Infrastructure status

**Metrics Collected**
- TCP connectivity status
- HTTP health check results
- Response time measurements
- Service availability percentage
- Error tracking and categorization

### Health Check Endpoints

**Service Level**
```bash
# Individual service health
curl http://localhost:11100/health

# Service metrics
curl http://localhost:11100/metrics

# Service info
curl http://localhost:11100/info
```

**Infrastructure Level**
```bash
# Overall network health
curl http://localhost:11091/health

# Detailed metrics
curl http://localhost:11091/metrics

# Service trends
curl http://localhost:11091/service/postgres/trend?minutes=60
```

## Deployment and Operations

### Deployment Process

1. **Prerequisites Check**
   - Docker and Docker Compose installed
   - sutazai-network exists
   - Port availability verification

2. **Infrastructure Deployment**
   ```bash
   cd /opt/sutazaiapp
   ./scripts/network/deploy-mcp-network.sh
   ```

3. **Validation**
   - Container status verification
   - Network connectivity testing
   - Service registration confirmation
   - Load balancer functionality check

### Operations Commands

**Start MCP Network**
```bash
docker-compose -f docker/docker-compose.mcp-network.yml up -d
```

**Stop MCP Network**
```bash
docker-compose -f docker/docker-compose.mcp-network.yml down
```

**View Logs**
```bash
docker-compose -f docker/docker-compose.mcp-network.yml logs -f
```

**Scale Services** (when needed)
```bash
docker-compose -f docker/docker-compose.mcp-network.yml up -d --scale mcp-postgres=2
```

## Troubleshooting

### Common Issues

#### Service Discovery Problems
**Symptoms**: Services not appearing in Consul
**Solutions**:
1. Check container network membership
2. Verify Consul agent connectivity
3. Review service registration logs

#### Load Balancer Issues
**Symptoms**: 503 errors, connection refused
**Solutions**:
1. Check HAProxy backend status
2. Verify service health checks
3. Review HAProxy configuration

#### Network Connectivity
**Symptoms**: Services unreachable, timeouts
**Solutions**:
1. Test TCP connectivity with netcat
2. Check Docker network configuration
3. Verify firewall rules

### Diagnostic Commands

**Network Inspection**
```bash
# Check network membership
docker network inspect sutazai-network
docker network inspect mcp-internal

# Container connectivity test
docker exec sutazai-mcp-consul ping mcp-postgres

# Port accessibility
nc -zv localhost 11100
```

**Service Diagnostics**
```bash
# Check service logs
docker logs sutazai-mcp-postgres

# Container process status
docker exec sutazai-mcp-postgres ps aux

# Health check testing
docker exec sutazai-mcp-postgres curl -f http://localhost:11100/health
```

**Load Balancer Status**
```bash
# HAProxy stats
curl http://localhost:11099/stats

# Backend server status
curl http://localhost:11099/stats | grep -A5 "postgres-servers"
```

## Security Considerations

### Network Isolation
- MCP services run in isolated network segment
- Access only through authenticated load balancer
- No direct external access to MCP services
- Network traffic encryption where applicable

### Access Controls
- Service discovery authentication
- Load balancer access controls
- Health monitoring secured endpoints
- Container-level security policies

## Performance Optimization

### Load Balancing
- Round-robin distribution prevents hotspots
- Health checks ensure optimal routing
- Connection pooling for efficiency
- Automatic failover for reliability

### Monitoring
- Real-time performance metrics
- Historical trend analysis
- Proactive alerting capabilities
- Capacity planning insights

## Maintenance Procedures

### Regular Tasks
1. Monitor service health dashboards
2. Review load balancer statistics
3. Check network connectivity trends
4. Update service configurations as needed

### Periodic Tasks
1. Review and rotate service credentials
2. Update container images and configurations
3. Analyze performance trends and optimize
4. Test disaster recovery procedures

## Future Enhancements

### Planned Improvements
- TLS encryption for service communication
- Advanced load balancing strategies (least connections, IP hash)
- Service mesh integration (Istio/Linkerd)
- Advanced monitoring and alerting
- Automated scaling based on load

### Integration Opportunities
- Kubernetes orchestration
- Advanced service discovery features
- Cross-datacenter service federation
- Advanced security policies and encryption