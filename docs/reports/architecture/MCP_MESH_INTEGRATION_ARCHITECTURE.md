# MCP-Mesh Integration Architecture Design

**Document Version**: 1.0.0  
**Date**: 2025-08-16  
**Author**: Backend API Architect  
**Status**: DESIGN PROPOSAL

## Executive Summary

This document outlines the architecture for integrating Model Context Protocol (MCP) servers with the SutazAI service mesh, enabling MCP servers to function as first-class mesh services with full service discovery, load balancing, circuit breaking, and observability capabilities.

## Current State Analysis

### MCP Infrastructure (17 Servers)
- **Deployment**: STDIO-based processes managed by wrapper scripts
- **Communication**: Direct process invocation via shell scripts
- **Discovery**: Static configuration in `.mcp.json`
- **Load Balancing**: None (single instance per server)
- **Fault Tolerance**: None (process crashes require manual restart)
- **Observability**: Limited to process logs

### Service Mesh Infrastructure
- **Service Discovery**: Consul-based (partially connected)
- **API Gateway**: Kong (configured but not integrated)
- **Load Balancing**: 5 strategies implemented
- **Circuit Breaking**: PyBreaker integration
- **Observability**: Prometheus metrics, distributed tracing

## Integration Architecture

### 1. MCP Service Adapter Layer

Create an adapter layer that bridges MCP servers with the mesh:

```python
class MCPServiceAdapter:
    """Adapter to expose MCP servers as mesh services"""
    
    def __init__(self, mcp_server_name: str, mcp_config: Dict):
        self.mcp_server_name = mcp_server_name
        self.mcp_config = mcp_config
        self.process = None
        self.port = self._allocate_port()  # Dynamic port allocation
        
    async def start(self):
        """Start MCP server with HTTP/REST wrapper"""
        # 1. Launch MCP process using wrapper script
        # 2. Create HTTP server to expose MCP functionality
        # 3. Register with service mesh
        # 4. Enable health checks
        
    async def handle_request(self, request: ServiceRequest):
        """Convert HTTP requests to MCP protocol"""
        # 1. Parse HTTP request
        # 2. Convert to MCP protocol
        # 3. Send to MCP process via STDIO
        # 4. Convert response to HTTP
        # 5. Return response
```

### 2. MCP Registry Service

Centralized registry for MCP server management:

```yaml
# /opt/sutazaiapp/config/mcp-mesh-registry.yaml
mcp_services:
  - name: language-server
    service_name: mcp-language-server
    instances: 3  # Scale to 3 instances
    port_range: [11100, 11102]
    health_check: /health
    tags: ["mcp", "language", "code-analysis"]
    load_balancer: ROUND_ROBIN
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 30
      
  - name: github
    service_name: mcp-github
    instances: 2
    port_range: [11103, 11104]
    health_check: /health
    tags: ["mcp", "github", "vcs"]
    load_balancer: LEAST_CONNECTIONS
    
  - name: postgres
    service_name: mcp-postgres
    instances: 5  # Database connection pool
    port_range: [11105, 11109]
    health_check: /health
    tags: ["mcp", "database", "postgres"]
    load_balancer: WEIGHTED
```

### 3. MCP-Mesh Bridge Service

New service to manage MCP-mesh integration:

```python
# /opt/sutazaiapp/backend/app/mesh/mcp_bridge.py

class MCPMeshBridge:
    """Bridge between MCP servers and service mesh"""
    
    def __init__(self, mesh: ServiceMesh, registry_path: str):
        self.mesh = mesh
        self.registry = self._load_registry(registry_path)
        self.adapters: Dict[str, List[MCPServiceAdapter]] = {}
        
    async def initialize(self):
        """Initialize all MCP services in the mesh"""
        for mcp_config in self.registry['mcp_services']:
            await self._deploy_mcp_service(mcp_config)
            
    async def _deploy_mcp_service(self, config: Dict):
        """Deploy MCP service with multiple instances"""
        service_name = config['service_name']
        instances = config.get('instances', 1)
        
        for i in range(instances):
            # Create adapter for each instance
            adapter = MCPServiceAdapter(
                mcp_server_name=config['name'],
                mcp_config=config
            )
            
            # Start the adapter
            await adapter.start()
            
            # Register with mesh
            instance = await self.mesh.register_service(
                service_name=service_name,
                address="localhost",
                port=adapter.port,
                tags=config.get('tags', []),
                metadata={
                    'mcp_server': config['name'],
                    'instance_id': i,
                    'version': '1.0.0'
                }
            )
            
            # Store adapter reference
            if service_name not in self.adapters:
                self.adapters[service_name] = []
            self.adapters[service_name].append(adapter)
```

### 4. HTTP/REST API Layer for MCP

Expose MCP functionality through RESTful APIs:

```python
# API Endpoints for MCP services
@app.post("/api/v1/mcp/{service_name}/execute")
async def execute_mcp_command(
    service_name: str,
    command: Dict[str, Any]
):
    """Execute MCP command through mesh"""
    request = ServiceRequest(
        service_name=f"mcp-{service_name}",
        method="POST",
        path="/execute",
        body=command
    )
    
    result = await service_mesh.call_service(request)
    return result

@app.get("/api/v1/mcp/{service_name}/status")
async def get_mcp_service_status(service_name: str):
    """Get MCP service status from mesh"""
    instances = await service_mesh.discover_services(f"mcp-{service_name}")
    return {
        "service": service_name,
        "instances": len(instances),
        "healthy": sum(1 for i in instances if i['state'] == 'HEALTHY'),
        "details": instances
    }
```

### 5. MCP Load Balancing Strategy

Implement MCP-specific load balancing:

```python
class MCPLoadBalancer(LoadBalancer):
    """MCP-aware load balancer"""
    
    def select_instance(self, instances, service_name, context=None):
        # Consider MCP-specific factors:
        # - Current process memory usage
        # - Request queue depth
        # - Response time history
        # - Capability matching (e.g., language-specific)
        
        if context and 'capability' in context:
            # Filter instances by capability
            capable_instances = [
                i for i in instances 
                if context['capability'] in i.metadata.get('capabilities', [])
            ]
            if capable_instances:
                instances = capable_instances
                
        return super().select_instance(instances, service_name)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
1. Create MCP Service Adapter base class
2. Implement HTTP/STDIO bridge for MCP communication
3. Create MCP Registry configuration schema
4. Add MCP-specific health checks

### Phase 2: Integration (Week 2)
1. Implement MCP-Mesh Bridge service
2. Add MCP service registration to mesh
3. Create REST API endpoints for MCP access
4. Implement MCP-specific load balancing

### Phase 3: Scaling & Resilience (Week 3)
1. Enable multi-instance MCP deployments
2. Implement circuit breakers for MCP services
3. Add retry logic with exponential backoff
4. Create MCP service orchestration

### Phase 4: Observability (Week 4)
1. Add Prometheus metrics for MCP services
2. Implement distributed tracing for MCP calls
3. Create Grafana dashboards for MCP monitoring
4. Add alerting for MCP service failures

## Benefits of Integration

### 1. **Scalability**
- Run multiple instances of MCP servers
- Automatic load distribution
- Horizontal scaling based on demand

### 2. **Resilience**
- Circuit breakers prevent cascade failures
- Automatic failover to healthy instances
- Retry logic with backoff

### 3. **Observability**
- Unified monitoring across all services
- Distributed tracing for debugging
- Performance metrics and alerting

### 4. **Service Discovery**
- Dynamic MCP service discovery
- No hardcoded endpoints
- Automatic instance registration

### 5. **API Gateway Integration**
- Kong routing for MCP services
- Rate limiting and authentication
- Request/response transformation

## Migration Strategy

### Step 1: Parallel Deployment
- Keep existing MCP infrastructure running
- Deploy MCP-mesh bridge alongside
- Route new requests through mesh

### Step 2: Gradual Migration
- Migrate one MCP server at a time
- Monitor performance and stability
- Rollback capability for each service

### Step 3: Deprecation
- Remove direct MCP invocation
- Update all clients to use mesh APIs
- Decommission old wrapper scripts

## Configuration Examples

### MCP Service Registration
```bash
curl -X POST http://localhost:10010/api/v1/mesh/v2/register \
  -H "Content-Type: application/json" \
  -d '{
    "service_id": "mcp-language-server-1",
    "service_name": "mcp-language-server",
    "address": "localhost",
    "port": 11100,
    "tags": ["mcp", "language", "code-analysis"],
    "metadata": {
      "mcp_server": "language-server",
      "instance": 1,
      "capabilities": ["python", "javascript", "typescript"]
    }
  }'
```

### MCP Service Discovery
```bash
curl http://localhost:10010/api/v1/mesh/v2/services?name=mcp-language-server
```

### MCP Command Execution
```bash
curl -X POST http://localhost:10010/api/v1/mcp/language-server/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "analyze",
    "params": {
      "file": "/opt/sutazaiapp/backend/app/main.py",
      "language": "python"
    }
  }'
```

## Monitoring & Metrics

### Key Metrics to Track
1. **MCP Service Metrics**
   - Request rate per MCP service
   - Response time percentiles (p50, p95, p99)
   - Error rates by service and error type
   - Instance utilization

2. **Resource Metrics**
   - Memory usage per MCP instance
   - CPU utilization
   - Process lifecycle events
   - Queue depths

3. **Business Metrics**
   - MCP command success rate
   - Most used MCP services
   - Command execution time by type
   - Concurrent MCP operations

## Security Considerations

### 1. **Authentication**
- JWT tokens for MCP API access
- Service-to-service authentication
- API key management for external access

### 2. **Authorization**
- Role-based access control for MCP commands
- Service-level permissions
- Resource quotas per client

### 3. **Isolation**
- Process isolation for MCP servers
- Network segmentation
- Resource limits per instance

### 4. **Audit Logging**
- Log all MCP command executions
- Track service access patterns
- Security event monitoring

## Conclusion

The MCP-Mesh integration architecture provides a robust, scalable, and observable platform for MCP services. By treating MCP servers as first-class mesh services, we gain:

- **High Availability**: Multiple instances with automatic failover
- **Performance**: Load balancing and circuit breaking
- **Observability**: Unified monitoring and tracing
- **Scalability**: Horizontal scaling based on demand
- **Maintainability**: Centralized configuration and management

This architecture transforms MCP servers from isolated processes into a distributed, resilient service ecosystem that can scale with the demands of the SutazAI platform.

## Next Steps

1. **Review and Approval**: Architecture review by team leads
2. **Prototype Development**: Build proof-of-concept for one MCP server
3. **Performance Testing**: Benchmark mesh overhead vs. direct invocation
4. **Security Review**: Validate security controls and access patterns
5. **Production Rollout**: Phased deployment with monitoring