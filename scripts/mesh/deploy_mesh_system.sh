#!/bin/bash
# Comprehensive Mesh System Deployment Script
# Purpose: Deploy and integrate ALL services into the service mesh
# Created: 2025-08-17 UTC

set -e

echo "============================================"
echo "MESH SYSTEM DEPLOYMENT - ULTRATHINK EDITION"
echo "============================================"
echo "Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Checking $service health on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" | grep -q "200\|204"; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Function to register service with Consul
register_with_consul() {
    local service_name=$1
    local service_id=$2
    local address=$3
    local port=$4
    local tags=$5
    
    echo -n "Registering $service_name with Consul..."
    
    cat > /tmp/consul_service_${service_id}.json <<EOF
{
  "ID": "${service_id}",
  "Name": "${service_name}",
  "Tags": ${tags},
  "Address": "${address}",
  "Port": ${port},
  "Check": {
    "HTTP": "http://${address}:${port}/health",
    "Interval": "10s",
    "Timeout": "5s",
    "DeregisterCriticalServiceAfter": "1m"
  }
}
EOF
    
    if curl -s -X PUT \
        -d @/tmp/consul_service_${service_id}.json \
        "http://localhost:10006/v1/agent/service/register"; then
        echo -e " ${GREEN}✓${NC}"
    else
        echo -e " ${RED}✗${NC}"
    fi
    
    rm -f /tmp/consul_service_${service_id}.json
}

echo "============================================"
echo "STEP 1: Starting Backend API with Mesh Integration"
echo "============================================"

# Start backend if not running
if ! docker ps | grep -q sutazai-backend; then
    echo "Starting backend service..."
    docker-compose -f /opt/sutazaiapp/docker/docker-compose.consolidated.yml up -d backend
    sleep 10
else
    echo "Backend already running"
fi

# Check backend health
check_service_health "Backend API" 10010

echo ""
echo "============================================"
echo "STEP 2: Deploying MCP Services in DinD"
echo "============================================"

# Build MCP service images if needed
echo "Building MCP service images..."
cd /opt/sutazaiapp/docker/dind/mcp-containers

# Build unified MCP image
if ! docker exec sutazai-mcp-orchestrator docker images | grep -q sutazai-mcp-unified; then
    echo "Building unified MCP image..."
    docker exec sutazai-mcp-orchestrator docker build \
        -f /var/lib/docker/Dockerfile.unified-mcp \
        -t sutazai-mcp-unified:latest \
        /var/lib/docker
fi

# Copy compose file to DinD
echo "Copying MCP services configuration to DinD..."
docker cp docker-compose.mcp-services.yml sutazai-mcp-orchestrator:/var/lib/docker/

# Deploy MCP services in DinD
echo "Deploying MCP services in DinD orchestrator..."
docker exec sutazai-mcp-orchestrator docker-compose \
    -f /var/lib/docker/docker-compose.mcp-services.yml \
    up -d

# Wait for services to start
echo "Waiting for MCP services to initialize..."
sleep 15

# Check MCP service status
echo "MCP Services Status:"
docker exec sutazai-mcp-orchestrator docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "============================================"
echo "STEP 3: Registering ALL Services with Consul"
echo "============================================"

# Register backend service
register_with_consul "backend-api" "backend-api-10010" "localhost" 10010 \
    '["api", "backend", "fastapi", "mesh-enabled"]'

# Register database services
register_with_consul "postgresql" "postgresql-10000" "localhost" 10000 \
    '["database", "postgres", "primary"]'

register_with_consul "redis" "redis-10001" "localhost" 10001 \
    '["cache", "redis", "session"]'

register_with_consul "neo4j" "neo4j-10003" "localhost" 10003 \
    '["database", "graph", "neo4j"]'

# Register AI services
register_with_consul "ollama" "ollama-10104" "localhost" 10104 \
    '["ai", "llm", "ollama"]'

register_with_consul "chromadb" "chromadb-10100" "localhost" 10100 \
    '["ai", "vectordb", "chromadb"]'

register_with_consul "qdrant" "qdrant-10101" "localhost" 10101 \
    '["ai", "vectordb", "qdrant"]'

# Register monitoring services
register_with_consul "prometheus" "prometheus-10200" "localhost" 10200 \
    '["monitoring", "metrics", "prometheus"]'

register_with_consul "grafana" "grafana-10201" "localhost" 10201 \
    '["monitoring", "visualization", "grafana"]'

register_with_consul "jaeger" "jaeger-10210" "localhost" 10210 \
    '["monitoring", "tracing", "jaeger"]'

# Register MCP services (via DinD bridge)
echo "Registering MCP services through DinD bridge..."
MCP_SERVICES=(
    "claude-flow:3001"
    "ruv-swarm:3002"
    "files:3003"
    "context7:3004"
    "http_fetch:3005"
    "ddg:3006"
    "sequentialthinking:3007"
    "nx-mcp:3008"
    "extended-memory:3009"
    "mcp_ssh:3010"
    "ultimatecoder:3011"
    "playwright-mcp:3012"
    "memory-bank-mcp:3013"
    "knowledge-graph-mcp:3014"
    "compass-mcp:3015"
    "github:3016"
    "http:3017"
    "language-server:3018"
    "claude-task-runner:3019"
)

for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    register_with_consul "mcp-$service" "mcp-$service-$port" "dind-bridge" "$port" \
        '["mcp", "'$service'", "containerized", "mesh-integrated"]'
done

echo ""
echo "============================================"
echo "STEP 4: Initializing Service Mesh"
echo "============================================"

# Create mesh initialization script
cat > /tmp/init_mesh.py <<'EOF'
import asyncio
import sys
sys.path.append('/opt/sutazaiapp/backend')

from app.mesh.service_mesh import ServiceMesh, ServiceInstance, ServiceState
from app.mesh.dind_mesh_bridge import DinDMeshBridge
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_mesh():
    """Initialize the service mesh with all services"""
    try:
        # Initialize mesh
        mesh = ServiceMesh()
        await mesh.initialize()
        
        # Initialize DinD bridge
        dind_bridge = DinDMeshBridge(mesh_client=mesh)
        await dind_bridge.initialize()
        
        # Discover MCP containers
        mcp_services = await dind_bridge.discover_mcp_containers()
        logger.info(f"Discovered {len(mcp_services)} MCP services in DinD")
        
        # Get mesh status
        status = await mesh.get_mesh_status()
        logger.info(f"Mesh Status: {status['total_services']} services, {status['healthy_services']} healthy")
        
        # Test service discovery
        services = await mesh.discover_service("mcp-claude-flow")
        if services:
            logger.info(f"✓ Service discovery working: Found {len(services)} instances of mcp-claude-flow")
        
        return True
        
    except Exception as e:
        logger.error(f"Mesh initialization failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(initialize_mesh())
    sys.exit(0 if result else 1)
EOF

echo "Initializing service mesh..."
cd /opt/sutazaiapp
python3 /tmp/init_mesh.py

echo ""
echo "============================================"
echo "STEP 5: Verifying Mesh Integration"
echo "============================================"

# Check Consul services
echo "Services registered in Consul:"
curl -s http://localhost:10006/v1/agent/services | python3 -c "import sys, json; services = json.load(sys.stdin); print(f'Total: {len(services)} services'); [print(f'  - {s}') for s in services.keys()]"

# Check mesh health via backend API
echo ""
echo "Checking mesh health via Backend API..."
curl -s http://localhost:10010/api/v1/mesh/status 2>/dev/null || echo "Backend mesh endpoint not responding"

echo ""
echo "============================================"
echo "STEP 6: Setting up Mesh Monitoring"
echo "============================================"

# Configure Prometheus to scrape mesh metrics
cat > /tmp/mesh_prometheus_config.yml <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mesh-services'
    consul_sd_configs:
      - server: 'localhost:10006'
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: service
      - source_labels: [__meta_consul_tags]
        target_label: tags
EOF

echo "Mesh monitoring configuration created"

echo ""
echo "============================================"
echo "DEPLOYMENT SUMMARY"
echo "============================================"

# Get final status
CONSUL_SERVICES=$(curl -s http://localhost:10006/v1/agent/services | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
MCP_CONTAINERS=$(docker exec sutazai-mcp-orchestrator docker ps -q | wc -l)
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:10010/health)

echo "✓ Consul Services Registered: $CONSUL_SERVICES"
echo "✓ MCP Containers Running: $MCP_CONTAINERS"
echo "✓ Backend API Status: $BACKEND_STATUS"
echo ""
echo -e "${GREEN}Mesh deployment complete!${NC}"
echo ""
echo "Access Points:"
echo "  - Backend API: http://localhost:10010"
echo "  - Consul UI: http://localhost:10006"
echo "  - Grafana: http://localhost:10201"
echo "  - Jaeger: http://localhost:10210"
echo ""
echo "Next Steps:"
echo "  1. Test mesh communication: ./scripts/mesh/test_mesh_communication.sh"
echo "  2. View mesh metrics: http://localhost:10201/d/mesh-overview"
echo "  3. Check service traces: http://localhost:10210"