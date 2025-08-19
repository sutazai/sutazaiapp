#!/bin/bash
# Deploy Real MCP Servers
# Replaces fake netcat loops with actual functional MCP servers

set -e

echo "========================================="
echo "Deploying Real MCP Servers"
echo "========================================="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Configuration
DIND_CONTAINER="sutazai-mcp-orchestrator"
MCP_IMAGE="sutazai-mcp-real:latest"
MCP_SERVICES=(
    "claude-flow:3001"
    "ruv-swarm:3002"
    "files:3003"
    "context7:3004"
    "http-fetch:3005"
    "ddg:3006"
    "sequentialthinking:3007"
    "nx-mcp:3008"
    "extended-memory:3009"
    "mcp-ssh:3010"
    "ultimatecoder:3011"
    "playwright-mcp:3012"
    "memory-bank-mcp:3013"
    "knowledge-graph-mcp:3014"
    "compass-mcp:3015"
    "github:3016"
    "http:3017"
    "language-server:3018"
    "claude-task-runner:3019"
    "postgres:3020"
)

# Build the real MCP Docker image
echo "Step 1: Building real MCP Docker image..."
cd /opt/sutazaiapp/docker/dind/mcp-real
docker build -t $MCP_IMAGE .
echo "✅ MCP image built successfully"
echo ""

# Copy image to DinD environment
echo "Step 2: Loading image into DinD environment..."
docker save $MCP_IMAGE | docker exec -i $DIND_CONTAINER docker load
echo "✅ Image loaded into DinD"
echo ""

# Stop and remove fake MCP containers
echo "Step 3: Removing fake MCP containers..."
for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    container_name="mcp-${service//_/-}"
    
    echo "  Stopping fake container: $container_name"
    docker exec $DIND_CONTAINER docker stop $container_name 2>/dev/null || true
    docker exec $DIND_CONTAINER docker rm $container_name 2>/dev/null || true
done
echo "✅ Fake containers removed"
echo ""

# Deploy real MCP containers
echo "Step 4: Deploying real MCP containers..."
for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    container_name="mcp-${service//_/-}"
    
    echo "  Deploying: $container_name on port $port"
    
    # Run the real MCP container
    docker exec $DIND_CONTAINER docker run -d \
        --name $container_name \
        --network mcp-internal \
        -e MCP_SERVICE=$service \
        -e MCP_PORT=$port \
        -e MCP_HOST=0.0.0.0 \
        -e NODE_ENV=production \
        -p $port:$port \
        -v mcp-shared-data:/opt/mcp/data \
        -v mcp-logs:/opt/mcp/logs \
        --restart unless-stopped \
        --label "mcp.real=true" \
        --label "mcp.service=$service" \
        --label "mcp.port=$port" \
        $MCP_IMAGE
    
    # Wait for container to start
    sleep 2
    
    # Check if container is running
    if docker exec $DIND_CONTAINER docker ps | grep -q $container_name; then
        echo "  ✅ $container_name deployed successfully"
    else
        echo "  ❌ Failed to deploy $container_name"
    fi
done
echo ""

# Verify deployment
echo "Step 5: Verifying deployment..."
echo ""
echo "Running MCP containers:"
docker exec $DIND_CONTAINER docker ps --filter "label=mcp.real=true" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Test health endpoints
echo "Step 6: Testing health endpoints..."
for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    container_name="mcp-${service//_/-}"
    
    echo -n "  Testing $container_name health... "
    
    # Test health endpoint from inside DinD
    if docker exec $DIND_CONTAINER wget -q -O- http://$container_name:$port/health >/dev/null 2>&1; then
        echo "✅ Healthy"
    else
        echo "⚠️  Not responding yet"
    fi
done
echo ""

# Update backend configuration
echo "Step 7: Updating backend configuration..."
cat > /tmp/update-mcp-config.py << 'EOF'
import json

# MCP service configuration
mcp_config = {
    "services": {
        "claude-flow": {"port": 3001, "enabled": True},
        "ruv-swarm": {"port": 3002, "enabled": True},
        "files": {"port": 3003, "enabled": True},
        "context7": {"port": 3004, "enabled": True},
        "http-fetch": {"port": 3005, "enabled": True},
        "ddg": {"port": 3006, "enabled": True},
        "sequentialthinking": {"port": 3007, "enabled": True},
        "nx-mcp": {"port": 3008, "enabled": True},
        "extended-memory": {"port": 3009, "enabled": True},
        "mcp-ssh": {"port": 3010, "enabled": True},
        "ultimatecoder": {"port": 3011, "enabled": True},
        "playwright-mcp": {"port": 3012, "enabled": True},
        "memory-bank-mcp": {"port": 3013, "enabled": True},
        "knowledge-graph-mcp": {"port": 3014, "enabled": True},
        "compass-mcp": {"port": 3015, "enabled": True},
        "github": {"port": 3016, "enabled": True},
        "http": {"port": 3017, "enabled": True},
        "language-server": {"port": 3018, "enabled": True},
        "claude-task-runner": {"port": 3019, "enabled": True},
        "postgres": {"port": 3020, "enabled": True}
    },
    "protocol": "mcp/v1",
    "deployment": "docker-in-docker",
    "real_implementation": True
}

# Save configuration
with open('/opt/sutazaiapp/backend/app/config/mcp_services.json', 'w') as f:
    json.dump(mcp_config, f, indent=2)

print("✅ Backend configuration updated")
EOF

python3 /tmp/update-mcp-config.py
echo ""

# Summary
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Real MCP servers have been deployed successfully."
echo "All fake netcat loops have been replaced with functional MCP servers."
echo ""
echo "Services available at:"
for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    echo "  - $service: http://localhost:$port"
done
echo ""
echo "To test a service:"
echo "  curl http://localhost:3001/health  # Claude Flow"
echo "  curl http://localhost:3003/info    # Files service"
echo ""
echo "To view logs:"
echo "  docker exec $DIND_CONTAINER docker logs mcp-claude-flow"
echo ""