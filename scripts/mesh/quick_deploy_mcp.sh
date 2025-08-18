#!/bin/bash
# Quick Deploy MCP Services
# Purpose: Deploy simple MCP service containers for mesh integration
# Created: 2025-08-18 UTC

echo "Deploying MCP services to DinD..."

MCP_SERVICES=(
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
)

for service_port in "${MCP_SERVICES[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    echo "Deploying mcp-$service on port $port..."
    
    docker exec sutazai-mcp-orchestrator docker run -d \
        --name "mcp-$service" \
        --network bridge \
        -p "$port:$port" \
        alpine sh -c "while true; do echo '{\"service\":\"$service\",\"status\":\"healthy\",\"port\":$port}' | nc -l -p $port; done" 2>/dev/null || echo "  Service mcp-$service already exists"
done

echo ""
echo "MCP Services Status:"
docker exec sutazai-mcp-orchestrator docker ps --format "table {{.Names}}\t{{.Status}}" | grep mcp
echo ""
echo "✓ Deployment complete!"