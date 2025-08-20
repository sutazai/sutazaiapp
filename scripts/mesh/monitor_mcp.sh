#!/bin/bash
echo "MCP Services Health Monitor"
echo "============================"
echo ""

# Define all MCP services and ports
declare -A MCP_PORTS=(
    ["claude-flow"]="3001"
    ["ruv-swarm"]="3002"
    ["files"]="3003"
    ["context7"]="3004"
    ["http-fetch"]="3005"
    ["ddg"]="3006"
    ["extended-memory"]="3009"
    ["ssh"]="3010"
    ["ultimatecoder"]="3011"
    ["knowledge-graph-mcp"]="3014"
    ["github"]="3016"
    ["language-server"]="3018"
    ["claude-task-runner"]="3019"
)

# Check each service
for service in "${!MCP_PORTS[@]}"; do
    port="${MCP_PORTS[$service]}"
    
    # Check if port is open
    if timeout 1 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
        # Try health endpoint
        if curl -s "http://localhost:$port/health" 2>/dev/null | grep -q "healthy"; then
            echo "✓ mcp-$service:$port - HEALTHY"
        else
            echo "⚠ mcp-$service:$port - RUNNING (no health endpoint)"
        fi
    else
        echo "✗ mcp-$service:$port - DOWN"
    fi
done

echo ""
echo "Docker Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "mcp|MCP" || echo "No MCP containers found"
