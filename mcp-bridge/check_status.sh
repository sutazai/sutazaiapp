#!/bin/bash

# MCP Bridge Status Check Script

echo "================================================"
echo "MCP Bridge Status Check"
echo "================================================"
echo ""

# Check if service is running on port 11100
if lsof -i:11100 >/dev/null 2>&1; then
    echo "✅ MCP Bridge is running on port 11100"
    
    # Get process info
    echo ""
    echo "Process Info:"
    ps aux | grep -E "mcp_bridge|11100" | grep -v grep | head -2
    
    # Check health endpoint
    echo ""
    echo "Health Check:"
    HEALTH=$(curl -s http://localhost:11100/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$HEALTH" | jq . 2>/dev/null || echo "$HEALTH"
    else
        echo "⚠️ Health endpoint not responding"
    fi
    
    # Check status endpoint
    echo ""
    echo "Service Status:"
    STATUS=$(curl -s http://localhost:11100/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$STATUS" | jq '.services | keys' 2>/dev/null || echo "Services available"
    fi
    
    # Check API endpoints
    echo ""
    echo "API Endpoints:"
    echo "  ✓ Root: http://localhost:11100/"
    echo "  ✓ Health: http://localhost:11100/health"
    echo "  ✓ Status: http://localhost:11100/status"
    echo "  ✓ Services: http://localhost:11100/api/services"
    echo "  ✓ Agents: http://localhost:11100/api/agents"
    
else
    echo "❌ MCP Bridge is NOT running on port 11100"
    echo ""
    echo "To start the MCP Bridge, run one of:"
    echo "  1. ./start_fastapi.sh      (Recommended - with FastAPI)"
    echo "  2. ./start_simple.sh       (Basic HTTP server)"
    echo "  3. docker-compose -f docker-compose-standalone.yml up -d"
fi

echo ""
echo "================================================"