#!/bin/bash

# FastAPI MCP Bridge Startup
echo "Starting MCP Bridge with FastAPI..."

cd /opt/sutazaiapp/mcp-bridge

# Kill any existing process on port 11100
lsof -ti:11100 | xargs kill -9 2>/dev/null || true

# Set environment
export PYTHONUNBUFFERED=1
export LOG_LEVEL=INFO
export MCP_BRIDGE_PORT=11100

# Activate virtual environment
source venv/bin/activate

# Start the server in background
echo "Starting MCP Bridge Server..."
nohup python services/mcp_bridge_simple.py > logs/mcp_bridge_fastapi.log 2>&1 &

echo "MCP Bridge started with PID $!"
echo "Waiting for server to start..."
sleep 5

# Check if it's running
if curl -f http://localhost:11100/health 2>/dev/null; then
    echo "✅ MCP Bridge is running on port 11100"
    
    # Show available endpoints
    echo ""
    echo "Available endpoints:"
    echo "  - http://localhost:11100/         (Root)"
    echo "  - http://localhost:11100/health   (Health check)"
    echo "  - http://localhost:11100/status   (Status)"
    echo "  - http://localhost:11100/api/services (Service registry)"
    echo "  - http://localhost:11100/api/agents   (Agent registry)"
    echo ""
    
    # Test all endpoints
    echo "Testing endpoints:"
    curl -s http://localhost:11100/ | jq -c .
    curl -s http://localhost:11100/api/services | jq -c .
else
    echo "⚠️ MCP Bridge may be starting up, check logs at logs/mcp_bridge_fastapi.log"
fi