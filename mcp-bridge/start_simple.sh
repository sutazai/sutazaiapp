#!/bin/bash

# Simple MCP Bridge Startup
echo "Starting MCP Bridge (Simple Mode)..."

cd /opt/sutazaiapp/mcp-bridge

# Kill any existing process on port 11100
lsof -ti:11100 | xargs kill -9 2>/dev/null || true

# Set environment
export PYTHONUNBUFFERED=1
export LOG_LEVEL=INFO
export MCP_BRIDGE_PORT=11100

# Try with system Python first
echo "Attempting to start MCP Bridge..."

# Check if we have Python 3
if command -v python3 &> /dev/null; then
    echo "Using Python 3..."
    
    # Try to install minimal dependencies
    python3 -m pip install fastapi uvicorn 2>/dev/null || true
    
    # Start the simple server
    nohup python3 services/mcp_bridge_simple.py > logs/mcp_bridge.log 2>&1 &
    
    echo "MCP Bridge started with PID $!"
    echo "Waiting for server to start..."
    sleep 3
    
    # Check if it's running
    if curl -f http://localhost:11100/health 2>/dev/null; then
        echo "✅ MCP Bridge is running on port 11100"
        echo "Health check successful!"
    else
        echo "⚠️ MCP Bridge may be starting up, please check logs"
    fi
else
    echo "❌ Python 3 not found"
    exit 1
fi