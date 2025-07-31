#!/bin/bash
# Start MCP servers for SutazAI

# Start SutazAI MCP server
echo "Starting SutazAI MCP server..."
cd /opt/sutazaiapp/mcp_server
nohup node index.js > /opt/sutazaiapp/logs/mcp-server.log 2>&1 &
echo $! > /tmp/sutazai-mcp-server.pid

# Start Task Master MCP (if needed)
echo "Task Master MCP is configured for on-demand startup via npx"

echo "MCP servers started. Check logs at /opt/sutazaiapp/logs/mcp-server.log"
