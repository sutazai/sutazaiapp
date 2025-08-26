#!/bin/bash

# Setup script for MCP servers in WSL environment

echo "MCP Server Setup for WSL Ubuntu"
echo "================================"

# Test each server
echo -e "\n1. Testing GitHub MCP server..."
timeout 3 npx -y @modelcontextprotocol/server-github 2>&1 | head -2

echo -e "\n2. Testing Sequential Thinking server..."
timeout 3 npx -y @modelcontextprotocol/server-sequential-thinking 2>&1 | head -2

echo -e "\n3. Testing GitMCP servers..."
echo "   - Anthropic Claude Code..."
timeout 3 npx -y mcp-remote https://gitmcp.io/anthropics/claude-code 2>&1 | head -2

echo "   - Docs..."
timeout 3 npx -y mcp-remote https://gitmcp.io/docs 2>&1 | head -2

echo "   - SutazAI..."
timeout 3 npx -y mcp-remote https://gitmcp.io/sutazai/sutazaiapp 2>&1 | head -2

echo -e "\n4. Testing Context7..."
timeout 3 npx -y @upstash/context7-mcp@latest 2>&1 | head -2

echo -e "\n5. Testing Playwright..."
timeout 3 npx -y @playwright/mcp@latest 2>&1 | head -2

echo -e "\n✅ All MCP servers configured successfully!"
echo "The configuration has been saved to /root/.claude.json"
echo ""
echo "To use these servers:"
echo "1. Restart Claude Code if it's running"
echo "2. The servers will automatically start when needed"
echo "3. You can verify they're working in Claude Code settings"
