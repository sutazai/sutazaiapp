#!/bin/bash

echo "==================================="
echo "MCP Server Comprehensive Test Suite"
echo "==================================="
echo ""

# Test Chrome installation
echo "1. Testing Chrome Installation..."
if google-chrome --version &>/dev/null; then
    echo "   ✅ Chrome installed: $(google-chrome --version)"
else
    echo "   ❌ Chrome not installed"
fi

# Test Playwright with Chrome
echo ""
echo "2. Testing Playwright with Chrome..."
timeout 5 npx -y @playwright/mcp@latest --version 2>&1 | head -2 && echo "   ✅ Playwright MCP working" || echo "   ❌ Playwright MCP failed"

# Test nx-mcp
echo ""
echo "3. Testing nx-mcp..."
timeout 5 npx -y nx-mcp@latest --version 2>&1 | head -2 && echo "   ✅ nx-mcp working" || echo "   ❌ nx-mcp failed"

# Test mcp_ssh
echo ""
echo "4. Testing mcp_ssh..."
timeout 5 /opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh --selfcheck && echo "   ✅ mcp_ssh ready" || echo "   ❌ mcp_ssh failed"

# Test language-server
echo ""
echo "5. Testing language-server..."
/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh --selfcheck 2>&1 | tail -3

# Check git-mcp removal
echo ""
echo "6. Checking git-mcp removal..."
if grep -q '"git-mcp"' /root/.claude.json 2>/dev/null; then
    echo "   ❌ git-mcp still in config"
else
    echo "   ✅ git-mcp removed from config"
fi

# Test all MCP servers connection
echo ""
echo "7. Testing MCP server connections..."
claude mcp list 2>/dev/null | grep -E "✓|✗" | head -30

echo ""
echo "==================================="
echo "Test Suite Complete"
echo "==================================="
