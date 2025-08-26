#!/bin/bash

echo "========================================="
echo "GitMCP Installation Verification"
echo "========================================="
echo ""

SUCCESS=true

# Test 1: Check if git-mcp is in Claude's MCP list
echo "1. Checking Claude MCP configuration..."
if claude mcp get git-mcp 2>/dev/null | grep -q "npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp"; then
    echo "   ✅ git-mcp is configured in Claude"
else
    echo "   ❌ git-mcp is NOT configured in Claude"
    SUCCESS=false
fi

# Test 2: Test direct connection
echo ""
echo "2. Testing direct connection to GitMCP..."
TEMP_LOG="/tmp/git-mcp-test-$$.log"
timeout 10 npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp &> "$TEMP_LOG" &
TEST_PID=$!
sleep 5

if grep -q "Proxy established successfully" "$TEMP_LOG" 2>/dev/null; then
    echo "   ✅ Direct connection successful"
else
    echo "   ❌ Direct connection failed"
    SUCCESS=false
fi
kill $TEST_PID 2>/dev/null
rm -f "$TEMP_LOG"

# Test 3: Check configuration files
echo ""
echo "3. Checking configuration files..."
ALL_FILES_EXIST=true
for file in \
    "/opt/sutazaiapp/.mcp/git-mcp-config.json" \
    "/opt/sutazaiapp/.mcp/git-mcp-service.json" \
    "/opt/sutazaiapp/scripts/mcp/servers/git-mcp/package.json" \
    "/opt/sutazaiapp/scripts/mcp/wrappers/git-mcp.sh"
do
    if [ -f "$file" ]; then
        echo "   ✅ $(basename $file) exists"
    else
        echo "   ❌ $(basename $file) missing"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    SUCCESS=false
fi

# Test 4: Check repository URL
echo ""
echo "4. Verifying repository URL..."
EXPECTED_URL="https://gitmcp.io/sutazai/sutazaiapp"
if claude mcp get git-mcp 2>/dev/null | grep -q "$EXPECTED_URL"; then
    echo "   ✅ Correct URL: $EXPECTED_URL"
else
    echo "   ❌ Incorrect URL configuration"
    SUCCESS=false
fi

# Test 5: Check if mcp-remote is available
echo ""
echo "5. Checking mcp-remote availability..."
if which npx >/dev/null && npm list -g mcp-remote 2>/dev/null | grep -q "mcp-remote"; then
    echo "   ✅ mcp-remote is installed globally"
else
    echo "   ❌ mcp-remote is not properly installed"
    SUCCESS=false
fi

# Summary
echo ""
echo "========================================="
if [ "$SUCCESS" = true ]; then
    echo "✅ GitMCP Installation VERIFIED"
    echo ""
    echo "The git-mcp server is properly installed and configured."
    echo "Note: The server may show as 'Failed to connect' in 'claude mcp list'"
    echo "due to startup time, but it will work when actually used."
    echo ""
    echo "To use GitMCP:"
    echo "  - It will automatically connect when you start a new Claude session"
    echo "  - Manual test: npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp"
    echo ""
    echo "The server provides access to:"
    echo "  - SutazAI repository documentation"
    echo "  - Source code with intelligent search"
    echo "  - Real-time updates from GitHub"
else
    echo "⚠️  GitMCP Installation INCOMPLETE"
    echo ""
    echo "Please review the issues above and run:"
    echo "  bash /opt/sutazaiapp/scripts/mcp/integrate-git-mcp.sh"
fi
echo "========================================="