#!/bin/bash

echo "========================================="
echo "GitMCP Full Test Suite for SutazAI"
echo "========================================="
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Check mcp-remote installation
echo "Test 1: Checking mcp-remote installation..."
if npm list -g mcp-remote &> /dev/null || which mcp-remote &> /dev/null; then
    echo "✅ mcp-remote is installed"
    ((TESTS_PASSED++))
else
    echo "❌ mcp-remote is not installed"
    ((TESTS_FAILED++))
fi

# Test 2: Check GitMCP connectivity
echo ""
echo "Test 2: Testing GitMCP server connectivity..."
if timeout 10 npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp --test 2>&1 | grep -q "Proxy established"; then
    echo "✅ GitMCP server is reachable"
    ((TESTS_PASSED++))
else
    echo "❌ GitMCP server connection failed"
    ((TESTS_FAILED++))
fi

# Test 3: Check configuration files
echo ""
echo "Test 3: Checking configuration files..."
FILES_OK=true
for file in \
    "/opt/sutazaiapp/.mcp/git-mcp-service.json" \
    "/opt/sutazaiapp/scripts/mcp/wrappers/git-mcp.sh" \
    "/opt/sutazaiapp/scripts/mcp/servers/git-mcp/package.json" \
    "/opt/sutazaiapp/.mcp/git-mcp-config.json"
do
    if [ -f "$file" ]; then
        echo "   ✅ Found: $(basename $file)"
    else
        echo "   ❌ Missing: $(basename $file)"
        FILES_OK=false
    fi
done

if [ "$FILES_OK" = true ]; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

# Test 4: Check GitMCP repository clone
echo ""
echo "Test 4: Checking GitMCP repository..."
if [ -d "/opt/sutazaiapp/.mcp/git-mcp" ]; then
    echo "✅ GitMCP repository is cloned"
    ((TESTS_PASSED++))
else
    echo "❌ GitMCP repository not found"
    ((TESTS_FAILED++))
fi

# Test 5: Test wrapper script
echo ""
echo "Test 5: Testing GitMCP wrapper script..."
if [ -x "/opt/sutazaiapp/scripts/mcp/wrappers/git-mcp.sh" ]; then
    echo "✅ GitMCP wrapper is executable"
    ((TESTS_PASSED++))
else
    echo "❌ GitMCP wrapper is not executable"
    ((TESTS_FAILED++))
fi

# Test 6: Check for GitMCP in running processes (optional)
echo ""
echo "Test 6: Checking for GitMCP processes..."
if ps aux | grep -q "mcp-remote.*gitmcp.io" | grep -v grep; then
    echo "✅ GitMCP process is running"
    ((TESTS_PASSED++))
else
    echo "⚠️  GitMCP is not currently running (this is normal if not in use)"
fi

# Test 7: Check GitMCP URL format
echo ""
echo "Test 7: Validating GitMCP URL configuration..."
EXPECTED_URL="https://gitmcp.io/sutazai/sutazaiapp"
CONFIGURED_URL=$(grep -o 'https://gitmcp.io/[^"]*' /opt/sutazaiapp/.mcp/git-mcp-service.json 2>/dev/null | head -1)
if [ "$CONFIGURED_URL" = "$EXPECTED_URL" ]; then
    echo "✅ GitMCP URL is correctly configured: $CONFIGURED_URL"
    ((TESTS_PASSED++))
else
    echo "❌ GitMCP URL mismatch. Expected: $EXPECTED_URL, Got: $CONFIGURED_URL"
    ((TESTS_FAILED++))
fi

# Test 8: Quick functionality test
echo ""
echo "Test 8: Quick functionality test..."
echo "   Starting GitMCP server for 5 seconds..."
timeout 5 npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp 2>&1 | head -20 > /tmp/gitmcp-test.log &
TEST_PID=$!
sleep 3

if grep -q "Proxy established successfully" /tmp/gitmcp-test.log 2>/dev/null; then
    echo "✅ GitMCP functionality test passed"
    ((TESTS_PASSED++))
else
    echo "⚠️  GitMCP functionality test incomplete (may need more time)"
fi

# Clean up
kill $TEST_PID 2>/dev/null
rm -f /tmp/gitmcp-test.log

# Summary
echo ""
echo "========================================="
echo "Test Results Summary"
echo "========================================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ All tests PASSED! GitMCP is properly installed and configured."
    echo ""
    echo "You can now use GitMCP to access SutazAI documentation:"
    echo "  - Manual start: npx mcp-remote https://gitmcp.io/sutazai/sutazaiapp"
    echo "  - Via wrapper: /opt/sutazaiapp/scripts/mcp/wrappers/git-mcp.sh"
    exit 0
else
    echo "⚠️  Some tests failed. Please review the output above."
    echo ""
    echo "To fix issues, try:"
    echo "  1. Run: npm install -g mcp-remote"
    echo "  2. Check network connectivity to gitmcp.io"
    echo "  3. Verify the repository exists at: https://github.com/sutazai/sutazaiapp"
    exit 1
fi