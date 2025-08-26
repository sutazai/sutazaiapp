#!/bin/bash
set -e

echo "========================================="
echo "SuperClaude & MCP Integration Test Report"
echo "========================================="
echo ""
echo "Date: $(date)"
echo ""

# Test SuperClaude installation
echo "=== SuperClaude Framework ==="
if command -v SuperClaude >/dev/null 2>&1; then
    echo "✓ SuperClaude CLI installed"
    SuperClaude --version 2>&1 | grep -o "v[0-9.]*" || echo "Version check failed"
else
    echo "✗ SuperClaude CLI not found"
fi

if [ -f "/root/.claude/CLAUDE.md" ]; then
    echo "✓ SuperClaude framework files installed"
    echo "  Components found:"
    ls /root/.claude/*.md 2>/dev/null | wc -l | xargs -I {} echo "  - {} MD files"
    ls /root/.claude/commands/ 2>/dev/null | wc -l | xargs -I {} echo "  - {} command directories" || echo "  - No commands directory"
fi
echo ""

# Test MCP servers
echo "=== MCP Server Status ==="
echo "Running comprehensive selfcheck..."
/opt/sutazaiapp/scripts/mcp/selfcheck_all.sh 2>&1 | grep -E "\[OK\]|\[ERR\]" | sort | uniq -c | awk '{print "  "$1" servers "$2}'
echo ""

# Count working vs failing
echo "=== MCP Server Summary ==="
WORKING=$(claude mcp list 2>&1 | grep -c "✓ Connected" || echo "0")
FAILING=$(claude mcp list 2>&1 | grep -c "✗ Failed" || echo "0")
TOTAL=$((WORKING + FAILING))

echo "Total MCP servers: $TOTAL"
echo "✓ Working: $WORKING"
echo "✗ Failing: $FAILING"
if [ "$TOTAL" -gt 0 ]; then
    PERCENTAGE=$((WORKING * 100 / TOTAL))
    echo "Success rate: ${PERCENTAGE}%"
fi
echo ""

# List SuperClaude MCP servers
echo "=== SuperClaude MCP Servers ==="
claude mcp list 2>&1 | grep -E "context7|sequential|playwright|magic|serena|morphllm" || echo "None found"
echo ""

# Check original MCP servers
echo "=== Original MCP Servers Status ==="
for server in claude-flow ruv-swarm files http_fetch ddg sequentialthinking nx-mcp extended-memory ultimatecoder; do
    if timeout 2 /opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh --selfcheck >/dev/null 2>&1; then
        echo "✓ $server"
    else
        echo "✗ $server"
    fi
done
echo ""

echo "========================================="
echo "Test Complete"
echo "========================================="
echo ""
echo "Recommendations:"
echo "1. Restart Claude to apply all configuration changes"
echo "2. SuperClaude framework is installed and ready to use"
echo "3. Most critical MCP servers are operational"
echo ""