#!/usr/bin/env bash
set -Eeuo pipefail

echo "===== MCP Server Test Report ====="
echo "Date: $(date)"
echo

TOTAL=0
PASSING=0
FAILED=0

# Test each wrapper with selfcheck
for wrapper in /opt/sutazaiapp/scripts/mcp/wrappers/*.sh; do
    if [ ! -x "$wrapper" ]; then continue; fi
    
    name=$(basename "$wrapper" .sh)
    # Skip known non-MCP scripts
    if [[ "$name" == "_common" || "$name" == "unified-memory" ]]; then continue; fi
    
    TOTAL=$((TOTAL + 1))
    
    if timeout 10 "$wrapper" --selfcheck >/dev/null 2>&1; then
        echo "✅ $name: PASSED"
        PASSING=$((PASSING + 1))
    else
        echo "❌ $name: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "===== Summary ====="
echo "Total servers: $TOTAL"
echo "Passing: $PASSING"
echo "Failed: $FAILED"
echo "Success rate: $(( (PASSING * 100) / TOTAL ))%"

# Special checks
echo
echo "===== Special Checks ====="

# Check postgres connection
if docker exec sutazai-postgres pg_isready >/dev/null 2>&1; then
    echo "✅ PostgreSQL container: READY"
else
    echo "❌ PostgreSQL container: NOT READY"
fi

# Check TypeScript for language-server
if npm list typescript >/dev/null 2>&1; then
    echo "✅ TypeScript: INSTALLED"
else
    echo "❌ TypeScript: NOT INSTALLED"
fi

# Check for stale MCP containers
mcp_containers=$(docker ps --filter label=mcp-service -q | wc -l)
if [ "$mcp_containers" -eq 0 ]; then
    echo "✅ No stale MCP containers"
else
    echo "⚠️  Found $mcp_containers stale MCP containers"
fi

echo
echo "===== Recommendations ====="
if [ "$FAILED" -gt 0 ]; then
    echo "• Some servers are failing selfcheck"
    echo "• Run: /opt/sutazaiapp/scripts/mcp/selfcheck_all.sh for details"
fi

if [ "$mcp_containers" -gt 0 ]; then
    echo "• Clean up stale containers with:"
    echo "  docker ps --filter label=mcp-service -q | xargs -r docker stop"
    echo "  docker ps -a --filter label=mcp-service -q | xargs -r docker rm"
fi

echo
echo "Note: Claude may need to be restarted to pick up configuration changes."
echo "The postgres and task-runner issues in 'claude mcp list' will resolve after restart."