#!/bin/bash

# Comprehensive MCP Server Test
# Date: 2025-08-27
# Tests all MCP servers individually

echo "======================================"
echo "MCP COMPREHENSIVE TEST SUITE"
echo "======================================"
echo "Testing all MCP servers..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results
declare -A results
total=0
passed=0
failed=0

# List of all MCP servers to test
servers=(
    "ddg"
    "http_fetch"
    "http"
    "sequentialthinking"
    "language-server"
    "files"
    "context7"
    "extended-memory"
    "ultimatecoder"
    "github"
    "git-mcp"
    "claude-task-runner-v2"
    "nx-mcp-official"
    "playwright-mcp-official"
    "memory-bank-mcp"
    "knowledge-graph-mcp"
    "compass-mcp"
    "mcp_ssh"
    "claude-flow"
    "ruv-swarm"
)

echo "Testing individual MCP servers..."
echo "=================================="

for server in "${servers[@]}"; do
    ((total++))
    
    # Special handling for npm-based tools
    if [[ "$server" == "claude-flow" ]]; then
        if timeout 2 npx claude-flow@alpha --version &>/dev/null; then
            echo -e "${GREEN}✓${NC} $server - NPM package working"
            results["$server"]="PASS"
            ((passed++))
        else
            echo -e "${RED}✗${NC} $server - NPM package failed"
            results["$server"]="FAIL"
            ((failed++))
        fi
    elif [[ "$server" == "ruv-swarm" ]]; then
        if timeout 2 npx ruv-swarm@latest --version &>/dev/null; then
            echo -e "${GREEN}✓${NC} $server - NPM package working"
            results["$server"]="PASS"
            ((passed++))
        else
            echo -e "${RED}✗${NC} $server - NPM package failed"
            results["$server"]="FAIL"
            ((failed++))
        fi
    else
        # Test wrapper-based servers
        wrapper="/opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh"
        if [ -f "$wrapper" ]; then
            if timeout 2 "$wrapper" --selfcheck &>/dev/null; then
                output=$("$wrapper" --selfcheck 2>/dev/null || echo "{}")
                if echo "$output" | grep -q '"healthy":true'; then
                    service=$(echo "$output" | grep -oP '"service":"[^"]*"' | cut -d'"' -f4)
                    echo -e "${GREEN}✓${NC} $server - Service: ${service:-unknown}"
                    results["$server"]="PASS"
                    ((passed++))
                else
                    echo -e "${YELLOW}⚠${NC} $server - Wrapper exists but unhealthy"
                    results["$server"]="WARN"
                    ((failed++))
                fi
            else
                echo -e "${RED}✗${NC} $server - Wrapper failed"
                results["$server"]="FAIL"
                ((failed++))
            fi
        else
            echo -e "${RED}✗${NC} $server - No wrapper found"
            results["$server"]="MISSING"
            ((failed++))
        fi
    fi
done

echo ""
echo "======================================"
echo "TEST SUMMARY"
echo "======================================"
echo -e "Total Tests: ${total}"
echo -e "Passed: ${GREEN}${passed}${NC}"
echo -e "Failed: ${RED}${failed}${NC}"
echo -e "Success Rate: $((passed * 100 / total))%"
echo ""

# Show failed servers for debugging
if [ $failed -gt 0 ]; then
    echo "Failed servers:"
    for server in "${!results[@]}"; do
        if [[ "${results[$server]}" == "FAIL" ]] || [[ "${results[$server]}" == "MISSING" ]]; then
            echo "  - $server"
        fi
    done
    echo ""
fi

# Final status
if [ $passed -eq $total ]; then
    echo -e "${GREEN}✅ All MCP servers are working!${NC}"
    exit 0
elif [ $passed -ge $((total * 80 / 100)) ]; then
    echo -e "${YELLOW}⚠️  MCP integration is mostly functional (>80%)${NC}"
    exit 0
else
    echo -e "${RED}❌ MCP integration needs attention (<80%)${NC}"
    exit 1
fi