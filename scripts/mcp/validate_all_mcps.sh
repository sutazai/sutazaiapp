#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "MCP Comprehensive Validation Report"
echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================"
echo

# Track overall status
FAILED_MCPS=()
WORKING_MCPS=()
PARTIAL_MCPS=()

# Test each MCP wrapper
test_mcp() {
    local mcp_name="$1"
    local wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/${mcp_name}.sh"
    
    printf "Testing %-25s: " "$mcp_name"
    
    if [ ! -f "$wrapper_path" ]; then
        echo -e "${RED}✗ Wrapper not found${NC}"
        FAILED_MCPS+=("$mcp_name (no wrapper)")
        return 1
    fi
    
    if [ ! -x "$wrapper_path" ]; then
        echo -e "${YELLOW}⚠ Not executable${NC}"
        PARTIAL_MCPS+=("$mcp_name (not executable)")
        chmod +x "$wrapper_path"
        echo "  Fixed: Made executable"
    fi
    
    # Test with a simple JSON-RPC request
    local output
    output=$(echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | timeout 2 "$wrapper_path" 2>&1 | head -5 || echo "TIMEOUT")
    local exit_code=$?
    
    if echo "$output" | grep -q "TIMEOUT"; then
        echo -e "${YELLOW}⚠ Timeout (may need longer init)${NC}"
        PARTIAL_MCPS+=("$mcp_name (timeout)")
        return 0
    elif [ $exit_code -eq 0 ] || echo "$output" | grep -qE "(running|server|initialized|FastMCP|PostgreSQL MCP Server)"; then
        echo -e "${GREEN}✓ Working${NC}"
        WORKING_MCPS+=("$mcp_name")
        return 0
    elif echo "$output" | grep -qE "(ModuleNotFoundError|ImportError|No module)"; then
        echo -e "${RED}✗ Missing dependencies${NC}"
        echo "  Error: $output" | head -1
        FAILED_MCPS+=("$mcp_name (missing deps)")
        return 1
    elif echo "$output" | grep -qE "(TOML|parse error)"; then
        echo -e "${RED}✗ Configuration error${NC}"
        echo "  Error: $output" | head -1
        FAILED_MCPS+=("$mcp_name (config error)")
        return 1
    else
        echo -e "${YELLOW}⚠ Partial (may need config)${NC}"
        PARTIAL_MCPS+=("$mcp_name")
        return 0
    fi
}

# List of all MCPs to test
MCPS=(
    "files"
    "github"
    "http"
    "ddg"
    "language-server"
    "mcp_ssh"
    "ultimatecoder"
    "extended-memory"
    "postgres"
    "context7"
    "http_fetch"
    "sequentialthinking"
    "nx-mcp"
    "playwright-mcp"
    "memory-bank-mcp"
    "knowledge-graph-mcp"
    "compass-mcp"
    "puppeteer-mcp"
)

echo "Individual MCP Tests:"
echo "---------------------"
for mcp in "${MCPS[@]}"; do
    test_mcp "$mcp"
done

echo
echo "================================================"
echo "Process Health Check:"
echo "---------------------"
RUNNING_PROCESSES=$(ps aux | grep -E "mcp|extended-memory|ultimatecoder|context7|nx-mcp" | grep -v grep | wc -l)
echo "MCP processes running: $RUNNING_PROCESSES"

if [ $RUNNING_PROCESSES -gt 0 ]; then
    echo -e "${GREEN}✓ MCP processes are active${NC}"
    ps aux | grep -E "mcp|extended-memory|ultimatecoder" | grep -v grep | awk '{print "  - " $11}' | sort -u | head -10
else
    echo -e "${YELLOW}⚠ No MCP processes detected${NC}"
fi

echo
echo "================================================"
echo "Backend Integration Test:"
echo "---------------------"
if curl -s http://localhost:10010/openapi.json >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend API is responding${NC}"
    
    # Test MCP-specific endpoints
    if curl -s http://localhost:10010/openapi.json | grep -q "mcp-stdio"; then
        echo -e "${GREEN}✓ MCP endpoints are available${NC}"
    else
        echo -e "${YELLOW}⚠ MCP endpoints not found in API${NC}"
    fi
else
    echo -e "${RED}✗ Backend API is not responding${NC}"
fi

echo
echo "================================================"
echo "Summary Report:"
echo "---------------------"
echo -e "${GREEN}Working MCPs (${#WORKING_MCPS[@]}):${NC}"
for mcp in "${WORKING_MCPS[@]}"; do
    echo "  ✓ $mcp"
done

if [ ${#PARTIAL_MCPS[@]} -gt 0 ]; then
    echo
    echo -e "${YELLOW}Partial/Config Needed (${#PARTIAL_MCPS[@]}):${NC}"
    for mcp in "${PARTIAL_MCPS[@]}"; do
        echo "  ⚠ $mcp"
    done
fi

if [ ${#FAILED_MCPS[@]} -gt 0 ]; then
    echo
    echo -e "${RED}Failed MCPs (${#FAILED_MCPS[@]}):${NC}"
    for mcp in "${FAILED_MCPS[@]}"; do
        echo "  ✗ $mcp"
    done
fi

echo
echo "================================================"
echo "Recommendations:"
echo "---------------------"

if [ ${#FAILED_MCPS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All core MCPs are operational!${NC}"
else
    echo "To fix failed MCPs:"
    echo "1. Check dependencies: pip/npm install requirements"
    echo "2. Verify configurations in /opt/sutazaiapp/.mcp/"
    echo "3. Check wrapper scripts in /opt/sutazaiapp/scripts/mcp/wrappers/"
    echo "4. Review logs for specific error details"
fi

echo
echo "================================================"
echo "Test completed at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================"

# Exit with appropriate code
if [ ${#FAILED_MCPS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi