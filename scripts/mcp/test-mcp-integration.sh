#!/bin/bash

# MCP Integration Test Suite
# Tests all MCP servers and integration methods

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WRAPPER_DIR="$SCRIPT_DIR/wrappers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}     MCP Integration Test Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo "Date: $(date)"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_SERVERS=()

# Function to test MCP server
test_mcp_server() {
    local server_name=$1
    local wrapper_path=$2
    local test_command=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing $server_name... "
    
    if [ -f "$wrapper_path" ]; then
        if timeout 5 $wrapper_path --selfcheck > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo -e "${RED}❌ FAILED${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_SERVERS+=("$server_name")
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  MISSING WRAPPER${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_SERVERS+=("$server_name (missing wrapper)")
        return 1
    fi
}

# Test direct MCP tool usage
test_direct_mcp() {
    echo -e "\n${BLUE}Testing Direct MCP Tool Usage:${NC}"
    
    # Test extended-memory
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing extended-memory save... "
    if timeout 10 $WRAPPER_DIR/extended-memory.sh --selfcheck > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    # Test code-index
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing code-index search... "
    if timeout 10 $WRAPPER_DIR/code-index-mcp.sh --selfcheck > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Test NPX CLI commands
test_npx_cli() {
    echo -e "\n${BLUE}Testing NPX CLI Commands:${NC}"
    
    # Test claude-flow version
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing claude-flow version... "
    if timeout 10 npx claude-flow@alpha --version > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    # Test MCP status
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing MCP status command... "
    if timeout 10 npx claude-flow@alpha mcp status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    # Test memory operations
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing memory list command... "
    if timeout 10 npx claude-flow@alpha memory list > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Test shell wrappers
test_shell_wrappers() {
    echo -e "\n${BLUE}Testing Shell Wrapper Scripts:${NC}"
    
    # Core MCP servers
    test_mcp_server "claude-flow" "$WRAPPER_DIR/claude-flow.sh"
    test_mcp_server "github" "$WRAPPER_DIR/github.sh"
    test_mcp_server "sequential-thinking" "$WRAPPER_DIR/sequentialthinking.sh"
    test_mcp_server "context7" "$WRAPPER_DIR/context7.sh"
    test_mcp_server "code-index" "$WRAPPER_DIR/code-index-mcp.sh"
    test_mcp_server "ultimatecoder" "$WRAPPER_DIR/ultimatecoder.sh"
    test_mcp_server "extended-memory" "$WRAPPER_DIR/extended-memory.sh"
    test_mcp_server "files" "$WRAPPER_DIR/files.sh"
    test_mcp_server "ddg" "$WRAPPER_DIR/ddg.sh"
    test_mcp_server "http" "$WRAPPER_DIR/http.sh"
    test_mcp_server "http_fetch" "$WRAPPER_DIR/http_fetch.sh"
    test_mcp_server "playwright" "$WRAPPER_DIR/playwright-mcp-official.sh"
    test_mcp_server "git-mcp" "$WRAPPER_DIR/git-mcp.sh"
    test_mcp_server "task-runner" "$WRAPPER_DIR/claude-task-runner-v2.sh"
}

# Test database connections
test_database_connections() {
    echo -e "\n${BLUE}Testing Database Connections:${NC}"
    
    # PostgreSQL container test removed - using standard PostgreSQL, not MCP
    
    # Test Redis
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing Redis connection... "
    if docker exec sutazai-redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Test API endpoints
test_api_endpoints() {
    echo -e "\n${BLUE}Testing API Endpoints:${NC}"
    
    # Test Backend Health
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing Backend API health... "
    if curl -s http://localhost:10010/health | grep -q "healthy"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Test SPARC modes
test_sparc_modes() {
    echo -e "\n${BLUE}Testing SPARC Modes:${NC}"
    
    local modes=("specification" "architecture" "coding" "testing" "mcp")
    
    for mode in "${modes[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo -n "Testing SPARC mode: $mode... "
        
        # Test if mode help is accessible
        if timeout 10 npx claude-flow@alpha sparc run $mode --help > /dev/null 2>&1; then
            echo -e "${GREEN}✅ AVAILABLE${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${YELLOW}⚠️  UNTESTED${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    done
}

# Run all tests
echo -e "${BLUE}Starting MCP Integration Tests...${NC}\n"

test_shell_wrappers
test_direct_mcp
test_npx_cli
test_database_connections
test_api_endpoints
test_sparc_modes

# Generate summary report
echo -e "\n${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}                TEST SUMMARY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Success Rate: $SUCCESS_RATE%"

if [ ${#FAILED_SERVERS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Failed Servers:${NC}"
    for server in "${FAILED_SERVERS[@]}"; do
        echo "  • $server"
    done
fi

# Recommendations
echo -e "\n${BLUE}Recommendations:${NC}"

if [ $SUCCESS_RATE -lt 80 ]; then
    echo "⚠️  Success rate below 80%. Critical issues detected."
    echo "   Run: /opt/sutazaiapp/scripts/mcp/fix-all-mcp-servers.sh"
elif [ $SUCCESS_RATE -lt 95 ]; then
    echo "⚠️  Some MCP servers need attention."
    echo "   Check failed servers and run individual fixes."
else
    echo "✅ MCP integration is healthy!"
fi

if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "\n${YELLOW}To debug failed tests:${NC}"
    echo "1. Check logs: journalctl -xe"
    echo "2. Test individual wrappers: <wrapper>.sh --selfcheck"
    echo "3. Verify Docker containers: docker ps"
fi

echo -e "\n${BLUE}═══════════════════════════════════════════════${NC}"
echo "Test completed at: $(date)"

exit $FAILED_TESTS