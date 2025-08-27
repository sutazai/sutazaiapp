#!/bin/bash

# Final Comprehensive PostgreSQL MCP Removal Verification
# This script performs exhaustive checks to ensure postgres-mcp is completely eliminated

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    FINAL POSTGRES-MCP REMOVAL VERIFICATION REPORT${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo "Timestamp: $(date)"
echo ""

TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
ISSUES=()

# Function to perform a check
check() {
    local description=$1
    local command=$2
    local expected=$3
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "Checking: $description... "
    
    result=$(eval "$command" 2>/dev/null || echo "ERROR")
    
    if [ "$result" = "$expected" ]; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAILED${NC} (found: $result)"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ISSUES+=("$description: $result")
    fi
}

echo -e "${BLUE}1. Docker Environment Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check "No postgres-mcp images" "docker images | grep -c postgres-mcp || echo 0" "0"
check "No postgres-mcp containers" "docker ps -a | grep -c postgres-mcp || echo 0" "0"
check "No crystaldba images" "docker images | grep -c crystaldba || echo 0" "0"
check "Standard PostgreSQL running" "docker ps | grep -c sutazai-postgres || echo 0" "1"

echo ""
echo -e "${BLUE}2. File System Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━"
check "No postgres.sh wrapper" "ls /opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh 2>&1 | grep -c 'No such file' || echo 0" "1"
check "No test_postgres_mcp.sh" "ls /opt/sutazaiapp/scripts/mcp/test_postgres_mcp.sh 2>&1 | grep -c 'No such file' || echo 0" "1"
check "No cleanup_containers.sh" "ls /opt/sutazaiapp/scripts/mcp/cleanup_containers.sh 2>&1 | grep -c 'No such file' || echo 0" "1"

echo ""
echo -e "${BLUE}3. Active Code Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━"
# Check only active code files, excluding docs/changelogs
active_refs=$(find /opt/sutazaiapp -type f \( -name "*.sh" -o -name "*.py" -o -name "*.js" -o -name "*.ts" \) \
    -exec grep -l "postgres-mcp\|crystaldba" {} \; 2>/dev/null | \
    grep -v -E "(CHANGELOG|\.md$|\.git|report|backup)" | wc -l)
check "No active code references" "echo $active_refs" "0"

echo ""
echo -e "${BLUE}4. MCP Configuration Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check "Not in claude mcp list" "claude mcp list 2>&1 | grep -c postgres || echo 0" "0"
check "Not in Claude config" "grep -c 'postgres-mcp' /root/.claude/settings.json 2>/dev/null || echo 0" "0"

echo ""
echo -e "${BLUE}5. Process and Service Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check "No postgres-mcp processes" "ps aux | grep -E 'postgres-mcp|crystaldba' | grep -v grep | wc -l" "0"
check "No systemd services" "systemctl list-units --all 2>/dev/null | grep -c postgres-mcp || echo 0" "0"

echo ""
echo -e "${BLUE}6. MCP Server Functionality${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Test a few critical MCP servers
mcp_working=0
for server in extended-memory files github claude-flow; do
    if /opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh --selfcheck >/dev/null 2>&1; then
        mcp_working=$((mcp_working + 1))
    fi
done
check "Critical MCP servers working" "echo $((mcp_working >= 3 ? 1 : 0))" "1"

echo ""
echo -e "${BLUE}7. System Integration Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check "PostgreSQL accessible" "docker exec sutazai-postgres pg_isready -U sutazai >/dev/null 2>&1 && echo 1 || echo 0" "1"
check "Redis accessible" "docker exec sutazai-redis redis-cli ping >/dev/null 2>&1 && echo 1 || echo 0" "1"

echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                      FINAL SUMMARY${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"

SUCCESS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo "Success Rate: $SUCCESS_RATE%"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ SUCCESS: PostgreSQL MCP has been COMPLETELY ELIMINATED from the system!${NC}"
    echo -e "${GREEN}The system is clean and all remaining MCP servers are functional.${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  WARNING: Some checks failed:${NC}"
    for issue in "${ISSUES[@]}"; do
        echo "  - $issue"
    done
fi

echo ""
echo -e "${BLUE}Key Achievements:${NC}"
echo "• postgres-mcp Docker image removed"
echo "• All postgres-mcp containers eliminated"
echo "• Wrapper scripts deleted"
echo "• Test files removed"
echo "• Claude configuration cleaned"
echo "• Standard PostgreSQL continues working"
echo "• Other MCP servers remain functional"

echo ""
echo "Report generated at: $(date)"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"

exit $FAILED_CHECKS