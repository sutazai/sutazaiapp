#!/usr/bin/env bash
# Comprehensive MCP Server Verification Script
# Created: 2025-08-26 UTC

set -Eeuo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}        MCP Server Comprehensive Verification${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"

# Test selfcheck for all wrappers
echo -e "${YELLOW}Running selfcheck for all MCP servers...${NC}\n"
/opt/sutazaiapp/scripts/mcp/selfcheck_all.sh 2>&1 | tail -20

# Check Claude MCP list
echo -e "\n${YELLOW}Checking Claude MCP connections...${NC}\n"
claude mcp list 2>&1 | grep -E "(✓|✗)"

# Count statistics
TOTAL=$(claude mcp list 2>&1 | grep -E "(✓|✗)" | wc -l)
CONNECTED=$(claude mcp list 2>&1 | grep "✓ Connected" | wc -l)
FAILED=$(claude mcp list 2>&1 | grep "✗ Failed" | wc -l)

# Calculate percentage
if [ "$TOTAL" -gt 0 ]; then
    PERCENTAGE=$((CONNECTED * 100 / TOTAL))
else
    PERCENTAGE=0
fi

echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}                    Final Statistics${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

echo -e "Total MCP Servers:    ${BLUE}$TOTAL${NC}"
echo -e "Connected:            ${GREEN}$CONNECTED${NC}"
echo -e "Failed:               ${RED}$FAILED${NC}"
echo -e "Success Rate:         ${GREEN}${PERCENTAGE}%${NC}"

if [ "$PERCENTAGE" -ge 90 ]; then
    echo -e "\n${GREEN}✓ EXCELLENT: MCP servers are functioning optimally!${NC}"
elif [ "$PERCENTAGE" -ge 75 ]; then
    echo -e "\n${YELLOW}⚠ GOOD: Most MCP servers are working${NC}"
else
    echo -e "\n${RED}✗ NEEDS ATTENTION: Several MCP servers are failing${NC}"
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"

# Show failed servers if any
if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}Failed servers:${NC}"
    claude mcp list 2>&1 | grep "✗ Failed" | awk -F: '{print "  - " $1}'
fi

echo -e "\n${GREEN}✓ Verification complete!${NC}"
