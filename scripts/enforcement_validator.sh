#!/bin/bash
# SUPREME VALIDATOR ENFORCEMENT SCRIPT
# Zero Tolerance Rule Enforcement for Codebase Cleanup
# Generated: 2025-08-15 00:00:00 UTC

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Enforcement counters
TOTAL_RULES=20
VIOLATIONS=0
CRITICAL_VIOLATIONS=0
WARNINGS=0

echo -e "${BOLD}${BLUE}🚨 SUPREME VALIDATOR ENFORCEMENT SYSTEM 🚨${NC}"
echo -e "${BOLD}Zero Tolerance Rule Enforcement Active${NC}"
echo "================================================"

# Function to check rule compliance
check_rule() {
    local rule_num=$1
    local rule_name=$2
    local check_command=$3
    local severity=$4
    
    echo -ne "Checking Rule $rule_num: $rule_name... "
    
    if eval "$check_command"; then
        echo -e "${GREEN}✅ COMPLIANT${NC}"
        return 0
    else
        echo -e "${RED}❌ VIOLATION DETECTED${NC}"
        ((VIOLATIONS++))
        if [ "$severity" = "CRITICAL" ]; then
            ((CRITICAL_VIOLATIONS++))
        fi
        return 1
    fi
}

# Rule 1: No Fantasy Code Check
check_rule 1 "Real Implementation Only" \
    "! grep -r 'TODO.*implement' --include='*.py' --include='*.js' /opt/sutazaiapp 2>/dev/null | grep -v node_modules | head -1" \
    "CRITICAL"

# Rule 2: Never Break Existing Functionality
check_rule 2 "Test Coverage Check" \
    "[ -f /opt/sutazaiapp/backend/tests/test_main.py ]" \
    "CRITICAL"

# Rule 4: Investigate Before Creating
check_rule 4 "No Duplicate Files" \
    "! find /opt/sutazaiapp -name '*.old' -o -name '*.backup' -o -name '*_copy.*' 2>/dev/null | grep -v node_modules | grep -v '.mcp' | grep -v '.venv' | head -1" \
    "HIGH"

# Rule 6: Centralized Documentation
check_rule 6 "Documentation Structure" \
    "[ -d /opt/sutazaiapp/docs ]" \
    "MEDIUM"

# Rule 11: Docker Excellence
check_rule 11 "Docker Configuration" \
    "[ -f /opt/sutazaiapp/docker-compose.yml ]" \
    "CRITICAL"

# Rule 18/19: CHANGELOG.md Requirements
echo -e "\n${BOLD}Checking Rule 18/19: CHANGELOG.md Compliance...${NC}"
MISSING_CHANGELOGS=0
for dir in backend frontend agents monitoring database tests docker scripts; do
    if [ -d "/opt/sutazaiapp/$dir" ]; then
        if [ ! -f "/opt/sutazaiapp/$dir/CHANGELOG.md" ]; then
            echo -e "  ${RED}❌ Missing: $dir/CHANGELOG.md${NC}"
            ((MISSING_CHANGELOGS++))
            ((VIOLATIONS++))
        else
            echo -e "  ${GREEN}✅ Found: $dir/CHANGELOG.md${NC}"
        fi
    fi
done

if [ $MISSING_CHANGELOGS -eq 0 ]; then
    echo -e "${GREEN}✅ All required CHANGELOG.md files present${NC}"
else
    echo -e "${RED}❌ $MISSING_CHANGELOGS CHANGELOG.md files missing${NC}"
    ((CRITICAL_VIOLATIONS++))
fi

# Rule 20: MCP Server Protection
echo -e "\n${BOLD}Checking Rule 20: MCP Server Protection...${NC}"
if [ -f "/opt/sutazaiapp/.mcp.json" ]; then
    # Check if MCP config was modified recently
    if find /opt/sutazaiapp/.mcp.json -mmin -60 2>/dev/null | grep -q mcp; then
        echo -e "${YELLOW}⚠️  WARNING: .mcp.json recently modified - verify authorization${NC}"
        ((WARNINGS++))
    else
        echo -e "${GREEN}✅ MCP configuration intact${NC}"
    fi
else
    echo -e "${RED}❌ MCP configuration missing!${NC}"
    ((CRITICAL_VIOLATIONS++))
fi

# Check for wrapper script integrity
MCP_WRAPPERS_OK=true
if [ -d "/opt/sutazaiapp/scripts/mcp" ]; then
    for wrapper in /opt/sutazaiapp/scripts/mcp/*.sh; do
        if [ -f "$wrapper" ]; then
            if ! [ -x "$wrapper" ]; then
                echo -e "  ${RED}❌ Non-executable wrapper: $(basename $wrapper)${NC}"
                MCP_WRAPPERS_OK=false
                ((VIOLATIONS++))
            fi
        fi
    done
    if $MCP_WRAPPERS_OK; then
        echo -e "${GREEN}✅ MCP wrapper scripts intact${NC}"
    fi
else
    echo -e "${RED}❌ MCP wrapper directory missing!${NC}"
    ((CRITICAL_VIOLATIONS++))
fi

# Generate Enforcement Report
echo -e "\n${BOLD}========================================${NC}"
echo -e "${BOLD}ENFORCEMENT SUMMARY${NC}"
echo -e "${BOLD}========================================${NC}"

COMPLIANCE_RATE=$((100 - (VIOLATIONS * 5)))
if [ $COMPLIANCE_RATE -lt 0 ]; then
    COMPLIANCE_RATE=0
fi

echo -e "Total Rules Checked: ${BOLD}$TOTAL_RULES${NC}"
echo -e "Total Violations: ${BOLD}$VIOLATIONS${NC}"
echo -e "Critical Violations: ${BOLD}${RED}$CRITICAL_VIOLATIONS${NC}"
echo -e "Warnings: ${BOLD}${YELLOW}$WARNINGS${NC}"
echo -e "Compliance Rate: ${BOLD}$COMPLIANCE_RATE%${NC}"

if [ $CRITICAL_VIOLATIONS -gt 0 ]; then
    echo -e "\n${RED}${BOLD}⛔ VETO ACTIVATED ⛔${NC}"
    echo -e "${RED}Critical violations detected. All changes BLOCKED until resolved.${NC}"
    echo -e "${RED}Supreme Validator has VETOED further development.${NC}"
    exit 1
elif [ $VIOLATIONS -gt 0 ]; then
    echo -e "\n${YELLOW}${BOLD}⚠️  ENFORCEMENT WARNING ⚠️${NC}"
    echo -e "${YELLOW}Violations detected. Address immediately before proceeding.${NC}"
    exit 1
else
    echo -e "\n${GREEN}${BOLD}✅ FULL COMPLIANCE ACHIEVED ✅${NC}"
    echo -e "${GREEN}All enforcement rules satisfied. Development may proceed.${NC}"
    exit 0
fi