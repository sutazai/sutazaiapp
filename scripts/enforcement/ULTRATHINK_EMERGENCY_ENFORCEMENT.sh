#!/bin/bash
# ULTRATHINK EMERGENCY ENFORCEMENT SCRIPT
# Generated: 2025-08-19 10:31:00 UTC
# Purpose: Fix CRITICAL rule violations identified in audit
# Authority: Rules 1, 4, 11 - ZERO TOLERANCE

set -e  # Exit on any error

echo "================================================"
echo "🚨 ULTRATHINK EMERGENCY ENFORCEMENT BEGINNING 🚨"
echo "================================================"
echo "Fixing CRITICAL violations of Rules 1, 4, 11"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track fixes
FIXES_APPLIED=0
ERRORS=0

echo -e "${YELLOW}[RULE 11]${NC} Fixing root directory violations..."

# Create proper directory structure if missing
mkdir -p /opt/sutazaiapp/docker
mkdir -p /opt/sutazaiapp/backups/docker
mkdir -p /opt/sutazaiapp/docs/reports
mkdir -p /opt/sutazaiapp/docs/changelog

# Fix Docker files in root (Rule 4 + Rule 11)
if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
    echo -e "${RED}✗${NC} Found docker-compose.yml in root - VIOLATION!"
    
    # Check if consolidated version exists
    if [ -f "/opt/sutazaiapp/docker/docker-compose.consolidated.yml" ]; then
        echo -e "${YELLOW}→${NC} Consolidated version exists, backing up root version..."
        mv /opt/sutazaiapp/docker-compose.yml /opt/sutazaiapp/backups/docker/docker-compose.yml.root.$(date +%Y%m%d_%H%M%S)
        ((FIXES_APPLIED++))
    else
        echo -e "${YELLOW}→${NC} Moving to proper location..."
        mv /opt/sutazaiapp/docker-compose.yml /opt/sutazaiapp/docker/docker-compose.yml
        ((FIXES_APPLIED++))
    fi
fi

# Move docker-compose backups
for backup in /opt/sutazaiapp/docker-compose.yml.backup.*; do
    if [ -f "$backup" ]; then
        echo -e "${YELLOW}→${NC} Moving backup: $(basename $backup)"
        mv "$backup" /opt/sutazaiapp/backups/docker/
        ((FIXES_APPLIED++))
    fi
done

# Move reports from root to proper location
if [ -f "/opt/sutazaiapp/COMPREHENSIVE_CACHE_CONSOLIDATION_REPORT.md" ]; then
    echo -e "${YELLOW}→${NC} Moving cache consolidation report to /docs/reports/"
    mv /opt/sutazaiapp/COMPREHENSIVE_CACHE_CONSOLIDATION_REPORT.md /opt/sutazaiapp/docs/reports/
    ((FIXES_APPLIED++))
fi

if [ -f "/opt/sutazaiapp/RULE_VIOLATIONS_REPORT.md" ]; then
    echo -e "${YELLOW}→${NC} Moving violations report to /docs/reports/"
    mv /opt/sutazaiapp/RULE_VIOLATIONS_REPORT.md /opt/sutazaiapp/docs/reports/
    ((FIXES_APPLIED++))
fi

if [ -f "/opt/sutazaiapp/CHANGELOG_CONSOLIDATED.md" ]; then
    echo -e "${YELLOW}→${NC} Moving consolidated changelog to /docs/changelog/"
    mv /opt/sutazaiapp/CHANGELOG_CONSOLIDATED.md /opt/sutazaiapp/docs/changelog/
    ((FIXES_APPLIED++))
fi

echo ""
echo -e "${YELLOW}[RULE 1]${NC} Scanning for mock/stub violations in tests..."

# Create a list of files with mock/stub violations
MOCK_FILES=$(grep -rl "mock\|Mock\|stub\|Stub\|fake\|Fake" /opt/sutazaiapp/tests/ 2>/dev/null | head -20 || true)

if [ ! -z "$MOCK_FILES" ]; then
    echo -e "${RED}✗${NC} Found $(echo "$MOCK_FILES" | wc -l) test files with mock/stub violations"
    echo "These files need manual rewriting to use REAL implementations:"
    echo "$MOCK_FILES" | head -10
    echo ""
    echo -e "${YELLOW}WARNING:${NC} Test suite is built on FANTASY CODE - tests are NOT valid!"
else
    echo -e "${GREEN}✓${NC} No mock/stub violations found in initial scan"
fi

echo ""
echo -e "${YELLOW}[RULE 4]${NC} Checking Docker consolidation..."

# Count docker-compose references
DOCKER_REFS=$(grep -r "docker-compose" /opt/sutazaiapp --exclude-dir=.git --exclude-dir=node_modules 2>/dev/null | wc -l || echo "0")
echo "Found $DOCKER_REFS references to docker-compose across codebase"

if [ "$DOCKER_REFS" -gt 50 ]; then
    echo -e "${RED}✗${NC} CRITICAL: Docker configuration is NOT consolidated!"
    echo "All Docker configs should use: /opt/sutazaiapp/docker/docker-compose.consolidated.yml"
fi

echo ""
echo "================================================"
echo "📊 ENFORCEMENT SUMMARY"
echo "================================================"
echo -e "Fixes Applied: ${GREEN}$FIXES_APPLIED${NC}"
echo -e "Errors: ${RED}$ERRORS${NC}"
echo ""

if [ $FIXES_APPLIED -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Emergency fixes applied successfully"
else
    echo -e "${YELLOW}⚠${NC} No automatic fixes were needed/possible"
fi

echo ""
echo "🔥 CRITICAL ACTIONS STILL REQUIRED:"
echo "-----------------------------------"
echo "1. REWRITE all test files to remove mock/stub/fake code"
echo "2. CONSOLIDATE all Docker configs to /docker/docker-compose.consolidated.yml"
echo "3. UPDATE all references to use consolidated Docker config"
echo "4. REMOVE all TODO/FIXME comments older than 30 days"
echo "5. ENFORCE no files in root directory policy"
echo ""
echo "Use /opt/sutazaiapp/ULTRATHINK_RULE_VIOLATIONS_AUDIT_REPORT.md for full details"
echo ""
echo "================================================"
echo "ENFORCEMENT COMPLETE - COMPLIANCE MANDATORY"
echo "================================================"