#!/bin/bash
# SutazAI Reorganization Setup Validation Script
# Validates that all reorganization scripts are properly configured

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 SutazAI Reorganization Setup Validation${NC}"
echo "=============================================="
echo

# Check script directory
echo "📁 Checking reorganization scripts..."

SCRIPT_DIR="/opt/sutazaiapp/scripts/reorganization"
if [ ! -d "$SCRIPT_DIR" ]; then
    echo -e "${RED}❌ Reorganization directory not found: $SCRIPT_DIR${NC}"
    exit 1
fi

# Expected scripts
EXPECTED_SCRIPTS=(
    "reorganize_codebase.sh"
    "01_backup_system.sh"
    "02_create_archive_structure.sh"
    "03_identify_files_to_move.sh"
    "04_move_files_safely.sh"
    "05_test_system_health.sh"
    "validate_setup.sh"
)

echo "Checking required scripts:"
for script in "${EXPECTED_SCRIPTS[@]}"; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        if [ -x "$SCRIPT_DIR/$script" ]; then
            echo -e "  ✅ $script (executable)"
        else
            echo -e "  ⚠️  $script (not executable)"
            chmod +x "$SCRIPT_DIR/$script"
            echo -e "     Fixed: Made executable"
        fi
    else
        echo -e "  ❌ $script (missing)"
    fi
done

echo

# Check critical system files
echo "🔒 Checking critical system files..."

CRITICAL_FILES=(
    "/opt/sutazaiapp/backend/app/main.py"
    "/opt/sutazaiapp/frontend/app.py"
    "/opt/sutazaiapp/docker-compose.minimal.yml"
    "/opt/sutazaiapp/scripts/live_logs.sh"
    "/opt/sutazaiapp/health_check.sh"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ✅ $(basename $file)"
    else
        echo -e "  ❌ $(basename $file) (missing: $file)"
    fi
done

echo

# Check system prerequisites
echo "🔧 Checking system prerequisites..."

# Docker
if command -v docker >/dev/null 2>&1; then
    if docker ps >/dev/null 2>&1; then
        echo -e "  ✅ Docker (running)"
    else
        echo -e "  ⚠️  Docker (installed but not running)"
    fi
else
    echo -e "  ❌ Docker (not installed)"
fi

# jq
if command -v jq >/dev/null 2>&1; then
    echo -e "  ✅ jq (available)"
else
    echo -e "  ⚠️  jq (not installed - needed for JSON parsing)"
fi

# curl
if command -v curl >/dev/null 2>&1; then
    echo -e "  ✅ curl (available)"
else
    echo -e "  ❌ curl (not installed - needed for health checks)"
fi

# bc
if command -v bc >/dev/null 2>&1; then
    echo -e "  ✅ bc (available)"
else
    echo -e "  ⚠️  bc (not installed - needed for calculations)"
fi

echo

# Check disk space
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df /opt/sutazaiapp | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

if [ "$AVAILABLE_SPACE" -ge 1000000 ]; then  # 1GB in KB
    echo -e "  ✅ Disk space: ${AVAILABLE_GB}GB available (sufficient)"
else
    echo -e "  ⚠️  Disk space: ${AVAILABLE_GB}GB available (recommend 1GB+)"
fi

echo

# Check permissions
echo "🔑 Checking permissions..."
if [ -w "/opt/sutazaiapp" ]; then
    echo -e "  ✅ Write permissions to /opt/sutazaiapp"
else
    echo -e "  ❌ No write permissions to /opt/sutazaiapp"
fi

# Create log directory if needed
if [ ! -d "/opt/sutazaiapp/logs" ]; then
    mkdir -p "/opt/sutazaiapp/logs"
    echo -e "  ✅ Created logs directory"
else
    echo -e "  ✅ Logs directory exists"
fi

echo

# Check running containers
echo "🐳 Checking running containers..."
if docker ps >/dev/null 2>&1; then
    RUNNING_CONTAINERS=$(docker ps --format "{{.Names}}" | grep "sutazai" | wc -l)
    echo -e "  ℹ️  SutazAI containers running: $RUNNING_CONTAINERS"
    
    # Check essential containers
    ESSENTIAL_CONTAINERS=(
        "sutazai-backend-minimal"
        "sutazai-ollama-minimal"
        "sutazai-postgres-minimal"
        "sutazai-redis-minimal"
    )
    
    for container in "${ESSENTIAL_CONTAINERS[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            echo -e "    ✅ $container"
        else
            echo -e "    ⚠️  $container (not running)"
        fi
    done
else
    echo -e "  ❌ Cannot check containers (Docker not accessible)"
fi

echo

# Test backend health (if available)
echo "🏥 Testing backend health..."
if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
    echo -e "  ✅ Backend health endpoint accessible"
else
    echo -e "  ⚠️  Backend health endpoint not accessible"
fi

echo

# Count files to be reorganized
echo "📊 Analyzing files for reorganization..."
if [ -d "/opt/sutazaiapp/scripts" ]; then
    TOTAL_SCRIPTS=$(find /opt/sutazaiapp/scripts -name "*.sh" -o -name "*.py" | wc -l)
    echo -e "  ℹ️  Total scripts found: $TOTAL_SCRIPTS"
    
    # Estimate files to be moved (rough calculation)
    ESTIMATED_MOVES=$((TOTAL_SCRIPTS * 40 / 100))  # Approximately 40% will be moved
    echo -e "  ℹ️  Estimated files to reorganize: ~$ESTIMATED_MOVES"
else
    echo -e "  ⚠️  Scripts directory not found"
fi

echo

# Summary
echo -e "${BLUE}📋 Validation Summary${NC}"
echo "==================="

# Count issues
ERRORS=0
WARNINGS=0

# Simple validation results
if [ ! -f "$SCRIPT_DIR/reorganize_codebase.sh" ]; then
    ((ERRORS++))
fi

if ! docker ps >/dev/null 2>&1; then
    ((ERRORS++))
fi

if [ ! -f "/opt/sutazaiapp/backend/app/main.py" ]; then
    ((ERRORS++))
fi

if [ "$AVAILABLE_SPACE" -lt 1000000 ]; then
    ((WARNINGS++))
fi

if ! curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
    ((WARNINGS++))
fi

echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo

if [ "$ERRORS" -eq 0 ]; then
    if [ "$WARNINGS" -eq 0 ]; then
        echo -e "${GREEN}✅ READY: System is ready for reorganization${NC}"
        echo -e "${GREEN}   Run: ./reorganize_codebase.sh${NC}"
    else
        echo -e "${YELLOW}⚠️  READY WITH WARNINGS: System can proceed but check warnings${NC}"
        echo -e "${YELLOW}   Run: ./reorganize_codebase.sh${NC}"
    fi
else
    echo -e "${RED}❌ NOT READY: Fix errors before proceeding${NC}"
    echo -e "${RED}   Address the issues above before running reorganization${NC}"
fi

echo

# Quick start instructions  
echo -e "${BLUE}🚀 Quick Start${NC}"
echo "============="
echo "To begin reorganization:"
echo "1. cd /opt/sutazaiapp/scripts/reorganization"
echo "2. ./reorganize_codebase.sh"
echo
echo "For help:"
echo "- cat README.md"
echo "- ./reorganize_codebase.sh --help"

exit $ERRORS