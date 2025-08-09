#!/bin/bash
#
# Purpose: Comprehensive Docker compliance validation orchestrator
# Usage: ./validate-docker-compliance.sh [--auto-fix] [--full-test]
# Requirements: Docker, Python 3.8+
#
# This script orchestrates Docker structure validation and container health checks

set -euo pipefail

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
PROJECT_ROOT="/opt/sutazaiapp"
REPORTS_DIR="${PROJECT_ROOT}/reports"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Parse arguments
AUTO_FIX=false
FULL_TEST=false
STRICT_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-fix)
            AUTO_FIX=true
            shift
            ;;
        --full-test)
            FULL_TEST=true
            shift
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--auto-fix] [--full-test] [--strict]"
            exit 1
            ;;
    esac
done

# Ensure directories exist
mkdir -p "${REPORTS_DIR}" "${LOGS_DIR}"

echo -e "${BLUE}🐳 Docker Compliance Validation Suite${NC}"
echo "======================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Auto-fix: ${AUTO_FIX}"
echo "Full Test: ${FULL_TEST}"
echo "Strict Mode: ${STRICT_MODE}"
echo

# Step 1: Run Docker Structure Validator
echo -e "${YELLOW}📋 Step 1: Validating Docker Structure (Rule 11)${NC}"
STRUCTURE_ARGS=""
if [ "$AUTO_FIX" = true ]; then
    STRUCTURE_ARGS="--auto-fix"
fi
if [ "$STRICT_MODE" = true ]; then
    STRUCTURE_ARGS="${STRUCTURE_ARGS} --strict"
fi

python3 "${PROJECT_ROOT}/scripts/agents/docker-structure-validator.py" \
    ${STRUCTURE_ARGS} \
    --report-format markdown || {
    echo -e "${RED}❌ Docker structure validation failed${NC}"
    if [ "$STRICT_MODE" = true ]; then
        exit 1
    fi
}

# Step 2: Container Infrastructure Validation (if requested)
if [ "$FULL_TEST" = true ]; then
    echo -e "\n${YELLOW}📋 Step 2: Validating Container Infrastructure${NC}"
    
    # Start health monitor in background
    echo "Starting health monitor..."
    python3 "${PROJECT_ROOT}/scripts/container-health-monitor.py" \
        --watch-cleanup \
        --alert-threshold 85 &
    MONITOR_PID=$!
    
    # Run container validation
    python3 "${PROJECT_ROOT}/scripts/validate-container-infrastructure.py" \
        --critical-only \
        --report-format markdown || {
        echo -e "${RED}❌ Container infrastructure validation failed${NC}"
        kill $MONITOR_PID 2>/dev/null || true
        if [ "$STRICT_MODE" = true ]; then
            exit 1
        fi
    }
    
    # Stop health monitor
    kill $MONITOR_PID 2>/dev/null || true
fi

# Step 3: Generate consolidated report
echo -e "\n${YELLOW}📋 Step 3: Generating Consolidated Report${NC}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONSOLIDATED_REPORT="${REPORTS_DIR}/docker_compliance_${TIMESTAMP}.md"

cat > "${CONSOLIDATED_REPORT}" << EOF
# Docker Compliance Validation Report

Generated: $(date)

## Validation Summary

### Docker Structure Validation (Rule 11)
EOF

# Find and append latest docker validation report
LATEST_DOCKER_REPORT=$(ls -t "${REPORTS_DIR}"/docker_validation_*.md 2>/dev/null | head -1)
if [ -n "$LATEST_DOCKER_REPORT" ]; then
    echo "Latest report: ${LATEST_DOCKER_REPORT}" >> "${CONSOLIDATED_REPORT}"
    tail -n +3 "$LATEST_DOCKER_REPORT" >> "${CONSOLIDATED_REPORT}"
fi

if [ "$FULL_TEST" = true ]; then
    cat >> "${CONSOLIDATED_REPORT}" << EOF

### Container Infrastructure Validation
EOF
    
    # Find and append latest container validation report
    LATEST_CONTAINER_REPORT=$(ls -t "${REPORTS_DIR}"/container_validation_*.md 2>/dev/null | head -1)
    if [ -n "$LATEST_CONTAINER_REPORT" ]; then
        echo "Latest report: ${LATEST_CONTAINER_REPORT}" >> "${CONSOLIDATED_REPORT}"
        tail -n +3 "$LATEST_CONTAINER_REPORT" >> "${CONSOLIDATED_REPORT}"
    fi
fi

# Step 4: Quick validation check
echo -e "\n${YELLOW}📋 Step 4: Quick Validation Status${NC}"

# Check for critical Dockerfiles
CRITICAL_DOCKERFILES=(
    "${PROJECT_ROOT}/docker/backend/Dockerfile"
    "${PROJECT_ROOT}/docker/frontend/Dockerfile"
)

MISSING_CRITICAL=0
for dockerfile in "${CRITICAL_DOCKERFILES[@]}"; do
    if [ ! -f "$dockerfile" ]; then
        echo -e "${RED}❌ Missing critical Dockerfile: $dockerfile${NC}"
        MISSING_CRITICAL=$((MISSING_CRITICAL + 1))
    else
        echo -e "${GREEN}✅ Found: $dockerfile${NC}"
    fi
done

# Check docker-compose.yml
if [ -f "${PROJECT_ROOT}/docker-compose.yml" ]; then
    echo -e "${GREEN}✅ Found docker-compose.yml${NC}"
    
    # Quick service check
    SERVICES=$(grep -E '^\s+\w+:' "${PROJECT_ROOT}/docker-compose.yml" | wc -l)
    echo "   Services defined: $SERVICES"
else
    echo -e "${RED}❌ Missing docker-compose.yml${NC}"
fi

# Check .dockerignore
if [ -f "${PROJECT_ROOT}/docker/.dockerignore" ]; then
    echo -e "${GREEN}✅ Found docker/.dockerignore${NC}"
else
    echo -e "${YELLOW}⚠️  Missing docker/.dockerignore${NC}"
fi

# Final summary
echo -e "\n${BLUE}======================================"
echo -e "📊 Validation Complete${NC}"
echo -e "Report saved: ${CONSOLIDATED_REPORT}"

if [ $MISSING_CRITICAL -eq 0 ]; then
    echo -e "${GREEN}✅ All critical Docker files present${NC}"
else
    echo -e "${RED}❌ Missing $MISSING_CRITICAL critical Docker files${NC}"
    if [ "$STRICT_MODE" = true ]; then
        exit 1
    fi
fi

# Display recommendations if auto-fix was not used
if [ "$AUTO_FIX" = false ] && [ $MISSING_CRITICAL -gt 0 ]; then
    echo -e "\n${YELLOW}💡 Run with --auto-fix to automatically fix common issues${NC}"
fi

echo -e "\n${GREEN}✨ Docker compliance validation completed!${NC}"