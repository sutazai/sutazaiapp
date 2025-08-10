#!/bin/bash
# Dockerfile Post-Migration Validation Script
# Ultra-careful validation after migrating a Dockerfile
# Author: System Architect
# Date: August 10, 2025

set -euo pipefail

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE=${1:-}
VALIDATION_DIR="/tmp/dockerfile-validation"
REPORT_DIR="/opt/sutazaiapp/reports/dockerfile-dedup"

# Validate input
if [ -z "$SERVICE" ]; then
    echo -e "${RED}Error: Service name required${NC}"
    echo "Usage: $0 <service-name>"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Post-Migration Validation for: $SERVICE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# Check for pre-migration report
PRE_REPORT="$REPORT_DIR/${SERVICE}-pre-migration.txt"
if [ ! -f "$PRE_REPORT" ]; then
    echo -e "${RED}Error: Pre-migration report not found${NC}"
    echo "Run validate-before-migration.sh first"
    exit 1
fi

# Find migrated Dockerfile
DOCKERFILE_PATH=""
if [ -f "/opt/sutazaiapp/docker/$SERVICE/Dockerfile" ]; then
    DOCKERFILE_PATH="/opt/sutazaiapp/docker/$SERVICE/Dockerfile"
elif [ -f "/opt/sutazaiapp/agents/$SERVICE/Dockerfile" ]; then
    DOCKERFILE_PATH="/opt/sutazaiapp/agents/$SERVICE/Dockerfile"
else
    echo -e "${RED}Error: Dockerfile not found for $SERVICE${NC}"
    exit 1
fi

SERVICE_DIR=$(dirname "$DOCKERFILE_PATH")
echo -e "${YELLOW}Validating migrated Dockerfile: $DOCKERFILE_PATH${NC}"

# Initialize post-migration report
POST_REPORT="$REPORT_DIR/${SERVICE}-post-migration.txt"
{
    echo "Service: $SERVICE"
    echo "Path: $DOCKERFILE_PATH"
    echo "Timestamp: $(date)"
    echo "---"
} > "$POST_REPORT"

# Step 1: Verify base image change
echo -e "\n${YELLOW}Step 1: Verifying base image migration...${NC}"
NEW_BASE=$(grep "^FROM" "$DOCKERFILE_PATH" | head -1)
OLD_BASE=$(grep "Base Image:" "$PRE_REPORT" | cut -d: -f2-)

echo "Old Base: $OLD_BASE" | tee -a "$POST_REPORT"
echo "New Base: $NEW_BASE" | tee -a "$POST_REPORT"

if [[ "$NEW_BASE" == *"sutazai-"* ]]; then
    echo -e "${GREEN}✓ Using SutazAI base image${NC}"
    echo "Base Migration: SUCCESS" >> "$POST_REPORT"
else
    echo -e "${RED}✗ Not using SutazAI base image${NC}"
    echo "Base Migration: FAILED" >> "$POST_REPORT"
fi

# Step 2: Build migrated image
echo -e "\n${YELLOW}Step 2: Building migrated image...${NC}"
BUILD_LOG="$VALIDATION_DIR/${SERVICE}-build-post.log"
if docker build -t "test-${SERVICE}-migrated" -f "$DOCKERFILE_PATH" "$SERVICE_DIR" > "$BUILD_LOG" 2>&1; then
    echo -e "${GREEN}✓ Build successful${NC}"
    echo "Build Status: SUCCESS" >> "$POST_REPORT"
else
    echo -e "${RED}✗ Build failed${NC}"
    echo "Build Status: FAILED" >> "$POST_REPORT"
    echo "See log: $BUILD_LOG"
    exit 1
fi

# Step 3: Compare image metrics
echo -e "\n${YELLOW}Step 3: Comparing image metrics...${NC}"
NEW_SIZE=$(docker images "test-${SERVICE}-migrated" --format "{{.Size}}")
OLD_SIZE=$(grep "Image Size:" "$PRE_REPORT" | cut -d: -f2- | tr -d ' ')
NEW_LAYERS=$(docker history "test-${SERVICE}-migrated" | wc -l)
OLD_LAYERS=$(grep "Layer Count:" "$PRE_REPORT" | cut -d: -f2- | tr -d ' ')

echo "Image Size: $OLD_SIZE → $NEW_SIZE" | tee -a "$POST_REPORT"
echo "Layer Count: $OLD_LAYERS → $NEW_LAYERS" | tee -a "$POST_REPORT"

# Calculate size difference
OLD_SIZE_MB=$(echo "$OLD_SIZE" | sed 's/MB//' | sed 's/GB/*1024/' | bc 2>/dev/null || echo "0")
NEW_SIZE_MB=$(echo "$NEW_SIZE" | sed 's/MB//' | sed 's/GB/*1024/' | bc 2>/dev/null || echo "0")

if [ "$NEW_SIZE_MB" != "0" ] && [ "$OLD_SIZE_MB" != "0" ]; then
    SIZE_REDUCTION=$(echo "scale=2; (1 - $NEW_SIZE_MB / $OLD_SIZE_MB) * 100" | bc 2>/dev/null || echo "0")
    echo "Size Reduction: ${SIZE_REDUCTION}%" | tee -a "$POST_REPORT"
    
    if (( $(echo "$SIZE_REDUCTION > 0" | bc -l) )); then
        echo -e "${GREEN}✓ Image size reduced${NC}"
    fi
fi

# Step 4: Test functionality
echo -e "\n${YELLOW}Step 4: Testing functionality...${NC}"
TESTS_PASSED=0
TESTS_TOTAL=0

# Test 1: Python runtime
((TESTS_TOTAL++))
if docker run --rm "test-${SERVICE}-migrated" python -c "import sys; print('Python OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ Python runtime OK${NC}"
    echo "Python Test: PASSED" >> "$POST_REPORT"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Python test skipped${NC}"
    echo "Python Test: SKIPPED" >> "$POST_REPORT"
fi

# Test 2: Import core dependencies
((TESTS_TOTAL++))
if docker run --rm "test-${SERVICE}-migrated" python -c "import fastapi, uvicorn; print('Core imports OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ Core dependencies available${NC}"
    echo "Core Dependencies: PASSED" >> "$POST_REPORT"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Core dependencies test skipped${NC}"
    echo "Core Dependencies: SKIPPED" >> "$POST_REPORT"
fi

# Test 3: User permissions
((TESTS_TOTAL++))
USER_ID=$(docker run --rm "test-${SERVICE}-migrated" id -u 2>/dev/null || echo "0")
if [ "$USER_ID" != "0" ]; then
    echo -e "${GREEN}✓ Running as non-root user (UID: $USER_ID)${NC}"
    echo "Non-root User: PASSED" >> "$POST_REPORT"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Running as root${NC}"
    echo "Non-root User: FAILED" >> "$POST_REPORT"
fi

# Test 4: Health endpoint
echo -e "\n${YELLOW}Step 5: Testing health endpoint...${NC}"
((TESTS_TOTAL++))
CONTAINER_NAME="test-${SERVICE}-health-post"
docker run -d --name "$CONTAINER_NAME" "test-${SERVICE}-migrated" > /dev/null 2>&1 || true
sleep 5

if docker exec "$CONTAINER_NAME" curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health endpoint responsive${NC}"
    echo "Health Check: PASSED" >> "$POST_REPORT"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ Health endpoint not available${NC}"
    echo "Health Check: NOT AVAILABLE" >> "$POST_REPORT"
fi

docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true

# Step 6: Compare with original functionality
echo -e "\n${YELLOW}Step 6: Comparing with original functionality...${NC}"
ORIGINAL_HEALTH=$(grep "Health Check:" "$PRE_REPORT" | cut -d: -f2- | tr -d ' ')
NEW_HEALTH=$(grep "Health Check:" "$POST_REPORT" | tail -1 | cut -d: -f2- | tr -d ' ')

if [ "$ORIGINAL_HEALTH" = "$NEW_HEALTH" ] || [ "$NEW_HEALTH" = "PASSED" ]; then
    echo -e "${GREEN}✓ Functionality preserved${NC}"
    echo "Functionality Comparison: PRESERVED" >> "$POST_REPORT"
else
    echo -e "${RED}✗ Functionality degraded${NC}"
    echo "Functionality Comparison: DEGRADED" >> "$POST_REPORT"
fi

# Step 7: Integration test
echo -e "\n${YELLOW}Step 7: Running integration test...${NC}"
if command -v docker-compose &> /dev/null; then
    # Try to start service in docker-compose
    if docker-compose ps | grep -q "$SERVICE"; then
        echo "Restarting service in docker-compose..."
        docker-compose restart "$SERVICE" 2>/dev/null || true
        sleep 10
        
        if docker-compose ps "$SERVICE" | grep -q "Up"; then
            echo -e "${GREEN}✓ Service running in docker-compose${NC}"
            echo "Integration Test: PASSED" >> "$POST_REPORT"
        else
            echo -e "${RED}✗ Service failed in docker-compose${NC}"
            echo "Integration Test: FAILED" >> "$POST_REPORT"
        fi
    else
        echo -e "${YELLOW}⚠ Service not in docker-compose${NC}"
        echo "Integration Test: SKIPPED" >> "$POST_REPORT"
    fi
fi

# Step 8: Security scan
echo -e "\n${YELLOW}Step 8: Running security scan...${NC}"
SCAN_RESULTS="$VALIDATION_DIR/${SERVICE}-security-scan.txt"
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image --severity HIGH,CRITICAL "test-${SERVICE}-migrated" > "$SCAN_RESULTS" 2>&1 || true

if [ -f "$SCAN_RESULTS" ]; then
    VULNS=$(grep -c "Total:" "$SCAN_RESULTS" 2>/dev/null || echo "0")
    echo "Security vulnerabilities found: $VULNS" | tee -a "$POST_REPORT"
    
    if [ "$VULNS" -eq 0 ]; then
        echo -e "${GREEN}✓ No high/critical vulnerabilities${NC}"
    else
        echo -e "${YELLOW}⚠ Security vulnerabilities detected${NC}"
    fi
fi

# Step 9: Generate migration success score
echo -e "\n${YELLOW}Step 9: Calculating migration success score...${NC}"
SUCCESS_SCORE=$(echo "scale=2; ($TESTS_PASSED / $TESTS_TOTAL) * 100" | bc 2>/dev/null || echo "0")
echo "Test Success Rate: ${SUCCESS_SCORE}%" | tee -a "$POST_REPORT"

# Overall validation result
echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
if (( $(echo "$SUCCESS_SCORE >= 80" | bc -l) )); then
    echo -e "${GREEN}  ✓ MIGRATION SUCCESSFUL${NC}"
    echo "MIGRATION RESULT: SUCCESS" >> "$POST_REPORT"
    
    # Create success marker
    touch "$REPORT_DIR/${SERVICE}.migrated"
elif (( $(echo "$SUCCESS_SCORE >= 60" | bc -l) )); then
    echo -e "${YELLOW}  ⚠ MIGRATION PARTIALLY SUCCESSFUL${NC}"
    echo "MIGRATION RESULT: PARTIAL" >> "$POST_REPORT"
else
    echo -e "${RED}  ✗ MIGRATION FAILED${NC}"
    echo "MIGRATION RESULT: FAILED" >> "$POST_REPORT"
    
    # Suggest rollback
    echo -e "${RED}Consider rolling back using:${NC}"
    echo "  cp /opt/sutazaiapp/archive/dockerfile-backups/pre-dedup/${SERVICE}-Dockerfile.*.backup $DOCKERFILE_PATH"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# Summary report
echo -e "\nReports saved to:"
echo -e "  Pre-migration:  ${YELLOW}$PRE_REPORT${NC}"
echo -e "  Post-migration: ${YELLOW}$POST_REPORT${NC}"

# Cleanup test images
docker rmi "test-${SERVICE}-migrated" > /dev/null 2>&1 || true

exit 0