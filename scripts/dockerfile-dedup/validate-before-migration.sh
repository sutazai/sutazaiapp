#!/bin/bash
# Dockerfile Pre-Migration Validation Script
# Ultra-careful validation before migrating any Dockerfile
# Author: System Architect
# Date: August 10, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pre-Migration Validation for: $SERVICE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

# Create directories
mkdir -p "$VALIDATION_DIR"
mkdir -p "$REPORT_DIR"

# Find service Dockerfile
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
echo -e "${YELLOW}Found Dockerfile: $DOCKERFILE_PATH${NC}"

# Step 1: Analyze current Dockerfile
echo -e "\n${YELLOW}Step 1: Analyzing current Dockerfile...${NC}"
{
    echo "Service: $SERVICE"
    echo "Path: $DOCKERFILE_PATH"
    echo "Timestamp: $(date)"
    echo "---"
    echo "Base Image: $(grep "^FROM" "$DOCKERFILE_PATH" | head -1)"
    echo "User: $(grep "^USER" "$DOCKERFILE_PATH" || echo "No USER directive (runs as root)")"
    echo "Healthcheck: $(grep "^HEALTHCHECK" "$DOCKERFILE_PATH" || echo "No HEALTHCHECK")"
    echo "Line Count: $(wc -l < "$DOCKERFILE_PATH")"
    echo "MD5: $(md5sum "$DOCKERFILE_PATH" | awk '{print $1}')"
} > "$REPORT_DIR/${SERVICE}-pre-migration.txt"

# Step 2: Build current image
echo -e "\n${YELLOW}Step 2: Building current image...${NC}"
BUILD_LOG="$VALIDATION_DIR/${SERVICE}-build.log"
if docker build -t "test-${SERVICE}-original" -f "$DOCKERFILE_PATH" "$SERVICE_DIR" > "$BUILD_LOG" 2>&1; then
    echo -e "${GREEN}✓ Build successful${NC}"
    echo "Build Status: SUCCESS" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
else
    echo -e "${RED}✗ Build failed${NC}"
    echo "Build Status: FAILED" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
    echo "See log: $BUILD_LOG"
    exit 1
fi

# Step 3: Record image metrics
echo -e "\n${YELLOW}Step 3: Recording image metrics...${NC}"
IMAGE_SIZE=$(docker images "test-${SERVICE}-original" --format "{{.Size}}")
IMAGE_LAYERS=$(docker history "test-${SERVICE}-original" | wc -l)
echo "Image Size: $IMAGE_SIZE" | tee -a "$REPORT_DIR/${SERVICE}-pre-migration.txt"
echo "Layer Count: $IMAGE_LAYERS" | tee -a "$REPORT_DIR/${SERVICE}-pre-migration.txt"

# Step 4: Test basic functionality
echo -e "\n${YELLOW}Step 4: Testing basic functionality...${NC}"
if docker run --rm "test-${SERVICE}-original" python -c "import sys; print('Python OK')" 2>/dev/null; then
    echo -e "${GREEN}✓ Python runtime OK${NC}"
    echo "Python Test: PASSED" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
else
    echo -e "${YELLOW}⚠ Python test skipped (may not be Python service)${NC}"
    echo "Python Test: SKIPPED" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
fi

# Step 5: Test health endpoint
echo -e "\n${YELLOW}Step 5: Testing health endpoint...${NC}"
CONTAINER_NAME="test-${SERVICE}-health"
docker run -d --name "$CONTAINER_NAME" "test-${SERVICE}-original" > /dev/null 2>&1 || true
sleep 5

if docker exec "$CONTAINER_NAME" curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health endpoint responsive${NC}"
    echo "Health Check: PASSED" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
else
    echo -e "${YELLOW}⚠ Health endpoint not available${NC}"
    echo "Health Check: NOT AVAILABLE" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
fi

docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true

# Step 6: Check for dependencies
echo -e "\n${YELLOW}Step 6: Analyzing dependencies...${NC}"
if [ -f "$SERVICE_DIR/requirements.txt" ]; then
    echo "Requirements.txt: FOUND" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
    echo "Package Count: $(wc -l < "$SERVICE_DIR/requirements.txt")" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
    cp "$SERVICE_DIR/requirements.txt" "$REPORT_DIR/${SERVICE}-requirements-backup.txt"
elif [ -f "$SERVICE_DIR/package.json" ]; then
    echo "Package.json: FOUND" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
    cp "$SERVICE_DIR/package.json" "$REPORT_DIR/${SERVICE}-package-backup.json"
else
    echo "Dependencies: NO MANIFEST FOUND" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
fi

# Step 7: Check docker-compose references
echo -e "\n${YELLOW}Step 7: Checking docker-compose references...${NC}"
COMPOSE_REFS=$(grep -r "$SERVICE" /opt/sutazaiapp/docker-compose*.yml 2>/dev/null | wc -l || echo "0")
echo "Docker-compose references: $COMPOSE_REFS" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"

# Step 8: Backup original Dockerfile
echo -e "\n${YELLOW}Step 8: Creating backup...${NC}"
BACKUP_DIR="/opt/sutazaiapp/archive/dockerfile-backups/pre-dedup"
mkdir -p "$BACKUP_DIR"
cp "$DOCKERFILE_PATH" "$BACKUP_DIR/${SERVICE}-Dockerfile.$(date +%Y%m%d-%H%M%S).backup"
echo -e "${GREEN}✓ Backup created${NC}"

# Step 9: Generate migration readiness score
echo -e "\n${YELLOW}Step 9: Calculating migration readiness...${NC}"
SCORE=0
TOTAL=5

# Check if using standard base image
if grep -q "^FROM python:3.11" "$DOCKERFILE_PATH"; then
    ((SCORE++))
fi

# Check if has USER directive
if grep -q "^USER" "$DOCKERFILE_PATH"; then
    ((SCORE++))
fi

# Check if has HEALTHCHECK
if grep -q "^HEALTHCHECK" "$DOCKERFILE_PATH"; then
    ((SCORE++))
fi

# Check if build succeeded
if [ -f "$BUILD_LOG" ]; then
    ((SCORE++))
fi

# Check if not too complex (under 100 lines)
if [ $(wc -l < "$DOCKERFILE_PATH") -lt 100 ]; then
    ((SCORE++))
fi

READINESS=$((SCORE * 100 / TOTAL))
echo "Migration Readiness Score: ${READINESS}%" | tee -a "$REPORT_DIR/${SERVICE}-pre-migration.txt"

if [ $READINESS -ge 80 ]; then
    echo -e "${GREEN}✓ Service is ready for migration${NC}"
    echo "Migration Status: READY" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
elif [ $READINESS -ge 60 ]; then
    echo -e "${YELLOW}⚠ Service needs review before migration${NC}"
    echo "Migration Status: REVIEW NEEDED" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
else
    echo -e "${RED}✗ Service requires significant work before migration${NC}"
    echo "Migration Status: NOT READY" >> "$REPORT_DIR/${SERVICE}-pre-migration.txt"
fi

# Summary
echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Validation Complete${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "Report saved to: ${YELLOW}$REPORT_DIR/${SERVICE}-pre-migration.txt${NC}"
echo -e "Backup saved to: ${YELLOW}$BACKUP_DIR/${NC}"
echo -e "Migration readiness: ${YELLOW}${READINESS}%${NC}"

# Cleanup test images
docker rmi "test-${SERVICE}-original" > /dev/null 2>&1 || true

exit 0