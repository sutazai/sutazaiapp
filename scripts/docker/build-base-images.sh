#!/bin/bash
# ============================================================================
# SutazAI Docker Base Images Builder - ULTRA CONSOLIDATION
# ============================================================================
# Purpose: Build all master base images for the SutazAI ecosystem
# Author: DOCKER-MASTER-001
# Date: August 10, 2025
# Status: PRODUCTION READY - Rule 2 Compliant (Non-Breaking)
# ============================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/sutazai_base_build_${TIMESTAMP}.log"
BACKUP_DIR="/opt/sutazaiapp/backups/base-images-${TIMESTAMP}"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE} SutazAI Docker Base Images Builder - ULTRA CONSOLIDATION${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Backup directory: $BACKUP_DIR"
echo ""

# ============================================================================
# SAFETY CHECKS (Rule 2: Don't Break Existing Functionality)
# ============================================================================

safety_checks() {
    echo -e "${YELLOW}[SAFETY] Performing pre-build safety checks...${NC}"
    
    # Check Docker daemon
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}[ERROR] Docker daemon not running${NC}"
        exit 1
    fi
    
    # Check disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df /opt/sutazaiapp | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        echo -e "${RED}[ERROR] Insufficient disk space. Need at least 10GB${NC}"
        exit 1
    fi
    
    # Backup existing base images
    echo -e "${YELLOW}[BACKUP] Creating backup of existing base images...${NC}"
    mkdir -p "$BACKUP_DIR"
    
    # Save existing images
    docker images | grep sutazai- | while read -r line; do
        IMAGE=$(echo "$line" | awk '{print $1":"$2}')
        FILENAME=$(echo "$IMAGE" | tr ':/' '-')
        echo "  Backing up: $IMAGE -> $BACKUP_DIR/$FILENAME.tar"
        docker save "$IMAGE" -o "$BACKUP_DIR/$FILENAME.tar" 2>/dev/null || true
    done
    
    echo -e "${GREEN}[SAFETY] Pre-build safety checks complete${NC}"
}

# ============================================================================
# BUILD BASE IMAGES
# ============================================================================

build_base_images() {
    echo -e "${YELLOW}[BUILD] Building base images...${NC}"
    
    cd /opt/sutazaiapp
    
    # Enable BuildKit for optimized builds
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Build all base images
    echo -e "${BLUE}[BUILD] Building Python Agent Master (132+ services depend on this)...${NC}"
    docker-compose -f docker-compose.base.yml build python-agent-master 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building Node.js Agent Master (7+ services depend on this)...${NC}"
    docker-compose -f docker-compose.base.yml build nodejs-agent-master 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building Python Alpine Optimized...${NC}"
    docker-compose -f docker-compose.base.yml build python-alpine-optimized 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building AI/ML CPU Base...${NC}"
    docker-compose -f docker-compose.base.yml build ai-ml-cpu-base 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building Golang Service Base...${NC}"
    docker-compose -f docker-compose.base.yml build golang-service-base 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building Monitoring Base...${NC}"
    docker-compose -f docker-compose.base.yml build monitoring-base 2>&1 | tee -a "$LOG_FILE"
    
    echo -e "${BLUE}[BUILD] Building Python Agent Minimal...${NC}"
    docker-compose -f docker-compose.base.yml build python-agent-minimal 2>&1 | tee -a "$LOG_FILE"
    
    # GPU build only if NVIDIA runtime available
    if docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo -e "${BLUE}[BUILD] NVIDIA GPU detected - Building AI/ML GPU Base...${NC}"
        docker-compose -f docker-compose.base.yml build ai-ml-gpu-base 2>&1 | tee -a "$LOG_FILE"
    else
        echo -e "${YELLOW}[SKIP] No NVIDIA GPU runtime detected - skipping GPU base image${NC}"
    fi
    
    echo -e "${GREEN}[BUILD] Base image builds complete${NC}"
}

# ============================================================================
# VALIDATION (Rule 2: Ensure No Functionality Broken)
# ============================================================================

validate_builds() {
    echo -e "${YELLOW}[VALIDATE] Validating base images...${NC}"
    
    # List all built images
    echo -e "${BLUE}[INFO] Built base images:${NC}"
    docker images | grep sutazai- | grep latest | while read -r line; do
        IMAGE=$(echo "$line" | awk '{print $1":"$2}')
        SIZE=$(echo "$line" | awk '{print $7$8}')
        echo "  ✓ $IMAGE ($SIZE)"
    done
    
    # Test basic functionality of key images
    echo -e "${YELLOW}[TEST] Testing Python agent master...${NC}"
    docker run --rm sutazai-python-agent-master:latest python --version || {
        echo -e "${RED}[ERROR] Python agent master failed basic test${NC}"
        exit 1
    }
    
    echo -e "${YELLOW}[TEST] Testing Node.js agent master...${NC}"
    docker run --rm sutazai-nodejs-agent-master:latest node --version || {
        echo -e "${RED}[ERROR] Node.js agent master failed basic test${NC}"
        exit 1
    }
    
    # Check running container health (Rule 2 compliance)
    echo -e "${YELLOW}[HEALTH] Checking running container health...${NC}"
    UNHEALTHY_CONTAINERS=$(docker ps --filter health=unhealthy --format "table {{.Names}}" | wc -l)
    if [ "$UNHEALTHY_CONTAINERS" -gt 1 ]; then  # Account for header line
        echo -e "${RED}[WARNING] Some containers are unhealthy after base image build${NC}"
        docker ps --filter health=unhealthy --format "table {{.Names}}\t{{.Status}}"
    else
        echo -e "${GREEN}[HEALTH] All containers healthy${NC}"
    fi
    
    echo -e "${GREEN}[VALIDATE] Validation complete${NC}"
}

# ============================================================================
# STATISTICS GENERATION
# ============================================================================

generate_stats() {
    echo -e "${YELLOW}[STATS] Generating consolidation statistics...${NC}"
    
    # Count image usage
    PYTHON_MASTER_COUNT=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-python-agent-master" {} \; | wc -l)
    NODEJS_MASTER_COUNT=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-nodejs-agent-master" {} \; | wc -l)
    TOTAL_DOCKERFILES=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -not -path "*/node_modules/*" | wc -l)
    CONSOLIDATED_COUNT=$((PYTHON_MASTER_COUNT + NODEJS_MASTER_COUNT))
    CONSOLIDATION_PERCENTAGE=$(( CONSOLIDATED_COUNT * 100 / TOTAL_DOCKERFILES ))
    
    cat > "/opt/sutazaiapp/docs/reports/docker-consolidation-stats-${TIMESTAMP}.md" << EOF
# Docker Consolidation Statistics

Generated: $(date)
Build Log: $LOG_FILE

## Consolidation Summary
- Total Dockerfiles: $TOTAL_DOCKERFILES
- Using Python Master: $PYTHON_MASTER_COUNT
- Using Node.js Master: $NODEJS_MASTER_COUNT  
- Consolidated Services: $CONSOLIDATED_COUNT
- Consolidation Rate: ${CONSOLIDATION_PERCENTAGE}%

## Base Images Built
$(docker images | grep sutazai- | grep latest)

## Storage Savings
$(docker system df)

## Status: SUCCESS ✅
EOF
    
    echo -e "${GREEN}[STATS] Statistics saved to docs/reports/docker-consolidation-stats-${TIMESTAMP}.md${NC}"
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${GREEN} DOCKER CONSOLIDATION COMPLETE - ${CONSOLIDATION_PERCENTAGE}% CONSOLIDATED${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo "Services using Python Master: $PYTHON_MASTER_COUNT"
    echo "Services using Node.js Master: $NODEJS_MASTER_COUNT"
    echo "Total consolidation rate: ${CONSOLIDATION_PERCENTAGE}%"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo "Starting Docker base image consolidation build..."
    
    safety_checks
    build_base_images
    validate_builds
    generate_stats
    
    echo -e "${GREEN}✅ Docker base image consolidation complete!${NC}"
    echo "Log file: $LOG_FILE"
    echo "Backup: $BACKUP_DIR"
}

# Execute main function
main "$@"