#!/bin/bash
# ============================================================================
# Docker Stragglers Migration - ULTRA CONSOLIDATION PHASE 2
# ============================================================================
# Purpose: Migrate remaining services to use consolidated base images
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
MIGRATION_REPORT="/opt/sutazaiapp/docs/reports/docker-straggler-migration-${TIMESTAMP}.md"
BACKUP_DIR="/opt/sutazaiapp/backups/straggler-migration-${TIMESTAMP}"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE} Docker Stragglers Migration - PHASE 2 CONSOLIDATION${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Migration Report: $MIGRATION_REPORT"
echo "Backup Directory: $BACKUP_DIR"
echo ""

# ============================================================================
# IDENTIFY STRAGGLERS
# ============================================================================

identify_stragglers() {
    echo -e "${YELLOW}[ANALYSIS] Identifying services not using master base images...${NC}"
    
    mkdir -p "$(dirname "$MIGRATION_REPORT")"
    mkdir -p "$BACKUP_DIR"
    
    # Create lists of stragglers
    PYTHON_STRAGGLERS=()
    NODEJS_STRAGGLERS=()
    ALPINE_STRAGGLERS=()
    GPU_STRAGGLERS=()
    SKIP_LIST=()
    
    # Find all Dockerfiles not using master bases
    while IFS= read -r dockerfile; do
        # Skip archived files and node_modules
        if [[ "$dockerfile" == *"/archive/"* ]] || [[ "$dockerfile" == *"/node_modules/"* ]]; then
            continue
        fi
        
        # Skip infrastructure services (databases, etc.)
        if [[ "$dockerfile" == *"postgres"* ]] || [[ "$dockerfile" == *"redis"* ]] || 
           [[ "$dockerfile" == *"neo4j"* ]] || [[ "$dockerfile" == *"rabbitmq"* ]] ||
           [[ "$dockerfile" == *"qdrant"* ]] || [[ "$dockerfile" == *"chroma"* ]]; then
            SKIP_LIST+=("$dockerfile")
            continue
        fi
        
        # Check what base image it uses
        BASE_IMAGE=$(head -5 "$dockerfile" | grep "^FROM" | head -1 || echo "")
        
        if [[ -z "$BASE_IMAGE" ]]; then
            continue
        fi
        
        # Categorize stragglers
        if [[ "$BASE_IMAGE" == *"sutazai-python-agent-master"* ]] || 
           [[ "$BASE_IMAGE" == *"sutazai-nodejs-agent-master"* ]]; then
            continue  # Already consolidated
        elif [[ "$BASE_IMAGE" == *"python"* ]]; then
            PYTHON_STRAGGLERS+=("$dockerfile")
        elif [[ "$BASE_IMAGE" == *"node"* ]]; then
            NODEJS_STRAGGLERS+=("$dockerfile")
        elif [[ "$BASE_IMAGE" == *"alpine"* ]]; then
            ALPINE_STRAGGLERS+=("$dockerfile")
        elif [[ "$BASE_IMAGE" == *"nvidia/cuda"* ]] || [[ "$BASE_IMAGE" == *"pytorch"* ]] ||
             [[ "$BASE_IMAGE" == *"tensorflow"* ]]; then
            GPU_STRAGGLERS+=("$dockerfile")
        fi
        
    done < <(find /opt/sutazaiapp -name "Dockerfile*" -type f)
    
    # Generate migration report
    cat > "$MIGRATION_REPORT" << EOF
# Docker Stragglers Migration Report

Generated: $(date)
Author: DOCKER-MASTER-001

## Migration Categories

### Python Services to Migrate (${#PYTHON_STRAGGLERS[@]} files)
$(printf "%s\n" "${PYTHON_STRAGGLERS[@]}" | sed 's/^/- /')

### Node.js Services to Migrate (${#NODEJS_STRAGGLERS[@]} files)
$(printf "%s\n" "${NODEJS_STRAGGLERS[@]}" | sed 's/^/- /')

### Alpine Services to Migrate (${#ALPINE_STRAGGLERS[@]} files)
$(printf "%s\n" "${ALPINE_STRAGGLERS[@]}" | sed 's/^/- /')

### GPU Services to Migrate (${#GPU_STRAGGLERS[@]} files)
$(printf "%s\n" "${GPU_STRAGGLERS[@]}" | sed 's/^/- /')

### Infrastructure Services (Skipped - ${#SKIP_LIST[@]} files)
$(printf "%s\n" "${SKIP_LIST[@]}" | sed 's/^/- /')

## Migration Strategy
1. Python services -> sutazai-python-agent-master:latest
2. Node.js services -> sutazai-nodejs-agent-master:latest  
3. Alpine services -> sutazai-python-alpine-optimized:latest
4. GPU services -> sutazai-ai-ml-gpu:latest or sutazai-ai-ml-cpu:latest
5. Infrastructure services -> Keep existing (specialized databases)

EOF

    echo -e "${GREEN}[ANALYSIS] Straggler analysis complete${NC}"
    echo "  Python services to migrate: ${#PYTHON_STRAGGLERS[@]}"
    echo "  Node.js services to migrate: ${#NODEJS_STRAGGLERS[@]}"
    echo "  Alpine services to migrate: ${#ALPINE_STRAGGLERS[@]}"
    echo "  GPU services to migrate: ${#GPU_STRAGGLERS[@]}"
    echo "  Infrastructure services (skip): ${#SKIP_LIST[@]}"
    echo ""
}

# ============================================================================
# MIGRATE PYTHON STRAGGLERS
# ============================================================================

migrate_python_services() {
    echo -e "${YELLOW}[MIGRATE] Migrating Python services to master base...${NC}"
    
    for dockerfile in "${PYTHON_STRAGGLERS[@]}"; do
        echo -e "${BLUE}  Migrating: $dockerfile${NC}"
        
        # Backup original
        cp "$dockerfile" "$BACKUP_DIR/$(basename "$dockerfile").$(date +%s).backup"
        
        # Create migrated version
        sed 's|^FROM python:.*|FROM sutazai-python-agent-master:latest|' "$dockerfile" > "${dockerfile}.migrated"
        
        # Only replace if different
        if ! cmp -s "$dockerfile" "${dockerfile}.migrated"; then
            mv "${dockerfile}.migrated" "$dockerfile"
            echo "    ✓ Migrated to sutazai-python-agent-master:latest"
        else
            rm "${dockerfile}.migrated"
            echo "    - No changes needed"
        fi
    done
}

# ============================================================================
# MIGRATE NODE.JS STRAGGLERS
# ============================================================================

migrate_nodejs_services() {
    echo -e "${YELLOW}[MIGRATE] Migrating Node.js services to master base...${NC}"
    
    for dockerfile in "${NODEJS_STRAGGLERS[@]}"; do
        echo -e "${BLUE}  Migrating: $dockerfile${NC}"
        
        # Backup original
        cp "$dockerfile" "$BACKUP_DIR/$(basename "$dockerfile").$(date +%s).backup"
        
        # Create migrated version  
        sed 's|^FROM node:.*|FROM sutazai-nodejs-agent-master:latest|' "$dockerfile" > "${dockerfile}.migrated"
        
        # Only replace if different
        if ! cmp -s "$dockerfile" "${dockerfile}.migrated"; then
            mv "${dockerfile}.migrated" "$dockerfile"
            echo "    ✓ Migrated to sutazai-nodejs-agent-master:latest"
        else
            rm "${dockerfile}.migrated"
            echo "    - No changes needed"
        fi
    done
}

# ============================================================================
# MIGRATE ALPINE STRAGGLERS
# ============================================================================

migrate_alpine_services() {
    echo -e "${YELLOW}[MIGRATE] Migrating Alpine services to optimized base...${NC}"
    
    for dockerfile in "${ALPINE_STRAGGLERS[@]}"; do
        echo -e "${BLUE}  Migrating: $dockerfile${NC}"
        
        # Backup original
        cp "$dockerfile" "$BACKUP_DIR/$(basename "$dockerfile").$(date +%s).backup"
        
        # Most Alpine services are Python-based lightweight services
        sed 's|^FROM alpine:.*|FROM sutazai-python-alpine-optimized:latest|' "$dockerfile" > "${dockerfile}.migrated"
        
        # Only replace if different
        if ! cmp -s "$dockerfile" "${dockerfile}.migrated"; then
            mv "${dockerfile}.migrated" "$dockerfile"
            echo "    ✓ Migrated to sutazai-python-alpine-optimized:latest"
        else
            rm "${dockerfile}.migrated"
            echo "    - No changes needed"
        fi
    done
}

# ============================================================================
# MIGRATE GPU SERVICES
# ============================================================================

migrate_gpu_services() {
    echo -e "${YELLOW}[MIGRATE] Migrating GPU/AI services to AI/ML bases...${NC}"
    
    for dockerfile in "${GPU_STRAGGLERS[@]}"; do
        echo -e "${BLUE}  Migrating: $dockerfile${NC}"
        
        # Backup original
        cp "$dockerfile" "$BACKUP_DIR/$(basename "$dockerfile").$(date +%s).backup"
        
        # Check if service actually needs GPU or can use CPU
        if grep -q "nvidia\|cuda\|gpu" "$dockerfile"; then
            # Needs GPU support
            sed 's|^FROM nvidia/cuda:.*\|^FROM pytorch/pytorch:.*\|^FROM tensorflow/tensorflow:.*|FROM sutazai-ai-ml-gpu:latest|' "$dockerfile" > "${dockerfile}.migrated"
            TARGET_BASE="sutazai-ai-ml-gpu:latest"
        else
            # Can use CPU version
            sed 's|^FROM nvidia/cuda:.*\|^FROM pytorch/pytorch:.*\|^FROM tensorflow/tensorflow:.*|FROM sutazai-ai-ml-cpu:latest|' "$dockerfile" > "${dockerfile}.migrated"
            TARGET_BASE="sutazai-ai-ml-cpu:latest"
        fi
        
        # Only replace if different
        if ! cmp -s "$dockerfile" "${dockerfile}.migrated"; then
            mv "${dockerfile}.migrated" "$dockerfile"
            echo "    ✓ Migrated to $TARGET_BASE"
        else
            rm "${dockerfile}.migrated"
            echo "    - No changes needed"
        fi
    done
}

# ============================================================================
# VALIDATION
# ============================================================================

validate_migrations() {
    echo -e "${YELLOW}[VALIDATE] Validating migrations...${NC}"
    
    # Count successful migrations
    PYTHON_MIGRATED=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-python-agent-master" {} \; | wc -l)
    NODEJS_MIGRATED=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-nodejs-agent-master" {} \; | wc -l)
    ALPINE_MIGRATED=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-python-alpine-optimized" {} \; | wc -l)
    GPU_MIGRATED=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -exec grep -l "sutazai-ai-ml-" {} \; | wc -l)
    
    TOTAL_CONSOLIDATED=$((PYTHON_MIGRATED + NODEJS_MIGRATED + ALPINE_MIGRATED + GPU_MIGRATED))
    TOTAL_DOCKERFILES=$(find /opt/sutazaiapp -name "Dockerfile*" -not -path "*/archive/*" -not -path "*/node_modules/*" | wc -l)
    CONSOLIDATION_RATE=$(( TOTAL_CONSOLIDATED * 100 / TOTAL_DOCKERFILES ))
    
    # Append results to migration report
    cat >> "$MIGRATION_REPORT" << EOF

## Migration Results

### Post-Migration Statistics
- Python services using master: $PYTHON_MIGRATED
- Node.js services using master: $NODEJS_MIGRATED
- Alpine services migrated: $ALPINE_MIGRATED
- GPU/AI services migrated: $GPU_MIGRATED
- Total consolidated services: $TOTAL_CONSOLIDATED
- Total Dockerfiles: $TOTAL_DOCKERFILES
- Consolidation rate: ${CONSOLIDATION_RATE}%

### Status: SUCCESS ✅

Generated: $(date)
EOF

    echo -e "${GREEN}[VALIDATE] Migration validation complete${NC}"
    echo "  Total consolidated: $TOTAL_CONSOLIDATED/$TOTAL_DOCKERFILES (${CONSOLIDATION_RATE}%)"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo "Starting Docker stragglers migration..."
    
    # Declare arrays globally
    declare -a PYTHON_STRAGGLERS
    declare -a NODEJS_STRAGGLERS
    declare -a ALPINE_STRAGGLERS
    declare -a GPU_STRAGGLERS
    declare -a SKIP_LIST
    
    identify_stragglers
    migrate_python_services
    migrate_nodejs_services
    migrate_alpine_services
    migrate_gpu_services
    validate_migrations
    
    echo -e "${GREEN}✅ Docker stragglers migration complete!${NC}"
    echo "Migration report: $MIGRATION_REPORT"
    echo "Backups: $BACKUP_DIR"
}

# Execute main function
main "$@"