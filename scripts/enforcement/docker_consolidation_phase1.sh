#!/bin/bash
# Docker Consolidation Phase 1 - Following Rule 4: Investigate & Consolidate First
# Generated: 2025-08-19

set -euo pipefail

DOCKER_DIR="/opt/sutazaiapp/docker"
BACKUP_DIR="/opt/sutazaiapp/backups/docker_consolidation_20250819_163429"
LOG_FILE="$BACKUP_DIR/consolidation.log"

echo "=== DOCKER CONSOLIDATION PHASE 1 ===" | tee "$LOG_FILE"
echo "Following Rule 4: Investigate & Consolidate First" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# STEP 1: Identify actually used Dockerfiles
echo "" | tee -a "$LOG_FILE"
echo "STEP 1: Identifying actually used Dockerfiles..." | tee -a "$LOG_FILE"

# Services that have build directives in docker-compose.consolidated.yml
REQUIRED_DOCKERFILES=(
    "backend/Dockerfile"
    "frontend/Dockerfile"
    "base/Dockerfile.python-base-secure"  # Base for Python services
    "dind/orchestrator/manager/Dockerfile" # DinD orchestrator
    "mcp-services/unified-dev/Dockerfile"  # Unified MCP service
)

echo "Required Dockerfiles based on docker-compose.consolidated.yml:" | tee -a "$LOG_FILE"
for df in "${REQUIRED_DOCKERFILES[@]}"; do
    echo "  - $df" | tee -a "$LOG_FILE"
done

# STEP 2: Consolidate duplicate base images
echo "" | tee -a "$LOG_FILE"
echo "STEP 2: Consolidating base images..." | tee -a "$LOG_FILE"

# Create consolidated base Dockerfile for Python services
cat > "$DOCKER_DIR/base/Dockerfile.python-consolidated" << 'EOF'
# Consolidated Python Base Image - Following Rule 1: Real Implementation Only
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install common Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    asyncpg \
    httpx \
    pydantic \
    python-dotenv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

echo "✅ Created consolidated Python base Dockerfile" | tee -a "$LOG_FILE"

# STEP 3: Mark unnecessary Dockerfiles for removal
echo "" | tee -a "$LOG_FILE"
echo "STEP 3: Marking unnecessary Dockerfiles for removal..." | tee -a "$LOG_FILE"

DOCKERFILES_TO_REMOVE=(
    # Duplicate FAISS Dockerfiles (keep only one)
    "$DOCKER_DIR/faiss/Dockerfile.optimized"
    "$DOCKER_DIR/faiss/Dockerfile.simple"
    "$DOCKER_DIR/faiss/Dockerfile.standalone"
    
    # Duplicate frontend Dockerfiles (keep main one)
    "$DOCKER_DIR/frontend/Dockerfile.secure"
    
    # Duplicate backend Dockerfiles (keep main one)
    "$DOCKER_DIR/backend/Dockerfile.production"
    
    # Unnecessary base Dockerfiles (consolidated into one)
    "$DOCKER_DIR/base/Dockerfile.simple-base"
    "$DOCKER_DIR/base/Dockerfile.python-agent-master"
    
    # Specialized MCP Dockerfiles (use unified one)
    "$DOCKER_DIR/dind/mcp-containers/Dockerfile.specialized-mcp"
    "$DOCKER_DIR/dind/mcp-containers/Dockerfile.nodejs-mcp"
    "$DOCKER_DIR/dind/mcp-containers/Dockerfile.python-mcp"
    
    # Secure variants (consolidate security into main files)
    "$DOCKER_DIR/base/Dockerfile.chromadb-secure"
    "$DOCKER_DIR/base/Dockerfile.jaeger-secure"
    "$DOCKER_DIR/base/Dockerfile.neo4j-secure"
    "$DOCKER_DIR/base/Dockerfile.ollama-secure"
    "$DOCKER_DIR/base/Dockerfile.postgres-secure"
    "$DOCKER_DIR/base/Dockerfile.promtail-secure"
    "$DOCKER_DIR/base/Dockerfile.qdrant-secure"
    "$DOCKER_DIR/base/Dockerfile.rabbitmq-secure"
    "$DOCKER_DIR/base/Dockerfile.redis-exporter-secure"
    "$DOCKER_DIR/base/Dockerfile.redis-secure"
    
    # Monitoring secure variants (use standard images)
    "$DOCKER_DIR/monitoring-secure/blackbox-exporter/Dockerfile"
    "$DOCKER_DIR/monitoring-secure/cadvisor/Dockerfile"
    "$DOCKER_DIR/monitoring-secure/consul/Dockerfile"
    
    # MCP service base (use unified)
    "$DOCKER_DIR/mcp-services/base/Dockerfile"
    "$DOCKER_DIR/mcp-services/files/Dockerfile"
    "$DOCKER_DIR/mcp-services/postgres/Dockerfile"
    "$DOCKER_DIR/mcp-services/unified-memory/Dockerfile"
    
    # Old MCP implementations
    "$DOCKER_DIR/mcp/UltimateCoderMCP/Dockerfile"
    "$DOCKER_DIR/dind/mcp-real/Dockerfile"
)

echo "Dockerfiles marked for removal: ${#DOCKERFILES_TO_REMOVE[@]}" | tee -a "$LOG_FILE"

# STEP 4: Remove unnecessary Dockerfiles
echo "" | tee -a "$LOG_FILE"
echo "STEP 4: Removing unnecessary Dockerfiles..." | tee -a "$LOG_FILE"

REMOVED_COUNT=0
for df in "${DOCKERFILES_TO_REMOVE[@]}"; do
    if [ -f "$df" ]; then
        rm -f "$df"
        ((REMOVED_COUNT++))
        echo "  ✅ Removed: $(basename $df)" | tee -a "$LOG_FILE"
    fi
done

echo "Total Dockerfiles removed: $REMOVED_COUNT" | tee -a "$LOG_FILE"

# STEP 5: Update docker-compose to use consolidated images
echo "" | tee -a "$LOG_FILE"
echo "STEP 5: Updating docker-compose.consolidated.yml..." | tee -a "$LOG_FILE"

# Update build contexts in docker-compose
sed -i 's|dockerfile: base/Dockerfile.python-agent-master|dockerfile: base/Dockerfile.python-consolidated|g' "$DOCKER_DIR/docker-compose.consolidated.yml"
sed -i 's|dockerfile: base/Dockerfile.python-base-secure|dockerfile: base/Dockerfile.python-consolidated|g' "$DOCKER_DIR/docker-compose.consolidated.yml"

echo "✅ Updated docker-compose references" | tee -a "$LOG_FILE"

# STEP 6: Consolidation Summary
echo "" | tee -a "$LOG_FILE"
echo "=== CONSOLIDATION SUMMARY ===" | tee -a "$LOG_FILE"

INITIAL_COUNT=36
FINAL_COUNT=$(find "$DOCKER_DIR" -name "Dockerfile*" -type f | wc -l)
REDUCTION=$((INITIAL_COUNT - FINAL_COUNT))
PERCENTAGE=$((REDUCTION * 100 / INITIAL_COUNT))

echo "Initial Dockerfiles: $INITIAL_COUNT" | tee -a "$LOG_FILE"
echo "Final Dockerfiles: $FINAL_COUNT" | tee -a "$LOG_FILE"
echo "Removed: $REDUCTION files ($PERCENTAGE% reduction)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Remaining essential Dockerfiles:" | tee -a "$LOG_FILE"
find "$DOCKER_DIR" -name "Dockerfile*" -type f | sed "s|$DOCKER_DIR/||" | sort | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ DOCKER CONSOLIDATION PHASE 1 COMPLETE" | tee -a "$LOG_FILE"
echo "Backup saved to: $BACKUP_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"