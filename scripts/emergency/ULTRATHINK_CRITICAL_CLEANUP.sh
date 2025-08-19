#!/bin/bash
# ULTRATHINK CRITICAL CLEANUP SCRIPT
# Created: 2025-08-19 by Elite System Reorganizer
# Mission: Fix critical infrastructure violations and restore order
# ==============================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/opt/sutazaiapp/backups/emergency_${TIMESTAMP}"
LOG_FILE="/opt/sutazaiapp/logs/cleanup_${TIMESTAMP}.log"

# Initialize logging
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ULTRATHINK CRITICAL CLEANUP INITIATED${NC}"
echo -e "${BLUE}Timestamp: ${TIMESTAMP}${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to log with color
log() {
    local level=$1
    shift
    local message="$@"
    case $level in
        ERROR) echo -e "${RED}[ERROR] ${message}${NC}" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS] ${message}${NC}" ;;
        WARNING) echo -e "${YELLOW}[WARNING] ${message}${NC}" ;;
        INFO) echo -e "${BLUE}[INFO] ${message}${NC}" ;;
    esac
}

# Create backup directory
log INFO "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# ==============================================================
# PHASE 1: EMERGENCY DOCKER SHUTDOWN
# ==============================================================
log INFO "PHASE 1: Emergency Docker Shutdown"

# Stop all non-essential containers
log WARNING "Stopping all non-sutazai containers..."
docker ps -q | while read container; do
    name=$(docker inspect --format='{{.Name}}' "$container" | sed 's/^\/\///')
    if [[ ! "$name" =~ ^sutazai- ]] && [[ "$name" != "portainer" ]]; then
        log INFO "Stopping container: $name"
        docker stop "$container" || log ERROR "Failed to stop $name"
    fi
done

# ==============================================================
# PHASE 2: DOCKER INFRASTRUCTURE REPAIR
# ==============================================================
log INFO "PHASE 2: Docker Infrastructure Repair"

# Backup existing docker-compose.yml
if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
    log INFO "Backing up root docker-compose.yml..."
    cp "/opt/sutazaiapp/docker-compose.yml" "$BACKUP_DIR/docker-compose.yml.backup"
fi

# Create sutazai-network if missing
if ! docker network ls | grep -q "sutazai-network"; then
    log INFO "Creating sutazai-network..."
    docker network create sutazai-network
else
    log SUCCESS "sutazai-network already exists"
fi

# Use the consolidated docker-compose
if [ -f "/opt/sutazaiapp/docker/docker-compose.consolidated.yml" ]; then
    log INFO "Installing consolidated docker-compose as primary..."
    cp "/opt/sutazaiapp/docker/docker-compose.consolidated.yml" "/opt/sutazaiapp/docker-compose.yml"
    log SUCCESS "Docker infrastructure configuration restored"
else
    log ERROR "Consolidated docker-compose not found!"
fi

# ==============================================================
# PHASE 3: CHANGELOG CLEANUP (179 files!)
# ==============================================================
log INFO "PHASE 3: CHANGELOG Cleanup"

# Find and backup all CHANGELOG.md files
log INFO "Backing up all CHANGELOG.md files..."
mkdir -p "$BACKUP_DIR/changelogs"
find /opt/sutazaiapp -name "CHANGELOG.md" -type f | while read file; do
    rel_path="${file#/opt/sutazaiapp/}"
    backup_path="$BACKUP_DIR/changelogs/$rel_path"
    mkdir -p "$(dirname "$backup_path")"
    cp "$file" "$backup_path"
done

# Keep only essential CHANGELOG.md files
KEEP_CHANGELOGS=(
    "/opt/sutazaiapp/CHANGELOG.md"
    "/opt/sutazaiapp/IMPORTANT/CHANGELOG.md"
    "/opt/sutazaiapp/docker/CHANGELOG.md"
    "/opt/sutazaiapp/backend/CHANGELOG.md"
    "/opt/sutazaiapp/frontend/CHANGELOG.md"
)

log INFO "Removing non-essential CHANGELOG.md files..."
find /opt/sutazaiapp -name "CHANGELOG.md" -type f | while read file; do
    keep=false
    for keeper in "${KEEP_CHANGELOGS[@]}"; do
        if [ "$file" = "$keeper" ]; then
            keep=true
            break
        fi
    done
    if [ "$keep" = false ]; then
        rm -f "$file"
        log INFO "Removed: $file"
    fi
done

# ==============================================================
# PHASE 4: REMOVE MOCK/FAKE/STUB FILES (Rule 1 Violation)
# ==============================================================
log INFO "PHASE 4: Remove Mock/Fake/Stub Files"

# Backup and remove mock/fake/stub files
mkdir -p "$BACKUP_DIR/mocks"
find /opt/sutazaiapp -type f \( -name "*mock*" -o -name "*fake*" -o -name "*stub*" \) \
    -not -path "*/node_modules/*" \
    -not -path "*/.venv/*" \
    -not -path "*/.venvs/*" | while read file; do
    rel_path="${file#/opt/sutazaiapp/}"
    backup_path="$BACKUP_DIR/mocks/$rel_path"
    mkdir -p "$(dirname "$backup_path")"
    cp "$file" "$backup_path"
    rm -f "$file"
    log INFO "Removed mock/fake/stub: $file"
done

# ==============================================================
# PHASE 5: CLEAN ROOT DIRECTORY
# ==============================================================
log INFO "PHASE 5: Clean Root Directory"

# Move .md files from root to docs
mkdir -p /opt/sutazaiapp/docs/relocated
for file in /opt/sutazaiapp/*.md; do
    if [ -f "$file" ] && [ "$file" != "/opt/sutazaiapp/CLAUDE.md" ] && [ "$file" != "/opt/sutazaiapp/AGENTS.md" ]; then
        filename=$(basename "$file")
        mv "$file" "/opt/sutazaiapp/docs/relocated/$filename"
        log INFO "Moved to docs: $filename"
    fi
done

# ==============================================================
# PHASE 6: START CRITICAL INFRASTRUCTURE
# ==============================================================
log INFO "PHASE 6: Start Critical Infrastructure"

cd /opt/sutazaiapp

# Start only essential services first
log INFO "Starting essential services..."
docker-compose up -d postgres redis neo4j ollama

# Wait for databases to be ready
log INFO "Waiting for databases to initialize..."
sleep 30

# Start remaining services
log INFO "Starting remaining services..."
docker-compose up -d

# ==============================================================
# PHASE 7: VALIDATION
# ==============================================================
log INFO "PHASE 7: Validation"

# Check running containers
running_count=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
log INFO "Running sutazai containers: $running_count"

# List all running services
docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ==============================================================
# SUMMARY
# ==============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CLEANUP COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "- Backup location: $BACKUP_DIR"
echo "- Log file: $LOG_FILE"
echo "- Running containers: $running_count"
echo ""
echo "Next steps:"
echo "1. Verify all services are healthy: docker ps"
echo "2. Check API endpoints: curl http://localhost:10010/health"
echo "3. Review cleanup log: less $LOG_FILE"
echo ""
log SUCCESS "ULTRATHINK cleanup completed successfully"