#!/bin/bash

# Security Migration Script: Transition to Non-Root Containers
# This script safely migrates Docker containers from root to non-root users

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/opt/sutazaiapp/backups/security-migration-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/opt/sutazaiapp/logs/security-migration-$(date +%Y%m%d-%H%M%S).log"

# Create directories
mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}✓ $1${NC}"
}

# Warning message
warning() {
    log "${YELLOW}⚠ $1${NC}"
}

# Check if running as root (required for some operations)
if [[ $EUID -ne 0 ]]; then
   warning "This script should be run with sudo for volume permission fixes"
fi

log "========================================="
log "Security Migration: Non-Root Containers"
log "Started at: $(date)"
log "========================================="

# Step 1: Pre-migration audit
log "\n${GREEN}Step 1: Pre-Migration Security Audit${NC}"
log "Checking current container users..."

declare -A CONTAINER_USERS
for container in $(docker ps --format "{{.Names}}"); do
    user=$(docker exec "$container" whoami 2>/dev/null || echo "ERROR")
    CONTAINER_USERS["$container"]=$user
    if [ "$user" = "root" ]; then
        log "  ${RED}✗${NC} $container: running as $user"
    else
        log "  ${GREEN}✓${NC} $container: running as $user"
    fi
done

# Step 2: Backup current configuration
log "\n${GREEN}Step 2: Backing Up Current Configuration${NC}"
cp /opt/sutazaiapp/docker-compose.yml "$BACKUP_DIR/docker-compose.yml.backup"
success "Backed up docker-compose.yml"

# Backup Dockerfiles
find /opt/sutazaiapp -name "Dockerfile" -exec cp --parents {} "$BACKUP_DIR" \; 2>/dev/null
success "Backed up Dockerfiles"

# Step 3: Build secure images
log "\n${GREEN}Step 3: Building Secure Docker Images${NC}"

# Build secure PostgreSQL
if [ -f "/opt/sutazaiapp/docker/postgres-secure/Dockerfile" ]; then
    log "Building secure PostgreSQL image..."
    docker build -t sutazai-postgres-secure:latest /opt/sutazaiapp/docker/postgres-secure/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-postgres-secure:latest"
fi

# Build secure Redis
if [ -f "/opt/sutazaiapp/docker/redis-secure/Dockerfile" ]; then
    log "Building secure Redis image..."
    docker build -t sutazai-redis-secure:latest /opt/sutazaiapp/docker/redis-secure/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-redis-secure:latest"
fi

# Build secure Ollama
if [ -f "/opt/sutazaiapp/docker/ollama-secure/Dockerfile" ]; then
    log "Building secure Ollama image..."
    docker build -t sutazai-ollama-secure:latest /opt/sutazaiapp/docker/ollama-secure/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-ollama-secure:latest"
fi

# Build secure ChromaDB
if [ -f "/opt/sutazaiapp/docker/chromadb-secure/Dockerfile" ]; then
    log "Building secure ChromaDB image..."
    docker build -t sutazai-chromadb-secure:latest /opt/sutazaiapp/docker/chromadb-secure/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-chromadb-secure:latest"
fi

# Build secure Qdrant
if [ -f "/opt/sutazaiapp/docker/qdrant-secure/Dockerfile" ]; then
    log "Building secure Qdrant image..."
    docker build -t sutazai-qdrant-secure:latest /opt/sutazaiapp/docker/qdrant-secure/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-qdrant-secure:latest"
fi

# Build secure AI Agent Orchestrator
if [ -f "/opt/sutazaiapp/agents/ai_agent_orchestrator/Dockerfile.secure" ]; then
    log "Building secure AI Agent Orchestrator image..."
    docker build -t sutazai-ai-agent-orchestrator-secure:latest \
        -f /opt/sutazaiapp/agents/ai_agent_orchestrator/Dockerfile.secure \
        /opt/sutazaiapp/agents/ai_agent_orchestrator/ >> "$LOG_FILE" 2>&1
    success "Built sutazai-ai-agent-orchestrator-secure:latest"
fi

# Step 4: Fix volume permissions
log "\n${GREEN}Step 4: Fixing Volume Permissions${NC}"

# Function to fix volume permissions
fix_volume_permissions() {
    local container=$1
    local volume=$2
    local user=$3
    local group=$4
    
    log "Fixing permissions for $volume (user: $user:$group)..."
    
    # Create temporary container to fix permissions
    docker run --rm -v "$volume:/data" alpine:latest \
        sh -c "chown -R $user:$group /data" >> "$LOG_FILE" 2>&1
    
    success "Fixed permissions for $volume"
}

# Fix known volumes
fix_volume_permissions "postgres" "postgres_data" "70" "70"
fix_volume_permissions "redis" "redis_data" "999" "999"
fix_volume_permissions "neo4j" "neo4j_data" "7474" "7474"
fix_volume_permissions "ollama" "ollama_models" "1002" "1002"
fix_volume_permissions "chromadb" "chromadb_data" "1003" "1003"
fix_volume_permissions "qdrant" "qdrant_storage" "1004" "1004"
fix_volume_permissions "rabbitmq" "rabbitmq_data" "999" "999"

# Step 5: Migration strategy selection
log "\n${GREEN}Step 5: Migration Strategy${NC}"
echo ""
echo "Select migration strategy:"
echo "1) Gradual migration (one service at a time)"
echo "2) Full migration (all services at once)"
echo "3) Test mode (dry run only)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        log "Selected: Gradual migration"
        MIGRATION_MODE="gradual"
        ;;
    2)
        log "Selected: Full migration"
        MIGRATION_MODE="full"
        ;;
    3)
        log "Selected: Test mode"
        MIGRATION_MODE="test"
        ;;
    *)
        error_exit "Invalid choice"
        ;;
esac

# Step 6: Execute migration
log "\n${GREEN}Step 6: Executing Migration${NC}"

if [ "$MIGRATION_MODE" = "test" ]; then
    log "Running in test mode - no changes will be made"
    log "\nWould migrate the following services:"
    for container in "${!CONTAINER_USERS[@]}"; do
        if [ "${CONTAINER_USERS[$container]}" = "root" ]; then
            log "  - $container"
        fi
    done
    success "Test run completed"
    exit 0
fi

if [ "$MIGRATION_MODE" = "gradual" ]; then
    # Migrate one service at a time
    for service in postgres redis ollama chromadb qdrant rabbitmq ai-agent-orchestrator; do
        container="sutazai-$service"
        
        if [ "${CONTAINER_USERS[$container]}" = "root" ]; then
            log "\nMigrating $service..."
            
            # Stop the container
            docker stop "$container" >> "$LOG_FILE" 2>&1
            success "Stopped $container"
            
            # Remove the container
            docker rm "$container" >> "$LOG_FILE" 2>&1
            success "Removed $container"
            
            # Start with secure configuration
            docker-compose -f /opt/sutazaiapp/docker-compose.secure.yml up -d "$service" >> "$LOG_FILE" 2>&1
            success "Started secure $service"
            
            # Wait and verify
            sleep 10
            
            # Check if service is healthy
            if docker ps | grep -q "$container"; then
                new_user=$(docker exec "$container" whoami 2>/dev/null || echo "ERROR")
                if [ "$new_user" != "root" ] && [ "$new_user" != "ERROR" ]; then
                    success "$service migrated successfully (now running as $new_user)"
                else
                    warning "$service migration may have issues (user: $new_user)"
                fi
            else
                warning "$service container not running after migration"
            fi
            
            # Ask to continue
            read -p "Continue with next service? (y/n): " continue_choice
            if [ "$continue_choice" != "y" ]; then
                log "Migration paused by user"
                break
            fi
        fi
    done
    
elif [ "$MIGRATION_MODE" = "full" ]; then
    log "Performing full migration..."
    
    # Stop all services
    log "Stopping all services..."
    docker-compose down >> "$LOG_FILE" 2>&1
    success "All services stopped"
    
    # Start with secure configuration
    log "Starting services with secure configuration..."
    docker-compose -f /opt/sutazaiapp/docker-compose.secure.yml up -d >> "$LOG_FILE" 2>&1
    success "Services started with secure configuration"
fi

# Step 7: Post-migration validation
log "\n${GREEN}Step 7: Post-Migration Validation${NC}"
sleep 15  # Wait for services to stabilize

log "Validating container users..."
MIGRATION_SUCCESS=true

for container in $(docker ps --format "{{.Names}}"); do
    user=$(docker exec "$container" whoami 2>/dev/null || echo "ERROR")
    
    if [ "$user" = "root" ]; then
        log "  ${RED}✗${NC} $container: still running as root"
        MIGRATION_SUCCESS=false
    elif [ "$user" = "ERROR" ]; then
        log "  ${YELLOW}⚠${NC} $container: unable to verify user"
    else
        log "  ${GREEN}✓${NC} $container: running as $user"
    fi
done

# Check service health
log "\nChecking service health..."
for container in $(docker ps --format "{{.Names}}"); do
    health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no healthcheck")
    
    if [ "$health" = "healthy" ]; then
        log "  ${GREEN}✓${NC} $container: healthy"
    elif [ "$health" = "no healthcheck" ]; then
        log "  ${YELLOW}⚠${NC} $container: no healthcheck defined"
    else
        log "  ${RED}✗${NC} $container: $health"
        MIGRATION_SUCCESS=false
    fi
done

# Step 8: Generate report
log "\n${GREEN}Step 8: Migration Report${NC}"

REPORT_FILE="$BACKUP_DIR/migration-report.txt"
{
    echo "Security Migration Report"
    echo "========================="
    echo "Date: $(date)"
    echo ""
    echo "Pre-Migration Status:"
    for container in "${!CONTAINER_USERS[@]}"; do
        echo "  $container: ${CONTAINER_USERS[$container]}"
    done
    echo ""
    echo "Post-Migration Status:"
    for container in $(docker ps --format "{{.Names}}"); do
        user=$(docker exec "$container" whoami 2>/dev/null || echo "ERROR")
        echo "  $container: $user"
    done
    echo ""
    echo "Migration Success: $MIGRATION_SUCCESS"
} > "$REPORT_FILE"

success "Migration report saved to $REPORT_FILE"

# Step 9: Rollback instructions
if [ "$MIGRATION_SUCCESS" = false ]; then
    warning "\nMigration encountered issues. To rollback:"
    log "  1. docker-compose down"
    log "  2. cp $BACKUP_DIR/docker-compose.yml.backup /opt/sutazaiapp/docker-compose.yml"
    log "  3. docker-compose up -d"
else
    success "\nMigration completed successfully!"
    log "\nNext steps:"
    log "  1. Monitor services for 24 hours"
    log "  2. Update documentation"
    log "  3. Remove backup after confirmation: rm -rf $BACKUP_DIR"
fi

log "\n========================================="
log "Migration completed at: $(date)"
log "Log file: $LOG_FILE"
log "========================================="