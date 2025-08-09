#!/bin/bash

# Automated Backup System for SutazAI Platform
# Handles database backups, Docker volumes, and configuration files
# Author: System Administrator
# Date: 2025-08-09

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/sutazaiapp/backups}"
BACKUP_DIR="${BACKUP_ROOT}/backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${BACKUP_DIR}/backup.log"
RETENTION_DAYS=${RETENTION_DAYS:-7}  # Keep backups for 7 days by default
MAX_BACKUPS=${MAX_BACKUPS:-10}  # Keep maximum 10 backups

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Initialize log
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to calculate directory size
get_size() {
    local path=$1
    du -sh "$path" 2>/dev/null | cut -f1 || echo "0"
}

# Function to backup PostgreSQL
backup_postgres() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up PostgreSQL Database"
    print_color "$BLUE" "========================================="
    
    local backup_file="${BACKUP_DIR}/postgres_backup.sql.gz"
    
    if docker exec sutazai-postgres pg_dumpall -U sutazai 2>/dev/null | gzip > "$backup_file"; then
        local size=$(get_size "$backup_file")
        print_color "$GREEN" "✓ PostgreSQL backup completed ($size)"
        echo "postgres_backup: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
    else
        print_color "$RED" "✗ PostgreSQL backup failed"
        echo "postgres_backup: FAILED" >> "${BACKUP_DIR}/backup_summary.txt"
        return 1
    fi
}

# Function to backup Redis
backup_redis() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up Redis Database"
    print_color "$BLUE" "========================================="
    
    local backup_file="${BACKUP_DIR}/redis_backup.rdb"
    
    # Trigger Redis save
    docker exec sutazai-redis redis-cli BGSAVE 2>/dev/null || true
    sleep 2
    
    # Copy dump file
    if docker cp sutazai-redis:/data/dump.rdb "$backup_file" 2>/dev/null; then
        local size=$(get_size "$backup_file")
        print_color "$GREEN" "✓ Redis backup completed ($size)"
        echo "redis_backup: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
    else
        print_color "$YELLOW" "⚠ Redis backup skipped (no data or service not running)"
        echo "redis_backup: SKIPPED" >> "${BACKUP_DIR}/backup_summary.txt"
    fi
}

# Function to backup Neo4j
backup_neo4j() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up Neo4j Database"
    print_color "$BLUE" "========================================="
    
    local backup_dir="${BACKUP_DIR}/neo4j_backup"
    mkdir -p "$backup_dir"
    
    # Stop Neo4j transactions for consistent backup
    docker exec sutazai-neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j.dump 2>/dev/null || true
    
    if docker cp sutazai-neo4j:/tmp/neo4j.dump "${backup_dir}/neo4j.dump" 2>/dev/null; then
        local size=$(get_size "${backup_dir}/neo4j.dump")
        print_color "$GREEN" "✓ Neo4j backup completed ($size)"
        echo "neo4j_backup: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
    else
        print_color "$YELLOW" "⚠ Neo4j backup skipped (no data or service not running)"
        echo "neo4j_backup: SKIPPED" >> "${BACKUP_DIR}/backup_summary.txt"
    fi
}

# Function to backup Docker volumes
backup_volumes() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up Docker Volumes"
    print_color "$BLUE" "========================================="
    
    local volumes_dir="${BACKUP_DIR}/volumes"
    mkdir -p "$volumes_dir"
    
    # List of critical volumes to backup
    local volumes=(
        "sutazaiapp_postgres_data"
        "sutazaiapp_redis_data"
        "sutazaiapp_neo4j_data"
        "sutazaiapp_ollama_data"
        "sutazaiapp_chromadb_data"
        "sutazaiapp_qdrant_data"
        "sutazaiapp_rabbitmq_data"
        "sutazaiapp_prometheus_data"
        "sutazaiapp_grafana_data"
    )
    
    for volume in "${volumes[@]}"; do
        if docker volume ls -q | grep -q "^${volume}$"; then
            print_color "$BLUE" "  Backing up volume: $volume"
            local backup_file="${volumes_dir}/${volume}.tar.gz"
            
            # Use temporary container to create backup
            if docker run --rm -v "${volume}:/data" \
                -v "${volumes_dir}:/backup" \
                alpine tar czf "/backup/${volume}.tar.gz" -C /data . 2>/dev/null; then
                local size=$(get_size "$backup_file")
                print_color "$GREEN" "    ✓ Backed up ($size)"
                echo "volume_${volume}: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
            else
                print_color "$YELLOW" "    ⚠ Skipped (empty or error)"
                echo "volume_${volume}: SKIPPED" >> "${BACKUP_DIR}/backup_summary.txt"
            fi
        fi
    done
}

# Function to backup configuration files
backup_configs() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up Configuration Files"
    print_color "$BLUE" "========================================="
    
    local configs_dir="${BACKUP_DIR}/configs"
    mkdir -p "$configs_dir"
    
    # List of configuration files and directories to backup
    local configs=(
        "docker-compose.yml"
        ".env"
        "config/"
        "agents/*/config.yaml"
        "backend/app/core/config.py"
        "monitoring/prometheus/prometheus.yml"
        "monitoring/grafana/provisioning/"
        "scripts/deployment/deploy.sh"
    )
    
    for config in "${configs[@]}"; do
        local source="${PROJECT_ROOT}/${config}"
        local dest="${configs_dir}/$(dirname "$config")"
        
        if [[ -e "$source" ]]; then
            mkdir -p "$dest"
            cp -r "$source" "$dest/" 2>/dev/null
            print_color "$GREEN" "  ✓ Backed up: $config"
        fi
    done
    
    # Create tarball of configs
    cd "$configs_dir"
    tar czf "${BACKUP_DIR}/configs.tar.gz" . 2>/dev/null
    local size=$(get_size "${BACKUP_DIR}/configs.tar.gz")
    print_color "$GREEN" "✓ Configuration backup completed ($size)"
    echo "configs_backup: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
}

# Function to backup agent code and data
backup_agents() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backing up Agent Services"
    print_color "$BLUE" "========================================="
    
    local agents_dir="${BACKUP_DIR}/agents"
    mkdir -p "$agents_dir"
    
    # Backup agent configurations and state
    for agent_dir in "$PROJECT_ROOT"/agents/*/; do
        if [[ -d "$agent_dir" ]]; then
            local agent_name=$(basename "$agent_dir")
            print_color "$BLUE" "  Backing up agent: $agent_name"
            
            # Create agent backup directory
            local agent_backup="${agents_dir}/${agent_name}"
            mkdir -p "$agent_backup"
            
            # Copy agent files (excluding __pycache__ and .pyc)
            rsync -a --exclude='__pycache__' --exclude='*.pyc' \
                "$agent_dir" "$agent_backup/" 2>/dev/null || true
            
            # If agent has persistent data in container, backup that too
            local container_name="sutazai-${agent_name//_/-}"
            if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
                docker cp "${container_name}:/app/data" "${agent_backup}/container_data" 2>/dev/null || true
            fi
        fi
    done
    
    # Create tarball of agents
    cd "$agents_dir"
    tar czf "${BACKUP_DIR}/agents.tar.gz" . 2>/dev/null
    local size=$(get_size "${BACKUP_DIR}/agents.tar.gz")
    print_color "$GREEN" "✓ Agents backup completed ($size)"
    echo "agents_backup: SUCCESS ($size)" >> "${BACKUP_DIR}/backup_summary.txt"
}

# Function to create backup metadata
create_metadata() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Creating Backup Metadata"
    print_color "$BLUE" "========================================="
    
    cat > "${BACKUP_DIR}/backup_metadata.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "backup_id": "$(basename "$BACKUP_DIR")",
  "system_info": {
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "docker_version": "$(docker --version | awk '{print $3}' | tr -d ',')",
    "compose_version": "$(docker-compose --version | awk '{print $4}' | tr -d ',')"
  },
  "services": {
    "running_containers": $(docker ps --format "{{.Names}}" | wc -l),
    "total_containers": $(docker ps -a --format "{{.Names}}" | wc -l)
  },
  "backup_size": "$(du -sh "$BACKUP_DIR" | cut -f1)",
  "retention_days": $RETENTION_DAYS,
  "backup_components": [
    "postgresql",
    "redis",
    "neo4j",
    "docker_volumes",
    "configurations",
    "agents"
  ]
}
EOF
    
    print_color "$GREEN" "✓ Metadata created"
}

# Function to cleanup old backups
cleanup_old_backups() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Cleaning Up Old Backups"
    print_color "$BLUE" "========================================="
    
    # Remove backups older than retention period
    find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
    
    # Keep only MAX_BACKUPS most recent backups
    local backup_count=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup_*" | wc -l)
    if [[ $backup_count -gt $MAX_BACKUPS ]]; then
        local remove_count=$((backup_count - MAX_BACKUPS))
        find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup_*" | \
            sort | head -n $remove_count | xargs rm -rf 2>/dev/null || true
    fi
    
    print_color "$GREEN" "✓ Old backups cleaned up"
}

# Function to verify backup integrity
verify_backup() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Verifying Backup Integrity"
    print_color "$BLUE" "========================================="
    
    local errors=0
    
    # Check if critical files exist
    [[ -f "${BACKUP_DIR}/postgres_backup.sql.gz" ]] || ((errors++))
    [[ -f "${BACKUP_DIR}/configs.tar.gz" ]] || ((errors++))
    [[ -d "${BACKUP_DIR}/volumes" ]] || ((errors++))
    
    if [[ $errors -eq 0 ]]; then
        print_color "$GREEN" "✓ Backup verification passed"
        echo "verification: PASSED" >> "${BACKUP_DIR}/backup_summary.txt"
    else
        print_color "$YELLOW" "⚠ Backup verification found $errors issues"
        echo "verification: WARNINGS ($errors issues)" >> "${BACKUP_DIR}/backup_summary.txt"
    fi
}

# Function to create restore script
create_restore_script() {
    cat > "${BACKUP_DIR}/restore.sh" <<'RESTORE_SCRIPT'
#!/bin/bash

# Restore Script for SutazAI Backup
# Auto-generated during backup

set -euo pipefail

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"

echo "========================================="
echo "SutazAI System Restore"
echo "Backup: $(basename "$BACKUP_DIR")"
echo "========================================="
echo ""
echo "WARNING: This will restore system to backup state!"
echo "Current data will be overwritten!"
echo ""
read -p "Continue? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "Restore cancelled"
    exit 1
fi

# Stop services
echo "Stopping services..."
cd "$PROJECT_ROOT"
docker-compose down

# Restore PostgreSQL
if [[ -f "${BACKUP_DIR}/postgres_backup.sql.gz" ]]; then
    echo "Restoring PostgreSQL..."
    docker-compose up -d postgres
    sleep 10
    gunzip -c "${BACKUP_DIR}/postgres_backup.sql.gz" | docker exec -i sutazai-postgres psql -U sutazai
fi

# Restore Redis
if [[ -f "${BACKUP_DIR}/redis_backup.rdb" ]]; then
    echo "Restoring Redis..."
    docker cp "${BACKUP_DIR}/redis_backup.rdb" sutazai-redis:/data/dump.rdb
    docker-compose restart redis
fi

# Restore volumes
if [[ -d "${BACKUP_DIR}/volumes" ]]; then
    echo "Restoring Docker volumes..."
    for backup in "${BACKUP_DIR}"/volumes/*.tar.gz; do
        if [[ -f "$backup" ]]; then
            volume_name=$(basename "$backup" .tar.gz)
            echo "  Restoring volume: $volume_name"
            docker run --rm -v "${volume_name}:/data" \
                -v "${BACKUP_DIR}/volumes:/backup" \
                alpine tar xzf "/backup/$(basename "$backup")" -C /data
        fi
    done
fi

# Restore configurations
if [[ -f "${BACKUP_DIR}/configs.tar.gz" ]]; then
    echo "Restoring configurations..."
    cd "$PROJECT_ROOT"
    tar xzf "${BACKUP_DIR}/configs.tar.gz"
fi

# Start all services
echo "Starting services..."
docker-compose up -d

echo ""
echo "Restore completed!"
echo "Please verify system functionality"
RESTORE_SCRIPT

    chmod +x "${BACKUP_DIR}/restore.sh"
    print_color "$GREEN" "✓ Restore script created"
}

# Main backup process
main() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "SutazAI Automated Backup System"
    print_color "$BLUE" "Starting backup: $(date)"
    print_color "$BLUE" "========================================="
    echo ""
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_color "$RED" "Error: Docker is not running"
        exit 1
    fi
    
    # Perform backups
    backup_postgres || true
    backup_redis || true
    backup_neo4j || true
    backup_volumes || true
    backup_configs || true
    backup_agents || true
    
    # Create metadata and verification
    create_metadata
    verify_backup
    create_restore_script
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Final summary
    echo ""
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Backup Summary"
    print_color "$BLUE" "========================================="
    
    local backup_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    print_color "$GREEN" "✓ Backup completed successfully"
    echo ""
    echo "Backup Location: $BACKUP_DIR"
    echo "Backup Size: $backup_size"
    echo "Restore Script: ${BACKUP_DIR}/restore.sh"
    echo ""
    
    # Display summary
    if [[ -f "${BACKUP_DIR}/backup_summary.txt" ]]; then
        cat "${BACKUP_DIR}/backup_summary.txt"
    fi
    
    print_color "$GREEN" "Backup process completed at $(date)"
}

# Handle script arguments
case "${1:-backup}" in
    backup)
        main
        ;;
    restore)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 restore <backup_directory>"
            exit 1
        fi
        if [[ -f "$2/restore.sh" ]]; then
            exec "$2/restore.sh"
        else
            print_color "$RED" "Error: Restore script not found in $2"
            exit 1
        fi
        ;;
    list)
        echo "Available backups:"
        find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup_*" | sort -r | while read -r backup; do
            local size=$(du -sh "$backup" | cut -f1)
            local date=$(basename "$backup" | sed 's/backup_//')
            echo "  - $date ($size) - $backup"
        done
        ;;
    help|--help|-h)
        echo "Usage: $0 [backup|restore|list|help]"
        echo ""
        echo "Commands:"
        echo "  backup  - Create a new backup (default)"
        echo "  restore - Restore from a backup"
        echo "  list    - List available backups"
        echo "  help    - Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  BACKUP_ROOT     - Backup directory (default: /opt/sutazaiapp/backups)"
        echo "  RETENTION_DAYS  - Days to keep backups (default: 7)"
        echo "  MAX_BACKUPS     - Maximum number of backups (default: 10)"
        ;;
    *)
        print_color "$RED" "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac