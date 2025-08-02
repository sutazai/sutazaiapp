#!/bin/bash
# SutazAI Safe System Backup Script
# Creates comprehensive backup before reorganization

set -euo pipefail

# Configuration
BACKUP_ROOT="/opt/sutazaiapp/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${BACKUP_ROOT}/reorganization_backup_${TIMESTAMP}"

# Logging
LOG_FILE="${BACKUP_DIR}/backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Create backup directory structure
create_backup_structure() {
    log "Creating backup directory structure..."
    
    mkdir -p "$BACKUP_DIR"/{
        system_state,
        docker_state,
        database,
        scripts,
        configs,
        logs,
        running_services
    }
    
    touch "$LOG_FILE"
    
    log "Backup directory created: $BACKUP_DIR"
}

# Backup running system state
backup_system_state() {
    log "Backing up system state..."
    
    # Running processes
    ps aux > "$BACKUP_DIR/system_state/processes.txt"
    
    # System resources
    free -h > "$BACKUP_DIR/system_state/memory.txt"
    df -h > "$BACKUP_DIR/system_state/disk.txt"
    
    # Network state
    netstat -tulpn > "$BACKUP_DIR/system_state/network.txt" 2>/dev/null || true
    
    # Environment variables
    env > "$BACKUP_DIR/system_state/environment.txt"
    
    log "System state backed up"
}

# Backup Docker state
backup_docker_state() {
    log "Backing up Docker state..."
    
    # Running containers
    docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" > "$BACKUP_DIR/docker_state/containers.txt"
    
    # Docker compose files
    find /opt/sutazaiapp -name "docker-compose*.yml" -exec cp {} "$BACKUP_DIR/docker_state/" \;
    
    # Docker images
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" > "$BACKUP_DIR/docker_state/images.txt"
    
    # Docker networks
    docker network ls > "$BACKUP_DIR/docker_state/networks.txt"
    
    # Docker volumes
    docker volume ls > "$BACKUP_DIR/docker_state/volumes.txt"
    
    log "Docker state backed up"
}

# Backup critical files
backup_critical_files() {
    log "Backing up critical files..."
    
    # Critical application files
    critical_files=(
        "/opt/sutazaiapp/backend/app/main.py"
        "/opt/sutazaiapp/docker-compose.minimal.yml"
        "/opt/sutazaiapp/scripts/live_logs.sh"
        "/opt/sutazaiapp/health_check.sh"
        "/opt/sutazaiapp/requirements.txt"
        "/opt/sutazaiapp/.env"
        "/opt/sutazaiapp/.env.tinyllama"
    )
    
    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            mkdir -p "$BACKUP_DIR/critical_files/$(dirname "$file")"
            cp "$file" "$BACKUP_DIR/critical_files/$file"
            log "Backed up: $file"
        else
            log "Warning: Critical file not found: $file"
        fi
    done
}

# Backup all scripts
backup_scripts() {
    log "Backing up all scripts..."
    
    # Create complete scripts backup
    cp -r /opt/sutazaiapp/scripts "$BACKUP_DIR/scripts/original"
    
    # Create index of all scripts
    find /opt/sutazaiapp/scripts -type f \( -name "*.sh" -o -name "*.py" \) > "$BACKUP_DIR/scripts/script_inventory.txt"
    
    log "Scripts backed up"
}

# Backup configurations
backup_configs() {
    log "Backing up configurations..."
    
    # Config directories
    config_dirs=(
        "/opt/sutazaiapp/config"
        "/opt/sutazaiapp/agents/configs"
        "/opt/sutazaiapp/.claude"
    )
    
    for dir in "${config_dirs[@]}"; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$BACKUP_DIR/configs/"
            log "Backed up config: $dir"
        fi
    done
}

# Test system health before backup
test_system_health() {
    log "Testing system health before backup..."
    
    # Test Docker
    if ! docker ps >/dev/null 2>&1; then
        error "Docker is not running"
        return 1
    fi
    
    # Test running services
    running_containers=$(docker ps --format "{{.Names}}" | wc -l)
    log "Found $running_containers running containers"
    
    # Test disk space
    available_space=$(df /opt/sutazaiapp | awk 'NR==2 {print $4}')
    log "Available disk space: ${available_space}KB"
    
    if [ "$available_space" -lt 1000000 ]; then  # Less than 1GB
        error "Insufficient disk space for backup"
        return 1
    fi
    
    log "System health check passed"
    return 0
}

# Create restoration script
create_restoration_script() {
    log "Creating restoration script..."
    
    cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Automatic restoration script
# Generated during backup process

set -euo pipefail

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTORE_LOG="$BACKUP_DIR/restore.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESTORE_LOG"
}

log "Starting system restoration from: $BACKUP_DIR"

# Stop all containers safely
log "Stopping containers..."
docker-compose -f /opt/sutazaiapp/docker-compose.minimal.yml down || true

# Restore critical files
log "Restoring critical files..."
cp -r "$BACKUP_DIR/critical_files/opt/sutazaiapp/"* /opt/sutazaiapp/

# Restore scripts
log "Restoring scripts..."
rm -rf /opt/sutazaiapp/scripts
cp -r "$BACKUP_DIR/scripts/original" /opt/sutazaiapp/scripts

# Restore configs
log "Restoring configs..."
cp -r "$BACKUP_DIR/configs/"* /opt/sutazaiapp/

# Restart system
log "Restarting system..."
cd /opt/sutazaiapp
docker-compose -f docker-compose.minimal.yml up -d

log "System restoration completed"
EOF
    
    chmod +x "$BACKUP_DIR/restore.sh"
    
    log "Restoration script created: $BACKUP_DIR/restore.sh"
}

# Create backup summary
create_backup_summary() {
    log "Creating backup summary..."
    
    cat > "$BACKUP_DIR/backup_summary.md" << EOF
# SutazAI System Backup Summary

**Backup Created:** $(date)
**Backup Directory:** $BACKUP_DIR
**System:** SutazAI AGI Infrastructure

## Backup Contents

### System State
- Running processes snapshot
- Memory and disk usage
- Network configuration
- Environment variables

### Docker State
- Container status and configuration
- Docker Compose files
- Images, networks, and volumes inventory

### Critical Files
- Backend main application
- Docker Compose configurations
- Monitoring scripts
- Environment files

### Complete Scripts Backup
- All shell scripts ($(find "$BACKUP_DIR/scripts/original" -name "*.sh" | wc -l) files)
- All Python scripts ($(find "$BACKUP_DIR/scripts/original" -name "*.py" | wc -l) files)

### Configurations
- Application configurations
- Agent configurations
- Claude agent settings

## Restoration
To restore the system, run:
\`\`\`bash
$BACKUP_DIR/restore.sh
\`\`\`

## Backup Verification
- Total size: $(du -sh "$BACKUP_DIR" | cut -f1)
- Files backed up: $(find "$BACKUP_DIR" -type f | wc -l)
- Integrity: $(cd "$BACKUP_DIR" && find . -type f -exec md5sum {} \; | wc -l) checksums generated

EOF
    
    # Generate checksums for integrity verification
    cd "$BACKUP_DIR"
    find . -type f -exec md5sum {} \; > checksums.md5
    
    log "Backup summary created"
}

# Main backup function
main() {
    log "Starting SutazAI system backup..."
    
    # Pre-backup health check
    if ! test_system_health; then
        error "System health check failed. Aborting backup."
        exit 1
    fi
    
    # Create backup structure
    create_backup_structure
    
    # Perform backups
    backup_system_state
    backup_docker_state
    backup_critical_files
    backup_scripts
    backup_configs
    
    # Create utilities
    create_restoration_script
    create_backup_summary
    
    log "Backup completed successfully: $BACKUP_DIR"
    log "To restore: bash $BACKUP_DIR/restore.sh"
    
    # Set permissions
    chmod -R 755 "$BACKUP_DIR"
    
    echo "✅ Backup completed: $BACKUP_DIR"
    echo "📝 Summary: $BACKUP_DIR/backup_summary.md"
    echo "🔄 Restore: $BACKUP_DIR/restore.sh"
}

# Run main function
main "$@"