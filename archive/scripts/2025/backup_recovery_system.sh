#!/bin/bash
# 💾 SutazAI Agent Infrastructure Backup & Recovery System
# Comprehensive backup and disaster recovery for all 38 AI agents

set -euo pipefail

# Configuration
PROJECT_ROOT=$(pwd)
BACKUP_ROOT="$PROJECT_ROOT/data/backups"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_LOG="$LOG_DIR/backup_$TIMESTAMP.log"
RECOVERY_LOG="$LOG_DIR/recovery_$TIMESTAMP.log"

# Backup retention (days)
RETENTION_DAILY=7
RETENTION_WEEKLY=4
RETENTION_MONTHLY=12

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Ensure directories exist
mkdir -p "$BACKUP_ROOT"/{postgres,redis,neo4j,agent_configs,agent_workspaces,ollama_models,monitoring,daily,weekly,monthly}
mkdir -p "$LOG_DIR"

# ===============================================
# 🚀 LOGGING FUNCTIONS
# ===============================================

log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${BLUE}ℹ️  [$timestamp] $message${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}✅ [$timestamp] $message${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}⚠️  [$timestamp] WARNING: $message${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}❌ [$timestamp] ERROR: $message${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
}

log_header() {
    local message="$1"
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
    echo -e "${CYAN}║ $message${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}" | tee -a "${BACKUP_LOG:-/dev/null}"
}

# ===============================================
# 🚀 DATABASE BACKUP FUNCTIONS
# ===============================================

backup_postgresql() {
    log_info "Backing up PostgreSQL database..."
    
    local backup_file="$BACKUP_ROOT/postgres/sutazai_$TIMESTAMP.sql"
    local compressed_file="$backup_file.gz"
    
    if docker compose exec -T postgres pg_dump -U sutazai -d sutazai > "$backup_file" 2>/dev/null; then
        # Compress the backup
        gzip "$backup_file"
        
        local size=$(du -h "$compressed_file" | cut -f1)
        log_success "PostgreSQL backup completed: $compressed_file ($size)"
        
        # Verify backup integrity
        if gzip -t "$compressed_file" 2>/dev/null; then
            log_success "PostgreSQL backup integrity verified"
        else
            log_error "PostgreSQL backup integrity check failed"
            return 1
        fi
    else
        log_error "PostgreSQL backup failed"
        return 1
    fi
}

backup_redis() {
    log_info "Backing up Redis database..."
    
    local backup_file="$BACKUP_ROOT/redis/redis_$TIMESTAMP.rdb"
    
    # Trigger Redis save and copy the dump
    if docker compose exec redis redis-cli BGSAVE > /dev/null 2>&1; then
        # Wait for background save to complete
        local save_status=""
        local attempts=0
        while [[ "$save_status" != "OK" ]] && [[ $attempts -lt 30 ]]; do
            sleep 2
            save_status=$(docker compose exec redis redis-cli LASTSAVE 2>/dev/null | tr -d '\r' || echo "")
            ((attempts++))
        done
        
        # Copy the RDB file
        if docker cp sutazai-redis:/data/dump.rdb "$backup_file" 2>/dev/null; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_success "Redis backup completed: $backup_file ($size)"
        else
            log_error "Failed to copy Redis dump file"
            return 1
        fi
    else
        log_error "Redis backup failed"
        return 1
    fi
}

backup_neo4j() {
    log_info "Backing up Neo4j database..."
    
    local backup_dir="$BACKUP_ROOT/neo4j/neo4j_$TIMESTAMP"
    mkdir -p "$backup_dir"
    
    # Stop Neo4j for consistent backup
    log_info "Stopping Neo4j for backup..."
    docker compose stop neo4j
    
    # Copy Neo4j data
    if docker cp sutazai-neo4j:/data "$backup_dir/" 2>/dev/null; then
        # Compress the backup
        tar -czf "$backup_dir.tar.gz" -C "$BACKUP_ROOT/neo4j" "neo4j_$TIMESTAMP"
        rm -rf "$backup_dir"
        
        local size=$(du -h "$backup_dir.tar.gz" | cut -f1)
        log_success "Neo4j backup completed: $backup_dir.tar.gz ($size)"
        
        # Restart Neo4j
        log_info "Restarting Neo4j..."
        docker compose start neo4j
        
        # Wait for Neo4j to be ready
        local attempts=0
        while ! docker compose exec neo4j cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-sutazai_neo4j_password}" "RETURN 1" > /dev/null 2>&1 && [[ $attempts -lt 30 ]]; do
            sleep 5
            ((attempts++))
        done
        
        if [[ $attempts -lt 30 ]]; then
            log_success "Neo4j restarted successfully"
        else
            log_warn "Neo4j restart may have issues"
        fi
    else
        log_error "Neo4j backup failed"
        # Restart Neo4j even if backup failed
        docker compose start neo4j
        return 1
    fi
}

# ===============================================
# 🚀 APPLICATION DATA BACKUP FUNCTIONS
# ===============================================

backup_agent_configurations() {
    log_info "Backing up agent configurations..."
    
    local backup_file="$BACKUP_ROOT/agent_configs/configs_$TIMESTAMP.tar.gz"
    
    # Backup agent configs, Dockerfiles, and compose files
    tar -czf "$backup_file" \
        --exclude='*.log' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        agents/ \
        docker-compose*.yml \
        .env* \
        config/ \
        2>/dev/null || true
    
    if [[ -f "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "Agent configurations backed up: $backup_file ($size)"
    else
        log_error "Agent configurations backup failed"
        return 1
    fi
}

backup_agent_workspaces() {
    log_info "Backing up agent workspaces..."
    
    local backup_file="$BACKUP_ROOT/agent_workspaces/workspaces_$TIMESTAMP.tar.gz"
    
    # Create workspace backup (if data exists)
    if [[ -d "./data/agent_workspaces" ]] || docker volume ls | grep -q "agent_workspaces"; then
        # Backup using docker volume
        docker run --rm \
            -v sutazaiapp_agent_workspaces:/source:ro \
            -v "$BACKUP_ROOT/agent_workspaces:/backup" \
            alpine:latest \
            tar -czf "/backup/workspaces_$TIMESTAMP.tar.gz" -C /source . 2>/dev/null || true
        
        if [[ -f "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_success "Agent workspaces backed up: $backup_file ($size)"
        else
            log_warn "Agent workspaces backup completed with no data"
        fi
    else
        log_info "No agent workspaces found to backup"
    fi
}

backup_ollama_models() {
    log_info "Backing up Ollama models..."
    
    local backup_file="$BACKUP_ROOT/ollama_models/models_$TIMESTAMP.tar.gz"
    
    # Backup Ollama models (large files, may take time)
    if docker volume ls | grep -q "ollama_data"; then
        log_info "This may take several minutes for large models..."
        
        docker run --rm \
            -v sutazaiapp_ollama_data:/source:ro \
            -v "$BACKUP_ROOT/ollama_models:/backup" \
            alpine:latest \
            tar -czf "/backup/models_$TIMESTAMP.tar.gz" -C /source . 2>/dev/null || true
        
        if [[ -f "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_success "Ollama models backed up: $backup_file ($size)"
        else
            log_warn "Ollama models backup completed with warnings"
        fi
    else
        log_info "No Ollama models found to backup"
    fi
}

backup_monitoring_data() {
    log_info "Backing up monitoring data..."
    
    local backup_file="$BACKUP_ROOT/monitoring/monitoring_$TIMESTAMP.tar.gz"
    
    # Backup Prometheus, Grafana, and Loki data
    if docker volume ls | grep -q "prometheus_data\|grafana_data\|loki_data"; then
        docker run --rm \
            -v sutazaiapp_prometheus_data:/prometheus:ro \
            -v sutazaiapp_grafana_data:/grafana:ro \
            -v sutazaiapp_loki_data:/loki:ro \
            -v "$BACKUP_ROOT/monitoring:/backup" \
            alpine:latest \
            sh -c "tar -czf /backup/monitoring_$TIMESTAMP.tar.gz /prometheus /grafana /loki 2>/dev/null || true"
        
        if [[ -f "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log_success "Monitoring data backed up: $backup_file ($size)"
        else
            log_warn "Monitoring data backup completed with warnings"
        fi
    else
        log_info "No monitoring data found to backup"
    fi
}

# ===============================================
# 🚀 BACKUP ORCHESTRATION
# ===============================================

create_full_backup() {
    log_header "🚀 CREATING FULL SYSTEM BACKUP"
    
    local backup_start=$(date +%s)
    local backup_errors=0
    
    # Create backup manifest
    local manifest_file="$BACKUP_ROOT/backup_manifest_$TIMESTAMP.json"
    cat > "$manifest_file" << EOF
{
  "backup_timestamp": "$TIMESTAMP",
  "backup_date": "$(date -Iseconds)",
  "backup_type": "full",
  "components": []
}
EOF
    
    # Database backups
    log_info "Phase 1: Database backups"
    backup_postgresql || ((backup_errors++))
    backup_redis || ((backup_errors++))
    backup_neo4j || ((backup_errors++))
    
    # Application data backups
    log_info "Phase 2: Application data backups"
    backup_agent_configurations || ((backup_errors++))
    backup_agent_workspaces || ((backup_errors++))
    
    # Optional large data backups
    log_info "Phase 3: Large data backups"
    backup_ollama_models || true  # Non-critical
    backup_monitoring_data || true  # Non-critical
    
    # Create consolidated backup
    local consolidated_backup="$BACKUP_ROOT/daily/sutazai_full_backup_$TIMESTAMP.tar.gz"
    log_info "Creating consolidated backup archive..."
    
    tar -czf "$consolidated_backup" \
        -C "$BACKUP_ROOT" \
        postgres/sutazai_$TIMESTAMP.sql.gz \
        redis/redis_$TIMESTAMP.rdb \
        neo4j/neo4j_$TIMESTAMP.tar.gz \
        agent_configs/configs_$TIMESTAMP.tar.gz \
        backup_manifest_$TIMESTAMP.json \
        2>/dev/null || true
    
    # Calculate backup statistics
    local backup_end=$(date +%s)
    local backup_duration=$((backup_end - backup_start))
    local backup_size=$(du -h "$consolidated_backup" | cut -f1)
    
    # Update manifest with results
    jq --arg duration "$backup_duration" \
       --arg size "$backup_size" \
       --arg errors "$backup_errors" \
       '.backup_duration_seconds = ($duration | tonumber) | .backup_size = $size | .errors = ($errors | tonumber)' \
       "$manifest_file" > "${manifest_file}.tmp" && mv "${manifest_file}.tmp" "$manifest_file"
    
    log_header "🎉 BACKUP COMPLETED"
    log_success "Backup duration: ${backup_duration}s"
    log_success "Backup size: $backup_size"
    log_success "Consolidated backup: $consolidated_backup"
    
    if [[ $backup_errors -gt 0 ]]; then
        log_warn "Backup completed with $backup_errors errors"
        return 1
    else
        log_success "Backup completed successfully with no errors"
        return 0
    fi
}

# ===============================================
# 🚀 RECOVERY FUNCTIONS
# ===============================================

restore_postgresql() {
    local backup_file="$1"
    
    log_info "Restoring PostgreSQL from: $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Stop dependent services
    log_info "Stopping services that depend on PostgreSQL..."
    docker compose stop backend-agi agent-registry
    
    # Restore database
    if [[ "$backup_file" == *.gz ]]; then
        if zcat "$backup_file" | docker compose exec -T postgres psql -U sutazai -d sutazai; then
            log_success "PostgreSQL restore completed"
        else
            log_error "PostgreSQL restore failed"
            return 1
        fi
    else
        if docker compose exec -T postgres psql -U sutazai -d sutazai < "$backup_file"; then
            log_success "PostgreSQL restore completed"
        else
            log_error "PostgreSQL restore failed"
            return 1
        fi
    fi
    
    # Restart dependent services
    log_info "Restarting dependent services..."
    docker compose start backend-agi agent-registry
}

restore_redis() {
    local backup_file="$1"
    
    log_info "Restoring Redis from: $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Stop Redis
    docker compose stop redis
    
    # Replace dump file
    if docker cp "$backup_file" sutazai-redis:/data/dump.rdb; then
        log_success "Redis dump file restored"
    else
        log_error "Failed to restore Redis dump file"
        return 1
    fi
    
    # Start Redis
    docker compose start redis
    
    # Wait for Redis to be ready
    local attempts=0
    while ! docker compose exec redis redis-cli ping > /dev/null 2>&1 && [[ $attempts -lt 30 ]]; do
        sleep 2
        ((attempts++))
    done
    
    if [[ $attempts -lt 30 ]]; then
        log_success "Redis restored and running"
    else
        log_error "Redis restore may have failed"
        return 1
    fi
}

restore_neo4j() {
    local backup_file="$1"
    
    log_info "Restoring Neo4j from: $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Stop Neo4j
    docker compose stop neo4j
    
    # Extract and restore data
    local temp_dir=$(mktemp -d)
    if tar -xzf "$backup_file" -C "$temp_dir"; then
        # Copy data back to container
        if docker cp "$temp_dir/neo4j_"*/data sutazai-neo4j:/; then
            log_success "Neo4j data restored"
        else
            log_error "Failed to restore Neo4j data"
            rm -rf "$temp_dir"
            return 1
        fi
    else
        log_error "Failed to extract Neo4j backup"
        rm -rf "$temp_dir"
        return 1
    fi
    
    rm -rf "$temp_dir"
    
    # Start Neo4j
    docker compose start neo4j
    
    log_success "Neo4j restore completed"
}

# ===============================================
# 🚀 BACKUP RETENTION MANAGEMENT
# ===============================================

cleanup_old_backups() {
    log_header "🧹 CLEANING UP OLD BACKUPS"
    
    # Daily backups - keep last 7 days
    log_info "Cleaning daily backups (keeping last $RETENTION_DAILY days)..."
    find "$BACKUP_ROOT/daily" -name "*.tar.gz" -mtime +$RETENTION_DAILY -delete 2>/dev/null || true
    
    # Weekly backups - keep last 4 weeks
    log_info "Cleaning weekly backups (keeping last $RETENTION_WEEKLY weeks)..."
    find "$BACKUP_ROOT/weekly" -name "*.tar.gz" -mtime +$((RETENTION_WEEKLY * 7)) -delete 2>/dev/null || true
    
    # Monthly backups - keep last 12 months
    log_info "Cleaning monthly backups (keeping last $RETENTION_MONTHLY months)..."
    find "$BACKUP_ROOT/monthly" -name "*.tar.gz" -mtime +$((RETENTION_MONTHLY * 30)) -delete 2>/dev/null || true
    
    # Individual component backups - keep last 3 days
    local components=("postgres" "redis" "neo4j" "agent_configs" "agent_workspaces" "ollama_models" "monitoring")
    for component in "${components[@]}"; do
        if [[ -d "$BACKUP_ROOT/$component" ]]; then
            find "$BACKUP_ROOT/$component" -name "*" -mtime +3 -delete 2>/dev/null || true
        fi
    done
    
    log_success "Backup cleanup completed"
}

# ===============================================
# 🚀 COMMAND LINE INTERFACE
# ===============================================

show_help() {
    echo "SutazAI Agent Infrastructure Backup & Recovery System"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  backup [full|databases|configs]     Create backup"
    echo "  restore [component] [backup_file]   Restore from backup"
    echo "  list [component]                    List available backups"
    echo "  cleanup                             Clean up old backups"
    echo "  verify [backup_file]                Verify backup integrity"
    echo "  help                                Show this help message"
    echo
    echo "Examples:"
    echo "  $0 backup full                      # Full system backup"
    echo "  $0 backup databases                 # Databases only"
    echo "  $0 restore postgres backup.sql.gz  # Restore PostgreSQL"
    echo "  $0 list daily                       # List daily backups"
    echo "  $0 cleanup                          # Clean old backups"
    echo
}

list_backups() {
    local component="${1:-all}"
    
    log_header "📋 AVAILABLE BACKUPS"
    
    case "$component" in
        "daily"|"weekly"|"monthly")
            log_info "$component backups:"
            if [[ -d "$BACKUP_ROOT/$component" ]]; then
                ls -lh "$BACKUP_ROOT/$component"/*.tar.gz 2>/dev/null | while read -r line; do
                    log_info "  $line"
                done
            else
                log_info "  No $component backups found"
            fi
            ;;
        "all")
            for period in daily weekly monthly; do
                list_backups "$period"
            done
            
            log_info "Component backups:"
            for comp in postgres redis neo4j agent_configs agent_workspaces; do
                if [[ -d "$BACKUP_ROOT/$comp" ]] && ls "$BACKUP_ROOT/$comp"/* >/dev/null 2>&1; then
                    local count=$(ls "$BACKUP_ROOT/$comp"/* 2>/dev/null | wc -l)
                    log_info "  $comp: $count files"
                fi
            done
            ;;
        *)
            if [[ -d "$BACKUP_ROOT/$component" ]]; then
                ls -lh "$BACKUP_ROOT/$component"/* 2>/dev/null | while read -r line; do
                    log_info "  $line"
                done
            else
                log_error "Component not found: $component"
            fi
            ;;
    esac
}

# ===============================================
# 🚀 SCRIPT EXECUTION
# ===============================================

main() {
    local command="${1:-help}"
    
    case "$command" in
        "backup")
            local backup_type="${2:-full}"
            case "$backup_type" in
                "full")
                    create_full_backup
                    ;;
                "databases")
                    log_header "🚀 DATABASE BACKUP"
                    backup_postgresql
                    backup_redis
                    backup_neo4j
                    ;;
                "configs")
                    log_header "🚀 CONFIGURATION BACKUP"
                    backup_agent_configurations
                    ;;
                *)
                    log_error "Unknown backup type: $backup_type"
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        "restore")
            if [[ $# -lt 3 ]]; then
                log_error "Usage: $0 restore <component> <backup_file>"
                exit 1
            fi
            
            local component="$2"
            local backup_file="$3"
            
            # Set recovery log
            BACKUP_LOG="$RECOVERY_LOG"
            
            log_header "🔄 RESTORING $component"
            
            case "$component" in
                "postgres"|"postgresql")
                    restore_postgresql "$backup_file"
                    ;;
                "redis")
                    restore_redis "$backup_file"
                    ;;
                "neo4j")
                    restore_neo4j "$backup_file"
                    ;;
                *)
                    log_error "Unknown component: $component"
                    exit 1
                    ;;
            esac
            ;;
        "list")
            list_backups "$2"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "verify")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 verify <backup_file>"
                exit 1
            fi
            
            local backup_file="$2"
            log_header "🔍 VERIFYING BACKUP"
            
            if [[ -f "$backup_file" ]]; then
                if [[ "$backup_file" == *.gz ]]; then
                    if gzip -t "$backup_file" 2>/dev/null; then
                        log_success "Backup file integrity verified: $backup_file"
                    else
                        log_error "Backup file is corrupted: $backup_file"
                        exit 1
                    fi
                elif [[ "$backup_file" == *.tar.gz ]]; then
                    if tar -tzf "$backup_file" >/dev/null 2>&1; then
                        log_success "Backup archive integrity verified: $backup_file"
                    else
                        log_error "Backup archive is corrupted: $backup_file"
                        exit 1
                    fi
                else
                    log_success "Backup file exists: $backup_file"
                fi
            else
                log_error "Backup file not found: $backup_file"
                exit 1
            fi
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Ensure we're in the project directory
if [[ ! -f "docker-compose.yml" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"