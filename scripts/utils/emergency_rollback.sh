#!/bin/bash
# EMERGENCY ROLLBACK SYSTEM - SutazAI Cleanup Operation
# QA Testing Specialist - CRITICAL SYSTEM RECOVERY
# Created: August 10, 2025

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_BASE_DIR="$PROJECT_ROOT/backups"
LOG_FILE="$PROJECT_ROOT/logs/emergency_rollback_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Logging
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log_emergency() { echo -e "${RED}${BOLD}[EMERGENCY]${NC} $1"; }
log_critical() { echo -e "${RED}[CRITICAL]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Find latest backup
find_latest_backup() {
    local latest_backup=""
    local latest_timestamp=0
    
    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        log_critical "No backup directory found: $BACKUP_BASE_DIR"
        return 1
    fi
    
    while IFS= read -r -d '' backup_dir; do
        local backup_name=$(basename "$backup_dir")
        # Extract timestamp from directory names like "test-20250810_143022"
        local timestamp=$(echo "$backup_name" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1 || echo "0")
        
        if [ -n "$timestamp" ] && [ "$timestamp" -gt "$latest_timestamp" ]; then
            latest_timestamp="$timestamp"
            latest_backup="$backup_dir"
        fi
    done < <(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -print0 2>/dev/null)
    
    if [ -n "$latest_backup" ]; then
        echo "$latest_backup"
        return 0
    else
        log_critical "No timestamped backup found"
        return 1
    fi
}

# Level 1: Container Emergency Rollback (5 minutes max)
rollback_containers() {
    log_emergency "LEVEL 1: CONTAINER ROLLBACK INITIATED"
    
    # Stop all running containers immediately
    log_info "Stopping all containers..."
    docker-compose down --remove-orphans --timeout 30 || {
        log_warning "Docker-compose down failed, forcing container stop"
        docker stop $(docker ps -q) || true
        docker rm $(docker ps -aq) || true
    }
    
    # Check for container backup images
    local backup_dir
    if backup_dir=$(find_latest_backup); then
        log_info "Using backup: $backup_dir"
        
        # Restore docker-compose.yml if available
        if [ -f "$backup_dir/docker-compose.yml" ]; then
            cp "$backup_dir/docker-compose.yml" "$PROJECT_ROOT/docker-compose.yml"
            log_success "Docker compose file restored"
        fi
        
        # Restore .env if available
        if [ -f "$backup_dir/.env" ]; then
            cp "$backup_dir/.env" "$PROJECT_ROOT/.env"
            log_success "Environment file restored"
        fi
    fi
    
    # Start critical services first
    log_info "Starting critical database services..."
    docker-compose up -d postgres redis neo4j || {
        log_critical "Failed to start databases"
        return 1
    }
    
    # Wait for databases
    sleep 20
    
    # Start core application services
    log_info "Starting core application services..."
    docker-compose up -d backend frontend || {
        log_warning "Failed to start some application services"
    }
    
    # Start AI services if possible
    docker-compose up -d ollama hardware-resource-optimizer || {
        log_warning "Failed to start AI services (non-critical)"
    }
    
    log_success "Container rollback completed"
    
    # Quick health check
    sleep 30
    local healthy_services=0
    local critical_ports=(10000 10001 10010 10011)
    
    for port in "${critical_ports[@]}"; do
        if nc -z localhost "$port" 2>/dev/null; then
            ((healthy_services++))
            log_info "✓ Service on port $port responding"
        else
            log_warning "✗ Service on port $port not responding"
        fi
    done
    
    log_info "Rollback health check: $healthy_services/${#critical_ports[@]} services responding"
    
    if [ $healthy_services -ge 2 ]; then
        log_success "LEVEL 1 ROLLBACK SUCCESSFUL"
        return 0
    else
        log_critical "LEVEL 1 ROLLBACK INSUFFICIENT - ESCALATING"
        return 1
    fi
}

# Level 2: Script and Configuration Rollback (3 minutes max)
rollback_scripts() {
    log_emergency "LEVEL 2: SCRIPT AND CONFIGURATION ROLLBACK"
    
    local backup_dir
    if ! backup_dir=$(find_latest_backup); then
        log_critical "No backup available for script rollback"
        return 1
    fi
    
    # Restore scripts directory
    if [ -d "$backup_dir/scripts" ]; then
        log_info "Restoring scripts from backup..."
        rm -rf "$PROJECT_ROOT/scripts"
        cp -r "$backup_dir/scripts" "$PROJECT_ROOT/scripts"
        
        # Make scripts executable
        find "$PROJECT_ROOT/scripts" -name "*.sh" -exec chmod +x {} \;
        
        log_success "Scripts restored from backup"
    else
        log_warning "No scripts backup found, attempting Git restore"
        cd "$PROJECT_ROOT"
        
        # Try Git restore
        if git status > /dev/null 2>&1; then
            git checkout HEAD -- scripts/ || {
                log_warning "Git restore failed, trying previous commit"
                git checkout HEAD~1 -- scripts/ || {
                    log_critical "Git restore failed completely"
                    return 1
                }
            }
            log_success "Scripts restored from Git"
        else
            log_critical "No Git repository found and no script backup"
            return 1
        fi
    fi
    
    # Restore other configuration files
    local config_files=("docker-compose.yml" ".env" "Makefile")
    
    for config_file in "${config_files[@]}"; do
        if [ -f "$backup_dir/$config_file" ]; then
            cp "$backup_dir/$config_file" "$PROJECT_ROOT/$config_file"
            log_info "✓ Restored $config_file"
        fi
    done
    
    log_success "LEVEL 2 ROLLBACK COMPLETED"
    return 0
}

# Level 3: Full System Rollback (15 minutes max)
rollback_full_system() {
    log_emergency "LEVEL 3: FULL SYSTEM ROLLBACK INITIATED"
    
    # Stop everything
    log_info "Stopping all services..."
    docker-compose down --volumes --remove-orphans --timeout 60 || {
        log_warning "Graceful shutdown failed, forcing stop"
        docker kill $(docker ps -q) 2>/dev/null || true
        docker rm -f $(docker ps -aq) 2>/dev/null || true
        docker volume prune -f
        docker network prune -f
    }
    
    # Git-based rollback if available
    cd "$PROJECT_ROOT"
    if git status > /dev/null 2>&1; then
        log_info "Performing Git-based system restore..."
        
        # Find last known-good commit
        local commits=(
            "HEAD~1"  # Previous commit
            "HEAD~2"  # Two commits back
            "HEAD~3"  # Three commits back
        )
        
        for commit in "${commits[@]}"; do
            log_info "Attempting restore to $commit..."
            
            if git reset --hard "$commit"; then
                log_success "Restored to commit: $commit"
                break
            else
                log_warning "Failed to restore to $commit"
            fi
        done
    fi
    
    # Restore from backup if available
    local backup_dir
    if backup_dir=$(find_latest_backup); then
        log_info "Applying backup overlays from: $backup_dir"
        
        # Restore critical files
        local critical_items=(
            "scripts"
            "docker-compose.yml"
            ".env"
            "Makefile"
            "backend/requirements.txt"
        )
        
        for item in "${critical_items[@]}"; do
            if [ -e "$backup_dir/$item" ]; then
                if [ -d "$backup_dir/$item" ]; then
                    rm -rf "$PROJECT_ROOT/$item"
                    cp -r "$backup_dir/$item" "$PROJECT_ROOT/$item"
                else
                    cp "$backup_dir/$item" "$PROJECT_ROOT/$item"
                fi
                log_info "✓ Restored $item from backup"
            fi
        done
    fi
    
    # Attempt to restore database backups
    restore_databases
    
    # Rebuild and restart system
    log_info "Rebuilding and restarting system..."
    
    # Try using restored deployment scripts
    if [ -f "$PROJECT_ROOT/scripts/master/deploy.sh" ]; then
        log_info "Using master deployment script..."
        timeout 600 "$PROJECT_ROOT/scripts/master/deploy.sh" minimal || {
            log_warning "Master deploy failed, using fallback"
            fallback_system_start
        }
    else
        log_info "No master script available, using fallback deployment"
        fallback_system_start
    fi
    
    # Final validation
    sleep 60
    validate_system_after_rollback
    
    log_success "LEVEL 3 ROLLBACK COMPLETED"
}

# Database restoration
restore_databases() {
    log_info "Attempting database restoration..."
    
    # Start database containers first
    docker-compose up -d postgres redis neo4j || {
        log_warning "Failed to start some databases"
        return 1
    }
    
    sleep 30
    
    # Look for database backups
    local backup_dirs=(
        "$BACKUP_BASE_DIR/postgres"
        "$BACKUP_BASE_DIR/pre-consolidation-$(date +%Y%m%d)"
        "$BACKUP_BASE_DIR"
    )
    
    for backup_dir in "${backup_dirs[@]}"; do
        if [ -f "$backup_dir/postgres_backup.sql" ]; then
            log_info "Restoring PostgreSQL from: $backup_dir/postgres_backup.sql"
            
            if docker exec -i sutazai-postgres psql -U sutazai < "$backup_dir/postgres_backup.sql"; then
                log_success "PostgreSQL backup restored"
                break
            else
                log_warning "PostgreSQL restore failed from $backup_dir"
            fi
        fi
    done
    
    # Redis restore (if available)
    for backup_dir in "${backup_dirs[@]}"; do
        if [ -f "$backup_dir/redis_backup.rdb" ]; then
            log_info "Redis backup found, manual restoration may be needed"
            break
        fi
    done
}

# Fallback system start
fallback_system_start() {
    log_info "Starting fallback system deployment..."
    
    # Start services in order
    local service_groups=(
        "postgres redis neo4j"
        "rabbitmq"
        "backend frontend"
        "ollama"
        "prometheus grafana"
    )
    
    for services in "${service_groups[@]}"; do
        log_info "Starting: $services"
        docker-compose up -d $services || {
            log_warning "Failed to start some services in group: $services"
        }
        sleep 15
    done
}

# System validation after rollback
validate_system_after_rollback() {
    log_info "Validating system after rollback..."
    
    # Check critical services
    local critical_checks=(
        "10000:Database"
        "10001:Redis"
        "10010:Backend"
        "10011:Frontend"
    )
    
    local validation_score=0
    local total_checks=${#critical_checks[@]}
    
    for check in "${critical_checks[@]}"; do
        IFS=':' read -r port name <<< "$check"
        
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "✓ $name responding on port $port"
            ((validation_score++))
        else
            log_warning "✗ $name not responding on port $port"
        fi
    done
    
    local health_percentage=$((validation_score * 100 / total_checks))
    
    if [ $health_percentage -ge 75 ]; then
        log_success "System validation: $health_percentage% ($validation_score/$total_checks services)"
        log_success "🎉 ROLLBACK SUCCESSFUL - SYSTEM OPERATIONAL"
        return 0
    elif [ $health_percentage -ge 50 ]; then
        log_warning "System validation: $health_percentage% ($validation_score/$total_checks services)"
        log_warning "⚠️  ROLLBACK PARTIAL - MANUAL INTERVENTION REQUIRED"
        return 1
    else
        log_critical "System validation: $health_percentage% ($validation_score/$total_checks services)"
        log_critical "❌ ROLLBACK FAILED - CRITICAL MANUAL INTERVENTION REQUIRED"
        return 2
    fi
}

# Main rollback orchestration
main() {
    local rollback_level="${1:-auto}"
    
    log_emergency "🚨 EMERGENCY ROLLBACK SYSTEM ACTIVATED"
    log_info "Timestamp: $(date)"
    log_info "Rollback Level: $rollback_level"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Log File: $LOG_FILE"
    
    case "$rollback_level" in
        1|container|containers)
            rollback_containers || exit 1
            ;;
        2|script|scripts)
            rollback_scripts || {
                log_critical "Level 2 rollback failed, escalating to Level 3"
                rollback_full_system
            }
            ;;
        3|full|system)
            rollback_full_system || {
                log_critical "Full system rollback failed - manual intervention required"
                exit 2
            }
            ;;
        auto)
            log_info "Auto-escalating rollback based on system state..."
            
            # Try Level 1 first
            if rollback_containers; then
                log_success "Level 1 rollback successful"
            else
                log_warning "Level 1 failed, trying Level 2"
                
                if rollback_scripts; then
                    if rollback_containers; then
                        log_success "Level 2 rollback successful"
                    else
                        log_critical "Level 2 partial failure, escalating to Level 3"
                        rollback_full_system
                    fi
                else
                    log_critical "Level 2 failed, escalating to Level 3"
                    rollback_full_system
                fi
            fi
            ;;
        *)
            echo "Usage: $0 {1|2|3|auto|container|script|full}"
            echo ""
            echo "Rollback Levels:"
            echo "  1, container   - Container rollback only (5 min)"
            echo "  2, script      - Script and config rollback (3 min)"
            echo "  3, full        - Complete system rollback (15 min)"
            echo "  auto           - Auto-escalating rollback (default)"
            exit 1
            ;;
    esac
    
    log_success "✅ EMERGENCY ROLLBACK COMPLETED"
    log_info "Full log available at: $LOG_FILE"
    
    # Generate rollback report
    echo ""
    echo "=== ROLLBACK SUMMARY ==="
    echo "Date: $(date)"
    echo "Level: $rollback_level"
    echo "Status: $(validate_system_after_rollback > /dev/null 2>&1 && echo "SUCCESS" || echo "PARTIAL")"
    echo "Log: $LOG_FILE"
    echo "========================"
}

# Execute main function
main "$@"