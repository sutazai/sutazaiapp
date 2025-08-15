#!/bin/bash
# Docker Consolidation Migration Script - Rule 11 Compliance
# Purpose: Migrate all Docker files to /docker/ directory per Rule 11
# Author: Ultra System Architect
# Date: 2025-08-15
# WARNING: This script will reorganize Docker files - backup first!

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_DIR="/opt/sutazaiapp/docker"
BACKUP_DIR="/opt/sutazaiapp/backups/docker-consolidation-$(date +%Y%m%d_%H%M%S)"
DRY_RUN=${1:-true}
LOG_FILE="/opt/sutazaiapp/logs/docker-consolidation-$(date +%Y%m%d_%H%M%S).log"

# Function definitions
log() {
    echo -e "${2:-$NC}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1" "$RED"
    exit 1
}

warning() {
    log "WARNING: $1" "$YELLOW"
}

success() {
    log "SUCCESS: $1" "$GREEN"
}

info() {
    log "INFO: $1" "$BLUE"
}

# Create backup function
create_backup() {
    info "Creating backup at $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Backup all Docker-related files
    find /opt/sutazaiapp -type f \( -name "Dockerfile*" -o -name "docker-compose*.yml" -o -name ".dockerignore" \) \
        -not -path "*/node_modules/*" -not -path "*/.git/*" -exec cp --parents {} "$BACKUP_DIR" \; 2>/dev/null || true
    
    success "Backup created successfully"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if docker directory exists
    if [[ ! -d "$DOCKER_DIR" ]]; then
        error "Docker directory $DOCKER_DIR does not exist"
    fi
    
    # Check for running containers
    if docker ps -q | grep -q .; then
        warning "Docker containers are running. Please stop them before migration."
        docker ps --format "table {{.Names}}\t{{.Status}}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    success "Prerequisites check passed"
}

# Phase 1: Migrate root docker-compose files
migrate_root_compose_files() {
    info "Phase 1: Migrating root docker-compose files..."
    
    local compose_dir="$DOCKER_DIR/compose"
    mkdir -p "$compose_dir"
    
    # List of files to migrate with their target subdirectories
    declare -A compose_files=(
        ["docker-compose.yml"]="core/docker-compose.yml"
        ["docker-compose.base.yml"]="core/docker-compose.base.yml"
        ["docker-compose.secure.yml"]="security/docker-compose.secure.yml"
        ["docker-compose.security.yml"]="security/docker-compose.security.yml"
        ["docker-compose.security-hardening.yml"]="security/docker-compose.security-hardening.yml"
        ["docker-compose.security-monitoring.yml"]="monitoring/docker-compose.security-monitoring.yml"
        ["docker-compose.performance.yml"]="performance/docker-compose.performance.yml"
        ["docker-compose.ultra-performance.yml"]="performance/docker-compose.ultra-performance.yml"
        ["docker-compose.optimized.yml"]="performance/docker-compose.optimized.yml"
        ["docker-compose.standard.yml"]="deployment/docker-compose.standard.yml"
        ["docker-compose.minimal.yml"]="deployment/docker-compose.minimal.yml"
        ["docker-compose.override.yml"]="overrides/docker-compose.override.yml"
        ["docker-compose.mcp.yml"]="integrations/docker-compose.mcp.yml"
        ["docker-compose.mcp.override.yml"]="integrations/docker-compose.mcp.override.yml"
        ["docker-compose.skyvern.yml"]="integrations/docker-compose.skyvern.yml"
        ["docker-compose.skyvern.override.yml"]="integrations/docker-compose.skyvern.override.yml"
        ["docker-compose.documind.override.yml"]="integrations/docker-compose.documind.override.yml"
        ["docker-compose.public-images.override.yml"]="overrides/docker-compose.public-images.override.yml"
        ["docker-compose.secure.hardware-optimizer.yml"]="agents/docker-compose.secure.hardware-optimizer.yml"
    )
    
    for src_file in "${!compose_files[@]}"; do
        local src_path="/opt/sutazaiapp/$src_file"
        local dst_path="$compose_dir/${compose_files[$src_file]}"
        local dst_dir=$(dirname "$dst_path")
        
        if [[ -f "$src_path" ]]; then
            mkdir -p "$dst_dir"
            if [[ "$DRY_RUN" == "false" ]]; then
                mv "$src_path" "$dst_path"
                ln -s "$dst_path" "$src_path"  # Create symlink for backward compatibility
                success "Migrated $src_file to $dst_path"
            else
                info "[DRY RUN] Would migrate $src_file to $dst_path"
            fi
        fi
    done
}

# Phase 2: Migrate agent Dockerfiles
migrate_agent_dockerfiles() {
    info "Phase 2: Migrating agent Dockerfiles..."
    
    local agents_dir="$DOCKER_DIR/agents"
    mkdir -p "$agents_dir"
    
    # Find all agent directories with Dockerfiles
    find /opt/sutazaiapp/agents -name "Dockerfile*" -type f | while read -r dockerfile; do
        local agent_dir=$(dirname "$dockerfile")
        local agent_name=$(basename "$agent_dir")
        local dockerfile_name=$(basename "$dockerfile")
        local target_dir="$agents_dir/$agent_name"
        local target_file="$target_dir/$dockerfile_name"
        
        mkdir -p "$target_dir"
        if [[ "$DRY_RUN" == "false" ]]; then
            mv "$dockerfile" "$target_file"
            ln -s "$target_file" "$dockerfile"  # Create symlink
            success "Migrated $agent_name/$dockerfile_name"
        else
            info "[DRY RUN] Would migrate $agent_name/$dockerfile_name to $target_dir/"
        fi
    done
}

# Phase 3: Migrate service Dockerfiles
migrate_service_dockerfiles() {
    info "Phase 3: Migrating service Dockerfiles..."
    
    local services_dir="$DOCKER_DIR/services"
    mkdir -p "$services_dir"
    
    # Backend Dockerfiles
    if [[ -f "/opt/sutazaiapp/backend/Dockerfile" ]]; then
        mkdir -p "$services_dir/backend"
        if [[ "$DRY_RUN" == "false" ]]; then
            mv /opt/sutazaiapp/backend/Dockerfile* "$services_dir/backend/" 2>/dev/null || true
            success "Migrated backend Dockerfiles"
        else
            info "[DRY RUN] Would migrate backend Dockerfiles"
        fi
    fi
    
    # Frontend Dockerfiles
    if [[ -f "/opt/sutazaiapp/frontend/Dockerfile" ]]; then
        mkdir -p "$services_dir/frontend"
        if [[ "$DRY_RUN" == "false" ]]; then
            mv /opt/sutazaiapp/frontend/Dockerfile* "$services_dir/frontend/" 2>/dev/null || true
            success "Migrated frontend Dockerfiles"
        else
            info "[DRY RUN] Would migrate frontend Dockerfiles"
        fi
    fi
}

# Phase 4: Update references in scripts and configs
update_references() {
    info "Phase 4: Updating references in scripts and configs..."
    
    local files_to_update=(
        "/opt/sutazaiapp/Makefile"
        "/opt/sutazaiapp/.github/workflows/*.yml"
        "/opt/sutazaiapp/scripts/deployment/*.sh"
        "/opt/sutazaiapp/scripts/maintenance/*.sh"
    )
    
    for pattern in "${files_to_update[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                if [[ "$DRY_RUN" == "false" ]]; then
                    # Backup original
                    cp "$file" "$file.bak"
                    
                    # Update docker-compose references
                    sed -i 's|docker-compose\.yml|docker/compose/core/docker-compose.yml|g' "$file"
                    sed -i 's|backend/Dockerfile|docker/services/backend/Dockerfile|g' "$file"
                    sed -i 's|frontend/Dockerfile|docker/services/frontend/Dockerfile|g' "$file"
                    
                    success "Updated references in $(basename $file)"
                else
                    info "[DRY RUN] Would update references in $(basename $file)"
                fi
            fi
        done
    done
}

# Generate migration report
generate_report() {
    info "Generating migration report..."
    
    local report_file="/opt/sutazaiapp/reports/docker-consolidation-migration-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Docker Consolidation Migration Report
Date: $(date)
Dry Run: $DRY_RUN

## Files Migrated
$(grep "Migrated\|Would migrate" "$LOG_FILE" | wc -l) files processed

## Migration Details
$(grep "Migrated\|Would migrate" "$LOG_FILE")

## Next Steps
1. Review migrated files in $DOCKER_DIR
2. Test all Docker builds and deployments
3. Update CI/CD pipelines
4. Remove symlinks after verification
5. Delete backup after confirmation

## Backup Location
$BACKUP_DIR
EOF
    
    success "Report generated at $report_file"
}

# Main execution
main() {
    log "=== Docker Consolidation Migration Script ===" "$GREEN"
    log "Dry Run Mode: $DRY_RUN" "$YELLOW"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        warning "This is a DRY RUN - no files will be moved"
        warning "Run with 'false' parameter to perform actual migration"
    fi
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Execute migration phases
    check_prerequisites
    create_backup
    migrate_root_compose_files
    migrate_agent_dockerfiles
    migrate_service_dockerfiles
    update_references
    generate_report
    
    success "Migration completed successfully!"
    info "Log file: $LOG_FILE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo
        warning "This was a DRY RUN. To perform actual migration, run:"
        echo "  $0 false"
    fi
}

# Help function
show_help() {
    cat << EOF
Docker Consolidation Migration Script

Usage: $0 [true|false]

Arguments:
  true  - Perform dry run (default)
  false - Perform actual migration

This script migrates all Docker-related files to the /docker/ directory
in compliance with Rule 11: Docker Excellence.

WARNING: Always backup before running with 'false' parameter!
EOF
}

# Parse arguments
case "${1:-}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
    true|false)
        DRY_RUN="$1"
        ;;
    "")
        DRY_RUN="true"
        ;;
    *)
        error "Invalid argument: $1. Use 'true' or 'false'"
        ;;
esac

# Run main function
main