#!/bin/bash

# Migrate All 174 Agents to Use Ollama Cluster Load Balancer
# Updates all agent configurations to use the high-availability cluster

set -euo pipefail

# Color codes

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
OLD_OLLAMA_URL="http://ollama:10104"
NEW_OLLAMA_URL="http://ollama-lb:80"
BACKUP_DIR="${PROJECT_ROOT}/backups/agent_configs_$(date +%Y%m%d_%H%M%S)"

log "Starting migration of all agents to Ollama cluster..."

# Create backup directory
create_backup() {
    log "Creating backup of current configurations..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup all docker-compose files
    find "$PROJECT_ROOT" -name "docker-compose*.yml" -exec cp {} "$BACKUP_DIR/" \;
    
    # Backup all agent config files
    find "$PROJECT_ROOT" -name "config.json" -path "*/docker/*" -exec cp {} "$BACKUP_DIR/" \;
    
    # Backup environment files
    find "$PROJECT_ROOT" -name ".env*" -exec cp {} "$BACKUP_DIR/" \;
    
    success "Backup created at: $BACKUP_DIR"
}

# Update docker-compose files
update_docker_compose_files() {
    log "Updating docker-compose files..."
    
    local updated_files=0
    
    # Find all docker-compose files
    while IFS= read -r -d '' file; do
        if grep -q "$OLD_OLLAMA_URL" "$file"; then
            log "Updating $file..."
            
            # Replace old Ollama URL with load balancer URL
            sed -i.bak "s|${OLD_OLLAMA_URL}|${NEW_OLLAMA_URL}|g" "$file"
            
            # Update environment variables to use load balancer
            sed -i "s|OLLAMA_BASE_URL: http://ollama:10104|OLLAMA_BASE_URL: http://ollama-lb:80|g" "$file"
            sed -i "s|OLLAMA_BASE_URL=http://ollama:10104|OLLAMA_BASE_URL=http://ollama-lb:80|g" "$file"
            
            ((updated_files++))
            success "Updated $file"
        fi
    done < <(find "$PROJECT_ROOT" -name "docker-compose*.yml" -print0)
    
    log "Updated $updated_files docker-compose files"
}

# Update agent configuration files
update_agent_configs() {
    log "Updating agent configuration files..."
    
    local updated_configs=0
    
    # Find all agent config.json files
    while IFS= read -r -d '' file; do
        if grep -q "$OLD_OLLAMA_URL" "$file"; then
            log "Updating agent config: $file..."
            
            # Create backup
            cp "$file" "${file}.bak"
            
            # Update URLs in JSON config
            sed -i "s|\"ollama_base_url\": \"${OLD_OLLAMA_URL}\"|\"ollama_base_url\": \"${NEW_OLLAMA_URL}\"|g" "$file"
            sed -i "s|\"openai_api_base\": \"${OLD_OLLAMA_URL}/v1\"|\"openai_api_base\": \"${NEW_OLLAMA_URL}/v1\"|g" "$file"
            
            ((updated_configs++))
            success "Updated agent config: $file"
        fi
    done < <(find "$PROJECT_ROOT/docker" -name "config.json" -print0)
    
    log "Updated $updated_configs agent configuration files"
}

# Update main docker-compose.yml to include load balancer
update_main_compose() {
    log "Updating main docker-compose.yml to include load balancer..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    # Check if load balancer service already exists
    if grep -q "ollama-loadbalancer:" "$compose_file"; then
        warning "Load balancer service already exists in main compose file"
        return
    fi
    
    # Add load balancer service to main compose file
    cat >> "$compose_file" << 'EOF'

  # Ollama Load Balancer for High Availability
  ollama-loadbalancer:
    image: nginx:alpine
    container_name: sutazai-ollama-lb
    restart: unless-stopped
    ports:
      - "10107:80"
    volumes:
      - ./config/nginx/ollama-lb.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ollama
    networks:
      - sutazai-network
    healthcheck:
      test:
        - CMD
        - wget
        - --no-verbose
        - --tries=1
        - --spider
        - http://localhost/health
      interval: 30s
      timeout: 10s
      retries: 3
EOF

    success "Added load balancer service to main docker-compose.yml"
}

# Update environment variables
update_environment_variables() {
    log "Updating environment variables..."
    
    # Update .env files if they exist
    for env_file in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.production" "$PROJECT_ROOT/.env.local"; do
        if [ -f "$env_file" ]; then
            log "Updating $env_file..."
            
            # Backup
            cp "$env_file" "${env_file}.bak"
            
            # Update Ollama URL
            sed -i "s|OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=http://ollama-lb:80|g" "$env_file"
            sed -i "s|OLLAMA_URL=.*|OLLAMA_URL=http://ollama-lb:80|g" "$env_file"
            
            success "Updated $env_file"
        fi
    done
}

# Validate updated configurations
validate_configurations() {
    log "Validating updated configurations..."
    
    local validation_errors=0
    
    # Check docker-compose files for syntax errors
    while IFS= read -r -d '' file; do
        if ! docker-compose -f "$file" config >/dev/null 2>&1; then
            error "Docker compose validation failed for: $file"
            ((validation_errors++))
        fi
    done < <(find "$PROJECT_ROOT" -name "docker-compose*.yml" -print0)
    
    # Check JSON config files for syntax errors
    while IFS= read -r -d '' file; do
        if ! python3 -m json.tool "$file" >/dev/null 2>&1; then
            error "JSON validation failed for: $file"
            ((validation_errors++))
        fi
    done < <(find "$PROJECT_ROOT/docker" -name "config.json" -print0)
    
    if [ $validation_errors -eq 0 ]; then
        success "All configuration files validated successfully"
        return 0
    else
        error "Found $validation_errors validation errors"
        return 1
    fi
}

# Create agent configuration update summary
create_update_summary() {
    log "Creating update summary..."
    
    local summary_file="${PROJECT_ROOT}/logs/agent_migration_summary_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Agent Migration to Ollama Cluster Summary"
        echo "========================================"
        echo "Migration Date: $(date)"
        echo "Old Ollama URL: $OLD_OLLAMA_URL"
        echo "New Load Balancer URL: $NEW_OLLAMA_URL"
        echo "Backup Location: $BACKUP_DIR"
        echo ""
        echo "Updated Files:"
        echo "-------------"
        
        # List all updated docker-compose files
        find "$PROJECT_ROOT" -name "docker-compose*.yml.bak" 2>/dev/null | while read -r backup; do
            original="${backup%.bak}"
            echo "  - $original"
        done
        
        # List all updated config files
        find "$PROJECT_ROOT/docker" -name "config.json.bak" 2>/dev/null | while read -r backup; do
            original="${backup%.bak}"
            echo "  - $original"
        done
        
        echo ""
        echo "Services That Will Use Load Balancer:"
        echo "-----------------------------------"
        
        # Count services that use Ollama
        grep -r "OLLAMA_BASE_URL.*ollama-lb" "$PROJECT_ROOT" --include="*.yml" | cut -d: -f1 | sort -u | while read -r file; do
            service_count=$(grep -c "container_name:" "$file" 2>/dev/null || echo "0")
            echo "  - $(basename "$file"): $service_count services"
        done
        
        echo ""
        echo "Verification Steps:"
        echo "-----------------"
        echo "1. Deploy Ollama cluster: ./scripts/deploy-ollama-cluster.sh"
        echo "2. Restart all services: docker-compose down && docker-compose up -d"
        echo "3. Check cluster monitor: http://localhost:10108"
        echo "4. Test load balancer: curl http://localhost:10107/health"
        echo "5. Verify tinyllama default: docker exec sutazai-ollama-primary ollama list"
        
    } > "$summary_file"
    
    success "Update summary saved to: $summary_file"
}

# Rollback function in case of issues
create_rollback_script() {
    log "Creating rollback script..."
    
    local rollback_script="${PROJECT_ROOT}/scripts/rollback-agent-migration.sh"
    
    cat > "$rollback_script" << EOF
#!/bin/bash
# Rollback Agent Migration to Ollama Cluster
# Generated automatically during migration

set -euo pipefail

echo "Rolling back agent migration..."

# Restore from backup
if [ -d "$BACKUP_DIR" ]; then
    echo "Restoring configurations from backup..."
    
    # Restore docker-compose files
    find "$BACKUP_DIR" -name "docker-compose*.yml" -exec cp {} "$PROJECT_ROOT/" \\;
    
    # Restore agent configs
    find "$PROJECT_ROOT/docker" -name "config.json.bak" | while read -r backup; do
        original="\${backup%.bak}"
        cp "\$backup" "\$original"
        echo "Restored \$original"
    done
    
    # Restore environment files
    find "$PROJECT_ROOT" -name ".env*.bak" | while read -r backup; do
        original="\${backup%.bak}"
        cp "\$backup" "\$original"
        echo "Restored \$original"
    done
    
    echo "Rollback completed successfully"
    echo "You may need to restart services: docker-compose down && docker-compose up -d"
else
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi
EOF
    
    chmod +x "$rollback_script"
    success "Rollback script created: $rollback_script"
}

# Main migration function
main() {
    log "Starting agent migration to Ollama cluster load balancer..."
    
    # Pre-flight checks
    if [ ! -f "${PROJECT_ROOT}/docker-compose.yml" ]; then
        error "Main docker-compose.yml not found in $PROJECT_ROOT"
        exit 1
    fi
    
    if [ ! -f "${PROJECT_ROOT}/config/nginx/ollama-lb.conf" ]; then
        error "Load balancer configuration not found. Run deploy-ollama-cluster.sh first."
        exit 1
    fi
    
    # Create backup before making changes
    create_backup
    
    # Create rollback script
    create_rollback_script
    
    # Perform migrations
    update_docker_compose_files
    update_agent_configs
    update_main_compose
    update_environment_variables
    
    # Validate changes
    if validate_configurations; then
        success "All configurations updated and validated successfully"
    else
        error "Configuration validation failed. Check logs and consider rollback."
        exit 1
    fi
    
    # Generate summary
    create_update_summary
    
    echo ""
    success "Agent migration to Ollama cluster completed successfully!"
    echo ""
    log "Summary:"
    echo "  🔄 All 174+ agents now configured to use load balancer"
    echo "  📊 Load balancer URL: http://ollama-lb:80"
    echo "  💾 Backup created at: $BACKUP_DIR"
    echo "  🔙 Rollback script: ./scripts/rollback-agent-migration.sh"
    echo ""
    log "Next Steps:"
    echo "  1. Deploy the Ollama cluster: ./scripts/deploy-ollama-cluster.sh"
    echo "  2. Restart all services: docker-compose down && docker-compose up -d"
    echo "  3. Monitor cluster health: http://localhost:10108"
    echo "  4. Test load balancer: curl http://localhost:10107/health"
    echo ""
    warning "Remember: tinyllama is configured as the default model per Rule 16"
}

# Run main function
main "$@"