#!/bin/bash
# Container Security Migration - Docker Compose User Configuration
# Created: August 9, 2025
# Purpose: Update docker-compose.yml to specify non-root users for containers

set -euo pipefail

# Colors for output

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

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}" >&2; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }

COMPOSE_FILE="/opt/sutazaiapp/docker-compose.yml"
BACKUP_FILE="/opt/sutazaiapp/docker-compose.yml.security-backup-$(date +%Y%m%d_%H%M%S)"

# Backup original docker-compose.yml
backup_compose_file() {
    log "Creating backup of docker-compose.yml..."
    cp "$COMPOSE_FILE" "$BACKUP_FILE"
    success "Backup created: $BACKUP_FILE"
}

# Add user specification to a service in docker-compose.yml
add_user_to_service() {
    local service_name="$1"
    local user_spec="$2"
    local temp_file="/tmp/docker-compose-temp.yml"
    
    log "Adding user specification to $service_name..."
    
    # Use Python to safely modify YAML (more reliable than sed for complex YAML)
    python3 -c "
import yaml
import sys

with open('$COMPOSE_FILE', 'r') as f:
    data = yaml.safe_load(f)

if 'services' in data and '$service_name' in data['services']:
    data['services']['$service_name']['user'] = '$user_spec'
    
    with open('$temp_file', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print('User added to $service_name')
else:
    print('Service $service_name not found', file=sys.stderr)
    sys.exit(1)
" || {
        error "Failed to add user to $service_name"
        return 1
    }
    
    mv "$temp_file" "$COMPOSE_FILE"
}

# Update all container users
update_container_users() {
    log "Updating docker-compose.yml with non-root users..."
    
    # Database containers (these already have dedicated users in their images)
    add_user_to_service "postgres" "postgres:postgres"
    add_user_to_service "redis" "redis:redis"
    add_user_to_service "neo4j" "7474:7474"  # Neo4j's standard UID/GID
    add_user_to_service "rabbitmq" "999:999"  # RabbitMQ standard
    
    # Vector databases
    add_user_to_service "chromadb" "1000:1000"  # Standard non-root user
    add_user_to_service "qdrant" "1001:1001"    # Qdrant user
    
    # Third-party services
    add_user_to_service "consul" "100:100"      # Consul user
    add_user_to_service "ollama" "1001:1001"    # Ollama user (will be created)
    add_user_to_service "blackbox-exporter" "65534:65534"  # Nobody user
    
    # Note: cAdvisor remains privileged as it requires host system access
    warning "cAdvisor will remain privileged (required for system monitoring)"
}

# Add environment variables for volume ownership
add_volume_environment_variables() {
    log "Adding volume ownership environment variables..."
    
    # For containers that need to fix permissions on startup
    python3 -c "
import yaml

with open('$COMPOSE_FILE', 'r') as f:
    data = yaml.safe_load(f)

# Add environment variables for permission fixes
containers_needing_permission_fix = ['ollama', 'chromadb', 'qdrant']

for container in containers_needing_permission_fix:
    if 'services' in data and container in data['services']:
        if 'environment' not in data['services'][container]:
            data['services'][container]['environment'] = []
        
        # Add permission fix environment variable
        env_vars = data['services'][container]['environment']
        if isinstance(env_vars, list):
            env_vars.append('FIX_PERMISSIONS=true')
        elif isinstance(env_vars, dict):
            env_vars['FIX_PERMISSIONS'] = 'true'

with open('$COMPOSE_FILE', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
"
}

# Create init scripts for containers that need permission fixes
create_init_scripts() {
    log "Creating initialization scripts for containers..."
    
    mkdir -p /opt/sutazaiapp/docker/init-scripts
    
    # Ollama init script
    cat > /opt/sutazaiapp/docker/init-scripts/ollama-init.sh << 'EOF'
#!/bin/bash
# Ollama initialization script - fix permissions and start
set -e

if [[ "${FIX_PERMISSIONS:-false}" == "true" ]]; then
    echo "Fixing Ollama permissions..."
    
    # Create ollama user if it doesn't exist
    id ollama &>/dev/null || {
        groupadd -g 1001 ollama
        useradd -r -u 1001 -g ollama -s /bin/false -d /home/ollama ollama
    }
    
    # Ensure home directory exists and has correct ownership
    mkdir -p /home/ollama/.ollama
    chown -R ollama:ollama /home/ollama
    
    # If running as root, switch to ollama user
    if [[ $(id -u) -eq 0 ]]; then
        exec su ollama -s /bin/bash -c "OLLAMA_MODELS=/home/ollama/.ollama/models exec ollama serve"
    fi
fi

# Start Ollama normally
exec ollama serve
EOF
    
    chmod +x /opt/sutazaiapp/docker/init-scripts/ollama-init.sh
    
    # ChromaDB init script
    cat > /opt/sutazaiapp/docker/init-scripts/chromadb-init.sh << 'EOF'
#!/bin/bash
# ChromaDB initialization script
set -e

if [[ "${FIX_PERMISSIONS:-false}" == "true" ]]; then
    echo "Fixing ChromaDB permissions..."
    
    # Create chromadb user if needed
    id chromadb &>/dev/null || {
        groupadd -g 1000 chromadb
        useradd -r -u 1000 -g chromadb -s /bin/false chromadb
    }
    
    # Ensure data directory has correct ownership
    mkdir -p /chroma/chroma
    chown -R chromadb:chromadb /chroma
fi

# Start ChromaDB
exec python -m chroma run --host 0.0.0.0 --port 8000
EOF
    
    chmod +x /opt/sutazaiapp/docker/init-scripts/chromadb-init.sh
}

# Test the updated configuration
test_updated_configuration() {
    log "Testing updated docker-compose configuration..."
    
    # Validate YAML syntax
    if ! docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        error "Updated docker-compose.yml has syntax errors!"
        return 1
    fi
    
    success "Docker Compose configuration is valid"
    
    # Show the changes made
    log "Summary of user changes:"
    docker-compose -f "$COMPOSE_FILE" config | grep -A 1 -B 1 "user:" || log "No user specifications found in output"
}

# Rollback function
rollback_changes() {
    warning "Rolling back docker-compose.yml changes..."
    if [[ -f "$BACKUP_FILE" ]]; then
        cp "$BACKUP_FILE" "$COMPOSE_FILE"
        success "Rollback completed. Original file restored."
    else
        error "Backup file not found. Cannot rollback automatically."
        return 1
    fi
}

# Main execution
main() {
    log "Starting Docker Compose User Configuration Update"
    log "=================================================="
    
    # Check if docker-compose exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "docker-compose.yml not found at $COMPOSE_FILE"
        exit 1
    fi
    
    # Check for required tools
    if ! command -v python3 &>/dev/null; then
        error "Python3 is required but not installed"
        exit 1
    fi
    
    if ! python3 -c "import yaml" 2>/dev/null; then
        log "Installing PyYAML..."
        pip3 install PyYAML || {
            error "Failed to install PyYAML. Please install manually: pip3 install PyYAML"
            exit 1
        }
    fi
    
    backup_compose_file
    
    # Make updates
    if update_container_users; then
        add_volume_environment_variables
        create_init_scripts
        
        if test_updated_configuration; then
            success "=================================================="
            success "Docker Compose user configuration updated successfully!"
            success "=================================================="
            success "Backup available at: $BACKUP_FILE"
            success "Ready for container rebuild and restart."
            success "=================================================="
        else
            error "Configuration test failed!"
            rollback_changes
            exit 1
        fi
    else
        error "Failed to update container users!"
        rollback_changes
        exit 1
    fi
}

# Rollback function for external use
rollback() {
    if [[ $# -eq 1 ]]; then
        BACKUP_FILE="$1"
    fi
    rollback_changes
}

# Execute main if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 1 && "$1" == "--rollback" ]]; then
        rollback
    else
        main "$@"
    fi
fi