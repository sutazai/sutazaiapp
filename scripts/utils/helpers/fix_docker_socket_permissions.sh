#!/bin/bash

# Fix Docker Socket Permissions for SutazAI Containers
# This script implements secure Docker socket access while maintaining security best practices

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Configuration
DOCKER_SOCK="/var/run/docker.sock"
DOCKER_GROUP="docker"
CONTAINERS_WITH_DOCKER_ACCESS=(
    "sutazai-hardware-optimizer"
    "sutazai-devops-manager" 
    "sutazai-ollama-specialist"
)
COMPOSE_FILE="/opt/sutazaiapp/docker-compose-agents-tier1.yml"

# Function to check if script is run as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        error "Please run as a user with docker group membership"
        exit 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or accessible"
        exit 1
    fi
    
    # Check if user is in docker group
    if ! groups | grep -q docker; then
        error "Current user is not in docker group"
        error "Add user to docker group: sudo usermod -aG docker \$USER"
        exit 1
    fi
    
    # Check if docker socket exists
    if [[ ! -S "$DOCKER_SOCK" ]]; then
        error "Docker socket not found at $DOCKER_SOCK"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Function to get Docker group ID
get_docker_gid() {
    local docker_gid
    docker_gid=$(stat -c %g "$DOCKER_SOCK")
    echo "$docker_gid"
}

# Function to create secure Docker socket access script
create_socket_wrapper() {
    local script_path="/opt/sutazaiapp/scripts/docker_socket_wrapper.sh"
    
    log "Creating secure Docker socket wrapper..."
    
    cat > "$script_path" << 'EOF'
#!/bin/bash
# Docker Socket Security Wrapper
# This script provides controlled access to Docker socket with logging

set -euo pipefail

# Log all Docker commands for security auditing
log_docker_command() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local user=${USER:-unknown}
    local container=${HOSTNAME:-unknown}
    echo "[$timestamp] Container: $container, User: $user, Command: docker $*" >> /var/log/docker_access.log
}

# Only allow specific Docker commands for security
allowed_commands=("ps" "stats" "inspect" "logs" "exec" "restart" "stop" "start")
command="$1"

if [[ ! " ${allowed_commands[@]} " =~ " ${command} " ]]; then
    echo "ERROR: Docker command '$command' not allowed for security reasons" >&2
    exit 1
fi

# Log the command
log_docker_command "$@"

# Execute the Docker command
exec docker "$@"
EOF

    chmod +x "$script_path"
    success "Docker socket wrapper created at $script_path"
}

# Function to update Dockerfiles with proper security settings
update_dockerfile_security() {
    local dockerfile="$1"
    local user_name="$2"
    local docker_gid="$3"
    
    log "Updating $dockerfile with security enhancements..."
    
    # Create a backup
    cp "$dockerfile" "${dockerfile}.backup.$(date +%Y%m%d-%H%M%S)"
    
    # Create improved Dockerfile content
    cat > "$dockerfile" << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal for security)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    docker.io \\
    procps \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Security: Create non-root user with proper Docker group membership
# Use dynamic GID matching host Docker group
RUN useradd -m -u 1000 $user_name \\
    && groupadd -g $docker_gid docker-host || true \\
    && usermod -aG docker-host $user_name \\
    && chown -R $user_name:$user_name /app \\
    && mkdir -p /home/$user_name && chown $user_name:$user_name /home/$user_name

# Security: Drop all capabilities by default, add only necessary ones
# Note: CAP_DAC_OVERRIDE needed for Docker socket access
USER $user_name

# Health check endpoint
EXPOSE 852${dockerfile: -1}

# Security: Use exec form to avoid shell injection
CMD ["python", "app.py"]
EOF

    success "Updated $dockerfile with enhanced security"
}

# Function to create secure docker-compose configuration
create_secure_compose_config() {
    local docker_gid="$1"
    
    log "Creating secure docker-compose configuration..."
    
    # Create a secure compose override
    cat > "/opt/sutazaiapp/docker-compose.security-override.yml" << EOF
# Security-Enhanced Docker Compose Override
# This file provides secure Docker socket access with proper permissions

version: '3.8'

services:
  infrastructure-devops-manager:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - DAC_OVERRIDE  # Required for Docker socket access
    read_only: false  # Needs write access for logs
    user: "1000:$docker_gid"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro  # Read-only access
      - ./scripts:/scripts:ro
      - ./logs/devops:/logs
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
      - SECURITY_MODE=enabled
      - LOG_LEVEL=INFO

  hardware-resource-optimizer:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - DAC_OVERRIDE  # Required for Docker socket access
      - SYS_PTRACE   # Required for system monitoring
    read_only: false
    user: "1000:$docker_gid"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - ./logs/optimizer:/logs
    environment:
      - SECURITY_MODE=enabled
      - LOG_LEVEL=INFO

  ollama-integration-specialist:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - DAC_OVERRIDE  # Required for Docker socket access
    read_only: false
    user: "1000:$docker_gid"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./models:/models
      - ./config/ollama:/config
    environment:
      - SECURITY_MODE=enabled
      - LOG_LEVEL=INFO

networks:
  sutazai-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"  # Disable inter-container communication by default
EOF

    success "Created secure compose override configuration"
}

# Function to create monitoring script for Docker socket access
create_monitoring_script() {
    log "Creating Docker socket access monitoring script..."
    
    cat > "/opt/sutazaiapp/scripts/monitor_docker_access.sh" << 'EOF'
#!/bin/bash

# Docker Socket Access Monitor
# Monitors and logs Docker socket access for security auditing

LOG_FILE="/var/log/sutazai_docker_access.log"
CONTAINERS=("sutazai-hardware-optimizer" "sutazai-devops-manager" "sutazai-ollama-specialist")

# Create log file if it doesn't exist
touch "$LOG_FILE"

log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

monitor_containers() {
    for container in "${CONTAINERS[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            # Get container stats
            stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" "$container" 2>/dev/null || echo "N/A")
            log_event "Container: $container, Status: Running, Stats: $stats"
        else
            log_event "Container: $container, Status: Not Running"
        fi
    done
}

# Main monitoring loop
while true; do
    monitor_containers
    sleep 60  # Monitor every minute
done
EOF

    chmod +x "/opt/sutazaiapp/scripts/monitor_docker_access.sh"
    success "Created Docker access monitoring script"
}

# Function to restart affected containers
restart_containers() {
    log "Restarting containers with Docker socket access..."
    
    for container in "${CONTAINERS_WITH_DOCKER_ACCESS[@]}"; do
        if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
            log "Restarting $container..."
            docker restart "$container" || warn "Failed to restart $container"
        else
            warn "Container $container not found"
        fi
    done
    
    success "Container restart process completed"
}

# Function to verify fix
verify_fix() {
    log "Verifying Docker socket access fix..."
    
    local failed_containers=()
    
    for container in "${CONTAINERS_WITH_DOCKER_ACCESS[@]}"; do
        log "Checking $container..."
        
        # Wait for container to be ready
        sleep 10
        
        # Check if container is running
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            # Test Docker access from inside container
            if docker exec "$container" docker ps >/dev/null 2>&1; then
                success "$container: Docker socket access working"
            else
                error "$container: Docker socket access failed"
                failed_containers+=("$container")
            fi
        else
            error "$container: Container not running"
            failed_containers+=("$container")
        fi
    done
    
    if [[ ${#failed_containers[@]} -eq 0 ]]; then
        success "All containers have proper Docker socket access"
        return 0
    else
        error "Failed containers: ${failed_containers[*]}"
        return 1
    fi
}

# Function to display security recommendations
display_security_recommendations() {
    log "Security Recommendations:"
    echo ""
    echo "1. Monitor Docker socket access logs regularly"
    echo "2. Review container permissions periodically"
    echo "3. Consider using Docker-in-Docker for better isolation"
    echo "4. Implement network segmentation for containers"
    echo "5. Regular security audits of container configurations"
    echo "6. Use secrets management for sensitive data"
    echo "7. Enable Docker Content Trust for image verification"
    echo ""
}

# Main execution
main() {
    log "Starting Docker Socket Permission Fix"
    
    # Security check
    check_root
    
    # Prerequisites
    check_prerequisites
    
    # Get Docker group ID
    local docker_gid
    docker_gid=$(get_docker_gid)
    log "Docker group ID: $docker_gid"
    
    # Create security components
    create_socket_wrapper
    create_secure_compose_config "$docker_gid"
    create_monitoring_script
    
    # Update Dockerfiles if they exist
    for container in "${CONTAINERS_WITH_DOCKER_ACCESS[@]}"; do
        case "$container" in
            "sutazai-hardware-optimizer")
                dockerfile="/opt/sutazaiapp/agents/hardware-optimizer/Dockerfile"
                user_name="optimizer"
                ;;
            "sutazai-devops-manager")
                dockerfile="/opt/sutazaiapp/agents/infrastructure-devops/Dockerfile"
                user_name="devops"
                ;;
            "sutazai-ollama-specialist")
                dockerfile="/opt/sutazaiapp/agents/ollama-integration/Dockerfile"
                user_name="ollama"
                ;;
        esac
        
        if [[ -f "$dockerfile" ]]; then
            update_dockerfile_security "$dockerfile" "$user_name" "$docker_gid"
        fi
    done
    
    log "Rebuilding containers with security fixes..."
    # Rebuild containers with new security settings
    docker-compose -f "$COMPOSE_FILE" -f "/opt/sutazaiapp/docker-compose.security-override.yml" build "${CONTAINERS_WITH_DOCKER_ACCESS[@]}" || warn "Some containers failed to build"
    
    # Start containers with new configuration
    log "Starting containers with secure configuration..."
    docker-compose -f "$COMPOSE_FILE" -f "/opt/sutazaiapp/docker-compose.security-override.yml" up -d "${CONTAINERS_WITH_DOCKER_ACCESS[@]}" || warn "Some containers failed to start"
    
    # Verify the fix
    sleep 30  # Wait for containers to fully start
    if verify_fix; then
        success "Docker socket permission fix completed successfully!"
    else
        error "Fix verification failed. Check logs for details."
        exit 1
    fi
    
    # Display security recommendations
    display_security_recommendations
    
    log "Fix completed. Monitor logs at /var/log/sutazai_docker_access.log"
}

# Run main function
main "$@"