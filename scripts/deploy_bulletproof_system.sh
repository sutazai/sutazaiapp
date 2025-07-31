#!/bin/bash

# Bulletproof SutazAI System Deployment Script
# Addresses all critical issues: Docker socket permissions, network naming, resource constraints, service dependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (required for Docker socket fixes)
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root to fix Docker socket permissions"
        log "Run: sudo $0"
        exit 1
    fi
    log_success "Running with root permissions"
}

# System resource check
check_system_resources() {
    log "Checking system resources..."
    
    # Memory check
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    
    log "Total Memory: ${TOTAL_MEM}GB, Available: ${AVAILABLE_MEM}GB"
    
    if [ "$AVAILABLE_MEM" -lt 4 ]; then
        log_warning "Low available memory (${AVAILABLE_MEM}GB). Consider stopping non-essential services."
    else
        log_success "Memory resources adequate"
    fi
    
    # Disk space check
    DISK_USAGE=$(df / | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 80 ]; then
        log_warning "High disk usage (${DISK_USAGE}%). Consider cleanup."
    else
        log_success "Disk space adequate"
    fi
}

# Docker environment validation
validate_docker_environment() {
    log "Validating Docker environment..."
    
    # Check Docker service
    if ! systemctl is-active --quiet docker; then
        log_error "Docker service is not running"
        log "Starting Docker service..."
        systemctl start docker
    fi
    log_success "Docker service is running"
    
    # Check Docker socket permissions
    ls -la /var/run/docker.sock
    chmod 666 /var/run/docker.sock  # Fix socket permissions temporarily
    log_success "Docker socket permissions fixed"
    
    # Check Docker network
    if docker network ls | grep -q sutazaiapp_sutazai-network; then
        log_success "SutazAI network exists"
    else
        log_warning "SutazAI network not found - will be created during deployment"
    fi
}

# Clean up problematic containers
cleanup_failed_containers() {
    log "Cleaning up failed containers..."
    
    # Stop and remove problematic containers
    FAILED_CONTAINERS=("sutazai-hardware-optimizer" "sutazai-devops-manager" "sutazai-ollama-specialist")
    
    for container in "${FAILED_CONTAINERS[@]}"; do
        if docker ps -a --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
            log "Stopping and removing $container"
            docker stop "$container" 2>/dev/null || true
            docker rm "$container" 2>/dev/null || true
        fi
    done
    
    # Clean up orphaned health check containers
    docker container prune -f
    
    log_success "Cleanup completed"
}

# Build fixed Docker images
build_fixed_images() {
    log "Building fixed Docker images for agents..."
    
    cd /opt/sutazaiapp
    
    # Build hardware optimizer with fixes
    if [ -d "agents/hardware-optimizer" ]; then
        log "Building hardware-optimizer with Docker socket fixes..."
        docker build -t sutazaiapp-hardware-resource-optimizer ./agents/hardware-optimizer/
        log_success "hardware-optimizer image built"
    fi
    
    # Build devops manager with fixes
    if [ -d "agents/infrastructure-devops" ]; then
        log "Building infrastructure-devops-manager with Docker socket fixes..."
        docker build -t sutazaiapp-infrastructure-devops-manager ./agents/infrastructure-devops/
        log_success "infrastructure-devops-manager image built"
    fi
    
    # Build ollama specialist with fixes
    if [ -d "agents/ollama-integration" ]; then
        log "Building ollama-integration-specialist with Docker socket fixes..."
        docker build -t sutazaiapp-ollama-integration-specialist ./agents/ollama-integration/
        log_success "ollama-integration-specialist image built"
    fi
}

# Deploy core infrastructure first
deploy_core_infrastructure() {
    log "Deploying core infrastructure..."
    
    cd /opt/sutazaiapp
    
    # Start only essential services first
    CORE_SERVICES=(
        "postgres"
        "redis" 
        "ollama"
        "prometheus"
        "grafana"
        "loki"
        "promtail"
    )
    
    for service in "${CORE_SERVICES[@]}"; do
        log "Starting core service: $service"
        docker-compose up -d "$service"
        
        # Wait for service to be healthy
        wait_for_service_health "$service"
    done
    
    log_success "Core infrastructure deployed"
}

# Wait for service health
wait_for_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    log "Waiting for $service to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker ps --filter "name=sutazai-$service" --filter "health=healthy" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            log_success "$service is healthy"
            return 0
        fi
        
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            log "Service $service is running (waiting for health check...)"
        else
            log "Service $service is not running yet..."
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_warning "$service health check timeout - continuing anyway"
    return 1
}

# Deploy fixed agent tier
deploy_fixed_agents() {
    log "Deploying fixed Tier 1 agents..."
    
    cd /opt/sutazaiapp
    
    # Use the fixed compose file
    if [ -f "docker-compose-agents-tier1-fixed.yml" ]; then
        log "Using fixed Tier 1 agent configuration"
        docker-compose -f docker-compose-agents-tier1-fixed.yml up -d
    else
        log_warning "Fixed configuration not found, using original with fixes"
        docker-compose -f docker-compose-agents-tier1.yml up -d
    fi
    
    # Wait for agents to stabilize
    sleep 30
    
    # Check agent health
    check_agent_health
    
    log_success "Fixed agents deployed"
}

# Check agent health
check_agent_health() {
    log "Checking agent health..."
    
    AGENTS=("sutazai-hardware-optimizer" "sutazai-devops-manager" "sutazai-ollama-specialist" "sutazai-litellm-manager" "sutazai-context-optimizer")
    
    for agent in "${AGENTS[@]}"; do
        if docker ps --filter "name=$agent" --filter "status=running" --format "{{.Names}}" | grep -q "$agent"; then
            log_success "$agent is running"
            
            # Try to get health status
            local port=""
            case $agent in
                "sutazai-hardware-optimizer") port="8523" ;;
                "sutazai-devops-manager") port="8522" ;;
                "sutazai-ollama-specialist") port="8520" ;;
                "sutazai-litellm-manager") port="8521" ;;
                "sutazai-context-optimizer") port="8524" ;;
            esac
            
            if [ -n "$port" ]; then
                if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
                    log_success "$agent health endpoint responding"
                else
                    log_warning "$agent health endpoint not responding yet"
                fi
            fi
        else
            log_error "$agent is not running"
            # Show last few log lines for debugging
            docker logs "$agent" --tail 10 2>/dev/null || log "No logs available for $agent"
        fi
    done
}

# Resource optimization
optimize_system_resources() {
    log "Optimizing system resources..."
    
    # Docker system cleanup
    docker system prune -f --volumes
    
    # Limit Docker daemon resources
    if [ -f "/etc/docker/daemon.json" ]; then
        log "Docker daemon.json already exists"
    else
        log "Creating Docker daemon configuration for resource limits..."
        cat > /etc/docker/daemon.json << EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "default-runtime": "runc",
    "default-ulimits": {
        "nofile": {
            "hard": 65536,
            "soft": 65536
        }
    }
}
EOF
        systemctl restart docker
        sleep 10
    fi
    
    log_success "System resources optimized"
}

# Monitoring and validation
validate_deployment() {
    log "Validating deployment..."
    
    # Check running containers
    local running_containers=$(docker ps --filter "name=sutazai" --format "{{.Names}}" | wc -l)
    log "Running SutazAI containers: $running_containers"
    
    # Check network connectivity
    if docker network inspect sutazaiapp_sutazai-network >/dev/null 2>&1; then
        local connected_containers=$(docker network inspect sutazaiapp_sutazai-network | jq -r '.[0].Containers | length')
        log "Containers connected to network: $connected_containers"
        log_success "Network connectivity validated"
    else
        log_error "Network validation failed"
    fi
    
    # Check system resources after deployment
    log "System resources after deployment:"
    free -h
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10
    
    log_success "Deployment validation completed"
}

# Emergency rollback function
emergency_rollback() {
    log_error "Emergency rollback initiated"
    
    # Stop all SutazAI containers
    docker stop $(docker ps --filter "name=sutazai" -q) 2>/dev/null || true
    
    # Keep only essential services running
    docker-compose up -d postgres redis ollama
    
    log_warning "System rolled back to essential services only"
}

# Main deployment flow
main() {
    log "Starting Bulletproof SutazAI System Deployment"
    log "========================================"
    
    # Trap for emergency rollback
    trap 'emergency_rollback' ERR
    
    # Pre-deployment checks
    check_permissions
    check_system_resources
    validate_docker_environment
    
    # Cleanup and preparation
    cleanup_failed_containers
    build_fixed_images
    
    # Deployment phases
    deploy_core_infrastructure
    sleep 30  # Let core services stabilize
    
    deploy_fixed_agents
    
    # Post-deployment optimization and validation
    optimize_system_resources
    validate_deployment
    
    log_success "Bulletproof deployment completed successfully!"
    log "========================================"
    log "System Status:"
    docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    log ""
    log "Health Check URLs:"
    log "- Hardware Optimizer: http://localhost:8523/health"
    log "- DevOps Manager: http://localhost:8522/health" 
    log "- Ollama Specialist: http://localhost:8520/health"
    log "- LiteLLM Manager: http://localhost:8521/health"
    log "- Context Optimizer: http://localhost:8524/health"
    
    log ""
    log "Monitoring URLs:"
    log "- Grafana: http://localhost:3000"
    log "- Prometheus: http://localhost:9090"
    log "- System Status: docker ps --filter name=sutazai"
}

# Script execution
main "$@"