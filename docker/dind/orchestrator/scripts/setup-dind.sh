#!/bin/bash
set -euo pipefail

# Docker-in-Docker Setup and Initialization Script
# Sets up the DinD environment for MCP orchestration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIND_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMPOSE_FILE="${DIND_DIR}/docker-compose.dind.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose is not available"
        return 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running"
        return 1
    fi
    
    # Check if user can run Docker without sudo
    if ! docker ps >/dev/null 2>&1; then
        error "User cannot run Docker commands (check permissions)"
        return 1
    fi
    
    success "Prerequisites check passed"
    return 0
}

# Create necessary directories and files
setup_directories() {
    log "Setting up directory structure..."
    
    local dirs=(
        "${DIND_DIR}/orchestrator/configs"
        "${DIND_DIR}/volumes/mcp-shared"
        "${DIND_DIR}/volumes/mcp-logs"
        "${DIND_DIR}/networks"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    success "Directory structure setup complete"
}

# Create external network if needed
setup_networks() {
    log "Setting up Docker networks..."
    
    # Check if sutazai-network exists
    if ! docker network ls | grep -q sutazai-network; then
        log "Creating sutazai-network..."
        docker network create sutazai-network \
            --driver bridge \
            --subnet=172.20.0.0/16 \
            --gateway=172.20.0.1 \
            --opt com.docker.network.bridge.name=sutazai-br0
        success "Created sutazai-network"
    else
        log "sutazai-network already exists"
    fi
}

# Pull required images
pull_images() {
    log "Pulling required Docker images..."
    
    local images=(
        "docker:25.0.5-dind-alpine3.19"
        "python:3.11-alpine3.19"
    )
    
    for image in "${images[@]}"; do
        log "Pulling image: $image"
        if docker pull "$image"; then
            success "Pulled image: $image"
        else
            error "Failed to pull image: $image"
            return 1
        fi
    done
    
    success "All required images pulled"
}

# Start DinD environment
start_dind() {
    log "Starting Docker-in-Docker environment..."
    
    cd "$DIND_DIR"
    
    # Start the services
    if docker compose -f "$COMPOSE_FILE" up -d; then
        success "DinD environment started"
    else
        error "Failed to start DinD environment"
        return 1
    fi
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        local healthy_services=$(docker compose -f "$COMPOSE_FILE" ps --format json | \
            jq -r 'select(.Health == "healthy") | .Name' | wc -l)
        local total_services=$(docker compose -f "$COMPOSE_FILE" ps --format json | wc -l)
        
        if [ "$healthy_services" -eq "$total_services" ] && [ "$total_services" -gt 0 ]; then
            success "All services are healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $healthy_services/$total_services services healthy"
        sleep 5
        ((attempt++))
    done
    
    error "Services failed to become healthy within timeout"
    docker compose -f "$COMPOSE_FILE" logs
    return 1
}

# Stop DinD environment
stop_dind() {
    log "Stopping Docker-in-Docker environment..."
    
    cd "$DIND_DIR"
    
    if docker compose -f "$COMPOSE_FILE" down; then
        success "DinD environment stopped"
    else
        error "Failed to stop DinD environment"
        return 1
    fi
}

# Show status
show_status() {
    log "DinD Environment Status:"
    
    cd "$DIND_DIR"
    docker compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log "Service Endpoints:"
    echo "  - MCP Orchestrator API: http://localhost:18080"
    echo "  - MCP Manager UI: http://localhost:18081"
    echo "  - Docker Daemon (TLS): tcp://localhost:12376"
    echo "  - Docker Daemon (no TLS): tcp://localhost:12375"
    echo "  - Metrics: http://localhost:19090"
}

# Show logs
show_logs() {
    local service="${1:-}"
    
    cd "$DIND_DIR"
    
    if [[ -n "$service" ]]; then
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Cleanup volumes and networks
cleanup() {
    log "Cleaning up DinD environment..."
    
    cd "$DIND_DIR"
    
    # Stop and remove containers
    docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
    
    # Remove volumes (optional)
    read -p "Remove all DinD volumes? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume rm $(docker volume ls -q --filter name=sutazai-mcp) 2>/dev/null || true
        success "Volumes removed"
    fi
    
    success "Cleanup complete"
}

# Test DinD functionality
test_dind() {
    log "Testing DinD functionality..."
    
    # Test Docker API access
    if curl -sf http://localhost:18081/health >/dev/null; then
        success "MCP Manager is accessible"
    else
        error "MCP Manager is not accessible"
        return 1
    fi
    
    # Test container deployment
    if "${SCRIPT_DIR}/deploy-mcp.sh" health; then
        success "DinD functionality test passed"
    else
        error "DinD functionality test failed"
        return 1
    fi
}

# Main function
main() {
    local command="${1:-start}"
    
    case "$command" in
        "start")
            check_prerequisites || exit 1
            setup_directories
            setup_networks
            pull_images || exit 1
            start_dind || exit 1
            show_status
            ;;
        "stop")
            stop_dind
            ;;
        "restart")
            stop_dind
            sleep 2
            start_dind || exit 1
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "cleanup")
            cleanup
            ;;
        "test")
            test_dind
            ;;
        "setup")
            check_prerequisites || exit 1
            setup_directories
            setup_networks
            pull_images || exit 1
            success "DinD setup complete. Run '$0 start' to start the environment."
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs|cleanup|test|setup}"
            echo ""
            echo "Commands:"
            echo "  start   - Start the DinD environment"
            echo "  stop    - Stop the DinD environment"
            echo "  restart - Restart the DinD environment"
            echo "  status  - Show current status"
            echo "  logs    - Show logs (optionally for specific service)"
            echo "  cleanup - Stop and clean up all resources"
            echo "  test    - Test DinD functionality"
            echo "  setup   - Setup only (no start)"
            exit 1
            ;;
    esac
}

# Make scripts executable
chmod +x "${SCRIPT_DIR}"/*.sh

# Run main function
main "$@"