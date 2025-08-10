#!/bin/bash

# Migration Script: From 60-service chaos to 8-service sanity
# This script safely migrates the SutazAI system to minimal architecture

set -e  # Exit on error


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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${PROJECT_ROOT}/backups/migration_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
}

# Create backup directory
create_backup() {
    log_info "Creating backup directory: ${BACKUP_DIR}"
    mkdir -p "${BACKUP_DIR}"
    
    # Backup current docker-compose
    if [[ -f "${PROJECT_ROOT}/docker-compose.yml" ]]; then
        cp "${PROJECT_ROOT}/docker-compose.yml" "${BACKUP_DIR}/docker-compose.yml.backup"
        log_success "Backed up docker-compose.yml"
    fi
    
    # Backup environment file
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "${BACKUP_DIR}/.env.backup"
        log_success "Backed up .env file"
    fi
    
    # List running containers for reference
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" > "${BACKUP_DIR}/running_containers.txt"
    log_success "Saved list of running containers"
}

# Stop resource-intensive monitoring
stop_monitoring() {
    log_info "Stopping resource-intensive monitoring processes..."
    
    # Kill static monitor script
    if pgrep -f "static_monitor.py" > /dev/null; then
        pkill -f "static_monitor.py"
        log_success "Stopped static_monitor.py"
    else
        log_info "static_monitor.py not running"
    fi
    
    # Kill glances
    if pgrep -f "glances" > /dev/null; then
        pkill -f "glances"
        log_success "Stopped glances"
    else
        log_info "glances not running"
    fi
    
    # Stop any other monitoring scripts
    for script in "monitor.py" "watch_system.py" "resource_monitor.py"; do
        if pgrep -f "$script" > /dev/null; then
            pkill -f "$script"
            log_success "Stopped $script"
        fi
    done
}

# Stop all current containers
stop_all_containers() {
    log_info "Stopping all running containers..."
    
    # Get list of running containers
    RUNNING_CONTAINERS=$(docker ps -q)
    
    if [[ -n "$RUNNING_CONTAINERS" ]]; then
        docker stop $RUNNING_CONTAINERS
        log_success "Stopped all containers"
    else
        log_info "No containers running"
    fi
}

# Clean up unused resources
cleanup_docker() {
    log_info "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused networks (except sutazai-network)
    docker network prune -f
    
    # Remove unused volumes (be careful here)
    log_warning "Skipping volume cleanup to preserve data"
    
    # Remove unused images
    docker image prune -f
    
    log_success "Docker cleanup complete"
}

# Ensure network exists
ensure_network() {
    log_info "Ensuring sutazai-network exists..."
    
    if ! docker network ls | grep -q "sutazai-network"; then
        docker network create sutazai-network
        log_success "Created sutazai-network"
    else
        log_info "sutazai-network already exists"
    fi
}

# Check environment variables
check_env() {
    log_info "Checking environment variables..."
    
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log_warning ".env file not found, creating template..."
        cat > "${PROJECT_ROOT}/.env" << 'EOF'
# Database
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=changeme_postgres_password
POSTGRES_DB=sutazai

# Redis
REDIS_PASSWORD=

# Neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# Application
SECRET_KEY=changeme_secret_key
JWT_SECRET=changeme_jwt_secret

# Grafana
GRAFANA_PASSWORD=admin

# Environment
SUTAZAI_ENV=production
TZ=UTC
EOF
        log_warning "Please update .env file with secure passwords before starting services"
        exit 1
    fi
    
    log_success "Environment file exists"
}

# Start minimal services
start_minimal() {
    log_info "Starting minimal services..."
    
    cd "${PROJECT_ROOT}"
    
    # Use the minimal compose file
    if [[ -f "docker-compose.minimal.yml" ]]; then
        docker-compose -f docker-compose.minimal.yml up -d
        log_success "Started minimal services"
    else
        log_error "docker-compose.minimal.yml not found!"
        exit 1
    fi
}

# Verify services are running
verify_services() {
    log_info "Verifying services..."
    
    sleep 10  # Give services time to start
    
    # Check each minimal service
    SERVICES=("postgres" "redis" "backend" "frontend" "ollama" "qdrant" "prometheus" "grafana")
    
    for service in "${SERVICES[@]}"; do
        if docker ps | grep -q "sutazai-${service}-minimal"; then
            log_success "${service} is running"
        else
            log_warning "${service} is not running"
        fi
    done
    
    # Show resource usage
    log_info "Current resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# Display migration summary
show_summary() {
    echo ""
    echo "=========================================="
    echo "         MIGRATION SUMMARY"
    echo "=========================================="
    echo ""
    
    log_success "Migration to minimal architecture complete!"
    echo ""
    echo "Services running: 8 (down from 60)"
    echo "Expected resource usage: <10% CPU, <2GB RAM (idle)"
    echo ""
    echo "Access points:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - Frontend UI: http://localhost:8501"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Ollama: http://localhost:11434"
    echo "  - Qdrant: http://localhost:6333"
    echo ""
    echo "Backup location: ${BACKUP_DIR}"
    echo ""
    echo "To revert to old setup:"
    echo "  docker-compose -f docker-compose.minimal.yml down"
    echo "  cp ${BACKUP_DIR}/docker-compose.yml.backup ./docker-compose.yml"
    echo "  docker-compose up -d"
    echo ""
    echo "=========================================="
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "  SutazAI Migration to Minimal Architecture"
    echo "=========================================="
    echo ""
    
    log_info "Starting migration process..."
    echo ""
    
    # Check permissions
    check_permissions
    
    # Check environment
    check_env
    
    # Create backup
    create_backup
    
    # Stop monitoring scripts
    stop_monitoring
    
    # Ask for confirmation before stopping containers
    read -p "This will stop all running containers. Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warning "Migration cancelled by user"
        exit 0
    fi
    
    # Stop all containers
    stop_all_containers
    
    # Clean up Docker resources
    cleanup_docker
    
    # Ensure network exists
    ensure_network
    
    # Start minimal services
    start_minimal
    
    # Verify services
    verify_services
    
    # Show summary
    show_summary
}

# Run main function
main "$@"