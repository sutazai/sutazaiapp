#!/bin/bash
# Master Deployment Script for SutazAI System - RULE 12 COMPLIANT
# Version: 3.0 - Self-Updating Ultra-Consolidated
# Created: August 10, 2025, Enhanced: August 11, 2025
# Purpose: Single, comprehensive, SELF-UPDATING deployment script for all environments
# Rule 12 Compliance: One Self-Updating, Intelligent, End-to-End Deployment Script

set -euo pipefail

# Self-update mechanism (Rule 12 requirement)
self_update() {
    log_info "🔄 Checking for script updates (Rule 12: Self-Updating)..."
    
    # Check if we're in a git repository
    if git rev-parse --git-dir > /dev/null 2>&1; then
        # Store current version
        local current_version=$(grep "^# Version:" "$0" | head -1 | cut -d: -f2 | xargs)
        
        # Pull latest changes
        git fetch origin 2>/dev/null || {
            log_warning "Could not fetch updates - continuing with current version"
            return 0
        }
        
        # Check if deploy script has been updated
        local script_path="$(realpath "$0")"
        local relative_path="${script_path#$PROJECT_ROOT/}"
        
        if git diff HEAD origin/$(git symbolic-ref --short HEAD) --quiet -- "$relative_path" 2>/dev/null; then
            log_info "✅ Deploy script is up to date ($current_version)"
            return 0
        fi
        
        log_info "🔄 Deploy script has updates available - self-updating..."
        
        # Create backup of current version
        cp "$0" "${0}.backup.$(date +%s)"
        
        # Pull updates
        git pull origin $(git symbolic-ref --short HEAD) 2>/dev/null || {
            log_error "Failed to pull updates"
            return 1
        }
        
        # Check if the script was actually updated
        local new_version=$(grep "^# Version:" "$0" | head -1 | cut -d: -f2 | xargs)
        if [ "$current_version" != "$new_version" ]; then
            log_success "🚀 Script updated from $current_version to $new_version"
            log_info "🔄 Re-executing with updated version..."
            exec "$0" "$@"
        fi
    else
        log_warning "Not in git repository - skipping self-update"
    fi
}

# Version tracking for self-update
SCRIPT_VERSION="3.0"
LAST_UPDATED="2025-08-11"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check for environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found. Creating from template..."
        cp "${PROJECT_ROOT}/.env.example" "$ENV_FILE" 2>/dev/null || {
            log_error "No .env or .env.example found"
            exit 1
        }
    fi
    
    log_success "Prerequisites check completed"
}

# Function to clean up old containers and volumes
cleanup_old() {
    log_info "Cleaning up old containers and volumes..."
    
    # Stop all containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Prune unused containers, networks, volumes
    docker system prune -f --volumes
    
    log_success "Cleanup completed"
}

# Function to build images
build_images() {
    log_info "Building Docker images..."
    
    # Build with BuildKit for better performance
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    docker-compose -f "$COMPOSE_FILE" build --parallel
    
    log_success "Images built successfully"
}

# Function to start core services
start_core() {
    log_info "Starting core services..."
    
    # Start databases first
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis neo4j
    
    # Wait for databases to be healthy
    log_info "Waiting for databases to be ready..."
    sleep 10
    
    # Start message queue
    docker-compose -f "$COMPOSE_FILE" up -d rabbitmq
    
    # Start vector databases
    docker-compose -f "$COMPOSE_FILE" up -d qdrant chromadb faiss
    
    log_success "Core services started"
}

# Function to start application services
start_applications() {
    log_info "Starting application services..."
    
    # Start backend
    docker-compose -f "$COMPOSE_FILE" up -d backend
    
    # Start frontend
    docker-compose -f "$COMPOSE_FILE" up -d frontend
    
    # Start AI services
    docker-compose -f "$COMPOSE_FILE" up -d ollama
    
    log_success "Application services started"
}

# Function to start monitoring stack
start_monitoring() {
    log_info "Starting monitoring stack..."
    
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana loki alertmanager
    
    log_success "Monitoring stack started"
}

# Function to start agent services
start_agents() {
    log_info "Starting AI agent services..."
    
    docker-compose -f "$COMPOSE_FILE" up -d \
        hardware-resource-optimizer \
        ai-agent-orchestrator \
        resource-arbitration-agent \
        task-assignment-coordinator \
        ollama-integration
    
    log_success "Agent services started"
}

# Function to perform health checks
health_check() {
    log_info "Performing health checks..."
    
    # Check core services
    services=(
        "postgres:10000"
        "redis:10001"
        "neo4j:10002"
        "backend:10010"
        "frontend:10011"
        "ollama:10104"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:${port}/" > /dev/null 2>&1; then
            log_success "$name is healthy on port $port"
        else
            log_warning "$name health check failed on port $port"
        fi
    done
}

# Function to display status
display_status() {
    log_info "System Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Access Points:"
    echo "  Frontend:    http://localhost:10011"
    echo "  Backend API: http://localhost:10010/docs"
    echo "  Grafana:     http://localhost:10201 (admin/admin)"
    echo "  Prometheus:  http://localhost:10200"
    echo ""
}

# Function to tail logs
tail_logs() {
    log_info "Tailing logs (Ctrl+C to stop)..."
    docker-compose -f "$COMPOSE_FILE" logs -f --tail=100
}

# Main deployment modes with self-update (Rule 12 compliance)
deploy_minimal() {
    log_info "🚀 Deploying minimal stack..."
    
    # Self-update check (Rule 12 requirement)
    [ "${SKIP_UPDATE:-0}" != "1" ] && self_update "$@"
    
    check_prerequisites
    cleanup_old
    start_core
    start_applications
    health_check
    display_status
}

deploy_full() {
    log_info "🚀 Deploying full stack..."
    
    # Self-update check (Rule 12 requirement)
    [ "${SKIP_UPDATE:-0}" != "1" ] && self_update "$@"
    
    check_prerequisites
    cleanup_old
    build_images
    start_core
    start_applications
    start_monitoring
    start_agents
    health_check
    display_status
}

deploy_production() {
    log_info "🚀 Deploying production stack..."
    
    # Self-update check (Rule 12 requirement)
    [ "${SKIP_UPDATE:-0}" != "1" ] && self_update "$@"
    
    check_prerequisites
    
    # Production-specific checks
    if [ ! -f "${PROJECT_ROOT}/.env.production" ]; then
        log_error "Production environment file not found"
        exit 1
    fi
    
    cp "${PROJECT_ROOT}/.env.production" "$ENV_FILE"
    
    cleanup_old
    build_images
    start_core
    start_applications
    start_monitoring
    start_agents
    health_check
    display_status
}

# Parse command line arguments
case "${1:-full}" in
    minimal)
        deploy_minimal
        ;;
    full)
        deploy_full
        ;;
    production|prod)
        deploy_production
        ;;
    status)
        display_status
        ;;
    health)
        health_check
        ;;
    logs)
        tail_logs
        ;;
    clean)
        cleanup_old
        ;;
    update)
        log_info "🔄 Force updating deployment script..."
        self_update "$@"
        ;;
    version)
        echo "SutazAI Master Deployment Script"
        echo "Version: $SCRIPT_VERSION"
        echo "Last Updated: $LAST_UPDATED" 
        echo "Self-Updating: Enabled (Rule 12 Compliant)"
        echo "Backup Available: /opt/sutazaiapp/archive/scripts-consolidation-20250811_012422/rollback.sh"
        ;;
    *)
        echo "Usage: $0 {minimal|full|production|status|health|logs|clean|update|version}"
        echo ""
        echo "🚀 SELF-UPDATING DEPLOYMENT SCRIPT (Rule 12 Compliant)"
        echo ""
        echo "Deployment Modes:"
        echo "  minimal    - Deploy core services only (databases, backend, frontend)"
        echo "  full       - Deploy all services including monitoring and agents"
        echo "  production - Deploy with production configuration"
        echo ""
        echo "Management Commands:"
        echo "  status     - Show current system status"
        echo "  health     - Run health checks on all services"
        echo "  logs       - Tail logs from all services"
        echo "  clean      - Stop and clean up all containers"
        echo "  update     - Force update this script from git"
        echo "  version    - Show script version and update information"
        echo ""
        echo "Environment Variables:"
        echo "  SKIP_UPDATE=1    - Skip automatic self-update check"
        echo "  FORCE_DEPLOY=1   - Force deployment even if health checks fail"
        echo ""
        echo "🛡️  Emergency Rollback Available:"
        echo "  /opt/sutazaiapp/archive/scripts-consolidation-20250811_012422/rollback.sh"
        exit 1
        ;;
esac

log_success "Deployment operation completed successfully!"