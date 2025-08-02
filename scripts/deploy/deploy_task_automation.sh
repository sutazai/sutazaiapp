#!/bin/bash
# SutazAI Task Automation Platform Deployment
# Comprehensive deployment script for multi-agent task automation system

# ===============================================
# CONFIGURATION
# ===============================================

# Enable Docker BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Set error handling
set -euo pipefail

# Project paths
PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/deployment_$TIMESTAMP.log"

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

log_success() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1" | tee -a "$LOG_FILE"
}

# ===============================================
# DEPLOYMENT FUNCTIONS
# ===============================================

check_prerequisites() {
    log "Checking prerequisites..."
    
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
    
    # Check environment file
    if [[ ! -f .env ]]; then
        log "Creating .env from template..."
        cp .env.example .env
    fi
    
    log_success "Prerequisites check completed"
}

deploy_core_services() {
    log "Deploying core services..."
    
    # Deploy PostgreSQL and Redis first
    docker-compose up -d postgres redis
    
    # Wait for services to be healthy
    log "Waiting for database to be ready..."
    sleep 10
    
    # Deploy backend services
    docker-compose up -d backend task-coordinator
    
    log_success "Core services deployed"
}

deploy_ollama() {
    log "Deploying Ollama service..."
    
    # Deploy Ollama
    docker-compose -f docker-compose.tinyllama.yml up -d ollama
    
    # Wait for Ollama to start
    log "Waiting for Ollama to initialize..."
    sleep 20
    
    # Pull TinyLlama model
    log "Pulling TinyLlama model..."
    docker exec sutazai-ollama-tiny ollama pull tinyllama || true
    
    log_success "Ollama deployed with TinyLlama"
}

deploy_agents() {
    log "Deploying task automation agents..."
    
    # Deploy agent services
    docker-compose -f docker-compose.agents.yml up -d
    
    log_success "Agents deployed"
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Backend API is healthy"
    else
        log_error "Backend API health check failed"
        return 1
    fi
    
    # Check Ollama
    if curl -f http://localhost:11435/api/tags > /dev/null 2>&1; then
        log_success "Ollama API is healthy"
    else
        log_error "Ollama API health check failed"
        return 1
    fi
    
    # List running containers
    log "Running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    return 0
}

# ===============================================
# MAIN DEPLOYMENT
# ===============================================

main() {
    log "Starting SutazAI Task Automation Platform deployment..."
    
    # Check prerequisites
    check_prerequisites
    
    # Deploy services
    deploy_core_services
    deploy_ollama
    deploy_agents
    
    # Verify deployment
    if verify_deployment; then
        log_success "Deployment completed successfully!"
        log "Access the platform at:"
        log "  - Backend API: http://localhost:8000"
        log "  - API Docs: http://localhost:8000/docs"
        log "  - Task Coordinator: http://localhost:8522"
    else
        log_error "Deployment verification failed"
        exit 1
    fi
}

# Run deployment
main "$@"