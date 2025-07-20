#!/bin/bash

# ==============================================================================
# SutazAI Production Deployment Script
# ==============================================================================
# Deploys the optimized, enterprise-grade SutazAI system

set -euo pipefail

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

# Check prerequisites
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
    
    # Check if secrets directory exists
    if [ ! -d "./secrets" ]; then
        log_warning "Secrets directory not found, creating..."
        mkdir -p secrets
        echo "change-this-password" > secrets/postgres_password.txt
        echo "admin123" > secrets/grafana_password.txt
        chmod 600 secrets/*
        log_warning "Default passwords created. Please change them for production!"
    fi
    
    log_success "Prerequisites check completed"
}

# Stop existing services
stop_existing_services() {
    log_info "Stopping existing services..."
    
    # Stop old Docker Compose setup
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    log_success "Existing services stopped"
}

# Build and deploy optimized system
deploy_optimized_system() {
    log_info "Deploying optimized SutazAI system..."
    
    # Use the optimized Docker Compose file
    export COMPOSE_FILE="docker-compose.optimized.yml"
    
    # Build services
    log_info "Building services..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    
    # Start core infrastructure first
    log_info "Starting core infrastructure..."
    docker-compose -f $COMPOSE_FILE up -d postgres redis
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    sleep 30
    
    # Start AI services
    log_info "Starting AI services..."
    docker-compose -f $COMPOSE_FILE up -d chromadb qdrant ollama
    
    # Wait for AI services
    log_info "Waiting for AI services to be ready..."
    sleep 20
    
    # Start application services
    log_info "Starting application services..."
    docker-compose -f $COMPOSE_FILE up -d backend frontend
    
    # Wait for application services
    log_info "Waiting for application services..."
    sleep 15
    
    # Start monitoring
    log_info "Starting monitoring services..."
    docker-compose -f $COMPOSE_FILE up -d prometheus grafana
    
    # Start reverse proxy
    log_info "Starting reverse proxy..."
    docker-compose -f $COMPOSE_FILE up -d nginx
    
    log_success "Optimized system deployed successfully"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check backend health
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_success "Backend is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Backend health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check frontend
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        log_success "Frontend is healthy"
    else
        log_warning "Frontend health check failed, but backend is working"
    fi
    
    log_success "Health checks completed"
}

# Download and setup models
setup_models() {
    log_info "Setting up AI models..."
    
    # Wait for Ollama to be ready
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:11434/api/tags &>/dev/null; then
            log_success "Ollama is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Ollama failed to start"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Pull essential models
    log_info "Pulling essential AI models..."
    docker exec sutazai-ollama ollama pull llama3 &
    docker exec sutazai-ollama ollama pull codellama &
    
    wait
    log_success "AI models setup completed"
}

# Show system status
show_status() {
    log_info "System Status:"
    echo "=================================="
    
    # Show running containers
    docker-compose -f docker-compose.optimized.yml ps
    
    echo ""
    log_info "Access URLs:"
    echo "🌐 Web UI: http://localhost:8501"
    echo "🔧 API Docs: http://localhost:8000/docs"
    echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
    echo "📈 Prometheus: http://localhost:9090"
    echo "🤖 Ollama API: http://localhost:11434"
    
    echo ""
    log_info "System Health:"
    
    # Check services
    services=("backend:8000/health" "frontend:8501/_stcore/health")
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        url="http://localhost:${service#*:}"
        
        if curl -f "$url" &>/dev/null; then
            echo "✅ $name: Healthy"
        else
            echo "❌ $name: Unhealthy"
        fi
    done
}

# Main deployment function
main() {
    log_info "Starting SutazAI optimized deployment..."
    echo "============================================="
    
    check_prerequisites
    stop_existing_services
    deploy_optimized_system
    health_check
    setup_models
    show_status
    
    echo ""
    log_success "🎉 SutazAI optimized deployment completed successfully!"
    log_info "The system is now running and ready to use."
    log_info "Visit http://localhost:8501 to access the web interface."
}

# Cleanup function for Ctrl+C
cleanup() {
    log_warning "Deployment interrupted by user"
    log_info "Cleaning up..."
    docker-compose -f docker-compose.optimized.yml down
    exit 1
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Run main function
main "$@"