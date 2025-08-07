#!/bin/bash
# JARVIS Unified Voice Interface Deployment Script
# Deploys the complete JARVIS system with all dependencies

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/jarvis_deployment_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

info() {
    log "${BLUE}INFO: $1${NC}"
}

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

info "Starting JARVIS deployment..."
info "Project root: $PROJECT_ROOT"
info "Log file: $LOG_FILE"

# Function to check if Docker is running
check_docker() {
    info "Checking Docker status..."
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker and try again."
    fi
    success "Docker is running"
}

# Function to check if required files exist
check_prerequisites() {
    info "Checking prerequisites..."
    
    local required_files=(
        "docker-compose.jarvis.yml"
        "services/jarvis/Dockerfile"
        "services/jarvis/main.py"
        "config/jarvis/config.yaml"
        "services/jarvis/requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            error "Required file missing: $file"
        fi
    done
    
    success "All required files found"
}

# Function to create necessary directories
create_directories() {
    info "Creating necessary directories..."
    
    local dirs=(
        "$PROJECT_ROOT/data/jarvis"
        "$PROJECT_ROOT/logs/jarvis"
        "$PROJECT_ROOT/services/jarvis/plugins"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        info "Created directory: $dir"
    done
    
    success "Directories created"
}

# Function to check system requirements
check_system_requirements() {
    info "Checking system requirements..."
    
    # Check available memory
    local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [[ $total_mem -lt 4096 ]]; then
        warning "System has less than 4GB RAM ($total_mem MB). JARVIS may not perform optimally."
    else
        success "Memory check passed: ${total_mem}MB available"
    fi
    
    # Check disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    if [[ $available_gb -lt 10 ]]; then
        warning "Less than 10GB disk space available (${available_gb}GB). Consider freeing up space."
    else
        success "Disk space check passed: ${available_gb}GB available"
    fi
    
    # Check if audio devices are available
    if [[ -c /dev/snd/controlC0 ]]; then
        success "Audio devices detected"
    else
        warning "No audio devices detected. Voice features may not work."
    fi
}

# Function to setup environment
setup_environment() {
    info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        cat > "$PROJECT_ROOT/.env" << EOF
# JARVIS Environment Configuration
SUTAZAI_ENV=production
TZ=UTC

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=sutazai

# Redis Configuration
REDIS_PASSWORD=

# JWT Secret
JWT_SECRET=$(openssl rand -base64 64)

# Ollama Configuration
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*

# JARVIS Configuration
JARVIS_DEBUG=false
JARVIS_LOG_LEVEL=INFO
EOF
        info "Created .env file with default configuration"
    else
        info "Using existing .env file"
    fi
    
    success "Environment setup complete"
}

# Function to pull required Docker images
pull_images() {
    info "Pulling required Docker images..."
    
    # Base images
    docker pull python:3.11-slim || warning "Failed to pull python:3.11-slim"
    docker pull postgres:16.3-alpine || warning "Failed to pull postgres:16.3-alpine"
    docker pull redis:7.2-alpine || warning "Failed to pull redis:7.2-alpine"
    
    success "Docker images pulled"
}

# Function to build JARVIS image
build_jarvis() {
    info "Building JARVIS Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the JARVIS image
    if docker build -t sutazai/jarvis:latest -f services/jarvis/Dockerfile services/jarvis/; then
        success "JARVIS image built successfully"
    else
        error "Failed to build JARVIS image"
    fi
}

# Function to setup Kong routes
setup_kong_routes() {
    info "Setting up Kong API Gateway routes..."
    
    # Wait for Kong to be ready
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8001/status >/dev/null 2>&1; then
            success "Kong is ready"
            break
        fi
        
        ((attempt++))
        info "Waiting for Kong... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        warning "Kong not available, skipping route setup"
        return
    fi
    
    # Configure Kong routes for JARVIS
    curl -X POST http://localhost:8001/services \
        --data name=jarvis-service \
        --data url=http://jarvis:8888 || warning "Failed to create Kong service"
        
    curl -X POST http://localhost:8001/services/jarvis-service/routes \
        --data paths[]=/jarvis \
        --data paths[]=/voice \
        --data paths[]=/assistant || warning "Failed to create Kong routes"
        
    success "Kong routes configured"
}

# Function to verify Ollama integration
verify_ollama() {
    info "Verifying Ollama integration..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:10104/api/tags >/dev/null 2>&1; then
            success "Ollama is available"
            
            # Check if tinyllama is available
            if curl -s http://localhost:10104/api/tags | grep -q "tinyllama"; then
                success "tinyllama model is available"
            else
                info "Pulling tinyllama model..."
                curl -X POST http://localhost:10104/api/pull \
                    -H "Content-Type: application/json" \
                    -d '{"name": "tinyllama"}' || warning "Failed to pull tinyllama"
            fi
            break
        fi
        
        ((attempt++))
        info "Waiting for Ollama... (attempt $attempt/$max_attempts)"
        sleep 2
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        warning "Ollama not available. JARVIS will use fallback modes."
    fi
}

# Function to deploy JARVIS
deploy_jarvis() {
    info "Deploying JARVIS services..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy using docker-compose
    if docker-compose -f docker-compose.jarvis.yml up -d; then
        success "JARVIS services deployed"
    else
        error "Failed to deploy JARVIS services"
    fi
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8888/health >/dev/null 2>&1; then
            success "JARVIS is ready and healthy"
            break
        fi
        
        ((attempt++))
        info "Waiting for JARVIS health check... (attempt $attempt/$max_attempts)"
        sleep 5
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "JARVIS health check failed"
    fi
}

# Function to run integration tests
run_tests() {
    info "Running integration tests..."
    
    # Test API endpoints
    local endpoints=(
        "http://localhost:8888/health"
        "http://localhost:8888/api/agents"
        "http://localhost:8888/metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" >/dev/null; then
            success "Endpoint test passed: $endpoint"
        else
            warning "Endpoint test failed: $endpoint"
        fi
    done
    
    # Test voice interface (if audio available)
    if [[ -c /dev/snd/controlC0 ]]; then
        info "Audio devices available - voice interface should work"
    else
        warning "No audio devices - voice interface will be limited"
    fi
    
    success "Integration tests completed"
}

# Function to display deployment summary
show_summary() {
    info "JARVIS Deployment Summary"
    echo "=================================="
    echo "JARVIS Voice Interface: http://localhost:8888"
    echo "Health Check: http://localhost:8888/health"
    echo "API Documentation: http://localhost:8888/docs"
    echo "Metrics: http://localhost:8888/metrics"
    echo ""
    echo "Kong Gateway Routes:"
    echo "- /jarvis -> JARVIS Service"
    echo "- /voice -> JARVIS Voice Interface"
    echo "- /assistant -> JARVIS Assistant"
    echo ""
    echo "Logs: $LOG_FILE"
    echo "Data Directory: $PROJECT_ROOT/data/jarvis"
    echo "Config Directory: $PROJECT_ROOT/config/jarvis"
    echo "=================================="
    
    success "JARVIS deployment completed successfully!"
}

# Function to cleanup on failure
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Deployment failed. Check logs: $LOG_FILE"
        info "Cleaning up partial deployment..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.jarvis.yml" down 2>/dev/null || true
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main deployment sequence
main() {
    info "Starting JARVIS deployment process..."
    
    check_docker
    check_prerequisites
    check_system_requirements
    create_directories
    setup_environment
    pull_images
    build_jarvis
    deploy_jarvis
    verify_ollama
    setup_kong_routes
    run_tests
    show_summary
    
    success "JARVIS deployment completed successfully!"
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        check_docker
        build_jarvis
        ;;
    "test")
        run_tests
        ;;
    "cleanup")
        info "Cleaning up JARVIS deployment..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.jarvis.yml" down
        docker rmi sutazai/jarvis:latest 2>/dev/null || true
        success "Cleanup completed"
        ;;
    "logs")
        docker-compose -f "$PROJECT_ROOT/docker-compose.jarvis.yml" logs -f jarvis
        ;;
    "status")
        docker-compose -f "$PROJECT_ROOT/docker-compose.jarvis.yml" ps
        ;;
    *)
        echo "Usage: $0 {deploy|build|test|cleanup|logs|status}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Full JARVIS deployment (default)"
        echo "  build   - Build JARVIS Docker image only"
        echo "  test    - Run integration tests"
        echo "  cleanup - Remove JARVIS deployment"
        echo "  logs    - Show JARVIS logs"
        echo "  status  - Show service status"
        exit 1
        ;;
esac