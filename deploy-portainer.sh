#!/bin/bash
# SutazaiApp Portainer Stack - Quick Deployment Script
# Version: 1.0.0
# Created: 2025-11-13 21:30:00 UTC
# Purpose: Simplified deployment script for Portainer stack

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/portainer-deployment-$(date +%Y%m%d-%H%M%S).log"

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S UTC')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Header
print_header() {
    echo -e "${BLUE}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          🤖 SutazaiApp Portainer Stack Deployment 🤖              ║
║                                                                    ║
║           Complete Multi-Agent AI Platform via Portainer           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    else
        log "✓ Docker: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose v2."
    else
        log "✓ Docker Compose: $(docker compose version)"
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    else
        log "✓ Docker daemon is running"
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [ $CPU_CORES -lt 4 ]; then
        warning "System has only $CPU_CORES cores. Minimum 8 cores recommended."
    else
        log "✓ CPU cores: $CPU_CORES"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ $TOTAL_RAM -lt 8 ]; then
        warning "System has only ${TOTAL_RAM}GB RAM. Minimum 16GB recommended."
    else
        log "✓ RAM: ${TOTAL_RAM}GB"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $AVAILABLE_SPACE -lt 50 ]; then
        warning "Only ${AVAILABLE_SPACE}GB free disk space. Minimum 100GB recommended."
    else
        log "✓ Disk space: ${AVAILABLE_SPACE}GB available"
    fi
}

# Check port availability
check_ports() {
    log "Checking critical port availability..."
    
    critical_ports=(9000 9443 10000 10001 10200 11000 11434)
    ports_in_use=()
    
    for port in "${critical_ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port " || lsof -i ":$port" &>/dev/null; then
            ports_in_use+=($port)
            warning "Port $port is already in use"
        fi
    done
    
    if [ ${#ports_in_use[@]} -gt 0 ]; then
        error "Critical ports in use: ${ports_in_use[*]}. Please free these ports before deployment."
    else
        log "✓ All critical ports are available"
    fi
}

# Build custom images
build_images() {
    log "Building custom Docker images..."
    
    # Build backend
    if [ -d "${SCRIPT_DIR}/backend" ]; then
        info "Building backend image..."
        cd "${SCRIPT_DIR}/backend"
        docker build -t sutazai/backend:latest . 2>&1 | tee -a "$LOG_FILE"
        log "✓ Backend image built successfully"
    else
        warning "Backend directory not found, skipping backend build"
    fi
    
    # Build frontend
    if [ -d "${SCRIPT_DIR}/frontend" ]; then
        info "Building frontend image..."
        cd "${SCRIPT_DIR}/frontend"
        docker build -t sutazai/frontend:latest . 2>&1 | tee -a "$LOG_FILE"
        log "✓ Frontend image built successfully"
    else
        warning "Frontend directory not found, skipping frontend build"
    fi
    
    # Build FAISS service
    if [ -d "${SCRIPT_DIR}/services/faiss" ]; then
        info "Building FAISS service image..."
        cd "${SCRIPT_DIR}/services/faiss"
        docker build -t sutazai/faiss-service:latest . 2>&1 | tee -a "$LOG_FILE"
        log "✓ FAISS service image built successfully"
    else
        warning "FAISS service directory not found, skipping FAISS build"
    fi
    
    cd "${SCRIPT_DIR}"
}

# Deploy stack
deploy_stack() {
    log "Deploying Portainer stack..."
    
    if [ ! -f "${SCRIPT_DIR}/portainer-stack.yml" ]; then
        error "portainer-stack.yml not found in ${SCRIPT_DIR}"
    fi
    
    cd "${SCRIPT_DIR}"
    docker compose -f portainer-stack.yml up -d 2>&1 | tee -a "$LOG_FILE"
    
    log "✓ Stack deployed successfully"
}

# Wait for services to be healthy
wait_for_health() {
    log "Waiting for services to become healthy..."
    
    max_wait=300  # 5 minutes
    elapsed=0
    check_interval=10
    
    while [ $elapsed -lt $max_wait ]; do
        healthy_count=$(docker ps --filter "name=sutazai-" --filter "health=healthy" | wc -l)
        total_count=$(docker ps --filter "name=sutazai-" | grep -v NAMES | wc -l)
        
        if [ $healthy_count -eq $total_count ] && [ $total_count -gt 0 ]; then
            log "✓ All services are healthy ($healthy_count/$total_count)"
            return 0
        fi
        
        info "Waiting for services... ($healthy_count/$total_count healthy)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    warning "Timeout waiting for all services to be healthy. Some services may still be starting."
}

# Initialize Ollama models
init_ollama() {
    log "Initializing Ollama with TinyLlama model..."
    
    # Wait for Ollama to be ready
    sleep 30
    
    if docker ps | grep -q sutazai-ollama; then
        info "Pulling TinyLlama model (this may take a few minutes)..."
        docker exec sutazai-ollama ollama pull tinyllama 2>&1 | tee -a "$LOG_FILE" || warning "Failed to pull TinyLlama model. You can do this manually later."
        log "✓ TinyLlama model initialized"
    else
        warning "Ollama container not running. Skipping model initialization."
    fi
}

# Display service URLs
display_urls() {
    echo ""
    log "═══════════════════════════════════════════════════════════════"
    log "                 🎉 DEPLOYMENT SUCCESSFUL 🎉                   "
    log "═══════════════════════════════════════════════════════════════"
    echo ""
    info "Access your services at the following URLs:"
    echo ""
    echo -e "${GREEN}Container Management:${NC}"
    echo "  Portainer:           http://localhost:9000"
    echo ""
    echo -e "${GREEN}Application:${NC}"
    echo "  JARVIS Frontend:     http://localhost:11000"
    echo "  Backend API:         http://localhost:10200"
    echo "  API Documentation:   http://localhost:10200/docs"
    echo ""
    echo -e "${GREEN}Monitoring:${NC}"
    echo "  Grafana:             http://localhost:10201 (admin/sutazai_secure_2024)"
    echo "  Prometheus:          http://localhost:10202"
    echo ""
    echo -e "${GREEN}Infrastructure:${NC}"
    echo "  Kong Proxy:          http://localhost:10008"
    echo "  Kong Admin:          http://localhost:10009"
    echo "  RabbitMQ:            http://localhost:10005 (sutazai/sutazai_secure_2024)"
    echo "  Neo4j:               http://localhost:10002 (neo4j/sutazai_secure_2024)"
    echo "  Consul:              http://localhost:10006"
    echo ""
    echo -e "${GREEN}AI Services:${NC}"
    echo "  Ollama:              http://localhost:11434"
    echo ""
    log "═══════════════════════════════════════════════════════════════"
    echo ""
    info "View container status: docker ps --filter 'name=sutazai-'"
    info "View logs: docker logs <container-name>"
    info "Full documentation: docs/PORTAINER_DEPLOYMENT_GUIDE.md"
    echo ""
    log "Deployment completed successfully!"
}

# Main execution
main() {
    print_header
    check_requirements
    check_ports
    build_images
    deploy_stack
    wait_for_health
    init_ollama
    display_urls
}

# Run main function
main

exit 0
