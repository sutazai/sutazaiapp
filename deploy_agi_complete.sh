#!/bin/bash
#
# SutazAI Complete AGI/ASI System Deployment
# Master script to deploy the entire autonomous AI system
#

set -euo pipefail

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires root privileges. Please run with sudo:"
    echo "sudo $0 $@"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/agi_deployment_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: Deployment failed at line $1" "$RED"
    log "Check log file: $LOG_FILE" "$RED"
    exit 1
}
trap 'handle_error $LINENO' ERR

# Header
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           SutazAI AGI/ASI System Deployment                  ║"
    echo "║           Enterprise-Grade Autonomous AI System              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..." "$BLUE"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "Docker not found. Please install Docker first." "$RED"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Docker Compose not found. Please install Docker Compose first." "$RED"
        exit 1
    fi
    
    # Check if user can run Docker
    if ! docker ps &> /dev/null; then
        if ! sudo docker ps &> /dev/null; then
            log "Cannot connect to Docker. Make sure Docker is running." "$RED"
            exit 1
        else
            log "Docker requires sudo. Please run this script with sudo." "$YELLOW"
            log "Usage: sudo ./deploy_agi_complete.sh" "$YELLOW"
            exit 1
        fi
    fi
    
    # Check disk space (require at least 50GB)
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 50 ]; then
        log "Insufficient disk space. At least 50GB required, found ${available_space}GB" "$RED"
        exit 1
    fi
    
    # Check memory (recommend at least 16GB)
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_memory" -lt 16 ]; then
        log "WARNING: Less than 16GB RAM detected (${total_memory}GB). Performance may be affected." "$YELLOW"
    fi
    
    log "Prerequisites check passed ✓" "$GREEN"
}

# Clean up existing containers
cleanup_existing() {
    log "Cleaning up existing containers..." "$BLUE"
    
    # Stop all sutazai containers
    containers=$(docker ps -a --format '{{.Names}}' | grep '^sutazai-' || true)
    if [ -n "$containers" ]; then
        echo "$containers" | while read container; do
            log "Stopping $container..." "$YELLOW"
            docker stop "$container" 2>/dev/null || true
            docker rm "$container" 2>/dev/null || true
        done
    else
        log "No existing containers to clean up" "$BLUE"
    fi
    
    # Remove dangling images
    docker image prune -f > /dev/null 2>&1 || true
    
    log "Cleanup completed ✓" "$GREEN"
}

# Create necessary directories
create_directories() {
    log "Creating directory structure..." "$BLUE"
    
    directories=(
        "$PROJECT_ROOT/agents/dockerfiles"
        "$PROJECT_ROOT/agents/configs"
        "$PROJECT_ROOT/agents/workspaces"
        "$PROJECT_ROOT/data/models"
        "$PROJECT_ROOT/data/embeddings"
        "$PROJECT_ROOT/data/knowledge"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/monitoring/prometheus"
        "$PROJECT_ROOT/monitoring/grafana"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log "Directory structure created ✓" "$GREEN"
}

# Generate environment file
generate_env_file() {
    log "Generating environment configuration..." "$BLUE"
    
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        cat > "${PROJECT_ROOT}/.env" << EOF
# SutazAI AGI/ASI System Environment Configuration

# Database
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
POSTGRES_DB=sutazai_main

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Neo4j
NEO4J_AUTH=neo4j/$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
NEO4J_URI=bolt://neo4j:7687

# ChromaDB
CHROMADB_API_KEY=$(openssl rand -hex 32)

# System
TZ=UTC
SUTAZAI_ENV=production
DEBUG=false
ENABLE_SELF_IMPROVEMENT=true

# API Keys (local)
OPENAI_API_BASE=http://ollama:11434/v1
OPENAI_API_KEY=local

# Monitoring
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
EOF
        log "Environment file created with secure passwords ✓" "$GREEN"
    else
        log "Environment file already exists ✓" "$GREEN"
    fi
}

# Deploy core infrastructure
deploy_core_infrastructure() {
    log "Deploying core infrastructure..." "$PURPLE"
    
    cd "$PROJECT_ROOT"
    
    # Start core services first
    log "Starting PostgreSQL..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d postgres
    sleep 10
    
    log "Starting Redis..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d redis
    sleep 5
    
    log "Starting Neo4j..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d neo4j
    sleep 15
    
    log "Core infrastructure deployed ✓" "$GREEN"
}

# Deploy vector databases
deploy_vector_databases() {
    log "Deploying vector databases..." "$PURPLE"
    
    log "Starting ChromaDB..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d chromadb
    sleep 10
    
    log "Starting Qdrant..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d qdrant
    sleep 10
    
    log "Vector databases deployed ✓" "$GREEN"
}

# Deploy Ollama and download models
deploy_ollama() {
    log "Deploying Ollama and AI models..." "$PURPLE"
    
    log "Starting Ollama..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d ollama
    sleep 15
    
    # Download essential models
    log "Downloading AI models (this may take a while)..." "$YELLOW"
    
    models=(
        "llama3.2:1b"
        "deepseek-r1:8b"
        "qwen2.5:3b"
        "codellama:7b"
        "nomic-embed-text"
    )
    
    for model in "${models[@]}"; do
        log "Pulling $model..." "$BLUE"
        docker exec sutazai-ollama ollama pull "$model" || log "Failed to pull $model, continuing..." "$YELLOW"
    done
    
    log "Ollama and models deployed ✓" "$GREEN"
}

# Build and deploy AGI backend
deploy_agi_backend() {
    log "Building and deploying AGI backend..." "$PURPLE"
    
    cd "$PROJECT_ROOT"
    
    # Build the AGI backend image
    log "Building backend-agi image..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml build backend-agi
    
    # Start the backend
    log "Starting AGI backend..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d backend-agi
    sleep 20
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "AGI backend is healthy ✓" "$GREEN"
    else
        log "WARNING: AGI backend health check failed" "$YELLOW"
    fi
}

# Build and deploy enhanced frontend
deploy_frontend() {
    log "Building and deploying enhanced frontend..." "$PURPLE"
    
    cd "$PROJECT_ROOT"
    
    # Build the frontend image
    log "Building frontend-agi image..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml build frontend-agi
    
    # Start the frontend
    log "Starting enhanced frontend..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d frontend-agi
    sleep 15
    
    # Check if accessible
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        log "Frontend is accessible ✓" "$GREEN"
    else
        log "WARNING: Frontend not accessible" "$YELLOW"
    fi
}

# Deploy all AI agents
deploy_agents() {
    log "Deploying AI agents..." "$PURPLE"
    
    # Run the agent deployment script
    if [ -f "${PROJECT_ROOT}/scripts/deploy_all_agents.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/deploy_all_agents.sh"
    else
        log "Agent deployment script not found, skipping..." "$YELLOW"
    fi
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..." "$PURPLE"
    
    log "Starting Prometheus..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d prometheus
    sleep 10
    
    log "Starting Grafana..." "$BLUE"
    docker-compose -f docker-compose-complete-agi.yml up -d grafana
    sleep 10
    
    log "Monitoring stack deployed ✓" "$GREEN"
}

# Initialize system
initialize_system() {
    log "Initializing AGI system..." "$PURPLE"
    
    # Wait for backend to be fully ready
    log "Waiting for backend to initialize..." "$BLUE"
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health 2>/dev/null | grep -q "healthy"; then
            log "Backend initialized ✓" "$GREEN"
            break
        fi
        
        attempt=$((attempt + 1))
        log "Waiting for backend... ($attempt/$max_attempts)" "$YELLOW"
        sleep 5
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log "Backend initialization timeout" "$RED"
    fi
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..." "$PURPLE"
    
    services=(
        "postgres:5432:PostgreSQL"
        "redis:6379:Redis"
        "neo4j:7474:Neo4j"
        "chromadb:8000:ChromaDB"
        "qdrant:6333:Qdrant"
        "ollama:11434:Ollama"
        "backend-agi:8000:AGI Backend"
        "frontend-agi:8501:Frontend"
        "prometheus:9090:Prometheus"
        "grafana:3003:Grafana"
    )
    
    all_healthy=true
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r container port name <<< "$service_info"
        
        if docker ps | grep -q "sutazai-$container"; then
            log "✓ $name is running" "$GREEN"
        else
            log "✗ $name is not running" "$RED"
            all_healthy=false
        fi
    done
    
    if $all_healthy; then
        log "All services verified ✓" "$GREEN"
    else
        log "Some services are not running properly" "$RED"
    fi
}

# Print access information
print_access_info() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                 SutazAI AGI/ASI System                       ║"
    echo "║                  Deployment Complete!                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${GREEN}Access Points:${NC}"
    echo -e "${BLUE}Main UI:${NC}          http://localhost:8501"
    echo -e "${BLUE}API:${NC}              http://localhost:8000"
    echo -e "${BLUE}API Docs:${NC}         http://localhost:8000/docs"
    echo -e "${BLUE}Neo4j Browser:${NC}    http://localhost:7474"
    echo -e "${BLUE}Prometheus:${NC}       http://localhost:9090"
    echo -e "${BLUE}Grafana:${NC}          http://localhost:3003"
    echo ""
    echo -e "${YELLOW}Credentials are stored in .env file${NC}"
    echo -e "${YELLOW}Logs are available in: $LOG_FILE${NC}"
    echo ""
    echo -e "${PURPLE}To stop the system:${NC} docker-compose -f docker-compose-complete-agi.yml down"
    echo -e "${PURPLE}To view logs:${NC} docker-compose -f docker-compose-complete-agi.yml logs -f"
}

# Main deployment function
main() {
    print_header
    
    log "Starting SutazAI AGI/ASI deployment..." "$GREEN"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Execute deployment steps
    check_prerequisites
    cleanup_existing
    create_directories
    generate_env_file
    
    deploy_core_infrastructure
    deploy_vector_databases
    deploy_ollama
    deploy_agi_backend
    deploy_frontend
    deploy_agents
    deploy_monitoring
    
    initialize_system
    verify_deployment
    
    log "Deployment completed successfully! 🎉" "$GREEN"
    print_access_info
}

# Run main function
main "$@" 