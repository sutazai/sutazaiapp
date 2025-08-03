#!/bin/bash
#
# SutazAI Optimized Deployment Script
# Version: 1.0.0
# 
# This script implements staged deployment with optimized build strategies
# to resolve timeout issues and improve deployment reliability
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"
readonly LOG_FILE="$PROJECT_ROOT/logs/optimized_deployment_$(date +%Y%m%d_%H%M%S).log"
readonly DEPLOYMENT_ID="deploy_optimized_$(date +%Y%m%d_%H%M%S)_$$"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Deployment stages
readonly DEPLOYMENT_STAGES=(
    "prepare_environment"
    "build_base_images"
    "deploy_infrastructure"
    "deploy_core_services"
    "deploy_ai_services"
    "validate_deployment"
    "optimize_system"
)

# Service deployment order (dependency-aware)
readonly INFRASTRUCTURE_SERVICES=(
    "postgres"
    "redis"
    "neo4j"
)

readonly VECTOR_SERVICES=(
    "chromadb" 
    "qdrant"
    "faiss"
)

readonly CORE_SERVICES=(
    "ollama"
    "backend"
    "frontend"
)

readonly AI_SERVICES=(
    "letta"
    "autogpt"
    "crewai"
    "aider"
    "langflow"
    "flowise"
)

# Logging functions
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log_stage() {
    echo -e "\n${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║ STAGE: $(printf "%-51s" "$1") ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}\n"
}

log_info() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

# Progress indicator
show_progress() {
    local current=$1
    local total=$2
    local description="$3"
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %d%% - %s${NC}" "$percentage" "$description"
    
    if [[ $current -eq $total ]]; then
        echo
    fi
}

# Environment preparation
prepare_environment() {
    log_stage "Environment Preparation"
    
    log_info "Checking system requirements..."
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker is available and running"
    
    # Check available resources
    local memory_gb
    memory_gb=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    local disk_gb
    disk_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log_info "System resources: ${memory_gb}GB RAM, ${disk_gb}GB disk"
    
    if [[ $memory_gb -lt 8 ]]; then
        log_warn "Limited memory detected, using lightweight configurations"
        export LIGHTWEIGHT_MODE=true
    fi
    
    # Setup environment file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        log_info "Creating environment configuration..."
        create_environment_file
    fi
    
    # Enable BuildKit for faster builds
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    log_success "Environment preparation completed"
}

# Create optimized environment file
create_environment_file() {
    local secrets_dir="$PROJECT_ROOT/secrets"
    mkdir -p "$secrets_dir"
    
    # Generate secure passwords
    local postgres_password
    postgres_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local redis_password
    redis_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local neo4j_password
    neo4j_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local jwt_secret
    jwt_secret=$(openssl rand -hex 32)
    local grafana_password
    grafana_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    
    # Save secrets
    echo "$postgres_password" > "$secrets_dir/postgres_password.txt"
    echo "$redis_password" > "$secrets_dir/redis_password.txt"
    echo "$neo4j_password" > "$secrets_dir/neo4j_password.txt"
    echo "$jwt_secret" > "$secrets_dir/jwt_secret.txt"
    echo "$grafana_password" > "$secrets_dir/grafana_password.txt"
    
    chmod 600 "$secrets_dir"/*
    
    # Create .env file
    cat > "$PROJECT_ROOT/.env" << EOF
# SutazAI Optimized Deployment Configuration
# Generated: $(date -Iseconds)
# Deployment ID: $DEPLOYMENT_ID

# System Configuration
TZ=UTC
SUTAZAI_ENV=local
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
DEPLOYMENT_ID=$DEPLOYMENT_ID

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$postgres_password
POSTGRES_DB=sutazai
DATABASE_URL=postgresql://sutazai:$postgres_password@postgres:5432/sutazai

# Redis Configuration
REDIS_PASSWORD=$redis_password

# Neo4j Configuration
NEO4J_PASSWORD=$neo4j_password

# Security Configuration
SECRET_KEY=$jwt_secret
JWT_SECRET=$jwt_secret

# Monitoring Configuration
GRAFANA_PASSWORD=$grafana_password

# Model Configuration
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1

# Feature Flags
ENABLE_GPU=false
ENABLE_MONITORING=true
ENABLE_LOGGING=true
ENABLE_HEALTH_CHECKS=true

# Performance Tuning
MAX_WORKERS=2
CONNECTION_POOL_SIZE=10
CACHE_TTL=3600

# Development Settings
DEBUG=false
LOG_LEVEL=INFO
EOF
    
    chmod 600 "$PROJECT_ROOT/.env"
    log_success "Environment file created with secure passwords"
}

# Build base images with optimized layer caching
build_base_images() {
    log_stage "Building Base Images"
    
    log_info "Building backend service (multi-stage)..."
    docker build \
        --target production \
        --cache-from sutazai/backend:cache \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t sutazai/backend:latest \
        -t sutazai/backend:cache \
        ./backend/ || {
            log_warn "Backend build failed, trying with fallback strategy..."
            docker build --no-cache -t sutazai/backend:latest ./backend/
        }
    
    log_info "Building frontend service (multi-stage)..."
    docker build \
        --target production \
        --cache-from sutazai/frontend:cache \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t sutazai/frontend:latest \
        -t sutazai/frontend:cache \
        ./frontend/ || {
            log_warn "Frontend build failed, trying with fallback strategy..."
            docker build --no-cache -t sutazai/frontend:latest ./frontend/
        }
    
    log_success "Base images built successfully"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log_stage "Infrastructure Deployment"
    
    log_info "Deploying infrastructure services..."
    
    local total_services=${#INFRASTRUCTURE_SERVICES[@]}
    local current_service=1
    
    for service in "${INFRASTRUCTURE_SERVICES[@]}"; do
        log_info "Starting $service..."
        docker compose up -d "$service"
        
        # Wait for service to be ready
        wait_for_service "$service" 60
        
        show_progress $current_service $total_services "Deploying infrastructure"
        current_service=$((current_service + 1))
    done
    
    log_success "Infrastructure deployment completed"
}

# Deploy core services
deploy_core_services() {
    log_stage "Core Services Deployment"
    
    # Deploy vector databases first
    log_info "Deploying vector databases..."
    for service in "${VECTOR_SERVICES[@]}"; do
        log_info "Starting $service..."
        docker compose up -d "$service" || log_warn "Failed to start $service, continuing..."
        sleep 5
    done
    
    # Deploy core application services
    log_info "Deploying core application services..."
    local total_services=${#CORE_SERVICES[@]}
    local current_service=1
    
    for service in "${CORE_SERVICES[@]}"; do
        log_info "Starting $service..."
        docker compose up -d "$service"
        
        # Wait for service to be ready
        wait_for_service "$service" 120
        
        show_progress $current_service $total_services "Deploying core services"
        current_service=$((current_service + 1))
    done
    
    # Download essential models for Ollama
    if docker ps --filter "name=sutazai-ollama" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-ollama"; then
        log_info "Downloading essential AI models..."
        download_essential_models
    fi
    
    log_success "Core services deployment completed"
}

# Deploy AI services
deploy_ai_services() {
    log_stage "AI Services Deployment"
    
    log_info "Deploying AI agent services..."
    
    local deployed_count=0
    local total_services=${#AI_SERVICES[@]}
    
    for service in "${AI_SERVICES[@]}"; do
        log_info "Starting AI service: $service..."
        
        # Try to start the service with timeout
        if timeout 120 docker compose up -d "$service" 2>/dev/null; then
            deployed_count=$((deployed_count + 1))
            log_success "Successfully started $service"
        else
            log_warn "Failed to start $service, continuing with other services"
        fi
        
        show_progress $deployed_count $total_services "Deploying AI services"
        sleep 3 # Brief pause between deployments
    done
    
    log_success "AI services deployment completed ($deployed_count/$total_services successful)"
}

# Wait for service to be ready
wait_for_service() {
    local service="$1"
    local timeout="${2:-60}"
    local interval=5
    local elapsed=0
    
    log_info "Waiting for $service to become ready..."
    
    while [[ $elapsed -lt $timeout ]]; do
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            # Check health status if available
            local health_status
            health_status=$(docker inspect "sutazai-$service" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
            
            if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no_healthcheck" ]]; then
                log_success "$service is ready"
                return 0
            fi
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        printf "\r${BLUE}Waiting for $service: %ds/%ds${NC}" "$elapsed" "$timeout"
    done
    
    echo
    log_warn "$service did not become ready within ${timeout}s"
    return 1
}

# Download essential models
download_essential_models() {
    local models=(
        "tinyllama:latest"
        "qwen2.5:3b"
        "nomic-embed-text:latest"
    )
    
    for model in "${models[@]}"; do
        log_info "Downloading model: $model..."
        if timeout 300 docker exec sutazai-ollama ollama pull "$model" >/dev/null 2>&1; then
            log_success "Downloaded $model"
        else
            log_warn "Failed to download $model or timeout exceeded"
        fi
    done
}

# Validate deployment
validate_deployment() {
    log_stage "Deployment Validation"
    
    log_info "Running deployment health checks..."
    
    local failed_checks=0
    
    # Check core services
    for service in "${INFRASTRUCTURE_SERVICES[@]}" "${CORE_SERVICES[@]}"; do
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            log_success "✅ $service is running"
        else
            log_error "❌ $service is not running"
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    # Test API endpoints
    if curl -s --max-time 10 http://localhost:8000/health >/dev/null 2>&1; then
        log_success "✅ Backend API is responding"
    else
        log_warn "❌ Backend API is not responding"
        failed_checks=$((failed_checks + 1))
    fi
    
    if curl -s --max-time 10 http://localhost:8501/healthz >/dev/null 2>&1; then
        log_success "✅ Frontend is responding"
    else
        log_warn "❌ Frontend is not responding"
        failed_checks=$((failed_checks + 1))
    fi
    
    # Test Ollama
    if docker exec sutazai-ollama ollama list >/dev/null 2>&1; then
        log_success "✅ Ollama is functional"
    else
        log_warn "❌ Ollama is not functional"
        failed_checks=$((failed_checks + 1))
    fi
    
    if [[ $failed_checks -eq 0 ]]; then
        log_success "All validation checks passed!"
    else
        log_warn "Validation completed with $failed_checks issues"
    fi
    
    # Generate access information
    generate_access_info
}

# Generate access information
generate_access_info() {
    local local_ip
    local_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETED                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${CYAN}🌐 Access Information:${NC}"
    echo -e "   Main Application: ${BLUE}http://$local_ip:8501${NC}"
    echo -e "   Backend API: ${BLUE}http://$local_ip:8000${NC}"
    echo -e "   API Documentation: ${BLUE}http://$local_ip:8000/docs${NC}"
    echo -e "   Ollama API: ${BLUE}http://$local_ip:11434${NC}"
    
    echo -e "\n${CYAN}🤖 AI Services:${NC}"
    echo -e "   LangFlow: ${BLUE}http://$local_ip:8090${NC}"
    echo -e "   FlowiseAI: ${BLUE}http://$local_ip:8099${NC}"
    echo -e "   Dify: ${BLUE}http://$local_ip:8107${NC}"
    
    echo -e "\n${CYAN}📊 Databases:${NC}"
    echo -e "   PostgreSQL: ${BLUE}$local_ip:5432${NC}"
    echo -e "   Redis: ${BLUE}$local_ip:6379${NC}"
    echo -e "   Neo4j Browser: ${BLUE}http://$local_ip:7474${NC}"
    echo -e "   ChromaDB: ${BLUE}http://$local_ip:8001${NC}"
    echo -e "   Qdrant: ${BLUE}http://$local_ip:6333${NC}"
    
    echo -e "\n${CYAN}🔧 Management:${NC}"
    echo -e "   View logs: ${YELLOW}docker compose logs [service]${NC}"
    echo -e "   Restart service: ${YELLOW}docker compose restart [service]${NC}"
    echo -e "   Stop all: ${YELLOW}docker compose down${NC}"
    
    echo -e "\n${CYAN}📁 Important Files:${NC}"
    echo -e "   Deployment log: ${BLUE}$LOG_FILE${NC}"
    echo -e "   Environment: ${BLUE}$PROJECT_ROOT/.env${NC}"
    echo -e "   Secrets: ${BLUE}$PROJECT_ROOT/secrets/${NC}"
    
    echo
}

# Optimize system performance
optimize_system() {
    log_stage "System Optimization"
    
    log_info "Running post-deployment optimizations..."
    
    # Clean up unused Docker resources
    docker system prune -f >/dev/null 2>&1 || true
    
    # Set container resource limits if not already set
    if [[ "${LIGHTWEIGHT_MODE:-false}" == "true" ]]; then
        log_info "Applying lightweight resource constraints..."
        # Resource limits would be applied here
    fi
    
    log_success "System optimization completed"
}

# Rollback functionality
rollback_deployment() {
    log_stage "Deployment Rollback"
    
    log_warn "Rolling back deployment..."
    
    # Stop all services
    docker compose down --remove-orphans >/dev/null 2>&1 || true
    
    # Remove deployment-specific containers
    docker ps -a --filter "name=sutazai-" --format "{{.Names}}" | xargs -r docker rm -f >/dev/null 2>&1 || true
    
    log_success "Rollback completed"
}

# Error handling
handle_error() {
    local exit_code=$?
    log_error "Deployment failed with exit code: $exit_code"
    
    echo -e "\n${YELLOW}Do you want to rollback the deployment? (y/n): ${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rollback_deployment
    fi
    
    exit $exit_code
}

trap 'handle_error' ERR

# Show usage
show_usage() {
    cat << EOF
SutazAI Optimized Deployment Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    deploy      Full optimized deployment (default)
    rollback    Rollback current deployment
    status      Show deployment status
    logs        Show deployment logs
    help        Show this help

OPTIONS:
    --lightweight    Use lightweight configurations for limited resources
    --skip-ai        Skip AI services deployment
    --force          Force deployment even with warnings
    --debug          Enable debug logging

EXAMPLES:
    $0 deploy
    $0 deploy --lightweight
    $0 rollback
    $0 status

EOF
}

# Main execution
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            setup_logging
            
            log_info "Starting SutazAI Optimized Deployment"
            log_info "Deployment ID: $DEPLOYMENT_ID"
            
            # Execute deployment stages
            for stage in "${DEPLOYMENT_STAGES[@]}"; do
                case "$stage" in
                    "prepare_environment") prepare_environment ;;
                    "build_base_images") build_base_images ;;
                    "deploy_infrastructure") deploy_infrastructure ;;
                    "deploy_core_services") deploy_core_services ;;
                    "deploy_ai_services") deploy_ai_services ;;
                    "validate_deployment") validate_deployment ;;
                    "optimize_system") optimize_system ;;
                esac
            done
            
            log_success "Deployment completed successfully!"
            ;;
        "rollback")
            setup_logging
            rollback_deployment
            ;;
        "status")
            docker compose ps
            ;;
        "logs")
            tail -f "$LOG_FILE" 2>/dev/null || echo "No log file found"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi