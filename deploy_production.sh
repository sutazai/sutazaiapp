#!/bin/bash
# Production Deployment Script for SutazAI AGI/ASI System
# This script deploys the complete system using the enhanced Docker Compose configuration

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_MODE="${1:-enhanced}"  # enhanced, minimal, or full
SUTAZAI_HOME="$(pwd)"
LOG_FILE="${SUTAZAI_HOME}/logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "${SUTAZAI_HOME}/logs"

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

success() {
    log "${GREEN}✅ ${1}${NC}"
}

info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

warn() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

progress() {
    log "${PURPLE}🔄 ${1}${NC}"
}

header() {
    log "${CYAN}"
    log "=============================================="
    log "🚀 SutazAI Production Deployment"
    log "=============================================="
    log "Mode: ${DEPLOYMENT_MODE}"
    log "Time: $(date)"
    log "=============================================="
    log "${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    progress "Running pre-deployment checks..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check available disk space (minimum 20GB)
    available_space=$(df "${SUTAZAI_HOME}" | tail -1 | awk '{print $4}')
    if [[ ${available_space} -lt 20971520 ]]; then  # 20GB in KB
        warn "Low disk space detected. Minimum 20GB recommended."
    fi
    
    # Check available RAM
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [[ ${total_ram} -lt 8 ]]; then
        warn "Low RAM detected (${total_ram}GB). 8GB+ recommended."
    fi
    
    success "Pre-deployment checks completed"
}

# Setup environment
setup_environment() {
    progress "Setting up deployment environment..."
    
    # Create required directories
    mkdir -p data/{postgres,redis,chromadb,qdrant,ollama,workspace,documents,logs}
    mkdir -p configs/{nginx,prometheus,grafana}
    mkdir -p secrets
    mkdir -p backups
    
    # Generate secrets if they don't exist
    if [[ ! -f secrets/postgres_password.txt ]]; then
        echo "$(openssl rand -base64 32)" > secrets/postgres_password.txt
    fi
    if [[ ! -f secrets/grafana_password.txt ]]; then
        echo "$(openssl rand -base64 32)" > secrets/grafana_password.txt
    fi
    if [[ ! -f secrets/jwt_secret.txt ]]; then
        echo "$(openssl rand -base64 32)" > secrets/jwt_secret.txt
    fi
    if [[ ! -f secrets/vault_token.txt ]]; then
        echo "$(openssl rand -base64 32)" > secrets/vault_token.txt
    fi
    
    # Set secure permissions
    chmod 600 secrets/*
    chmod 755 data
    chmod 777 data/workspace
    
    success "Environment setup completed"
}

# Deploy based on mode
deploy_system() {
    progress "Deploying SutazAI system in ${DEPLOYMENT_MODE} mode..."
    
    case "${DEPLOYMENT_MODE}" in
        "enhanced"|"production")
            COMPOSE_FILE="docker-compose.enhanced.yml"
            ;;
        "minimal")
            COMPOSE_FILE="docker-compose.yml"
            ;;
        "full")
            COMPOSE_FILE="docker-compose.enhanced.yml"
            ;;
        *)
            error_exit "Invalid deployment mode: ${DEPLOYMENT_MODE}"
            ;;
    esac
    
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        error_exit "Compose file not found: ${COMPOSE_FILE}"
    fi
    
    # Stop any existing containers
    info "Stopping existing containers..."
    docker-compose -f "${COMPOSE_FILE}" down --remove-orphans || true
    
    # Pull latest images
    info "Pulling latest Docker images..."
    docker-compose -f "${COMPOSE_FILE}" pull || warn "Some images may not be available"
    
    # Build custom images
    info "Building custom images..."
    docker-compose -f "${COMPOSE_FILE}" build --no-cache
    
    # Start infrastructure services first
    info "Starting infrastructure services..."
    docker-compose -f "${COMPOSE_FILE}" up -d postgres redis chromadb qdrant neo4j elasticsearch
    
    # Wait for infrastructure to be ready
    info "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Check infrastructure health
    check_service_health "postgres" "5432"
    check_service_health "redis" "6379"
    check_service_health "chromadb" "8001"
    check_service_health "qdrant" "6333"
    
    # Start AI model services
    info "Starting AI model services..."
    docker-compose -f "${COMPOSE_FILE}" up -d ollama tabbyml
    
    # Wait for model services
    sleep 60
    check_service_health "ollama" "11434"
    
    # Start application services
    info "Starting application services..."
    docker-compose -f "${COMPOSE_FILE}" up -d backend frontend
    
    # Wait for application services
    sleep 30
    check_service_health "backend" "8000"
    check_service_health "frontend" "8501"
    
    # Start AI agents
    info "Starting AI agents..."
    docker-compose -f "${COMPOSE_FILE}" up -d autogpt crewai aider gpt-engineer
    
    # Start monitoring services
    info "Starting monitoring services..."
    docker-compose -f "${COMPOSE_FILE}" up -d prometheus grafana
    
    # Start additional services
    info "Starting additional services..."
    docker-compose -f "${COMPOSE_FILE}" up -d nginx consul vault
    
    success "System deployment completed"
}

# Health check function
check_service_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose ps | grep -q "${service}.*Up"; then
            if nc -z localhost "${port}" 2>/dev/null; then
                success "${service} is healthy"
                return 0
            fi
        fi
        
        info "Waiting for ${service} (attempt ${attempt}/${max_attempts})..."
        sleep 10
        ((attempt++))
    done
    
    warn "${service} may not be fully ready"
    return 1
}

# Post-deployment configuration
post_deployment_setup() {
    progress "Running post-deployment configuration..."
    
    # Install AI models
    info "Installing AI models..."
    install_ai_models
    
    # Configure monitoring
    info "Configuring monitoring..."
    configure_monitoring
    
    # Run health checks
    info "Running comprehensive health checks..."
    run_health_checks
    
    success "Post-deployment setup completed"
}

# Install AI models
install_ai_models() {
    local models=("deepseek-r1:8b" "qwen3:8b" "codellama:7b" "llama2")
    
    for model in "${models[@]}"; do
        info "Installing model: ${model}"
        docker exec sutazai-ollama ollama pull "${model}" || warn "Failed to install ${model}"
        
        # Test model
        if docker exec sutazai-ollama ollama run "${model}" "Hello" &>/dev/null; then
            success "Model ${model} is working"
        else
            warn "Model ${model} may have issues"
        fi
    done
}

# Configure monitoring
configure_monitoring() {
    # Wait for Grafana to be ready
    local attempts=0
    while [[ $attempts -lt 30 ]]; do
        if curl -s http://localhost:3000/api/health &>/dev/null; then
            break
        fi
        sleep 5
        ((attempts++))
    done
    
    info "Grafana is accessible at http://localhost:3000"
    info "Prometheus is accessible at http://localhost:9090"
}

# Comprehensive health checks
run_health_checks() {
    local issues=0
    
    # Check core services
    info "Checking core services..."
    
    services=(
        "backend:8000:/health"
        "frontend:8501:/"
        "ollama:11434:/api/tags"
        "chromadb:8001:/api/v1/heartbeat"
        "qdrant:6333:/health"
        "prometheus:9090:/-/healthy"
        "grafana:3000:/api/health"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port endpoint <<< "$service_info"
        
        if curl -s -f "http://localhost:${port}${endpoint}" &>/dev/null; then
            success "${service} is responding"
        else
            warn "${service} is not responding properly"
            ((issues++))
        fi
    done
    
    # Check container status
    info "Checking container status..."
    docker-compose ps
    
    if [[ $issues -eq 0 ]]; then
        success "All health checks passed"
    else
        warn "${issues} issues detected in health checks"
    fi
}

# Generate deployment report
generate_deployment_report() {
    local report_file="deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "${report_file}" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "mode": "${DEPLOYMENT_MODE}",
    "version": "2.0.0",
    "status": "completed"
  },
  "services": {
    "backend": "http://localhost:8000",
    "frontend": "http://localhost:8501", 
    "api_docs": "http://localhost:8000/docs",
    "monitoring": "http://localhost:3000",
    "prometheus": "http://localhost:9090"
  },
  "credentials": {
    "grafana": {
      "username": "admin",
      "password_file": "secrets/grafana_password.txt"
    },
    "postgres": {
      "username": "sutazai",
      "password_file": "secrets/postgres_password.txt"
    }
  },
  "ai_models": [
    "deepseek-r1:8b",
    "qwen3:8b", 
    "codellama:7b",
    "llama2"
  ],
  "ai_agents": [
    "AutoGPT",
    "CrewAI",
    "Aider",
    "GPT-Engineer"
  ]
}
EOF
    
    info "Deployment report saved: ${report_file}"
}

# Print final information
print_deployment_info() {
    log "${GREEN}"
    log "=================================================================="
    log "🎉 SutazAI Production Deployment Complete!"
    log "=================================================================="
    log "${NC}"
    
    log "${CYAN}🌐 Access Points:${NC}"
    log "Frontend UI:      http://localhost:8501"
    log "Backend API:      http://localhost:8000"
    log "API Documentation: http://localhost:8000/docs"
    log "Monitoring:       http://localhost:3000"
    log "Prometheus:       http://localhost:9090"
    log ""
    
    log "${CYAN}🔐 Credentials:${NC}"
    log "Grafana:          admin / $(cat secrets/grafana_password.txt 2>/dev/null || echo 'See secrets/grafana_password.txt')"
    log "PostgreSQL:       sutazai / $(cat secrets/postgres_password.txt 2>/dev/null || echo 'See secrets/postgres_password.txt')"
    log ""
    
    log "${CYAN}🤖 AI Models Available:${NC}"
    log "• DeepSeek R1 8B (Reasoning & General)"
    log "• Qwen3 8B (Multilingual)"
    log "• CodeLlama 7B (Code Generation)"
    log "• Llama2 (General Purpose)"
    log ""
    
    log "${CYAN}🔧 Management Commands:${NC}"
    log "Status:           docker-compose ps"
    log "Logs:             docker-compose logs [service]"
    log "Stop:             docker-compose down"
    log "Restart:          docker-compose restart [service]"
    log ""
    
    log "${YELLOW}📋 Next Steps:${NC}"
    log "1. Test the system by visiting http://localhost:8501"
    log "2. Check monitoring at http://localhost:3000"
    log "3. Review logs for any issues: docker-compose logs"
    log "4. Configure additional models if needed"
    log ""
    
    log "${GREEN}✨ SutazAI AGI/ASI System is now running!${NC}"
}

# Main execution
main() {
    header
    
    pre_deployment_checks
    setup_environment
    deploy_system
    post_deployment_setup
    generate_deployment_report
    print_deployment_info
    
    success "Deployment completed successfully!"
    log "📄 Full deployment log: ${LOG_FILE}"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi