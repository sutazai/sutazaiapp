#!/bin/bash
# SutazAI Docker Deployment Script - Docker Excellence Compliant
# Deploys services according to environment specifications

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_DIR="${SCRIPT_DIR}/compose"
ENVIRONMENT=${ENVIRONMENT:-"production"}
DEPLOY_LOG="${SCRIPT_DIR}/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOY_LOG"
}

# Environment-specific compose file configurations
get_compose_files() {
    local env="$1"
    case "$env" in
        "production")
            echo "-f ${COMPOSE_DIR}/docker-compose.yml"
            ;;
        "development")
            echo "-f ${COMPOSE_DIR}/docker-compose.yml -f ${COMPOSE_DIR}/docker-compose.dev.yml"
            ;;
        "test")
            echo "-f ${COMPOSE_DIR}/docker-compose.test.yml"
            ;;
        "agents-only")
            echo "-f ${COMPOSE_DIR}/docker-compose.agents.yml"
            ;;
        *)
            log_error "Unknown environment: $env"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker and Docker Compose
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check required directories
    local required_dirs=(
        "${PROJECT_ROOT}/data"
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/secrets"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "Creating required directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Check secrets for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        local secret_files=(
            "${PROJECT_ROOT}/secrets/postgres_password.txt"
            "${PROJECT_ROOT}/secrets/redis_password.txt"
            "${PROJECT_ROOT}/secrets/jwt_secret.txt"
            "${PROJECT_ROOT}/secrets/grafana_password.txt"
        )
        
        for secret_file in "${secret_files[@]}"; do
            if [[ ! -f "$secret_file" ]]; then
                log_warning "Secret file missing: $secret_file"
                log "Creating default secret file (CHANGE IN PRODUCTION!)"
                echo "change-me-$(openssl rand -hex 16)" > "$secret_file"
                chmod 600 "$secret_file"
            fi
        done
    fi
}

# Create Docker network if it doesn't exist
create_network() {
    local network_name="sutazai-network"
    
    if ! docker network ls | grep -q "$network_name"; then
        log "Creating Docker network: $network_name"
        docker network create \
            --driver bridge \
            --subnet 172.20.0.0/16 \
            "$network_name" >> "$DEPLOY_LOG" 2>&1
        log_success "Created network: $network_name"
    else
        log "Network $network_name already exists"
    fi
}

# Deploy services
deploy_services() {
    local compose_files
    compose_files=$(get_compose_files "$ENVIRONMENT")
    
    log "Deploying SutazAI services for environment: $ENVIRONMENT"
    log "Using compose files: $compose_files"
    
    # Change to project root for relative paths in compose files
    cd "$PROJECT_ROOT"
    
    # Pull latest images if in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Pulling latest images..."
        eval "docker-compose $compose_files pull --ignore-pull-failures" >> "$DEPLOY_LOG" 2>&1 || {
            log_warning "Some images could not be pulled (using local builds)"
        }
    fi
    
    # Deploy services
    log "Starting services..."
    if eval "docker-compose $compose_files up -d --remove-orphans" >> "$DEPLOY_LOG" 2>&1; then
        log_success "Services deployed successfully"
    else
        log_error "Failed to deploy services"
        return 1
    fi
}

# Health check deployment
health_check() {
    local compose_files
    compose_files=$(get_compose_files "$ENVIRONMENT")
    
    log "Performing health checks..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Wait for services to be ready
    local max_wait=300  # 5 minutes
    local wait_time=0
    local check_interval=10
    
    while [[ $wait_time -lt $max_wait ]]; do
        local healthy_services=0
        local total_services=0
        
        # Get service status
        while IFS= read -r line; do
            if [[ "$line" =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+([^[:space:]]+) ]]; then
                local service_name="${BASH_REMATCH[1]}"
                local health_status="${BASH_REMATCH[2]}"
                
                ((total_services++))
                
                if [[ "$health_status" == "healthy" || "$health_status" == "Up" ]]; then
                    ((healthy_services++))
                elif [[ "$health_status" == "unhealthy" ]]; then
                    log_warning "Service $service_name is unhealthy"
                fi
            fi
        done < <(eval "docker-compose $compose_files ps --format table" 2>/dev/null | tail -n +3)
        
        if [[ $healthy_services -eq $total_services && $total_services -gt 0 ]]; then
            log_success "All $total_services services are healthy"
            return 0
        fi
        
        log "Health check: $healthy_services/$total_services services ready. Waiting..."
        sleep $check_interval
        ((wait_time += check_interval))
    done
    
    log_error "Health check timeout after ${max_wait}s"
    return 1
}

# Show deployment status
show_status() {
    local compose_files
    compose_files=$(get_compose_files "$ENVIRONMENT")
    
    cd "$PROJECT_ROOT"
    
    log "Deployment Status:"
    eval "docker-compose $compose_files ps"
    
    echo ""
    log "Service URLs (if accessible):"
    
    case "$ENVIRONMENT" in
        "production")
            echo "  Frontend: http://localhost:8501"
            echo "  Prometheus: http://localhost:9090"
            echo "  Grafana: http://localhost:3000"
            ;;
        "development")
            echo "  Frontend: http://localhost:8501"
            echo "  Backend: http://localhost:8000"
            echo "  Ollama: http://localhost:11434"
            echo "  Prometheus: http://localhost:9090"
            echo "  Grafana: http://localhost:3000"
            echo "  PostgreSQL: localhost:5432"
            echo "  Redis: localhost:6379"
            echo "  PgAdmin: http://localhost:8080"
            echo "  Redis Commander: http://localhost:8081"
            ;;
    esac
}

# Stop and cleanup deployment
cleanup_deployment() {
    local compose_files
    compose_files=$(get_compose_files "$ENVIRONMENT")
    
    cd "$PROJECT_ROOT"
    
    log "Stopping and cleaning up deployment..."
    eval "docker-compose $compose_files down --volumes --remove-orphans" >> "$DEPLOY_LOG" 2>&1
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    log "Starting SutazAI deployment process"
    log "Environment: $ENVIRONMENT"
    log "Action: $action"
    log "Deploy log: $DEPLOY_LOG"
    
    # Initialize deploy log
    echo "SutazAI Deployment Log - $(date)" > "$DEPLOY_LOG"
    
    case "$action" in
        "deploy")
            check_prerequisites
            create_network
            deploy_services
            health_check
            show_status
            ;;
        "status")
            show_status
            ;;
        "health")
            health_check
            ;;
        "cleanup"|"down")
            cleanup_deployment
            ;;
        "logs")
            local compose_files
            compose_files=$(get_compose_files "$ENVIRONMENT")
            cd "$PROJECT_ROOT"
            eval "docker-compose $compose_files logs -f"
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 [deploy|status|health|cleanup|logs]"
            exit 1
            ;;
    esac
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "${@}"
fi