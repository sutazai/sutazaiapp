#!/bin/bash
# Purpose: Deploy distributed AI services infrastructure
# Usage: ./deploy-distributed-ai.sh [--env dev|staging|prod] [--phase all|infra|ai|monitor]
# Requires: Docker, Docker Compose, curl

set -euo pipefail

# Configuration
PROJECT_NAME="sutazai-distributed"
COMPOSE_FILE="docker-compose.distributed-ai.yml"
ENV="${1:-dev}"
PHASE="${2:-all}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
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
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."
    
    directories=(
        "config/kong"
        "config/consul"
        "config/envoy"
        "config/prometheus"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/rabbitmq"
        "config/redis"
        "config/scaler"
        "data/chromadb"
        "data/qdrant"
        "data/n8n"
        "data/autogpt"
        "services/langchain"
        "services/scaler"
        "services/streamlit"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_info "Directories created"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log_info "Deploying infrastructure services..."
    
    # Start core services
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        consul \
        redis \
        rabbitmq \
        kong \
        envoy
    
    # Wait for services to be healthy
    log_info "Waiting for infrastructure services to be healthy..."
    sleep 30
    
    # Check service health
    services=("consul:8500" "redis:6379" "rabbitmq:5672" "kong:8001")
    for service in "${services[@]}"; do
        if ! nc -z ${service%:*} ${service#*:} 2>/dev/null; then
            log_error "Service $service is not responding"
            exit 1
        fi
    done
    
    log_info "Infrastructure services deployed successfully"
}

# Deploy AI services
deploy_ai_services() {
    log_info "Deploying AI services..."
    
    # Start persistent AI services
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        ollama \
        chromadb \
        qdrant \
        n8n
    
    # Wait for services to initialize
    log_info "Waiting for AI services to initialize..."
    sleep 45
    
    # Pull Ollama models
    log_info "Pulling Ollama models..."
    models=("tinyllama" "tinyllama" "tinyllama3:8b" "tinyllama:7b")
    
    for model in "${models[@]}"; do
        log_info "Pulling model: $model"
        docker exec sutazai-distributed_ollama_1 ollama pull "$model" || log_warn "Failed to pull $model"
    done
    
    log_info "AI services deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Start monitoring services
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        prometheus \
        grafana \
        jaeger
    
    # Start service scaler
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        service-scaler
    
    log_info "Monitoring stack deployed successfully"
}

# Configure services
configure_services() {
    log_info "Configuring services..."
    
    # Configure Kong routes
    if [ -f "config/kong/kong.yml" ]; then
        log_info "Configuring Kong API Gateway..."
        sleep 10
        curl -X POST http://localhost:8001/config \
            -F config=@config/kong/kong.yml || log_warn "Kong configuration might need manual setup"
    fi
    
    # Register services with Consul
    if [ -f "config/consul/services.json" ]; then
        log_info "Registering services with Consul..."
        curl -X PUT http://localhost:8500/v1/agent/service/register \
            -d @config/consul/services.json || log_warn "Consul registration might need manual setup"
    fi
    
    log_info "Service configuration completed"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    # Check core services
    endpoints=(
        "http://localhost:8500/v1/status/leader"  # Consul
        "http://localhost:8001/status"             # Kong
        "http://localhost:15672/api/overview"      # RabbitMQ
        "http://localhost:9090/-/healthy"          # Prometheus
        "http://localhost:3000/api/health"         # Grafana
        "http://localhost:10104/api/tags"          # Ollama
    )
    
    failed=0
    for endpoint in "${endpoints[@]}"; do
        if curl -s -f "$endpoint" > /dev/null 2>&1; then
            log_info "✓ ${endpoint%//*} is healthy"
        else
            log_error "✗ ${endpoint%//*} is not responding"
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log_info "All services are healthy!"
    else
        log_warn "$failed services are not responding"
    fi
}

# Display service URLs
display_urls() {
    echo
    log_info "Service URLs:"
    echo "  API Gateway:      http://localhost:8000"
    echo "  Kong Admin:       http://localhost:8001"
    echo "  Consul UI:        http://localhost:8500"
    echo "  RabbitMQ UI:      http://localhost:15672 (admin/admin)"
    echo "  Prometheus:       http://localhost:9090"
    echo "  Grafana:          http://localhost:3000 (admin/admin)"
    echo "  Jaeger UI:        http://localhost:16686"
    echo "  Ollama API:       http://localhost:10104"
    echo "  n8n Workflow:     http://localhost:5678"
    echo
}

# Main deployment flow
main() {
    log_info "Starting distributed AI services deployment..."
    log_info "Environment: $ENV"
    log_info "Phase: $PHASE"
    
    check_prerequisites
    create_directories
    
    case "$PHASE" in
        "all")
            deploy_infrastructure
            sleep 10
            deploy_ai_services
            sleep 10
            deploy_monitoring
            configure_services
            ;;
        "infra")
            deploy_infrastructure
            ;;
        "ai")
            deploy_ai_services
            ;;
        "monitor")
            deploy_monitoring
            ;;
        *)
            log_error "Invalid phase: $PHASE"
            exit 1
            ;;
    esac
    
    # Always run health check and display URLs
    health_check
    display_urls
    
    log_info "Deployment completed successfully!"
}

# Run main function
main "$@"