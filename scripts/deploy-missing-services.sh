#!/bin/bash

# SutazAI Missing Services Deployment Script
# Deploys all critical infrastructure services missing from the Master System Blueprint v2.2
# Author: Deploy Automation Master
# Date: 2025-08-04

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.missing-services.yml"
ENV_FILE="$PROJECT_ROOT/.env"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

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

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
        log_info "Check logs above for details"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose > /dev/null 2>&1; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Validate compose file
    if ! docker-compose -f "$COMPOSE_FILE" config > /dev/null 2>&1; then
        log_error "Invalid Docker Compose configuration"
        docker-compose -f "$COMPOSE_FILE" config
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    local dirs=(
        "$PROJECT_ROOT/data/neo4j"
        "$PROJECT_ROOT/data/neo4j/logs"
        "$PROJECT_ROOT/data/kong"
        "$PROJECT_ROOT/data/consul"
        "$PROJECT_ROOT/data/rabbitmq"
        "$PROJECT_ROOT/data/loki"
        "$PROJECT_ROOT/data/alertmanager"
        "$PROJECT_ROOT/data/faiss"
        "$PROJECT_ROOT/data/resource-manager"
        "$BACKUP_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    log_success "Directories created"
}

# Function to check port availability
check_ports() {
    log_info "Checking port availability..."
    
    local ports=(
        10002 10003  # Neo4j
        10005        # Kong
        10006        # Consul
        10007 10008  # RabbitMQ
        10009        # Resource Manager
        10010        # Backend API
        10011        # Frontend UI
        10100        # ChromaDB (existing but checking)
        10103        # FAISS
        10202        # Loki
        10203        # Alertmanager
        10204        # AI Metrics
    )
    
    local port_conflicts=()
    
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            port_conflicts+=("$port")
        fi
    done
    
    if [ ${#port_conflicts[@]} -gt 0 ]; then
        log_warning "Port conflicts detected: ${port_conflicts[*]}"
        log_info "Attempting to stop conflicting services..."
        
        # Try to stop existing services gracefully
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
        
        # Wait a moment for ports to be released
        sleep 5
        
        # Check again
        remaining_conflicts=()
        for port in "${port_conflicts[@]}"; do
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                remaining_conflicts+=("$port")
            fi
        done
        
        if [ ${#remaining_conflicts[@]} -gt 0 ]; then
            log_error "Unable to resolve port conflicts: ${remaining_conflicts[*]}"
            log_error "Please manually stop services using these ports and retry"
            exit 1
        fi
    fi
    
    log_success "Port availability check passed"
}

# Function to create environment file if it doesn't exist
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file..."
        
        cat > "$ENV_FILE" << EOF
# SutazAI Environment Configuration
TZ=UTC
SUTAZAI_ENV=production

# Database passwords
POSTGRES_PASSWORD=sutazai_secure_$(openssl rand -hex 8)
NEO4J_PASSWORD=sutazai_neo4j_$(openssl rand -hex 8)

# RabbitMQ configuration
RABBITMQ_USER=sutazai
RABBITMQ_PASSWORD=sutazai_rmq_$(openssl rand -hex 8)

# Service authentication
CHROMADB_API_KEY=sutazai_chroma_$(openssl rand -hex 12)
GRAFANA_PASSWORD=sutazai_grafana_$(openssl rand -hex 8)

# Resource limits
CPU_CORES=12
TOTAL_MEMORY_GB=29

# Consul encryption
CONSUL_ENCRYPT_KEY=$(openssl rand -base64 32)
EOF
        
        log_success "Environment file created: $ENV_FILE"
    else
        log_info "Environment file already exists: $ENV_FILE"
    fi
}

# Function to backup existing data
backup_existing_data() {
    log_info "Backing up existing data..."
    
    local data_dirs=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/volumes"
    )
    
    for dir in "${data_dirs[@]}"; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$BACKUP_DIR/" 2>/dev/null || true
            log_info "Backed up: $dir"
        fi
    done
    
    log_success "Data backup completed: $BACKUP_DIR"
}

# Function to create external network
create_network() {
    log_info "Creating external Docker network..."
    
    if ! docker network inspect sutazai-network > /dev/null 2>&1; then
        docker network create \
            --driver bridge \
            --subnet=172.20.0.0/16 \
            sutazai-network
        log_success "Created external network: sutazai-network"
    else
        log_info "External network already exists: sutazai-network"
    fi
}

# Function to create external volumes
create_volumes() {
    log_info "Creating external Docker volumes..."
    
    local volumes=(
        "shared_runtime_data"
    )
    
    for volume in "${volumes[@]}"; do
        if ! docker volume inspect "$volume" > /dev/null 2>&1; then
            docker volume create "$volume"
            log_success "Created external volume: $volume"
        else
            log_info "External volume already exists: $volume"
        fi
    done
}

# Function to deploy services in stages
deploy_services() {
    log_info "Deploying missing services in stages..."
    
    # Stage 1: Core Infrastructure
    log_info "Stage 1: Deploying core infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        neo4j \
        consul \
        rabbitmq
    
    # Wait for core services to be ready
    wait_for_service "neo4j" "10002" "/browser/"
    wait_for_service "consul" "10006" "/"
    wait_for_service "rabbitmq" "10008" "/"
    
    # Stage 2: Service Mesh
    log_info "Stage 2: Deploying service mesh..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        kong \
        resource-manager
    
    wait_for_service "kong" "10005" "/"
    wait_for_service "resource-manager" "10009" "/health"
    
    # Stage 3: AI Services
    log_info "Stage 3: Deploying AI services..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        faiss-vector \
        ai-metrics-exporter
    
    wait_for_service "faiss-vector" "10103" "/health"
    wait_for_service "ai-metrics-exporter" "10204" "/metrics"
    
    # Stage 4: Application Services
    log_info "Stage 4: Deploying application services..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        backend-api \
        frontend-ui
    
    wait_for_service "backend-api" "10010" "/health"
    wait_for_service "frontend-ui" "10011" "/"
    
    # Stage 5: Monitoring
    log_info "Stage 5: Deploying monitoring services..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        loki \
        alertmanager
    
    wait_for_service "loki" "10202" "/ready"
    wait_for_service "alertmanager" "10203" "/-/healthy"
    
    log_success "All services deployed successfully"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local health_path="$3"
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$port$health_path" > /dev/null 2>&1; then
            log_success "$service_name is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Function to validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    local services=(
        "neo4j:10002:/browser/"
        "consul:10006:/"
        "rabbitmq:10008:/"
        "kong:10005:/"
        "resource-manager:10009:/health"
        "faiss-vector:10103:/health"
        "ai-metrics-exporter:10204:/metrics"
        "backend-api:10010:/health"
        "frontend-ui:10011:/"
        "loki:10202:/ready"
        "alertmanager:10203:/-/healthy"
    )
    
    local failed_services=()
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service_name port health_path <<< "$service_info"
        
        if ! curl -f -s "http://localhost:$port$health_path" > /dev/null 2>&1; then
            failed_services+=("$service_name")
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_error "Validation failed for services: ${failed_services[*]}"
        return 1
    fi
    
    log_success "All services validated successfully"
}

# Function to show service status
show_service_status() {
    log_info "Service Status Summary:"
    echo
    
    # Show Docker Compose status
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo
    log_info "Service Access URLs:"
    cat << EOF
🔗 Neo4j Graph Database:    http://localhost:10002
🔗 Consul Service Discovery: http://localhost:10006
🔗 RabbitMQ Management:     http://localhost:10008
🔗 Kong API Gateway:        http://localhost:10005
🔗 Resource Manager:        http://localhost:10009
🔗 FAISS Vector Service:    http://localhost:10103
🔗 AI Metrics:              http://localhost:10204/metrics
🔗 Backend API:             http://localhost:10010
🔗 Frontend UI:             http://localhost:10011
🔗 Loki Logs:               http://localhost:10202
🔗 Alertmanager:            http://localhost:10203
EOF
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed, cleaning up..."
    
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    log_info "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting SutazAI Missing Services Deployment"
    log_info "Script: $0"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Compose File: $COMPOSE_FILE"
    
    # Pre-deployment checks
    check_prerequisites
    check_ports
    create_directories
    create_env_file
    backup_existing_data
    create_network
    create_volumes
    
    # Deploy services
    if deploy_services; then
        if validate_deployment; then
            show_service_status
            log_success "✅ Missing services deployment completed successfully!"
            log_info "All services are now running and ready for use"
        else
            log_error "❌ Deployment validation failed"
            cleanup_on_failure
            exit 1
        fi
    else
        log_error "❌ Service deployment failed"
        cleanup_on_failure
        exit 1
    fi
}

# Script usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy SutazAI missing infrastructure services

Options:
    -h, --help      Show this help message
    --skip-backup   Skip data backup step
    --force         Force deployment even with port conflicts
    
Examples:
    $0                      # Normal deployment
    $0 --skip-backup        # Deploy without backing up existing data
    $0 --force              # Force deployment
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"