#!/bin/bash
#
# SutazAI Missing Components Deployment Script
# Based on Master System Blueprint v2.2
# 
# This script deploys all missing infrastructure and agent containers
# to complete the SutazAI AGI Platform integration

set -euo pipefail

# Color codes for output
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
    
    # Check if network exists
    if ! docker network ls | grep -q sutazai-network; then
        log_info "Creating sutazai-network..."
        docker network create --driver bridge --subnet=172.20.0.0/16 sutazai-network
    fi
    
    # Check if volumes exist
    if ! docker volume ls | grep -q agent_workspaces; then
        log_info "Creating agent_workspaces volume..."
        docker volume create agent_workspaces
    fi
    
    if ! docker volume ls | grep -q shared_runtime_data; then
        log_info "Creating shared_runtime_data volume..."
        docker volume create shared_runtime_data
    fi
    
    log_success "Prerequisites check completed"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log_info "Deploying missing infrastructure services..."
    
    if [ -f /opt/sutazaiapp/docker-compose.missing-services.yml ]; then
        docker-compose -f /opt/sutazaiapp/docker-compose.missing-services.yml up -d
        log_success "Infrastructure services deployed"
    else
        log_error "docker-compose.missing-services.yml not found"
        return 1
    fi
}

# Deploy agent services
deploy_agents() {
    log_info "Deploying missing agent services..."
    
    if [ -f /opt/sutazaiapp/docker-compose.missing-agents.yml ]; then
        docker-compose -f /opt/sutazaiapp/docker-compose.missing-agents.yml up -d
        log_success "Agent services deployed"
    else
        log_error "docker-compose.missing-agents.yml not found"
        return 1
    fi
}

# Check service health
check_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    log_info "Checking health of $service on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 2
    done
    
    log_error "$service health check failed after $max_attempts attempts"
    return 1
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check critical infrastructure services
    local critical_services=(
        "consul:10006"
        "kong:10005"
        "rabbitmq:10007"
        "resource-manager:10009"
    )
    
    for service_port in "${critical_services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        check_health "$service" "$port" || return 1
    done
    
    # Check critical agents
    local critical_agents=(
        "agentzero-coordinator:10300"
        "agent-orchestrator:10301"
    )
    
    for agent_port in "${critical_agents[@]}"; do
        IFS=':' read -r agent port <<< "$agent_port"
        check_health "$agent" "$port" || log_warning "$agent health check failed (non-critical)"
    done
    
    log_success "Deployment validation completed"
}

# Show deployment summary
show_summary() {
    log_info "=== DEPLOYMENT SUMMARY ==="
    echo
    echo "Infrastructure Services:"
    echo "  - Neo4j Browser: http://localhost:10002"
    echo "  - Kong API Gateway: http://localhost:10005"
    echo "  - Consul UI: http://localhost:10006"
    echo "  - RabbitMQ Management: http://localhost:10008"
    echo "  - Resource Manager: http://localhost:10009"
    echo "  - Backend API: http://localhost:10010"
    echo "  - Frontend UI: http://localhost:10011"
    echo
    echo "Agent Services:"
    echo "  - AgentZero Coordinator: http://localhost:10300"
    echo "  - Agent Orchestrator: http://localhost:10301"
    echo "  - Deep Learning Brain: http://localhost:10320"
    echo "  - Model Training: http://localhost:10322"
    echo
    echo "Monitoring:"
    echo "  - Loki Logs: http://localhost:10202"
    echo "  - Alertmanager: http://localhost:10203"
    echo "  - AI Metrics: http://localhost:10204/metrics"
    echo
    log_success "All services deployed successfully!"
}

# Main deployment flow
main() {
    log_info "Starting SutazAI Missing Components Deployment..."
    
    check_prerequisites
    
    # Deploy in phases
    log_info "Phase 1: Infrastructure Services"
    deploy_infrastructure
    sleep 10  # Wait for infrastructure to stabilize
    
    log_info "Phase 2: Agent Services"
    deploy_agents
    sleep 10  # Wait for agents to initialize
    
    log_info "Phase 3: Validation"
    validate_deployment
    
    show_summary
    
    log_success "Deployment completed successfully!"
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"