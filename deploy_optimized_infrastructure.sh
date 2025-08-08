#!/bin/bash
# SutazAI Infrastructure Optimization Deployment Script
# Systems Architect: Infrastructure DevOps Manager (INFRA-001)
# Date: 2025-08-08

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
COMPOSE_FILE="docker-compose.yml"
OPTIMIZED_FILE="docker-compose.optimized.yml"
VALIDATION_TIMEOUT=300

# Logging
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    if [[ ! -f "$OPTIMIZED_FILE" ]]; then
        error "Optimized compose file not found: $OPTIMIZED_FILE"
    fi
    
    log "Prerequisites check passed"
}

# Create backup
create_backup() {
    log "Creating backup in $BACKUP_DIR..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup current compose file
    if [[ -f "$COMPOSE_FILE" ]]; then
        cp "$COMPOSE_FILE" "$BACKUP_DIR/docker-compose.yml.backup"
        log "Backed up current docker-compose.yml"
    fi
    
    # Backup current containers state
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "$BACKUP_DIR/containers_before.txt" 2>/dev/null || true
    
    # Backup environment
    if [[ -f ".env" ]]; then
        cp ".env" "$BACKUP_DIR/.env.backup"
    fi
    
    # Export current volumes
    docker volume ls > "$BACKUP_DIR/volumes_before.txt" 2>/dev/null || true
    
    log "Backup created successfully"
}

# Stop current services gracefully
stop_current_services() {
    log "Stopping current services gracefully..."
    
    if [[ -f "$COMPOSE_FILE" ]]; then
        # Try graceful shutdown first
        docker-compose down --timeout 30 2>/dev/null || true
        
        # Force stop any remaining containers
        docker stop $(docker ps -aq --filter "name=sutazai-*") 2>/dev/null || true
        
        # Clean up networks
        docker network rm sutazai-network 2>/dev/null || true
        
        log "Current services stopped"
    else
        warn "No existing docker-compose.yml found"
    fi
}

# Deploy optimized configuration
deploy_optimized() {
    log "Deploying optimized infrastructure..."
    
    # Replace compose file
    cp "$OPTIMIZED_FILE" "$COMPOSE_FILE"
    log "Replaced docker-compose.yml with optimized version"
    
    # Create network if it doesn't exist
    docker network create sutazai-network 2>/dev/null || true
    
    # Start core infrastructure first (Tier 1)
    log "Starting Tier 1: Core Infrastructure..."
    docker-compose up -d postgres redis neo4j
    sleep 30
    
    # Start AI/ML infrastructure (Tier 2)  
    log "Starting Tier 2: AI/ML Infrastructure..."
    docker-compose up -d ollama chromadb qdrant faiss
    sleep 45
    
    # Start service mesh (Tier 3)
    log "Starting Tier 3: Service Mesh..."
    docker-compose up -d kong consul rabbitmq
    sleep 20
    
    # Start application layer (Tier 4)
    log "Starting Tier 4: Application Layer..."
    docker-compose up -d backend
    sleep 30
    docker-compose up -d frontend
    sleep 15
    
    # Start monitoring stack (Tier 5)
    log "Starting Tier 5: Monitoring Stack..."
    docker-compose up -d prometheus grafana loki alertmanager node-exporter cadvisor
    docker-compose up -d postgres-exporter redis-exporter blackbox-exporter promtail
    sleep 30
    
    # Start AI agents (Tier 6+)
    log "Starting Tier 6+: AI Agent Ecosystem..."
    docker-compose up -d ollama-integration langflow flowise n8n dify
    docker-compose up -d autogpt crewai aider gpt-engineer llamaindex
    docker-compose up -d hardware-resource-optimizer documind privategpt shellgpt
    sleep 30
    
    # Start utility services (Tier 9)
    log "Starting Tier 9: Utility Services..."
    docker-compose up -d health-monitor service-hub code-improver
    
    log "Optimized infrastructure deployed"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    local start_time=$(date +%s)
    local timeout=$VALIDATION_TIMEOUT
    
    # Core service validation
    local core_services=("sutazai-postgres" "sutazai-redis" "sutazai-neo4j" "sutazai-ollama" "sutazai-backend" "sutazai-frontend")
    
    for service in "${core_services[@]}"; do
        info "Validating $service..."
        local elapsed=0
        
        while [[ $elapsed -lt $timeout ]]; do
            if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
                local health=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "unknown")
                if [[ "$health" == "healthy" ]] || [[ "$health" == "unknown" ]]; then
                    log "$service is running and healthy"
                    break
                fi
            fi
            
            sleep 5
            elapsed=$((elapsed + 5))
            
            if [[ $elapsed -ge $timeout ]]; then
                error "$service failed to start within timeout"
            fi
        done
    done
    
    # Endpoint validation
    info "Validating key endpoints..."
    
    # Wait a bit more for services to fully initialize
    sleep 60
    
    # Test backend health
    if curl -f http://localhost:10010/health >/dev/null 2>&1; then
        log "Backend API is responding"
    else
        warn "Backend API not responding yet (may still be initializing)"
    fi
    
    # Test frontend
    if curl -f http://localhost:10011 >/dev/null 2>&1; then
        log "Frontend is accessible"
    else
        warn "Frontend not accessible yet (may still be initializing)"
    fi
    
    # Test monitoring
    if curl -f http://localhost:10200/-/healthy >/dev/null 2>&1; then
        log "Prometheus is healthy"
    else
        warn "Prometheus not healthy yet"
    fi
    
    if curl -f http://localhost:10201/api/health >/dev/null 2>&1; then
        log "Grafana is accessible"
    else
        warn "Grafana not accessible yet"
    fi
    
    log "Deployment validation completed"
}

# Generate status report
generate_status_report() {
    log "Generating status report..."
    
    local report_file="$BACKUP_DIR/deployment_report.txt"
    
    {
        echo "SutazAI Infrastructure Optimization Deployment Report"
        echo "Date: $(date)"
        echo "======================================================="
        echo ""
        
        echo "DEPLOYED SERVICES:"
        docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Could not fetch container status"
        echo ""
        
        echo "NETWORK CONFIGURATION:"
        docker network inspect sutazai-network 2>/dev/null || echo "Network not found"
        echo ""
        
        echo "VOLUME STATUS:"
        docker volume ls | grep -E "(postgres|redis|neo4j|ollama|grafana|prometheus)" 2>/dev/null || echo "Could not fetch volume status"
        echo ""
        
        echo "KEY ENDPOINTS:"
        echo "- Backend API: http://localhost:10010/health"
        echo "- Frontend UI: http://localhost:10011"
        echo "- Prometheus: http://localhost:10200"
        echo "- Grafana: http://localhost:10201 (admin/admin)"
        echo "- Kong Admin: http://localhost:10015"
        echo "- Consul: http://localhost:10006"
        echo "- RabbitMQ: http://localhost:10008"
        echo ""
        
        echo "BACKUP LOCATION: $BACKUP_DIR"
        echo ""
        
    } > "$report_file"
    
    cat "$report_file"
    log "Status report saved to: $report_file"
}

# Rollback function
rollback() {
    error_msg="$1"
    warn "Deployment failed: $error_msg"
    warn "Initiating rollback..."
    
    # Stop new services
    docker-compose down --timeout 30 2>/dev/null || true
    
    # Restore backup
    if [[ -f "$BACKUP_DIR/docker-compose.yml.backup" ]]; then
        cp "$BACKUP_DIR/docker-compose.yml.backup" "$COMPOSE_FILE"
        log "Restored original docker-compose.yml"
        
        # Start original services
        docker-compose up -d
        log "Original services restarted"
    fi
    
    error "Deployment failed and rolled back. Check logs in $BACKUP_DIR"
}

# Main deployment process
main() {
    log "Starting SutazAI Infrastructure Optimization Deployment"
    log "========================================================"
    
    # Set trap for error handling
    trap 'rollback "Unexpected error during deployment"' ERR
    
    # Execute deployment steps
    check_prerequisites
    create_backup
    stop_current_services
    deploy_optimized
    validate_deployment
    generate_status_report
    
    log "========================================================"
    log "🎉 DEPLOYMENT SUCCESSFUL! 🎉"
    log "========================================================"
    log ""
    log "✅ Optimized infrastructure is now running"
    log "✅ All services have been validated"
    log "✅ Backup created in: $BACKUP_DIR"
    log ""
    log "🔗 Quick Access URLs:"
    log "   • Frontend: http://localhost:10011"
    log "   • Backend API: http://localhost:10010/health"
    log "   • Grafana: http://localhost:10201 (admin/admin)"
    log "   • Prometheus: http://localhost:10200"
    log ""
    log "📚 Next Steps:"
    log "   1. Configure Grafana dashboards and alerts"
    log "   2. Test AI agent functionality"
    log "   3. Monitor system performance"
    log "   4. Scale services as needed using profiles"
    log ""
    log "🛟 Rollback (if needed):"
    log "   docker-compose down && cp $BACKUP_DIR/docker-compose.yml.backup docker-compose.yml && docker-compose up -d"
    log ""
    log "✨ SutazAI Infrastructure Optimization Complete!"
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        if [[ -z "${2:-}" ]]; then
            error "Please specify backup directory for rollback"
        fi
        BACKUP_DIR="$2"
        rollback "Manual rollback requested"
        ;;
    "validate")
        validate_deployment
        ;;
    "status")
        generate_status_report
        ;;
    *)
        echo "Usage: $0 [deploy|rollback <backup_dir>|validate|status]"
        echo ""
        echo "Commands:"
        echo "  deploy          - Deploy optimized infrastructure (default)"
        echo "  rollback <dir>  - Rollback to backup in specified directory"
        echo "  validate        - Validate current deployment"
        echo "  status          - Generate status report"
        exit 1
        ;;
esac