#!/bin/bash

# Self-Healing Critical Services Deployment Script
# Deploys core infrastructure with automatic recovery mechanisms

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/self_healing_deployment_$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.self-healing-critical.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log_error "An error occurred on line $1"
    log_error "Rolling back any partial deployments..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    exit 1
}

trap 'handle_error $LINENO' ERR

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Check if network exists
    if ! docker network inspect sutazai-network >/dev/null 2>&1; then
        log "Creating sutazai-network..."
        docker network create sutazai-network
    fi
    
    log "Prerequisites check completed"
}

# Load environment variables
load_environment() {
    log "Loading environment variables..."
    
    # Load from .env file if it exists
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
        log "Environment variables loaded from .env"
    fi
    
    # Set default values if not provided
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    export NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(openssl rand -base64 32)}"
    export SECRET_KEY="${SECRET_KEY:-$(openssl rand -base64 32)}"
    export JWT_SECRET="${JWT_SECRET:-$(openssl rand -base64 32)}"
    
    # Save passwords to secure location
    mkdir -p "$PROJECT_ROOT/secrets_secure"
    echo "$POSTGRES_PASSWORD" > "$PROJECT_ROOT/secrets_secure/postgres_password.txt"
    echo "$REDIS_PASSWORD" > "$PROJECT_ROOT/secrets_secure/redis_password.txt"
    echo "$NEO4J_PASSWORD" > "$PROJECT_ROOT/secrets_secure/neo4j_password.txt"
    chmod 600 "$PROJECT_ROOT/secrets_secure"/*.txt
    
    log "Environment variables configured"
}

# Build required images
build_images() {
    log "Building required Docker images..."
    
    # Build self-healing monitor image
    if [[ -f "$PROJECT_ROOT/docker/Dockerfile.healthcheck" ]]; then
        log_info "Building self-healing monitor image..."
        docker build -f "$PROJECT_ROOT/docker/Dockerfile.healthcheck" -t sutazai-self-healing-monitor "$PROJECT_ROOT"
    fi
    
    # Build circuit breaker image
    if [[ -f "$PROJECT_ROOT/self-healing/Dockerfile" ]]; then
        log_info "Building circuit breaker image..."
        docker build -f "$PROJECT_ROOT/self-healing/Dockerfile" -t sutazai-circuit-breaker "$PROJECT_ROOT/self-healing"
    fi
    
    log "Docker images built successfully"
}

# Deploy critical services
deploy_critical_services() {
    log "Deploying critical self-healing services..."
    
    # Stop any existing services
    log_info "Stopping existing services..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Start core services first
    log_info "Starting core database services..."
    docker compose -f "$COMPOSE_FILE" up -d postgres redis neo4j
    
    # Wait for databases to be healthy
    log_info "Waiting for databases to become healthy..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        local healthy_count=0
        
        # Check PostgreSQL
        if docker compose -f "$COMPOSE_FILE" ps postgres | grep -q "healthy"; then
            ((healthy_count++))
        fi
        
        # Check Redis
        if docker compose -f "$COMPOSE_FILE" ps redis | grep -q "healthy"; then
            ((healthy_count++))
        fi
        
        # Check Neo4j (may take longer)
        if docker compose -f "$COMPOSE_FILE" ps neo4j | grep -q "healthy"; then
            ((healthy_count++))
        fi
        
        if [[ $healthy_count -eq 3 ]]; then
            log "All database services are healthy"
            break
        fi
        
        log_info "Waiting for databases... ($attempt/$max_attempts) - Healthy: $healthy_count/3"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        log_error "Databases failed to become healthy within timeout"
        return 1
    fi
    
    # Start Ollama
    log_info "Starting Ollama service..."
    docker compose -f "$COMPOSE_FILE" up -d ollama
    
    # Wait for Ollama to be healthy
    attempt=1
    max_attempts=20
    while [[ $attempt -le $max_attempts ]]; do
        if docker compose -f "$COMPOSE_FILE" ps ollama | grep -q "healthy"; then
            log "Ollama service is healthy"
            break
        fi
        
        log_info "Waiting for Ollama... ($attempt/$max_attempts)"
        sleep 15
        ((attempt++))
    done
    
    # Start monitoring services
    log_info "Starting self-healing monitoring services..."
    docker compose -f "$COMPOSE_FILE" up -d self-healing-monitor circuit-breaker
    
    log "Critical services deployed successfully"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    local services=("postgres" "redis" "neo4j" "ollama" "self-healing-monitor" "circuit-breaker")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if ! docker compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            failed_services+=("$service")
            log_warning "Service $service is not running"
        else
            log_info "Service $service is running"
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "The following services failed to start: ${failed_services[*]}"
        return 1
    fi
    
    # Test connectivity
    log_info "Testing service connectivity..."
    
    # Test PostgreSQL
    if docker exec sutazai-postgres pg_isready -U sutazai -d sutazai >/dev/null 2>&1; then
        log_info "PostgreSQL connectivity: OK"
    else
        log_warning "PostgreSQL connectivity: FAILED"
    fi
    
    # Test Redis
    if docker exec sutazai-redis redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q "PONG"; then
        log_info "Redis connectivity: OK"
    else
        log_warning "Redis connectivity: FAILED"
    fi
    
    # Test Ollama
    if curl -s http://localhost:10104/api/version >/dev/null 2>&1; then
        log_info "Ollama connectivity: OK"
    else
        log_warning "Ollama connectivity: FAILED"
    fi
    
    log "Deployment validation completed"
}

# Setup monitoring dashboard
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create monitoring configuration
    cat > "$PROJECT_ROOT/monitoring/self_healing_config.json" << EOF
{
    "monitoring": {
        "interval": 30,
        "services": ["postgres", "redis", "neo4j", "ollama"],
        "restart_threshold": 3,
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
    },
    "alerting": {
        "enabled": true,
        "webhook_url": "${SLACK_WEBHOOK_URL:-}",
        "critical_services": ["postgres", "redis", "neo4j", "ollama"]
    }
}
EOF
    
    log "Monitoring configuration created"
}

# Create system status script
create_status_script() {
    log "Creating system status script..."
    
    cat > "$PROJECT_ROOT/scripts/check-self-healing-status.sh" << 'EOF'
#!/bin/bash

# Self-Healing System Status Check

echo "=== SutazAI Self-Healing System Status ==="
echo "Timestamp: $(date)"
echo

# Check critical services
echo "Critical Services Status:"
echo "------------------------"

services=("sutazai-postgres" "sutazai-redis" "sutazai-neo4j" "sutazai-ollama")
for service in "${services[@]}"; do
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service.*healthy"; then
        echo "✅ $service: HEALTHY"
    elif docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service.*Up"; then
        echo "⚠️  $service: RUNNING (no health check)"
    else
        echo "❌ $service: DOWN"
    fi
done

echo
echo "Self-Healing Services:"
echo "---------------------"

monitoring_services=("sutazai-self-healing-monitor" "sutazai-circuit-breaker")
for service in "${monitoring_services[@]}"; do
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service.*Up"; then
        echo "✅ $service: RUNNING"
    else
        echo "❌ $service: DOWN"
    fi
done

echo
echo "Circuit Breaker Status:"
echo "----------------------"
if curl -s http://localhost:10099/status >/dev/null 2>&1; then
    curl -s http://localhost:10099/status | jq -r '.status | to_entries[] | "  \(.key): \(.value.state) (failures: \(.value.failure_count), success_rate: \(.value.success_rate | round)%)"'
else
    echo "❌ Circuit breaker API not accessible"
fi

echo
echo "Resource Usage:"
echo "--------------"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" sutazai-postgres sutazai-redis sutazai-neo4j sutazai-ollama 2>/dev/null || echo "Unable to get resource stats"

EOF
    
    chmod +x "$PROJECT_ROOT/scripts/check-self-healing-status.sh"
    
    log "Status script created at scripts/check-self-healing-status.sh"
}

# Main deployment function
main() {
    log "Starting SutazAI self-healing critical services deployment"
    log "=================================================="
    
    check_prerequisites
    load_environment
    build_images
    deploy_critical_services
    validate_deployment
    setup_monitoring
    create_status_script
    
    log "=================================================="
    log "🎉 Self-healing critical services deployment completed successfully!"
    log ""
    log "Service Endpoints:"
    log "  - PostgreSQL: localhost:10010"
    log "  - Redis: localhost:10011"
    log "  - Neo4j HTTP: localhost:10002"
    log "  - Neo4j Bolt: localhost:10003"
    log "  - Ollama: localhost:10104"
    log "  - Circuit Breaker API: localhost:10099"
    log ""
    log "Management Commands:"
    log "  - Check status: ./scripts/check-self-healing-status.sh"
    log "  - View logs: docker compose -f docker-compose.self-healing-critical.yml logs -f"
    log "  - Stop services: docker compose -f docker-compose.self-healing-critical.yml down"
    log ""
    log "Next Steps:"
    log "  1. Verify all services are healthy: ./scripts/check-self-healing-status.sh"
    log "  2. Deploy AI agents with updated service endpoints"
    log "  3. Monitor self-healing actions in logs/self_healing_monitor.log"
    log ""
    log "Deployment log saved to: $LOG_FILE"
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Run main function
main "$@"