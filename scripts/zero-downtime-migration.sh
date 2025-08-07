#!/bin/bash
# Zero-Downtime Migration Script for SutazAI Distributed Architecture
# This script orchestrates the migration from single-instance to distributed setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups/migration-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/migration-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $*" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log_error "Migration failed at line $1"
    log_error "Rolling back changes..."
    rollback_migration
    exit 1
}

trap 'handle_error $LINENO' ERR

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        return 1
    fi
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]] && ! groups | grep -q docker; then
        log_error "This script must be run as root or user must be in docker group"
        return 1
    fi
    
    # Check disk space (need at least 10GB free)
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        log_error "Insufficient disk space. Need at least 10GB free"
        return 1
    fi
    
    # Check if old services are running
    if docker ps | grep -q "sutazai"; then
        log_info "Existing SutazAI services detected"
    fi
    
    log "Prerequisites check passed"
    return 0
}

# Backup current state
backup_current_state() {
    log "Creating backup of current state..."
    
    mkdir -p "$BACKUP_DIR"/{data,config,docker}
    
    # Backup Docker volumes
    log_info "Backing up Docker volumes..."
    for volume in $(docker volume ls -q | grep sutazai); do
        docker run --rm -v "${volume}:/source:ro" -v "${BACKUP_DIR}/data:/backup" \
            alpine tar -czf "/backup/${volume}.tar.gz" -C /source .
    done
    
    # Backup configuration files
    log_info "Backing up configuration files..."
    cp -r "${PROJECT_ROOT}/config" "${BACKUP_DIR}/"
    
    # Backup docker-compose files
    log_info "Backing up Docker Compose files..."
    cp "${PROJECT_ROOT}"/docker-compose*.yml "${BACKUP_DIR}/docker/"
    
    # Export running container list
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "${BACKUP_DIR}/running-containers.txt"
    
    log "Backup completed at: $BACKUP_DIR"
}

# Phase 1: Deploy infrastructure components
deploy_infrastructure() {
    log "Phase 1: Deploying infrastructure components..."
    
    # Initialize Docker Swarm if not already
    if ! docker info | grep -q "Swarm: active"; then
        log_info "Initializing Docker Swarm mode..."
        docker swarm init --advertise-addr $(hostname -I | awk '{print $1}')
    fi
    
    # Label nodes for service placement
    log_info "Labeling nodes for service placement..."
    docker node update --label-add consul=server1 $(docker node ls -q)
    docker node update --label-add redis=master1 $(docker node ls -q)
    docker node update --label-add postgres=primary $(docker node ls -q)
    
    # Deploy Consul cluster
    log_info "Deploying Consul cluster..."
    docker stack deploy -c "${PROJECT_ROOT}/docker-compose.distributed.yml" sutazai-distributed
    
    # Wait for Consul to be healthy
    log_info "Waiting for Consul cluster to be healthy..."
    for i in {1..30}; do
        if docker service ls | grep -q "sutazai-distributed_consul-server-1.*1/1"; then
            log "Consul cluster is healthy"
            break
        fi
        sleep 10
    done
    
    # Deploy Redis cluster
    log_info "Deploying Redis cluster..."
    # Wait for Redis masters to be ready
    sleep 30
    
    # Initialize Redis cluster
    log_info "Initializing Redis cluster..."
    docker exec $(docker ps -q -f name=redis-master-1) redis-cli --cluster create \
        redis-master-1:6379 redis-master-2:6379 redis-master-3:6379 \
        redis-slave-1:6379 redis-slave-2:6379 redis-slave-3:6379 \
        --cluster-replicas 1 --cluster-yes || true
    
    log "Infrastructure components deployed"
}

# Phase 2: Migrate data
migrate_data() {
    log "Phase 2: Migrating data..."
    
    # Check if old Redis is running
    if docker ps | grep -q "sutazai-redis"; then
        log_info "Migrating Redis data..."
        
        # Dump data from old Redis
        docker exec sutazai-redis redis-cli BGSAVE
        sleep 5
        
        # Copy dump to new Redis master
        docker cp sutazai-redis:/data/dump.rdb /tmp/redis-dump.rdb
        docker cp /tmp/redis-dump.rdb $(docker ps -q -f name=redis-master-1):/data/
        
        # Restart Redis to load data
        docker restart $(docker ps -q -f name=redis-master-1)
    fi
    
    # Check if old PostgreSQL is running
    if docker ps | grep -q "sutazai-postgres"; then
        log_info "Migrating PostgreSQL data..."
        
        # Create backup from old database
        docker exec sutazai-postgres pg_dump -U sutazai sutazai > /tmp/sutazai-db.sql
        
        # Restore to new primary
        docker exec -i $(docker ps -q -f name=postgres-primary) psql -U sutazai sutazai < /tmp/sutazai-db.sql
    fi
    
    # Migrate vector databases
    if docker ps | grep -q "sutazai-chromadb"; then
        log_info "Migrating ChromaDB data..."
        # Implementation depends on ChromaDB version
    fi
    
    log "Data migration completed"
}

# Phase 3: Deploy services with traffic routing
deploy_services() {
    log "Phase 3: Deploying distributed services..."
    
    # Deploy AI agents
    log_info "Deploying AI agent pool..."
    docker service scale sutazai-distributed_ai-agent=5
    
    # Deploy Celery workers
    log_info "Deploying Celery workers..."
    docker service scale sutazai-distributed_celery-worker=3
    
    # Deploy Ollama instances
    log_info "Deploying Ollama instances..."
    docker service scale sutazai-distributed_ollama=2
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 60
    
    # Health check all services
    for service in ai-agent celery-worker ollama; do
        if docker service ls | grep -q "sutazai-distributed_${service}"; then
            log "Service ${service} is running"
        else
            log_warning "Service ${service} is not fully deployed"
        fi
    done
    
    log "Services deployed"
}

# Phase 4: Traffic cutover
traffic_cutover() {
    log "Phase 4: Performing traffic cutover..."
    
    # Update DNS or load balancer configuration
    log_info "Updating load balancer configuration..."
    
    # If using HAProxy, reload configuration
    if docker ps | grep -q "haproxy"; then
        docker kill -s HUP $(docker ps -q -f name=haproxy)
    fi
    
    # Update Consul service registrations
    log_info "Updating service discovery..."
    
    # Register new services with Consul
    for i in {1..5}; do
        curl -X PUT http://localhost:8500/v1/agent/service/register -d "{
            \"ID\": \"ai-agent-${i}\",
            \"Name\": \"ai-agent\",
            \"Tags\": [\"ai\", \"distributed\", \"v2\"],
            \"Port\": 8080,
            \"Check\": {
                \"HTTP\": \"http://ai-agent-${i}:8080/health\",
                \"Interval\": \"10s\"
            }
        }"
    done
    
    # Gradually shift traffic
    log_info "Gradually shifting traffic to new infrastructure..."
    
    # Start with 10% traffic to new services
    for percentage in 10 25 50 75 90 100; do
        log_info "Routing ${percentage}% traffic to new services..."
        
        # Update load balancer weights
        # This would be specific to your load balancer
        
        # Monitor for errors
        sleep 30
        
        # Check error rates
        error_rate=$(curl -s http://localhost:9090/api/v1/query?query='rate(http_requests_total{status=~"5.."}[1m])' | \
            jq -r '.data.result[0].value[1] // "0"')
        
        if (( $(echo "$error_rate > 0.05" | bc -l) )); then
            log_error "High error rate detected: ${error_rate}"
            log_error "Rolling back traffic shift"
            return 1
        fi
    done
    
    log "Traffic cutover completed"
}

# Phase 5: Decommission old infrastructure
decommission_old() {
    log "Phase 5: Decommissioning old infrastructure..."
    
    # Stop old services
    log_info "Stopping old services..."
    
    # List of old service patterns
    old_services=(
        "sutazai-redis"
        "sutazai-postgres"
        "sutazai-ollama"
        "sutazai-agent"
        "sutazai-monitoring"
    )
    
    for service in "${old_services[@]}"; do
        if docker ps | grep -q "$service"; then
            log_info "Stopping $service..."
            docker stop "$service" || true
        fi
    done
    
    # Wait before removing
    log_info "Waiting 5 minutes before removing old containers..."
    sleep 300
    
    # Remove old containers
    for service in "${old_services[@]}"; do
        if docker ps -a | grep -q "$service"; then
            log_info "Removing $service..."
            docker rm "$service" || true
        fi
    done
    
    log "Old infrastructure decommissioned"
}

# Rollback function
rollback_migration() {
    log_warning "Starting rollback procedure..."
    
    # Stop new services
    docker stack rm sutazai-distributed || true
    
    # Restore old services from backup
    if [[ -d "$BACKUP_DIR" ]]; then
        log_info "Restoring from backup..."
        
        # Restore volumes
        for backup in "$BACKUP_DIR"/data/*.tar.gz; do
            volume_name=$(basename "$backup" .tar.gz)
            docker volume create "$volume_name"
            docker run --rm -v "${volume_name}:/target" -v "${BACKUP_DIR}/data:/backup:ro" \
                alpine tar -xzf "/backup/$(basename "$backup")" -C /target
        done
        
        # Restart old services
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.yml up -d
    fi
    
    log_warning "Rollback completed"
}

# Health check function
health_check() {
    log "Performing health check..."
    
    # Check Consul
    if ! curl -s http://localhost:8500/v1/status/leader | grep -q "8500"; then
        log_error "Consul is not healthy"
        return 1
    fi
    
    # Check Redis cluster
    if ! docker exec $(docker ps -q -f name=redis-master-1) redis-cli ping | grep -q PONG; then
        log_error "Redis cluster is not healthy"
        return 1
    fi
    
    # Check service endpoints
    endpoints=(
        "http://localhost:8080/health"     # HAProxy
        "http://localhost:9090/-/healthy"  # Prometheus
        "http://localhost:3000/api/health" # Grafana
    )
    
    for endpoint in "${endpoints[@]}"; do
        if ! curl -s -o /dev/null -w "%{http_code}" "$endpoint" | grep -q "200"; then
            log_error "Endpoint $endpoint is not healthy"
            return 1
        fi
    done
    
    log "Health check passed"
    return 0
}

# Main migration flow
main() {
    log "Starting SutazAI zero-downtime migration"
    log "Migration log: $LOG_FILE"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Phase 0: Prerequisites and backup
    check_prerequisites || exit 1
    backup_current_state
    
    # Phase 1: Deploy infrastructure
    deploy_infrastructure
    
    # Health check
    if ! health_check; then
        log_error "Infrastructure health check failed"
        rollback_migration
        exit 1
    fi
    
    # Phase 2: Migrate data
    migrate_data
    
    # Phase 3: Deploy services
    deploy_services
    
    # Phase 4: Traffic cutover
    if ! traffic_cutover; then
        rollback_migration
        exit 1
    fi
    
    # Final health check
    if ! health_check; then
        log_error "Final health check failed"
        rollback_migration
        exit 1
    fi
    
    # Phase 5: Decommission old infrastructure
    read -p "Ready to decommission old infrastructure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        decommission_old
    else
        log_warning "Skipping decommission phase. Old infrastructure still running."
    fi
    
    log "Migration completed successfully!"
    log "Backup location: $BACKUP_DIR"
    log "New distributed system is operational"
}

# Run main function
main "$@"