#!/bin/bash

# SutazAI Migration Script - Transition to Tiered Architecture
# Safely migrates from current monolithic docker-compose.yml to tiered system

set -e

# Color codes

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="migration_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

print_info() {
    log "${BLUE}[INFO]${NC} $1"
}

print_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Function to create backup
create_backup() {
    print_info "Creating backup of current configuration..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup docker-compose files
    cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || true
    cp .env "$BACKUP_DIR/" 2>/dev/null || true
    
    # Backup data volumes list
    docker volume ls --format "{{.Name}}" | grep "sutazai" > "$BACKUP_DIR/volumes.txt" || true
    
    # Export running containers list
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | grep "sutazai" > "$BACKUP_DIR/running_containers.txt" || true
    
    # Backup Prometheus data if exists
    if docker volume ls | grep -q "prometheus_data"; then
        print_info "Backing up Prometheus data..."
        docker run --rm -v sutazaiapp_prometheus_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/prometheus_data.tar.gz -C /data . 2>/dev/null || true
    fi
    
    # Backup Grafana dashboards if exists
    if docker volume ls | grep -q "grafana_data"; then
        print_info "Backing up Grafana dashboards..."
        docker run --rm -v sutazaiapp_grafana_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/grafana_data.tar.gz -C /data . 2>/dev/null || true
    fi
    
    print_success "Backup created in $BACKUP_DIR"
}

# Function to analyze current deployment
analyze_current() {
    print_info "Analyzing current deployment..."
    
    echo ""
    echo "=== Current System Status ==="
    echo "Running Containers: $(docker ps --format '{{.Names}}' | grep -c 'sutazai-' || echo 0)"
    echo "Total Defined Services: $(grep -c '  [a-z].*:$' docker-compose.yml || echo 0)"
    echo ""
    
    # Check resource usage
    echo "=== Resource Usage ==="
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -20
    echo ""
    
    # Identify problematic services
    echo "=== Problematic Services ==="
    docker ps --format "{{.Names}}\t{{.Status}}" | grep -E "Restarting|unhealthy" || echo "None found"
    echo ""
    
    # List agent services
    echo "=== Agent Services (to be removed) ==="
    grep "container_name:" docker-compose.yml | grep -E "jarvis|agent|ai-" | sed 's/.*container_name: //' || echo "None found"
    echo ""
}

# Function to validate environment
validate_environment() {
    print_info "Validating environment..."
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cat > .env << 'EOF'
# SutazAI Environment Configuration
SUTAZAI_ENV=production
TZ=UTC

# Database
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_secure_password
POSTGRES_DB=sutazai

# Redis
REDIS_PASSWORD=

# Neo4j
NEO4J_PASSWORD=sutazai_neo4j

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=sutazai_grafana

# ChromaDB
CHROMADB_API_KEY=test-token
EOF
        print_warning "Please update .env file with secure passwords before continuing"
        exit 1
    fi
    
    # Validate Docker version
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' | cut -d. -f1)
    if [ "$DOCKER_VERSION" -lt 20 ]; then
        print_warning "Docker version is older than 20.x. Some features may not work."
    fi
    
    print_success "Environment validation completed"
}

# Function to stop unnecessary services
stop_unnecessary_services() {
    print_info "Stopping unnecessary services..."
    
    # List of services to stop
    SERVICES_TO_STOP=(
        "sutazai-jarvis"
        "sutazai-agent"
        "sutazai-ai-"
        "sutazai-hardware"
        "sutazai-crewai"
        "sutazai-autogpt"
        "sutazai-aider"
        "sutazai-gpt-engineer"
        "sutazai-tabbyml"
        "sutazai-pytorch"
        "sutazai-tensorflow"
        "sutazai-jax"
        "sutazai-kong"
        "sutazai-consul"
        "sutazai-cadvisor"
        "sutazai-chromadb"
        "sutazai-faiss"
    )
    
    for pattern in "${SERVICES_TO_STOP[@]}"; do
        docker ps --format "{{.Names}}" | grep "^$pattern" | while read container; do
            print_info "Stopping $container..."
            docker stop "$container" 2>/dev/null || true
            docker rm "$container" 2>/dev/null || true
        done
    done
    
    print_success "Unnecessary services stopped"
}

# Function to optimize running services
optimize_services() {
    print_info "Optimizing resource limits for running services..."
    
    # Update Neo4j if running
    if docker ps | grep -q "sutazai-neo4j"; then
        print_info "Optimizing Neo4j configuration..."
        docker update --memory="512m" --memory-swap="512m" --cpus="1" sutazai-neo4j 2>/dev/null || true
    fi
    
    # Update Ollama if running
    if docker ps | grep -q "sutazai-ollama"; then
        print_info "Optimizing Ollama configuration..."
        docker update --memory="2g" --memory-swap="2g" --cpus="2" sutazai-ollama 2>/dev/null || true
    fi
    
    # Update Backend if running
    if docker ps | grep -q "sutazai-backend"; then
        print_info "Optimizing Backend configuration..."
        docker update --memory="1g" --memory-swap="1g" --cpus="1" sutazai-backend 2>/dev/null || true
    fi
    
    # Update Frontend if running
    if docker ps | grep -q "sutazai-frontend"; then
        print_info "Optimizing Frontend configuration..."
        docker update --memory="512m" --memory-swap="512m" --cpus="0.5" sutazai-frontend 2>/dev/null || true
    fi
    
    print_success "Service optimization completed"
}

# Function to create migration report
create_migration_report() {
    print_info "Creating migration report..."
    
    cat > "migration_report_$(date +%Y%m%d_%H%M%S).md" << EOF
# SutazAI Migration Report
Date: $(date)

## Pre-Migration Status
- Running Containers: $(docker ps --format '{{.Names}}' | grep -c 'sutazai-' || echo 0)
- Total Defined Services: $(grep -c '  [a-z].*:$' docker-compose.yml || echo 0)
- Backup Location: $BACKUP_DIR

## Services Removed
$(docker ps -a --format '{{.Names}}' | grep -E 'jarvis|agent|ai-metrics' | sed 's/^/- /' || echo "None")

## Resource Optimization Applied
- Neo4j: Memory reduced to 512MB
- Ollama: Memory reduced to 2GB
- Backend: Memory reduced to 1GB
- Frontend: Memory reduced to 512MB

## Recommended Next Steps
1. Test   tier: ./scripts/deploy-tier.sh   up
2. Validate core functionality
3. If stable, proceed to standard tier: ./scripts/deploy-tier.sh standard up
4. Monitor resource usage and adjust as needed

## Rollback Instructions
If issues occur, restore previous configuration:
\`\`\`bash
cp $BACKUP_DIR/docker-compose.yml .
docker-compose up -d
\`\`\`
EOF
    
    print_success "Migration report created"
}

# Function to perform migration
perform_migration() {
    print_info "Starting migration to tiered architecture..."
    
    # Step 1: Create backup
    create_backup
    
    # Step 2: Analyze current deployment
    analyze_current
    
    # Step 3: Validate environment
    validate_environment
    
    # Step 4: Stop unnecessary services
    print_warning "This will stop all non-essential services. Continue? (yes/no)"
    read -r response
    if [ "$response" != "yes" ]; then
        print_info "Migration cancelled"
        exit 0
    fi
    
    stop_unnecessary_services
    
    # Step 5: Optimize remaining services
    optimize_services
    
    # Step 6: Create migration report
    create_migration_report
    
    # Step 7: Verify tier configuration files exist
    if [ ! -f "docker-compose. .yml" ]; then
        print_error "docker-compose. .yml not found. Cannot proceed with migration."
        print_info "Please ensure all tier configuration files are present."
        exit 1
    fi
    
    print_success "Migration preparation completed!"
    echo ""
    print_info "Next steps:"
    echo "  1. Review the migration report"
    echo "  2. Deploy   tier: ./scripts/deploy-tier.sh   up"
    echo "  3. Test core functionality"
    echo "  4. Gradually add services with standard/full tiers as needed"
    echo ""
    print_warning "Original configuration backed up to: $BACKUP_DIR"
}

# Function to show migration status
show_status() {
    print_info "Current Migration Status"
    echo "========================="
    
    # Check which tier files exist
    echo "Tier Configuration Files:"
    [ -f "docker-compose. .yml" ] && echo "  ✓   tier ready" || echo "  ✗   tier missing"
    [ -f "docker-compose.standard.yml" ] && echo "  ✓ Standard tier ready" || echo "  ✗ Standard tier missing"
    [ -f "docker-compose.full.yml" ] && echo "  ✓ Full tier ready" || echo "  ✗ Full tier missing"
    echo ""
    
    # Check deployment script
    echo "Deployment Tools:"
    [ -f "scripts/deploy-tier.sh" ] && echo "  ✓ Deployment script ready" || echo "  ✗ Deployment script missing"
    [ -f "scripts/migrate-to-tiered.sh" ] && echo "  ✓ Migration script ready" || echo "  ✗ Migration script missing"
    echo ""
    
    # Current system load
    echo "System Resources:"
    docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemPerc}}" $(docker ps -q --filter "name=sutazai-") 2>/dev/null | \
        awk '{gsub(/%/, "", $1); gsub(/%/, "", $2); cpu+=$1; mem+=$2} 
             END {printf "  CPU Usage: %.2f%%\n  Memory Usage: %.2f%%\n", cpu, mem}'
    echo ""
    
    # Container count
    echo "Container Status:"
    echo "  Running: $(docker ps --format '{{.Names}}' | grep -c 'sutazai-' || echo 0) containers"
    echo "  Defined: $(grep -c '  [a-z].*:$' docker-compose.yml || echo 0) services in docker-compose.yml"
}

# Main menu
case ${1:-migrate} in
    migrate)
        perform_migration
        ;;
    status)
        show_status
        ;;
    backup)
        create_backup
        ;;
    analyze)
        analyze_current
        ;;
    optimize)
        optimize_services
        ;;
    *)
        echo "Usage: $0 [migrate|status|backup|analyze|optimize]"
        echo ""
        echo "Commands:"
        echo "  migrate  - Perform full migration to tiered architecture (default)"
        echo "  status   - Show current migration status"
        echo "  backup   - Create backup only"
        echo "  analyze  - Analyze current deployment"
        echo "  optimize - Optimize resource limits only"
        exit 1
        ;;
esac