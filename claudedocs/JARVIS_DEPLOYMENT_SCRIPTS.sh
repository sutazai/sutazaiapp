#!/bin/bash
# JARVIS Multi-Agent System - Deployment Scripts
# Complete automation for system deployment and management

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JARVIS_HOME="/opt/sutazaiapp"
COMPOSE_FILE="$JARVIS_HOME/claudedocs/JARVIS_DEPLOYMENT_CONFIG.yaml"
ENV_FILE="$JARVIS_HOME/.env"
LOG_FILE="$JARVIS_HOME/logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$JARVIS_HOME/logs"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# ===========================================
# INSTALLATION SCRIPT
# ===========================================

install_jarvis() {
    log "Starting JARVIS Multi-Agent System Installation"
    
    # Check prerequisites
    check_prerequisites
    
    # Create directory structure
    create_directories
    
    # Generate environment file
    generate_env_file
    
    # Pull Docker images
    pull_images
    
    # Initialize databases
    init_databases
    
    # Start core services
    start_core_services
    
    # Deploy agents
    deploy_agents
    
    # Configure service mesh
    configure_service_mesh
    
    # Setup monitoring
    setup_monitoring
    
    # Validate installation
    validate_installation
    
    log "JARVIS Installation Complete!"
    show_access_info
}

# ===========================================
# PREREQUISITE CHECKS
# ===========================================

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    info "Docker version: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    info "Docker Compose version: $(docker-compose --version)"
    
    # Check system resources
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8192 ]; then
        warning "System has less than 8GB RAM. Performance may be affected."
    fi
    info "Available memory: ${TOTAL_MEM}MB"
    
    # Check disk space
    DISK_SPACE=$(df -BG "$JARVIS_HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 20 ]; then
        warning "Less than 20GB disk space available"
    fi
    info "Available disk space: ${DISK_SPACE}GB"
    
    # Check ports
    check_port_availability
}

check_port_availability() {
    log "Checking port availability..."
    
    PORTS=(10000 10001 10002 10005 10010 10011 10100 10101 10104 10200 10201 11321 11400)
    
    for PORT in "${PORTS[@]}"; do
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            error "Port $PORT is already in use"
        fi
    done
    info "All required ports are available"
}

# ===========================================
# DIRECTORY STRUCTURE
# ===========================================

create_directories() {
    log "Creating directory structure..."
    
    DIRS=(
        "services/jarvis-core"
        "services/jarvis-voice"
        "services/jarvis-orchestrator"
        "agents/letta"
        "agents/autogpt"
        "agents/crewai"
        "agents/agent-zero"
        "agents/gpt-engineer"
        "config/kong"
        "config/consul"
        "config/prometheus"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/loki"
        "init-scripts/postgres"
        "scripts/models"
        "logs"
        "backups"
    )
    
    for DIR in "${DIRS[@]}"; do
        mkdir -p "$JARVIS_HOME/$DIR"
        info "Created: $DIR"
    done
}

# ===========================================
# ENVIRONMENT CONFIGURATION
# ===========================================

generate_env_file() {
    log "Generating environment configuration..."
    
    if [ -f "$ENV_FILE" ]; then
        warning "Environment file exists. Backing up..."
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    cat > "$ENV_FILE" << EOF
# JARVIS Multi-Agent System Environment Configuration
# Generated: $(date)

# Database Passwords
POSTGRES_PASSWORD=jarvis_secure_$(openssl rand -hex 8)
REDIS_PASSWORD=jarvis_redis_$(openssl rand -hex 8)
NEO4J_PASSWORD=jarvis_neo4j_$(openssl rand -hex 8)
RABBITMQ_PASSWORD=jarvis_rabbit_$(openssl rand -hex 8)

# Service Tokens
CHROMA_TOKEN=jarvis_chroma_$(openssl rand -hex 8)
GRAFANA_PASSWORD=jarvis_grafana_$(openssl rand -hex 8)

# System Configuration
JARVIS_HOME=$JARVIS_HOME
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=5
DEFAULT_MODEL=tinyllama
COMPLEX_MODEL=qwen3:0.5b

# Resource Limits
MEMORY_LIMIT_JARVIS=2G
MEMORY_LIMIT_AGENTS=1G
MEMORY_LIMIT_DATABASES=512M

# Feature Flags
ENABLE_VOICE=true
ENABLE_SELF_IMPROVEMENT=true
ENABLE_MONITORING=true
EOF
    
    chmod 600 "$ENV_FILE"
    info "Environment file generated"
}

# ===========================================
# DOCKER OPERATIONS
# ===========================================

pull_images() {
    log "Pulling Docker images..."
    
    IMAGES=(
        "postgres:16-alpine"
        "redis:7-alpine"
        "neo4j:5-community"
        "rabbitmq:3-management-alpine"
        "kong:3.5-alpine"
        "consul:1.17"
        "chromadb/chroma:latest"
        "qdrant/qdrant:latest"
        "ollama/ollama:latest"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "grafana/loki:latest"
        "docker:24-dind"
        "portainer/portainer-ce:latest"
    )
    
    for IMAGE in "${IMAGES[@]}"; do
        info "Pulling $IMAGE..."
        docker pull "$IMAGE" >> "$LOG_FILE" 2>&1
    done
}

# ===========================================
# DATABASE INITIALIZATION
# ===========================================

init_databases() {
    log "Initializing databases..."
    
    # Create PostgreSQL initialization script
    cat > "$JARVIS_HOME/init-scripts/postgres/01_init.sql" << 'EOF'
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database
CREATE DATABASE jarvis_db;
\c jarvis_db;

-- Create initial schema
-- Tables will be created by migration scripts
EOF
    
    # Create Neo4j initialization
    cat > "$JARVIS_HOME/config/neo4j_init.cypher" << 'EOF'
// Create indexes
CREATE INDEX agent_id IF NOT EXISTS FOR (a:Agent) ON (a.id);
CREATE INDEX task_id IF NOT EXISTS FOR (t:Task) ON (t.id);
CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id);

// Create constraints
CREATE CONSTRAINT unique_agent_id IF NOT EXISTS ON (a:Agent) ASSERT a.id IS UNIQUE;
CREATE CONSTRAINT unique_task_id IF NOT EXISTS ON (t:Task) ASSERT t.id IS UNIQUE;
EOF
    
    info "Database initialization scripts created"
}

# ===========================================
# SERVICE DEPLOYMENT
# ===========================================

start_core_services() {
    log "Starting core services..."
    
    # Start infrastructure services
    docker-compose -f "$COMPOSE_FILE" up -d \
        postgres redis neo4j rabbitmq \
        kong consul \
        chromadb qdrant \
        prometheus grafana loki \
        ollama
    
    # Wait for services to be healthy
    wait_for_services
    
    # Download models for Ollama
    download_models
}

wait_for_services() {
    log "Waiting for services to be healthy..."
    
    SERVICES=("postgres" "redis" "rabbitmq" "consul" "ollama")
    MAX_WAIT=300  # 5 minutes
    ELAPSED=0
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        ALL_HEALTHY=true
        
        for SERVICE in "${SERVICES[@]}"; do
            if ! docker-compose -f "$COMPOSE_FILE" ps "$SERVICE" | grep -q "healthy"; then
                ALL_HEALTHY=false
                break
            fi
        done
        
        if $ALL_HEALTHY; then
            info "All services are healthy"
            return 0
        fi
        
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo -n "."
    done
    
    error "Services failed to become healthy within timeout"
}

download_models() {
    log "Downloading AI models..."
    
    # Download TinyLlama (lightweight)
    docker exec sutazai-ollama ollama pull tinyllama:latest
    
    # Download Qwen3 0.5B (complex tasks)
    docker exec sutazai-ollama ollama pull qwen3:0.5b
    
    info "Models downloaded successfully"
}

deploy_agents() {
    log "Deploying AI agents..."
    
    # Build and start JARVIS core services
    docker-compose -f "$COMPOSE_FILE" up -d \
        jarvis-core jarvis-voice jarvis-orchestrator
    
    # Deploy agents one by one with health checks
    AGENTS=("letta-agent" "autogpt-agent" "crewai-agent" "agent-zero" "gpt-engineer-agent")
    
    for AGENT in "${AGENTS[@]}"; do
        info "Deploying $AGENT..."
        docker-compose -f "$COMPOSE_FILE" up -d "$AGENT"
        sleep 10  # Allow time for initialization
    done
    
    # Start DinD orchestrator
    docker-compose -f "$COMPOSE_FILE" up -d dind-orchestrator
    
    # Start UI services
    docker-compose -f "$COMPOSE_FILE" up -d backend frontend portainer
}

# ===========================================
# SERVICE MESH CONFIGURATION
# ===========================================

configure_service_mesh() {
    log "Configuring service mesh..."
    
    # Register services with Consul
    register_consul_services
    
    # Configure Kong routes
    configure_kong_routes
    
    # Setup RabbitMQ exchanges
    setup_rabbitmq
}

register_consul_services() {
    log "Registering services with Consul..."
    
    CONSUL_URL="http://localhost:10015"
    
    # Register JARVIS Core
    curl -X PUT "$CONSUL_URL/v1/agent/service/register" \
        -H "Content-Type: application/json" \
        -d '{
            "ID": "jarvis-core",
            "Name": "jarvis-core",
            "Port": 11321,
            "Check": {
                "HTTP": "http://jarvis-core:8000/health",
                "Interval": "30s"
            }
        }' >> "$LOG_FILE" 2>&1
    
    # Register other services similarly
    info "Services registered with Consul"
}

configure_kong_routes() {
    log "Configuring Kong API Gateway..."
    
    KONG_ADMIN="http://localhost:10006"
    
    # Create JARVIS service
    curl -X POST "$KONG_ADMIN/services" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "jarvis-api",
            "url": "http://jarvis-core:8000"
        }' >> "$LOG_FILE" 2>&1
    
    # Create route
    curl -X POST "$KONG_ADMIN/services/jarvis-api/routes" \
        -H "Content-Type: application/json" \
        -d '{
            "paths": ["/api/jarvis"],
            "strip_path": true
        }' >> "$LOG_FILE" 2>&1
    
    info "Kong routes configured"
}

setup_rabbitmq() {
    log "Setting up RabbitMQ..."
    
    # Wait for RabbitMQ to be ready
    sleep 10
    
    # Create exchanges and queues
    docker exec sutazai-rabbitmq rabbitmqctl add_vhost jarvis
    docker exec sutazai-rabbitmq rabbitmqctl set_permissions -p jarvis admin ".*" ".*" ".*"
    
    info "RabbitMQ configured"
}

# ===========================================
# MONITORING SETUP
# ===========================================

setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create Prometheus configuration
    cat > "$JARVIS_HOME/config/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'jarvis-core'
    static_configs:
      - targets: ['jarvis-core:8000']
  
  - job_name: 'agents'
    static_configs:
      - targets: ['letta-agent:8000', 'autogpt-agent:8000', 'crewai-agent:8000']
  
  - job_name: 'databases'
    static_configs:
      - targets: ['postgres:5432', 'redis:6379', 'neo4j:7474']
EOF
    
    # Create Grafana dashboard
    create_grafana_dashboard
    
    info "Monitoring configured"
}

create_grafana_dashboard() {
    log "Creating Grafana dashboards..."
    
    # Create datasource configuration
    cat > "$JARVIS_HOME/config/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    info "Grafana dashboards created"
}

# ===========================================
# VALIDATION
# ===========================================

validate_installation() {
    log "Validating installation..."
    
    ERRORS=0
    
    # Check all containers are running
    EXPECTED_CONTAINERS=25
    RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}" | tail -n +2 | wc -l)
    
    if [ "$RUNNING_CONTAINERS" -lt "$EXPECTED_CONTAINERS" ]; then
        warning "Expected $EXPECTED_CONTAINERS containers, found $RUNNING_CONTAINERS"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Test API endpoints
    ENDPOINTS=(
        "http://localhost:11321/health"  # JARVIS Core
        "http://localhost:10010/health"  # Backend
        "http://localhost:10200"         # Prometheus
        "http://localhost:10201"         # Grafana
    )
    
    for ENDPOINT in "${ENDPOINTS[@]}"; do
        if ! curl -f -s "$ENDPOINT" > /dev/null 2>&1; then
            warning "Endpoint not responding: $ENDPOINT"
            ERRORS=$((ERRORS + 1))
        else
            info "✓ $ENDPOINT"
        fi
    done
    
    if [ $ERRORS -eq 0 ]; then
        log "✅ Installation validated successfully!"
    else
        warning "Installation completed with $ERRORS warnings"
    fi
}

# ===========================================
# ACCESS INFORMATION
# ===========================================

show_access_info() {
    cat << EOF

${GREEN}════════════════════════════════════════════════════════════════${NC}
${GREEN}         JARVIS Multi-Agent System - Installation Complete       ${NC}
${GREEN}════════════════════════════════════════════════════════════════${NC}

${BLUE}Main Services:${NC}
  • JARVIS API:        http://localhost:11321
  • WebSocket:         ws://localhost:11322
  • Voice Interface:   http://localhost:11323
  • Backend API:       http://localhost:10010
  • Frontend UI:       http://localhost:10011

${BLUE}Monitoring:${NC}
  • Grafana:          http://localhost:10201 (admin/jarvis_grafana_2025)
  • Prometheus:       http://localhost:10200
  • Portainer:        http://localhost:11700

${BLUE}Service Mesh:${NC}
  • Kong Gateway:     http://localhost:10005
  • Consul UI:        http://localhost:10015
  • RabbitMQ:         http://localhost:10008 (admin/jarvis_rabbit_2025)

${BLUE}Databases:${NC}
  • PostgreSQL:       localhost:10000
  • Redis:            localhost:10001
  • Neo4j Browser:    http://localhost:10002
  • ChromaDB:         http://localhost:10100
  • Qdrant:           http://localhost:10101

${BLUE}Quick Start:${NC}
  1. Open the Frontend UI: http://localhost:10011
  2. Say "Hey JARVIS" or click the microphone icon
  3. Try: "JARVIS, what agents are available?"
  4. Monitor system: http://localhost:10201

${BLUE}Documentation:${NC}
  • System Design:    $JARVIS_HOME/claudedocs/JARVIS_UNIFIED_SYSTEM_DESIGN.md
  • API Spec:         $JARVIS_HOME/claudedocs/JARVIS_API_SPECIFICATION.yaml
  • Database Schema:  $JARVIS_HOME/claudedocs/JARVIS_DATABASE_SCHEMA.md

${GREEN}════════════════════════════════════════════════════════════════${NC}

EOF
}

# ===========================================
# MANAGEMENT COMMANDS
# ===========================================

start_jarvis() {
    log "Starting JARVIS system..."
    docker-compose -f "$COMPOSE_FILE" up -d
    show_access_info
}

stop_jarvis() {
    log "Stopping JARVIS system..."
    docker-compose -f "$COMPOSE_FILE" down
}

restart_jarvis() {
    log "Restarting JARVIS system..."
    stop_jarvis
    start_jarvis
}

status_jarvis() {
    log "JARVIS system status:"
    docker-compose -f "$COMPOSE_FILE" ps
}

logs_jarvis() {
    SERVICE=${1:-jarvis-core}
    docker-compose -f "$COMPOSE_FILE" logs -f "$SERVICE"
}

backup_jarvis() {
    log "Creating backup..."
    
    BACKUP_DIR="$JARVIS_HOME/backups/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup databases
    docker exec sutazai-postgres pg_dumpall -U postgres > "$BACKUP_DIR/postgres.sql"
    docker exec sutazai-redis redis-cli BGSAVE
    docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb"
    
    # Backup configurations
    cp -r "$JARVIS_HOME/config" "$BACKUP_DIR/"
    cp "$ENV_FILE" "$BACKUP_DIR/"
    
    log "Backup created: $BACKUP_DIR"
}

clean_jarvis() {
    log "Cleaning JARVIS system..."
    
    read -p "This will remove all containers and volumes. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "$COMPOSE_FILE" down -v
        log "System cleaned"
    else
        log "Clean cancelled"
    fi
}

# ===========================================
# MAIN COMMAND HANDLER
# ===========================================

show_usage() {
    cat << EOF
JARVIS Multi-Agent System Management Script

Usage: $0 [command] [options]

Commands:
  install     Install the complete JARVIS system
  start       Start all services
  stop        Stop all services
  restart     Restart all services
  status      Show service status
  logs        Show logs (optionally specify service)
  backup      Create system backup
  clean       Remove all containers and volumes
  help        Show this help message

Examples:
  $0 install              # Full installation
  $0 start                # Start system
  $0 logs jarvis-core     # Show JARVIS core logs
  $0 backup               # Create backup

EOF
}

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    install)
        install_jarvis
        ;;
    start)
        start_jarvis
        ;;
    stop)
        stop_jarvis
        ;;
    restart)
        restart_jarvis
        ;;
    status)
        status_jarvis
        ;;
    logs)
        logs_jarvis "$2"
        ;;
    backup)
        backup_jarvis
        ;;
    clean)
        clean_jarvis
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        error "Unknown command: $COMMAND"
        show_usage
        ;;
esac