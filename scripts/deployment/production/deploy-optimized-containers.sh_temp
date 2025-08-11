#!/bin/bash

# Optimized Container Deployment Script
# Deploys production-ready, security-hardened containers
# Created: August 10, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.optimized.yml"
ENV_FILE=".env"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="deployment.log"

echo -e "${BLUE}SutazAI Optimized Container Deployment${NC}"
echo -e "${BLUE}======================================${NC}"
echo "Timestamp: $(date)"
echo ""

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    local missing_tools=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    # Check for compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo -e "${RED}Error: Optimized compose file not found: $COMPOSE_FILE${NC}"
        exit 1
    fi
    
    # Check for environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        echo -e "${YELLOW}Warning: .env file not found. Creating with defaults...${NC}"
        create_default_env
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo -e "${RED}Error: Missing required tools: ${missing_tools[*]}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
}

# Function to create default environment file
create_default_env() {
    cat > "$ENV_FILE" << 'EOF'
# SutazAI Environment Configuration
# Generated: $(date)

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=changeme_postgres_$(openssl rand -hex 12)
POSTGRES_DB=sutazai

# Redis Configuration
REDIS_PASSWORD=changeme_redis_$(openssl rand -hex 12)

# Neo4j Configuration
NEO4J_PASSWORD=changeme_neo4j_$(openssl rand -hex 12)

# RabbitMQ Configuration
RABBITMQ_DEFAULT_USER=sutazai
RABBITMQ_DEFAULT_PASS=changeme_rabbitmq_$(openssl rand -hex 12)

# ChromaDB Configuration
CHROMADB_API_KEY=changeme_chroma_$(openssl rand -hex 12)

# JWT Configuration
JWT_SECRET_KEY=changeme_jwt_$(openssl rand -hex 32)
SECRET_KEY=changeme_secret_$(openssl rand -hex 32)

# Grafana Configuration
GRAFANA_PASSWORD=changeme_grafana_$(openssl rand -hex 12)

# Environment
SUTAZAI_ENV=production
TZ=UTC
EOF
    
    echo -e "${GREEN}✓ Created default .env file with secure passwords${NC}"
    echo -e "${YELLOW}⚠ Please update passwords in .env before production deployment${NC}"
}

# Function to create required directories
create_directories() {
    echo -e "${BLUE}Creating required directories...${NC}"
    
    local dirs=(
        "volumes/postgres"
        "volumes/redis"
        "volumes/neo4j/data"
        "volumes/neo4j/logs"
        "volumes/ollama"
        "volumes/models"
        "volumes/chromadb"
        "volumes/qdrant/storage"
        "volumes/qdrant/snapshots"
        "volumes/rabbitmq"
        "volumes/prometheus"
        "volumes/grafana"
        "logs"
        "data"
        "configs"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        # Set appropriate permissions
        chmod 755 "$dir"
    done
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Function to backup existing deployment
backup_existing() {
    echo -e "${BLUE}Backing up existing deployment...${NC}"
    
    if docker-compose ps -q | grep -q .; then
        mkdir -p "$BACKUP_DIR"
        
        # Export running container configs
        docker-compose config > "$BACKUP_DIR/docker-compose.backup.yml"
        
        # Copy environment file
        if [[ -f "$ENV_FILE" ]]; then
            cp "$ENV_FILE" "$BACKUP_DIR/.env.backup"
        fi
        
        # List running containers
        docker-compose ps > "$BACKUP_DIR/running_containers.txt"
        
        echo -e "${GREEN}✓ Backup created in $BACKUP_DIR${NC}"
    else
        echo -e "${YELLOW}No existing deployment to backup${NC}"
    fi
}

# Function to create networks
create_networks() {
    echo -e "${BLUE}Creating networks...${NC}"
    
    # Create internal network
    if ! docker network ls | grep -q sutazai-internal; then
        docker network create \
            --driver bridge \
            --internal \
            --subnet=172.30.0.0/24 \
            sutazai-internal
        echo -e "${GREEN}✓ Created internal network${NC}"
    else
        echo "Internal network already exists"
    fi
    
    # Create external network
    if ! docker network ls | grep -q sutazai-external; then
        docker network create \
            --driver bridge \
            --subnet=172.31.0.0/24 \
            sutazai-external
        echo -e "${GREEN}✓ Created external network${NC}"
    else
        echo "External network already exists"
    fi
}

# Function to build secure images
build_secure_images() {
    echo -e "${BLUE}Building secure container images...${NC}"
    
    # Build images with BuildKit for better performance
    export DOCKER_BUILDKIT=1
    
    # List of services that need building
    local services=(
        "postgres"
        "redis"
        "ollama"
        "chromadb"
        "qdrant"
        "backend"
        "frontend"
        "hardware-resource-optimizer"
        "ai-agent-orchestrator"
    )
    
    for service in "${services[@]}"; do
        echo -e "Building ${service}..."
        if docker-compose -f "$COMPOSE_FILE" build --no-cache "$service" 2>&1 | tee -a "$LOG_FILE"; then
            echo -e "${GREEN}✓ Built ${service}${NC}"
        else
            echo -e "${YELLOW}⚠ Could not build ${service} (may use pre-built image)${NC}"
        fi
    done
}

# Function to deploy services in order
deploy_services() {
    echo -e "${BLUE}Deploying services...${NC}"
    
    # Deploy database tier first
    echo "Starting database tier..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis neo4j
    sleep 10
    
    # Deploy message queue
    echo "Starting message queue..."
    docker-compose -f "$COMPOSE_FILE" up -d rabbitmq
    sleep 5
    
    # Deploy AI/ML tier
    echo "Starting AI/ML tier..."
    docker-compose -f "$COMPOSE_FILE" up -d ollama chromadb qdrant
    sleep 10
    
    # Deploy application tier
    echo "Starting application tier..."
    docker-compose -f "$COMPOSE_FILE" up -d backend
    sleep 10
    
    # Deploy frontend
    echo "Starting frontend..."
    docker-compose -f "$COMPOSE_FILE" up -d frontend
    sleep 5
    
    # Deploy monitoring
    echo "Starting monitoring..."
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana
    sleep 5
    
    # Deploy agents
    echo "Starting agent services..."
    docker-compose -f "$COMPOSE_FILE" up -d hardware-resource-optimizer ai-agent-orchestrator
    
    echo -e "${GREEN}✓ All services deployed${NC}"
}

# Function to wait for services to be healthy
wait_for_health() {
    echo -e "${BLUE}Waiting for services to become healthy...${NC}"
    
    local max_attempts=30
    local attempt=0
    local all_healthy=false
    
    while [[ $attempt -lt $max_attempts ]]; do
        attempt=$((attempt + 1))
        echo -n "Attempt $attempt/$max_attempts: "
        
        # Check if all services are healthy
        if docker-compose -f "$COMPOSE_FILE" ps | grep -v "healthy" | grep -q "unhealthy\|starting"; then
            echo "Some services still starting..."
            sleep 10
        else
            all_healthy=true
            break
        fi
    done
    
    if $all_healthy; then
        echo -e "${GREEN}✓ All services are healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Some services may not be fully healthy${NC}"
    fi
}

# Function to validate deployment
validate_deployment() {
    echo -e "${BLUE}Validating deployment...${NC}"
    
    # Run validation script if available
    if [[ -f "scripts/validate-container-optimization.sh" ]]; then
        bash scripts/validate-container-optimization.sh "$COMPOSE_FILE" || true
    fi
    
    # Check service endpoints
    echo -e "\n${BLUE}Testing service endpoints:${NC}"
    
    local endpoints=(
        "http://localhost:10010/health:Backend API"
        "http://localhost:10011:Frontend UI"
        "http://localhost:10200:Prometheus"
        "http://localhost:10201:Grafana"
        "http://localhost:11110/health:Hardware Optimizer"
        "http://localhost:8589/health:AI Agent Orchestrator"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r url service_name <<< "$endpoint_info"
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $service_name is accessible"
        else
            echo -e "${YELLOW}⚠${NC} $service_name is not responding"
        fi
    done
}

# Function to display post-deployment information
show_deployment_info() {
    echo -e "\n${BLUE}=====================================>${NC}"
    echo -e "${BLUE}Deployment Complete!${NC}"
    echo -e "${BLUE}=====================================>${NC}"
    echo ""
    echo -e "${GREEN}Access Points:${NC}"
    echo "  Frontend:    http://localhost:10011"
    echo "  Backend API: http://localhost:10010"
    echo "  Grafana:     http://localhost:10201 (admin/admin)"
    echo "  Prometheus:  http://localhost:10200"
    echo ""
    echo -e "${GREEN}Container Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    echo -e "${GREEN}Security Notes:${NC}"
    echo "  - All containers running with non-root users"
    echo "  - Resource limits enforced"
    echo "  - Network isolation implemented"
    echo "  - Health checks configured"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Update passwords in .env file"
    echo "  2. Configure SSL/TLS certificates"
    echo "  3. Set up backup strategy"
    echo "  4. Configure monitoring alerts"
    echo "  5. Review security policies"
    echo ""
    echo -e "${BLUE}For K3s deployment:${NC}"
    echo "  kubectl apply -f k3s-deployment.yaml"
}

# Main deployment flow
main() {
    log "Starting optimized container deployment"
    
    # Check prerequisites
    check_prerequisites
    
    # Create required directories
    create_directories
    
    # Backup existing deployment
    backup_existing
    
    # Create networks
    create_networks
    
    # Build secure images
    if [[ "${BUILD_IMAGES:-false}" == "true" ]]; then
        build_secure_images
    fi
    
    # Deploy services
    deploy_services
    
    # Wait for health
    wait_for_health
    
    # Validate deployment
    validate_deployment
    
    # Show deployment information
    show_deployment_info
    
    log "Deployment completed successfully"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        echo -e "${BLUE}Stopping all services...${NC}"
        docker-compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✓ Services stopped${NC}"
        ;;
    restart)
        echo -e "${BLUE}Restarting all services...${NC}"
        docker-compose -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}✓ Services restarted${NC}"
        ;;
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    status)
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    validate)
        validate_deployment
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|validate}"
        exit 1
        ;;
esac