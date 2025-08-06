#!/bin/bash

# SutazAI Consolidated Deployment Script
# Version: 2.0
# Date: 2025-08-05
# Description: Single script to deploy ONLY working services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.consolidated.yml"
ENV_FILE=".env"
LOG_FILE="deployment-$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (at least 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=10485760  # 10GB in KB
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space detected. At least 10GB recommended."
    fi
    
    # Check available memory (at least 8GB)
    available_memory=$(free -m | awk 'NR==2{print $7}')
    required_memory=8192  # 8GB in MB
    if [ "$available_memory" -lt "$required_memory" ]; then
        print_warning "Low memory detected. At least 8GB recommended."
    fi
    
    print_status "Prerequisites check completed"
}

# Function to create network
create_network() {
    print_header "CREATING DOCKER NETWORK"
    
    if docker network ls | grep -q "sutazai-network"; then
        print_status "Network 'sutazai-network' already exists"
    else
        docker network create sutazai-network
        print_status "Created network 'sutazai-network'"
    fi
}

# Function to create environment file if it doesn't exist
create_env_file() {
    print_header "CHECKING ENVIRONMENT CONFIGURATION"
    
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Environment file not found. Creating default .env file..."
        cat > "$ENV_FILE" << 'EOF'
# SutazAI Environment Configuration
# IMPORTANT: Change these default values in production!

# Database Configuration
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai123_change_in_production
NEO4J_PASSWORD=sutazai123_change_in_production

# Vector Database
CHROMADB_API_KEY=test-token-change-in-production

# Application Security
SECRET_KEY=dev-secret-key-change-in-production-with-32-chars
JWT_SECRET=dev-jwt-secret-change-in-production-with-32-chars

# Monitoring
GRAFANA_PASSWORD=sutazai_grafana

# Application Settings
SUTAZAI_ENV=production
TZ=UTC

# Optional Webhook for Health Alerts
HEALTH_ALERT_WEBHOOK=

# Redis (optional password)
REDIS_PASSWORD=
EOF
        print_status "Created default .env file. Please review and update passwords!"
        print_warning "Default passwords are not secure. Update them before production use."
    else
        print_status "Environment file exists"
    fi
}

# Function to validate Docker Compose file
validate_compose() {
    print_header "VALIDATING DOCKER COMPOSE CONFIGURATION"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "Docker Compose file '$COMPOSE_FILE' not found"
        exit 1
    fi
    
    # Validate compose file syntax
    if docker-compose -f "$COMPOSE_FILE" config &> /dev/null; then
        print_status "Docker Compose file syntax is valid"
    else
        print_error "Docker Compose file has syntax errors"
        docker-compose -f "$COMPOSE_FILE" config
        exit 1
    fi
}

# Function to pull images
pull_images() {
    print_header "PULLING DOCKER IMAGES"
    
    print_status "Pulling required images..."
    docker-compose -f "$COMPOSE_FILE" pull --quiet
    print_status "Image pull completed"
}

# Function to build custom images
build_images() {
    print_header "BUILDING CUSTOM IMAGES"
    
    print_status "Building backend image..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache backend
    
    print_status "Building frontend image..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache frontend
    
    print_status "Building health-monitor image..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache health-monitor
    
    print_status "Custom image build completed"
}

# Function to start services in phases
start_services() {
    print_header "STARTING SERVICES"
    
    # Phase 1: Core infrastructure
    print_status "Phase 1: Starting core infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis neo4j
    
    # Wait for databases to be healthy
    print_status "Waiting for databases to be ready..."
    sleep 30
    
    # Check database health
    for i in {1..12}; do
        if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "healthy\|Up"; then
            print_status "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 12 ]; then
            print_error "PostgreSQL failed to start within timeout"
            exit 1
        fi
        sleep 10
    done
    
    # Phase 2: Vector databases and search
    print_status "Phase 2: Starting vector databases..."
    docker-compose -f "$COMPOSE_FILE" up -d chromadb qdrant
    sleep 20
    
    # Phase 3: LLM service
    print_status "Phase 3: Starting Ollama LLM service..."
    docker-compose -f "$COMPOSE_FILE" up -d ollama
    
    # Wait for Ollama to be ready and pull gpt-oss model
    print_status "Waiting for Ollama to be ready and pulling gpt-oss model..."
    sleep 60
    
    # Pull gpt-oss model
    docker exec sutazai-ollama ollama pull gpt-oss || print_warning "Could not pull gpt-oss model automatically"
    
    # Phase 4: Application services
    print_status "Phase 4: Starting application services..."
    docker-compose -f "$COMPOSE_FILE" up -d backend
    sleep 30
    
    docker-compose -f "$COMPOSE_FILE" up -d frontend
    sleep 20
    
    # Phase 5: Monitoring stack
    print_status "Phase 5: Starting monitoring services..."
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana loki node-exporter cadvisor
    sleep 20
    
    # Phase 6: Health monitoring
    print_status "Phase 6: Starting health monitor..."
    docker-compose -f "$COMPOSE_FILE" up -d health-monitor
    
    print_status "All services started successfully"
}

# Function to check service health
check_health() {
    print_header "CHECKING SERVICE HEALTH"
    
    # Give services time to initialize
    print_status "Waiting for services to initialize..."
    sleep 60
    
    # Check service status
    print_status "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    # Check specific service health
    services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "backend" "frontend")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up\|healthy"; then
            print_status "✓ $service is running"
        else
            print_warning "✗ $service may have issues"
            # Show logs for failed service
            echo "=== Logs for $service ==="
            docker-compose -f "$COMPOSE_FILE" logs --tail=10 "$service"
            echo "=========================="
        fi
    done
}

# Function to show access information
show_access_info() {
    print_header "ACCESS INFORMATION"
    
    echo -e "${GREEN}SutazAI services are now running!${NC}"
    echo ""
    echo "Web Interfaces:"
    echo "  🌐 Frontend:          http://localhost:10011"
    echo "  📊 Backend API:       http://localhost:10010"
    echo "  📈 Grafana:           http://localhost:10201 (admin/sutazai_grafana)"
    echo "  📊 Prometheus:        http://localhost:10200"
    echo "  🔍 Loki:              http://localhost:10202"
    echo ""
    echo "Database Connections:"
    echo "  🐘 PostgreSQL:        localhost:10000"
    echo "  🗃️  Redis:             localhost:10001"
    echo "  🕷️  Neo4j Browser:     http://localhost:10002"
    echo "  🔍 ChromaDB:          http://localhost:10100"
    echo "  🎯 Qdrant:            http://localhost:10101"
    echo ""
    echo "AI Services:"
    echo "  🤖 Ollama:            http://localhost:10104"
    echo ""
    echo "Monitoring:"
    echo "  ❤️  Health Monitor:    http://localhost:10210"
    echo "  📊 Node Exporter:     http://localhost:10205"
    echo "  📦 cAdvisor:          http://localhost:10206"
    echo ""
    echo -e "${YELLOW}Note: All services use the credentials from your .env file${NC}"
    echo -e "${YELLOW}Default credentials are NOT secure - change them in production!${NC}"
}

# Function to show useful commands
show_commands() {
    print_header "USEFUL COMMANDS"
    
    echo "View logs:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f [service_name]"
    echo ""
    echo "Stop all services:"
    echo "  docker-compose -f $COMPOSE_FILE down"
    echo ""
    echo "Restart a service:"
    echo "  docker-compose -f $COMPOSE_FILE restart [service_name]"
    echo ""
    echo "Check service status:"
    echo "  docker-compose -f $COMPOSE_FILE ps"
    echo ""
    echo "Scale a service:"
    echo "  docker-compose -f $COMPOSE_FILE up -d --scale [service_name]=2"
    echo ""
    echo "View resource usage:"
    echo "  docker stats"
}

# Function to cleanup on error
cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    exit 1
}

# Main deployment function
main() {
    print_header "SUTAZAI CONSOLIDATED DEPLOYMENT"
    print_status "Starting deployment at $(date)"
    
    # Set error handler
    trap cleanup_on_error ERR
    
    # Run deployment steps
    check_prerequisites
    create_network
    create_env_file
    validate_compose
    pull_images
    build_images
    start_services
    check_health
    show_access_info
    show_commands
    
    print_status "Deployment completed successfully at $(date)"
    print_status "Log file: $LOG_FILE"
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        print_status "Stopping all services..."
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    "restart")
        print_status "Restarting all services..."
        docker-compose -f "$COMPOSE_FILE" down
        sleep 5
        main
        ;;
    "status")
        print_status "Service status:"
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "logs")
        service=${2:-}
        if [ -n "$service" ]; then
            docker-compose -f "$COMPOSE_FILE" logs -f "$service"
        else
            docker-compose -f "$COMPOSE_FILE" logs -f
        fi
        ;;
    "health")
        check_health
        ;;
    "clean")
        print_warning "This will remove all containers, volumes, and networks!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
            docker system prune -f
            print_status "Cleanup completed"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "SutazAI Consolidated Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    Deploy all services (default)"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  status    Show service status"
        echo "  logs      Show logs (optionally for specific service)"
        echo "  health    Check service health"
        echo "  clean     Remove all containers and volumes (destructive)"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 deploy"
        echo "  $0 logs backend"
        echo "  $0 status"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac