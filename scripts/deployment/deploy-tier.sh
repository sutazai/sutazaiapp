#!/bin/bash

# SutazAI Tiered Deployment Script
# Manages deployment of minimal, standard, or full tier configurations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TIER=${1:-minimal}
ACTION=${2:-up}
PROJECT_NAME="sutazai"
COMPOSE_FILES=""
COMPOSE_CMD="docker-compose"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose not found, trying docker compose..."
        if docker compose version &> /dev/null; then
            COMPOSE_CMD="docker compose"
        else
            print_error "Neither docker-compose nor docker compose is available"
            exit 1
        fi
    fi
    
    # Create network if it doesn't exist
    if ! docker network ls | grep -q sutazai-network; then
        print_info "Creating sutazai-network..."
        docker network create sutazai-network
    fi
    
    # Create data directories if they don't exist
    mkdir -p data/{postgres,redis,ollama,models,qdrant,prometheus,grafana,loki}
    
    print_success "Prerequisites check completed"
}

# Function to validate tier selection
validate_tier() {
    case $TIER in
        minimal|min)
            TIER="minimal"
            COMPOSE_FILES="-f docker-compose.minimal.yml"
            print_info "Selected Minimal Tier (5 containers, 2 CPU cores, 4GB RAM)"
            ;;
        standard|std)
            TIER="standard"
            COMPOSE_FILES="-f docker-compose.minimal.yml -f docker-compose.standard.yml"
            print_info "Selected Standard Tier (10 containers, 4 CPU cores, 8GB RAM)"
            ;;
        full)
            TIER="full"
            COMPOSE_FILES="-f docker-compose.minimal.yml -f docker-compose.standard.yml -f docker-compose.full.yml"
            print_info "Selected Full Tier (15-20 containers, 8+ CPU cores, 16GB+ RAM)"
            ;;
        current)
            TIER="current"
            COMPOSE_FILES="-f docker-compose.yml"
            print_warning "Using current (unoptimized) docker-compose.yml"
            ;;
        *)
            print_error "Invalid tier: $TIER"
            echo "Usage: $0 [minimal|standard|full|current] [up|down|restart|status|logs|clean]"
            exit 1
            ;;
    esac
}

# Function to check if required compose files exist
check_compose_files() {
    if [ "$TIER" = "minimal" ] && [ ! -f "docker-compose.minimal.yml" ]; then
        print_error "docker-compose.minimal.yml not found"
        exit 1
    fi
    
    if [ "$TIER" = "standard" ] && [ ! -f "docker-compose.standard.yml" ]; then
        print_error "docker-compose.standard.yml not found"
        exit 1
    fi
    
    if [ "$TIER" = "full" ] && [ ! -f "docker-compose.full.yml" ]; then
        print_warning "docker-compose.full.yml not found - creating placeholder..."
        cat > docker-compose.full.yml << 'EOF'
# SutazAI Full Tier Configuration Overlay
# Placeholder - to be implemented
version: '3.8'
services: {}
EOF
    fi
}

# Function to stop all running containers
stop_all_containers() {
    print_info "Stopping all SutazAI containers..."
    
    # Stop containers gracefully
    docker ps --format "{{.Names}}" | grep "^sutazai-" | while read container; do
        print_info "Stopping $container..."
        docker stop "$container" 2>/dev/null || true
    done
    
    print_success "All containers stopped"
}

# Function to start tier
start_tier() {
    print_info "Starting $TIER tier..."
    
    # Load environment variables if .env exists
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    # Start services
    $COMPOSE_CMD $COMPOSE_FILES up -d
    
    # Wait for services to be healthy
    print_info "Waiting for services to become healthy..."
    sleep 10
    
    # Check health status
    check_health
}

# Function to check health of services
check_health() {
    print_info "Checking service health..."
    
    $COMPOSE_CMD $COMPOSE_FILES ps
    
    echo ""
    print_info "Container resource usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.Status}}" | grep "sutazai-" || true
    
    echo ""
    print_info "Service endpoints:"
    case $TIER in
        minimal|standard|full)
            echo "  PostgreSQL:  localhost:10000"
            echo "  Redis:       localhost:10001"
            echo "  Ollama:      localhost:10104"
            echo "  Backend API: http://localhost:10010"
            echo "  Frontend UI: http://localhost:10011"
            ;;
    esac
    
    if [ "$TIER" = "standard" ] || [ "$TIER" = "full" ]; then
        echo "  Qdrant:      http://localhost:10101"
        echo "  Prometheus:  http://localhost:10200"
        echo "  Grafana:     http://localhost:10201 (admin/sutazai_grafana)"
        echo "  Loki:        http://localhost:10202"
    fi
}

# Function to show logs
show_logs() {
    SERVICE=${3:-}
    if [ -z "$SERVICE" ]; then
        print_info "Showing logs for all services..."
        $COMPOSE_CMD $COMPOSE_FILES logs -f --tail=100
    else
        print_info "Showing logs for $SERVICE..."
        $COMPOSE_CMD $COMPOSE_FILES logs -f --tail=100 "$SERVICE"
    fi
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes. Are you sure? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        print_info "Cleaning up..."
        
        # Stop all containers
        stop_all_containers
        
        # Remove containers
        docker ps -a --format "{{.Names}}" | grep "^sutazai-" | while read container; do
            docker rm "$container" 2>/dev/null || true
        done
        
        # Remove volumes
        docker volume ls --format "{{.Name}}" | grep "^sutazai" | while read volume; do
            print_info "Removing volume $volume..."
            docker volume rm "$volume" 2>/dev/null || true
        done
        
        # Remove network
        docker network rm sutazai-network 2>/dev/null || true
        
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Function to show resource usage summary
show_resource_summary() {
    print_info "Resource Usage Summary for $TIER tier:"
    echo ""
    
    # Count running containers
    RUNNING_COUNT=$(docker ps --format "{{.Names}}" | grep "^sutazai-" | wc -l)
    echo "Running Containers: $RUNNING_COUNT"
    
    # Calculate total resource usage
    docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemPerc}}" $(docker ps -q --filter "name=sutazai-") 2>/dev/null | \
        awk '{gsub(/%/, "", $1); gsub(/%/, "", $2); cpu+=$1; mem+=$2; count++} 
             END {
                 if (count > 0) {
                     printf "Total CPU Usage: %.2f%%\n", cpu
                     printf "Total Memory Usage: %.2f%%\n", mem
                     printf "Average CPU per Container: %.2f%%\n", cpu/count
                     printf "Average Memory per Container: %.2f%%\n", mem/count
                 }
             }'
}

# Main execution
main() {
    print_info "SutazAI Tiered Deployment Manager"
    echo "========================================"
    
    # Check prerequisites
    check_prerequisites
    
    # Validate tier selection
    validate_tier
    
    # Check if compose files exist
    check_compose_files
    
    # Execute action
    case $ACTION in
        up|start)
            stop_all_containers
            start_tier
            show_resource_summary
            print_success "$TIER tier is running"
            ;;
        down|stop)
            print_info "Stopping $TIER tier..."
            $COMPOSE_CMD $COMPOSE_FILES down
            print_success "$TIER tier stopped"
            ;;
        restart)
            print_info "Restarting $TIER tier..."
            $COMPOSE_CMD $COMPOSE_FILES restart
            check_health
            print_success "$TIER tier restarted"
            ;;
        status)
            check_health
            show_resource_summary
            ;;
        logs)
            show_logs "$@"
            ;;
        clean)
            cleanup
            ;;
        pull)
            print_info "Pulling latest images for $TIER tier..."
            $COMPOSE_CMD $COMPOSE_FILES pull
            print_success "Images updated"
            ;;
        validate)
            print_info "Validating $TIER tier configuration..."
            $COMPOSE_CMD $COMPOSE_FILES config > /dev/null
            print_success "Configuration is valid"
            ;;
        *)
            print_error "Invalid action: $ACTION"
            echo "Usage: $0 [minimal|standard|full|current] [up|down|restart|status|logs|clean|pull|validate]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"