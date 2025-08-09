#!/bin/bash
# Service Mesh Startup Script for SutazAI
# Starts and configures the complete service mesh infrastructure
#
# DEPRECATION NOTICE: The Kong/Consul/RabbitMQ service-mesh stack is deprecated.
# See docs/decisions/2025-08-07-remove-service-mesh.md for context.
# This script remains for historical reference and may not function in current setups.
echo "[DEPRECATED] Service mesh scripts are deprecated and retained for reference."
echo "            See docs/decisions/2025-08-07-remove-service-mesh.md"

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="/opt/sutazaiapp/logs/service-mesh-startup.log"
DOCKER_COMPOSE_INFRA="${PROJECT_ROOT}/docker-compose.infrastructure.yml"
VALIDATION_SCRIPT="${PROJECT_ROOT}/scripts/service-mesh/validate-service-mesh.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Error handler
error_exit() {
    log ERROR "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed or not in PATH"
    fi
    
    # Check project structure
    if [[ ! -f "$DOCKER_COMPOSE_INFRA" ]]; then
        error_exit "Infrastructure docker-compose file not found: $DOCKER_COMPOSE_INFRA"
    fi
    
    # Check configuration files
    local config_files=(
        "${PROJECT_ROOT}/config/consul/services.json"
        "${PROJECT_ROOT}/config/kong/kong.yml"
        "${PROJECT_ROOT}/config/rabbitmq/definitions.json"
        "${PROJECT_ROOT}/config/rabbitmq/rabbitmq.conf"
    )
    
    for config_file in "${config_files[@]}"; do
        if [[ ! -f "$config_file" ]]; then
            error_exit "Required configuration file not found: $config_file"
        fi
    done
    
    log INFO "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log INFO "Creating necessary directories..."
    
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/config/consul"
    mkdir -p "${PROJECT_ROOT}/config/kong"
    mkdir -p "${PROJECT_ROOT}/config/rabbitmq"
    
    log INFO "Directories created"
}

# Stop existing services
stop_existing_services() {
    log INFO "Stopping existing service mesh services..."
    
    cd "$PROJECT_ROOT"
    
    # Stop infrastructure services
    docker-compose -f "$DOCKER_COMPOSE_INFRA" down --remove-orphans || {
        log WARN "Failed to stop some services, continuing..."
    }
    
    # Clean up any orphaned containers
    docker container prune -f || true
    
    log INFO "Existing services stopped"
}

# Start infrastructure services
start_infrastructure() {
    log INFO "Starting service mesh infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Build service mesh images
    log INFO "Building service mesh Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" build --no-cache \
        service-discovery health-check-server service-mesh-orchestrator || {
        error_exit "Failed to build service mesh images"
    }
    
    # Start infrastructure in stages
    log INFO "Starting Consul..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d consul
    
    # Wait for Consul
    wait_for_service "Consul" "http://localhost:10006/v1/status/leader" 60
    
    log INFO "Starting Kong database..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d kong-database
    
    # Wait for database
    sleep 15
    
    log INFO "Running Kong migrations..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up kong-migration
    
    log INFO "Starting Kong..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d kong
    
    # Wait for Kong
    wait_for_service "Kong" "http://localhost:10007/status" 60
    
    log INFO "Starting RabbitMQ..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d rabbitmq
    
    # Wait for RabbitMQ
    wait_for_service "RabbitMQ" "http://localhost:10042/api/overview" 90 "admin:adminpass"
    
    log INFO "Starting Redis for service mesh..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d redis-service-mesh
    
    # Wait for Redis
    sleep 10
    
    log INFO "Starting service mesh components..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d service-discovery health-check-server
    
    # Wait for health check server
    wait_for_service "Health Check Server" "http://localhost:10008/health" 60
    
    log INFO "Starting service mesh orchestrator..."
    docker-compose -f "$DOCKER_COMPOSE_INFRA" up -d service-mesh-orchestrator
    
    # Wait for orchestrator to complete initial configuration
    sleep 30
    
    log INFO "Service mesh infrastructure started successfully"
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-60}
    local auth=${4:-""}
    
    log INFO "Waiting for $service_name to be ready..."
    
    local count=0
    local auth_arg=""
    
    if [[ -n "$auth" ]]; then
        auth_arg="-u $auth"
    fi
    
    while [[ $count -lt $timeout ]]; do
        if curl -s $auth_arg -f "$url" > /dev/null 2>&1; then
            log INFO "$service_name is ready"
            return 0
        fi
        
        sleep 2
        ((count+=2))
        
        if [[ $((count % 20)) -eq 0 ]]; then
            log INFO "Still waiting for $service_name... (${count}s elapsed)"
        fi
    done
    
    error_exit "$service_name failed to become ready within ${timeout}s"
}

# Validate service mesh
validate_service_mesh() {
    log INFO "Validating service mesh configuration..."
    
    if [[ -f "$VALIDATION_SCRIPT" ]]; then
        cd "$PROJECT_ROOT"
        
        # Install Python dependencies if needed
        if ! python3 -c "import aiohttp" 2>/dev/null; then
            log INFO "Installing Python dependencies for validation..."
            pip3 install -q aiohttp pyyaml redis psutil || {
                log WARN "Failed to install validation dependencies, skipping validation"
                return 0
            }
        fi
        
        # Set environment variables for validation
        export CONSUL_URL="http://localhost:10006"
        export KONG_ADMIN_URL="http://localhost:10007"
        export KONG_PROXY_URL="http://localhost:10005"
        export RABBITMQ_URL="http://localhost:10042"
        export HEALTH_CHECK_URL="http://localhost:10008"
        
        # Run validation
        if python3 "$VALIDATION_SCRIPT"; then
            log INFO "Service mesh validation passed"
        else
            log WARN "Service mesh validation failed - some components may not be working correctly"
            log INFO "Check the validation output above for details"
        fi
    else
        log WARN "Validation script not found, skipping validation"
    fi
}

# Show service status
show_status() {
    log INFO "Service mesh status:"
    echo ""
    echo "=== Service Mesh Infrastructure ==="
    echo "Consul UI:           http://localhost:10006"
    echo "Kong Admin API:      http://localhost:10007"
    echo "Kong Proxy:          http://localhost:10005"
    echo "RabbitMQ Management: http://localhost:10042 (admin/adminpass)"
    echo "Health Check API:    http://localhost:10008"
    echo ""
    echo "=== Container Status ==="
    docker-compose -f "$DOCKER_COMPOSE_INFRA" ps
    echo ""
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-validation    Skip service mesh validation"
    echo "  --stop-only        Only stop existing services"
    echo "  --status           Show service status and exit"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  LOG_LEVEL          Set log level (INFO, WARN, ERROR, DEBUG)"
    echo ""
}

# Main function
main() {
    local skip_validation=false
    local stop_only=false
    local show_status_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-validation)
                skip_validation=true
                shift
                ;;
            --stop-only)
                stop_only=true
                shift
                ;;
            --status)
                show_status_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Show status only
    if [[ "$show_status_only" == true ]]; then
        show_status
        exit 0
    fi
    
    log INFO "Starting SutazAI Service Mesh setup..."
    
    # Check prerequisites
    check_prerequisites
    
    # Create directories
    create_directories
    
    # Stop existing services
    stop_existing_services
    
    # Exit if stop-only
    if [[ "$stop_only" == true ]]; then
        log INFO "Services stopped (stop-only mode)"
        exit 0
    fi
    
    # Start infrastructure
    start_infrastructure
    
    # Validate if requested
    if [[ "$skip_validation" == false ]]; then
        validate_service_mesh
    fi
    
    # Show final status
    show_status
    
    log INFO "Service mesh setup completed successfully!"
    log INFO "The service mesh is now ready for SutazAI services integration"
}

# Run main function
main "$@"
