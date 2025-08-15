#!/bin/bash
################################################################################
# EMERGENCY SYSTEM RECOVERY SCRIPT
# Purpose: Bring SutazAI system back online from complete down state
# Author: ULTRA-REMEDIATION-MASTER-001  
# Date: August 13, 2025
# Follows: CLAUDE.md Rules 1, 2, 3, 4, 19 (NO fantasy elements, real only)
################################################################################

set -euo pipefail

# Script configuration
readonly SCRIPT_NAME="Emergency System Recovery"
readonly SCRIPT_VERSION="1.0.0"
readonly PROJECT_ROOT="/opt/sutazaiapp"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/emergency_recovery_${TIMESTAMP}.log"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

################################################################################
# LOGGING FUNCTIONS
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" >&2
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
    esac
}

################################################################################
# VALIDATION FUNCTIONS
################################################################################

validate_prerequisites() {
    log "INFO" "Validating prerequisites..."
    
    # Check docker command availability
    if ! command -v docker >/dev/null 2>&1; then
        log "ERROR" "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check docker-compose command availability
    if ! command -v docker-compose >/dev/null 2>&1; then
        log "ERROR" "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Verify project root exists
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log "ERROR" "Project root directory not found: $PROJECT_ROOT"
        exit 1
    fi
    
    # Verify docker-compose.yml exists
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        log "ERROR" "Docker Compose file not found: $PROJECT_ROOT/docker-compose.yml"
        exit 1
    fi
    
    # Check if sutazai-network exists (REAL network from analysis)
    if ! docker network inspect sutazai-network >/dev/null 2>&1; then
        log "WARN" "sutazai-network not found, will be created by docker-compose"
    fi
    
    log "SUCCESS" "Prerequisites validation passed"
}

cleanup_existing_containers() {
    log "INFO" "Cleaning up existing containers..."
    
    # Get all sutazai containers (both running and stopped)
    local containers
    containers=$(docker ps -a --filter "name=sutazai" --format "{{.Names}}" 2>/dev/null || true)
    
    if [[ -z "$containers" ]]; then
        log "INFO" "No existing sutazai containers found"
        return 0
    fi
    
    log "INFO" "Found existing containers: $containers"
    
    # Stop any running containers
    for container in $containers; do
        if docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null | grep -q "running"; then
            log "INFO" "Stopping container: $container"
            docker stop "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # Remove all containers
    for container in $containers; do
        log "INFO" "Removing container: $container"
        docker rm "$container" >/dev/null 2>&1 || true
    done
    
    log "SUCCESS" "Container cleanup completed"
}

################################################################################
# RECOVERY FUNCTIONS  
################################################################################

start_core_services() {
    log "INFO" "Starting core database services..."
    
    cd "$PROJECT_ROOT"
    
    # Start core services first (databases)
    local core_services=("postgres" "redis" "neo4j")
    
    for service in "${core_services[@]}"; do
        log "INFO" "Starting service: $service"
        
        # Start service and capture any errors
        if docker-compose up -d "$service" 2>&1 | tee -a "$LOG_FILE"; then
            log "SUCCESS" "Service $service started"
            
            # Wait a moment for container to initialize
            sleep 5
            
            # Verify container is running
            if docker ps --filter "name=sutazai-$service" --format "{{.Names}}" | grep -q "sutazai-$service"; then
                log "SUCCESS" "Container sutazai-$service is running"
            else
                log "WARN" "Container sutazai-$service may not be running properly"
            fi
        else
            log "ERROR" "Failed to start service: $service"
            return 1
        fi
    done
}

start_application_services() {
    log "INFO" "Starting application services..."
    
    cd "$PROJECT_ROOT"
    
    # Start application services
    local app_services=("ollama" "backend" "frontend")
    
    for service in "${app_services[@]}"; do
        log "INFO" "Starting service: $service"
        
        if docker-compose up -d "$service" 2>&1 | tee -a "$LOG_FILE"; then
            log "SUCCESS" "Service $service started"
            sleep 3
        else
            log "WARN" "Service $service failed to start, continuing..."
        fi
    done
}

################################################################################
# HEALTH CHECK FUNCTIONS
################################################################################

verify_system_health() {
    log "INFO" "Performing system health checks..."
    
    local health_passed=0
    local health_total=0
    
    # Check PostgreSQL
    ((health_total++))
    if docker exec sutazai-postgres pg_isready -U sutazai >/dev/null 2>&1; then
        log "SUCCESS" "PostgreSQL health check passed"
        ((health_passed++))
    else
        log "WARN" "PostgreSQL health check failed"
    fi
    
    # Check Redis  
    ((health_total++))
    if docker exec sutazai-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log "SUCCESS" "Redis health check passed"
        ((health_passed++))
    else
        log "WARN" "Redis health check failed"
    fi
    
    # Check Neo4j (if running)
    if docker ps --filter "name=sutazai-neo4j" --format "{{.Names}}" | grep -q "sutazai-neo4j"; then
        ((health_total++))
        if curl -s http://localhost:10002 >/dev/null 2>&1; then
            log "SUCCESS" "Neo4j health check passed"
            ((health_passed++))
        else
            log "WARN" "Neo4j health check failed"
        fi
    fi
    
    # Check Backend API (if running)
    if docker ps --filter "name=sutazai-backend" --format "{{.Names}}" | grep -q "sutazai-backend"; then
        ((health_total++))
        if curl -s http://localhost:10010/health >/dev/null 2>&1; then
            log "SUCCESS" "Backend API health check passed"
            ((health_passed++))
        else
            log "WARN" "Backend API health check failed"
        fi
    fi
    
    log "INFO" "Health checks completed: $health_passed/$health_total services healthy"
    
    if [[ $health_passed -eq $health_total ]] && [[ $health_total -gt 0 ]]; then
        return 0
    else
        return 1
    fi
}

display_system_status() {
    log "INFO" "Current system status:"
    
    # Show running containers
    echo -e "\n${BOLD}Running Containers:${NC}"
    docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
    
    # Show service endpoints
    echo -e "\n${BOLD}Service Endpoints:${NC}"
    echo "PostgreSQL: localhost:10000 (if running)"
    echo "Redis: localhost:10001 (if running)"  
    echo "Neo4j: http://localhost:10002 (if running)"
    echo "Backend API: http://localhost:10010 (if running)"
    echo "Frontend UI: http://localhost:10011 (if running)"
    echo "Ollama: http://localhost:10104 (if running)"
    
    # Show logs location
    echo -e "\n${BOLD}Recovery Log:${NC} $LOG_FILE"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    log "INFO" "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    log "INFO" "Recovery log: $LOG_FILE"
    
    echo -e "${BOLD}SutazAI Emergency System Recovery${NC}"
    echo -e "Log: $LOG_FILE\n"
    
    # Step 1: Validate prerequisites
    validate_prerequisites
    
    # Step 2: Clean up existing containers
    cleanup_existing_containers
    
    # Step 3: Start core services
    start_core_services || {
        log "ERROR" "Failed to start core services"
        exit 1
    }
    
    # Step 4: Start application services (best effort)
    start_application_services
    
    # Step 5: Wait for services to stabilize
    log "INFO" "Waiting 30 seconds for services to stabilize..."
    sleep 30
    
    # Step 6: Verify system health
    if verify_system_health; then
        log "SUCCESS" "System recovery completed successfully"
        display_system_status
        exit 0
    else
        log "WARN" "System recovery completed with warnings"
        display_system_status
        exit 2
    fi
}

# Signal handlers
trap 'log "ERROR" "Script interrupted by user"; exit 130' INT
trap 'log "ERROR" "Script terminated"; exit 143' TERM

# Execute main function
main "$@"