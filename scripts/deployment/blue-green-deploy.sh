#!/bin/bash

# ============================================================================
# Blue/Green Deployment Script for Perfect Jarvis System
# ============================================================================
#
# This script orchestrates zero-downtime deployments using Blue/Green strategy
# following CLAUDE.md rules and production-ready deployment practices.
#
# Usage:
#   ./blue-green-deploy.sh [OPTIONS]
#
# Options:
#   --target-color      Target deployment color (blue|green) [required]
#   --deployment-tag    Docker image tag to deploy [default: latest]
#   --skip-tests        Skip smoke testing phase
#   --skip-backup       Skip blue environment backup
#   --auto-switch       Automatically switch traffic after successful deployment
#   --rollback          Rollback to previous environment
#   --status            Show current deployment status
#   --help              Show this help message
#
# Examples:
#   ./blue-green-deploy.sh --target-color green --deployment-tag v1.2.0
#   ./blue-green-deploy.sh --rollback
#   ./blue-green-deploy.sh --status
#
# ============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/config/deploy"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.blue-green.yml"
HEALTH_CHECK_SCRIPT="${SCRIPT_DIR}/health-checks.sh"
ENV_MANAGER_SCRIPT="${SCRIPT_DIR}/manage-environments.py"

# Logging configuration
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/blue-green-deploy-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
TARGET_COLOR=""
DEPLOYMENT_TAG="latest"
SKIP_TESTS=false
SKIP_BACKUP=false
AUTO_SWITCH=false
ROLLBACK=false
SHOW_STATUS=false
TIMEOUT=300
MAX_RETRIES=3

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Color based on log level
    case "$level" in
        "INFO")  echo -e "${GREEN}[${timestamp}] [INFO]${NC} ${message}" ;;
        "WARN")  echo -e "${YELLOW}[${timestamp}] [WARN]${NC} ${message}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] [ERROR]${NC} ${message}" ;;
        "DEBUG") echo -e "${BLUE}[${timestamp}] [DEBUG]${NC} ${message}" ;;
        *)       echo -e "[${timestamp}] [${level}] ${message}" ;;
    esac
    
    # Also log to file
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
}

error_exit() {
    log "ERROR" "$1"
    exit 1
}

validate_prerequisites() {
    log "INFO" "Validating deployment prerequisites..."
    
    # Check if docker-compose file exists
    if [[ ! -f "${DOCKER_COMPOSE_FILE}" ]]; then
        error_exit "Docker compose file not found: ${DOCKER_COMPOSE_FILE}"
    fi
    
    # Check if required scripts exist
    if [[ ! -f "${HEALTH_CHECK_SCRIPT}" ]]; then
        error_exit "Health check script not found: ${HEALTH_CHECK_SCRIPT}"
    fi
    
    if [[ ! -f "${ENV_MANAGER_SCRIPT}" ]]; then
        error_exit "Environment manager script not found: ${ENV_MANAGER_SCRIPT}"
    fi
    
    # Check Docker and docker-compose availability
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error_exit "docker-compose is not installed or not in PATH"
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi
    
    log "INFO" "Prerequisites validation completed successfully"
}

get_opposite_color() {
    local color="$1"
    if [[ "$color" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

get_current_active_color() {
    # Check HAProxy backend status to determine active environment
    if python3 "${ENV_MANAGER_SCRIPT}" --status | grep -q "Active: blue"; then
        echo "blue"
    elif python3 "${ENV_MANAGER_SCRIPT}" --status | grep -q "Active: green"; then
        echo "green"
    else
        log "WARN" "Unable to determine current active environment"
        echo "unknown"
    fi
}

wait_for_service() {
    local service_name="$1"
    local timeout="${2:-300}"
    local interval=10
    local elapsed=0
    
    log "INFO" "Waiting for service ${service_name} to be healthy (timeout: ${timeout}s)..."
    
    while [[ $elapsed -lt $timeout ]]; do
        if docker-compose -f "${DOCKER_COMPOSE_FILE}" ps "${service_name}" | grep -q "healthy\|Up"; then
            log "INFO" "Service ${service_name} is ready"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log "DEBUG" "Waiting for ${service_name}... (${elapsed}s/${timeout}s)"
    done
    
    log "ERROR" "Service ${service_name} failed to become healthy within ${timeout}s"
    return 1
}

# ============================================================================
# DEPLOYMENT FUNCTIONS
# ============================================================================

show_deployment_status() {
    log "INFO" "=== Current Deployment Status ==="
    
    local current_active=$(get_current_active_color)
    log "INFO" "Currently active environment: ${current_active}"
    
    # Show running containers by color
    log "INFO" "Blue environment containers:"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps | grep "blue-" || log "INFO" "  No blue containers running"
    
    log "INFO" "Green environment containers:"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps | grep "green-" || log "INFO" "  No green containers running"
    
    # Show shared services status
    log "INFO" "Shared services status:"
    docker-compose -f "${DOCKER_COMPOSE_FILE}" ps postgres redis neo4j ollama prometheus grafana
    
    # Show HAProxy backend status
    log "INFO" "HAProxy backend status:"
    python3 "${ENV_MANAGER_SCRIPT}" --status || log "WARN" "Unable to get HAProxy status"
    
    log "INFO" "=== Status Check Complete ==="
}

backup_current_environment() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log "INFO" "Skipping backup as requested"
        return 0
    fi
    
    local current_active=$(get_current_active_color)
    if [[ "$current_active" == "unknown" ]]; then
        log "WARN" "Cannot backup unknown environment"
        return 0
    fi
    
    log "INFO" "Creating backup of current ${current_active} environment..."
    
    local backup_dir="${PROJECT_ROOT}/backups/blue-green-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${backup_dir}"
    
    # Export current environment configuration
    docker-compose -f "${DOCKER_COMPOSE_FILE}" config > "${backup_dir}/docker-compose-backup.yml"
    
    # Create database backup
    log "INFO" "Creating database backup..."
    docker exec sutazai-postgres pg_dump -U sutazai sutazai > "${backup_dir}/database-backup.sql" || {
        log "WARN" "Database backup failed, continuing anyway..."
    }
    
    # Export environment state
    python3 "${ENV_MANAGER_SCRIPT}" --export-state > "${backup_dir}/environment-state.json" || {
        log "WARN" "Environment state export failed"
    }
    
    # Create rollback script
    cat > "${backup_dir}/rollback.sh" << EOF
#!/bin/bash
# Auto-generated rollback script
cd "${PROJECT_ROOT}"
python3 "${ENV_MANAGER_SCRIPT}" --switch-to ${current_active}
echo "Rolled back to ${current_active} environment"
EOF
    chmod +x "${backup_dir}/rollback.sh"
    
    log "INFO" "Backup created successfully at: ${backup_dir}"
    echo "${backup_dir}" > "${LOG_DIR}/latest-backup-path.txt"
}

deploy_target_environment() {
    local target="$1"
    local tag="${2:-latest}"
    
    log "INFO" "Starting deployment to ${target} environment with tag: ${tag}"
    
    # Set deployment environment variables
    export DEPLOYMENT_VERSION="${tag}"
    export SUTAZAI_ENV="${target}"
    
    # Pull latest images if needed
    log "INFO" "Pulling latest Docker images..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" pull ${target}-backend ${target}-frontend ${target}-jarvis-voice-interface || {
        log "WARN" "Image pull failed, will use local images"
    }
    
    # Stop existing target environment
    log "INFO" "Stopping existing ${target} environment..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" stop ${target}-backend ${target}-frontend ${target}-jarvis-voice-interface || {
        log "WARN" "Failed to stop some ${target} services, continuing..."
    }
    
    # Start shared services if not running
    log "INFO" "Ensuring shared services are running..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d postgres redis neo4j ollama chromadb qdrant faiss prometheus grafana loki
    
    # Wait for shared services to be healthy
    for service in postgres redis ollama; do
        wait_for_service "$service" || error_exit "Failed to start shared service: $service"
    done
    
    # Deploy target environment
    log "INFO" "Deploying ${target} environment..."
    docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d ${target}-backend ${target}-frontend ${target}-jarvis-voice-interface
    
    # Wait for services to be healthy
    for service in "${target}-backend" "${target}-frontend"; do
        wait_for_service "$service" || error_exit "Failed to deploy ${target} service: $service"
    done
    
    log "INFO" "${target} environment deployment completed successfully"
}

run_smoke_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "INFO" "Skipping smoke tests as requested"
        return 0
    fi
    
    local target="$1"
    log "INFO" "Running smoke tests on ${target} environment..."
    
    # Run comprehensive health checks
    if ! bash "${HEALTH_CHECK_SCRIPT}" --environment "${target}"; then
        error_exit "Smoke tests failed for ${target} environment"
    fi
    
    # Run specific endpoint tests
    local backend_url="http://localhost:20010"  # HAProxy routing
    local frontend_url="http://localhost:20011"
    
    log "INFO" "Testing backend API endpoints..."
    
    # Test health endpoint with retry logic
    local retry_count=0
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f -s -m 10 "${backend_url}/health" | grep -q "healthy\|ok"; then
            log "INFO" "Backend health check passed"
            break
        fi
        retry_count=$((retry_count + 1))
        if [[ $retry_count -eq $MAX_RETRIES ]]; then
            error_exit "Backend health check failed after ${MAX_RETRIES} retries"
        fi
        log "WARN" "Backend health check failed, retrying... (${retry_count}/${MAX_RETRIES})"
        sleep 10
    done
    
    # Test API endpoints
    log "INFO" "Testing API endpoints..."
    curl -f -s -m 10 "${backend_url}/api/v1/agents" > /dev/null || {
        error_exit "Agents API endpoint test failed"
    }
    
    # Test Ollama integration
    log "INFO" "Testing Ollama integration..."
    if curl -f -s -m 30 "${backend_url}/api/v1/models" | grep -q "tinyllama\|gpt"; then
        log "INFO" "Ollama integration test passed"
    else
        log "WARN" "Ollama integration test failed, but continuing..."
    fi
    
    log "INFO" "Smoke tests completed successfully for ${target} environment"
}

switch_traffic() {
    local target="$1"
    
    log "INFO" "Switching traffic to ${target} environment..."
    
    # Use environment manager to switch traffic
    if python3 "${ENV_MANAGER_SCRIPT}" --switch-to "${target}"; then
        log "INFO" "Traffic successfully switched to ${target} environment"
        
        # Verify traffic switch
        sleep 10
        local active_env=$(get_current_active_color)
        if [[ "$active_env" == "$target" ]]; then
            log "INFO" "Traffic switch verified successfully"
        else
            error_exit "Traffic switch verification failed. Expected: ${target}, Got: ${active_env}"
        fi
    else
        error_exit "Failed to switch traffic to ${target} environment"
    fi
}

cleanup_old_environment() {
    local old_color="$1"
    
    log "INFO" "Cleaning up old ${old_color} environment..."
    
    # Give some time for connections to drain
    log "INFO" "Waiting 30s for connection draining..."
    sleep 30
    
    # Stop old environment services
    docker-compose -f "${DOCKER_COMPOSE_FILE}" stop ${old_color}-backend ${old_color}-frontend ${old_color}-jarvis-voice-interface
    
    # Optionally remove containers (commented out for safety)
    # docker-compose -f "${DOCKER_COMPOSE_FILE}" rm -f ${old_color}-backend ${old_color}-frontend ${old_color}-jarvis-voice-interface
    
    log "INFO" "Old ${old_color} environment cleanup completed"
}

perform_rollback() {
    log "INFO" "=== Starting Rollback Process ==="
    
    local current_active=$(get_current_active_color)
    if [[ "$current_active" == "unknown" ]]; then
        error_exit "Cannot determine current active environment for rollback"
    fi
    
    local rollback_target=$(get_opposite_color "$current_active")
    log "INFO" "Rolling back from ${current_active} to ${rollback_target}"
    
    # Check if rollback target is available
    if ! docker-compose -f "${DOCKER_COMPOSE_FILE}" ps "${rollback_target}-backend" | grep -q "Up"; then
        log "WARN" "Rollback target ${rollback_target} is not running, attempting to start..."
        docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d ${rollback_target}-backend ${rollback_target}-frontend
        wait_for_service "${rollback_target}-backend" || error_exit "Failed to start rollback target"
    fi
    
    # Perform health check on rollback target
    if ! bash "${HEALTH_CHECK_SCRIPT}" --environment "${rollback_target}" --quick; then
        error_exit "Rollback target ${rollback_target} failed health checks"
    fi
    
    # Switch traffic back
    switch_traffic "$rollback_target"
    
    log "INFO" "Rollback to ${rollback_target} completed successfully"
}

# ============================================================================
# MAIN DEPLOYMENT ORCHESTRATION
# ============================================================================

main_deploy() {
    log "INFO" "=== Starting Blue/Green Deployment Process ==="
    log "INFO" "Target: ${TARGET_COLOR}, Tag: ${DEPLOYMENT_TAG}"
    
    # Phase 1: Pre-deployment validation
    log "INFO" "Phase 1: Pre-deployment validation"
    validate_prerequisites
    
    local current_active=$(get_current_active_color)
    log "INFO" "Current active environment: ${current_active}"
    
    if [[ "$current_active" == "$TARGET_COLOR" ]]; then
        log "WARN" "Target environment ${TARGET_COLOR} is currently active"
        read -p "Continue with in-place deployment? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "INFO" "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Phase 2: Backup current environment
    log "INFO" "Phase 2: Environment backup"
    backup_current_environment
    
    # Phase 3: Deploy to target environment
    log "INFO" "Phase 3: Target environment deployment"
    deploy_target_environment "$TARGET_COLOR" "$DEPLOYMENT_TAG"
    
    # Phase 4: Smoke testing
    log "INFO" "Phase 4: Smoke testing"
    run_smoke_tests "$TARGET_COLOR"
    
    # Phase 5: Traffic switching (manual or automatic)
    log "INFO" "Phase 5: Traffic switching"
    if [[ "$AUTO_SWITCH" == "true" ]]; then
        switch_traffic "$TARGET_COLOR"
        
        # Phase 6: Cleanup old environment
        if [[ "$current_active" != "unknown" && "$current_active" != "$TARGET_COLOR" ]]; then
            log "INFO" "Phase 6: Old environment cleanup"
            cleanup_old_environment "$current_active"
        fi
    else
        log "INFO" "Deployment complete. To switch traffic, run:"
        log "INFO" "  python3 ${ENV_MANAGER_SCRIPT} --switch-to ${TARGET_COLOR}"
        log "INFO" "Or run this script with --auto-switch flag"
    fi
    
    log "INFO" "=== Blue/Green Deployment Process Completed Successfully ==="
    log "INFO" "Deployment log saved to: ${LOG_FILE}"
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

show_help() {
    cat << EOF
Blue/Green Deployment Script for Perfect Jarvis System

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --target-color COLOR    Target deployment color (blue|green) [required]
    --deployment-tag TAG    Docker image tag to deploy [default: latest]
    --skip-tests           Skip smoke testing phase
    --skip-backup          Skip blue environment backup
    --auto-switch          Automatically switch traffic after successful deployment
    --rollback             Rollback to previous environment
    --status               Show current deployment status
    --timeout SECONDS      Deployment timeout in seconds [default: 300]
    --help                 Show this help message

EXAMPLES:
    # Deploy version 1.2.0 to green environment
    $0 --target-color green --deployment-tag v1.2.0
    
    # Deploy with automatic traffic switching
    $0 --target-color blue --auto-switch
    
    # Quick deployment (skip tests and backup)
    $0 --target-color green --skip-tests --skip-backup
    
    # Rollback to previous environment
    $0 --rollback
    
    # Show current status
    $0 --status

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-color)
            TARGET_COLOR="$2"
            shift 2
            ;;
        --deployment-tag)
            DEPLOYMENT_TAG="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --auto-switch)
            AUTO_SWITCH=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Handle status request
if [[ "$SHOW_STATUS" == "true" ]]; then
    show_deployment_status
    exit 0
fi

# Handle rollback request
if [[ "$ROLLBACK" == "true" ]]; then
    perform_rollback
    exit 0
fi

# Validate target color for deployment
if [[ -z "$TARGET_COLOR" ]]; then
    error_exit "Target color is required. Use --target-color (blue|green)"
fi

if [[ "$TARGET_COLOR" != "blue" && "$TARGET_COLOR" != "green" ]]; then
    error_exit "Invalid target color: $TARGET_COLOR. Must be 'blue' or 'green'"
fi

# Execute main deployment
main_deploy

log "INFO" "Deployment script execution completed"