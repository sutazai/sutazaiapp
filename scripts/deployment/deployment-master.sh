#!/bin/bash
# SutazAI Master Deployment Script
# Consolidates 47+ deployment script variations into one parameterized script
# Author: DevOps Manager - Deduplication Operation  
# Date: August 10, 2025

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Default values
DEPLOYMENT_TYPE="minimal"
ENVIRONMENT="development" 
BUILD_IMAGES=false
FORCE_REBUILD=false
SKIP_HEALTH_CHECK=false
PARALLEL_BUILDS=4
TIMEOUT=300

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Usage information
usage() {
    cat << EOF
SutazAI Master Deployment Script - Consolidated Deployment System

USAGE:
    $0 [OPTIONS] <deployment-type>

DEPLOYMENT TYPES:
    minimal        Deploy minimal stack (8 containers) - RECOMMENDED
    standard       Deploy standard stack (14+ containers)
    full           Deploy full system (50+ containers) 
    agents-only    Deploy only agent services
    infrastructure Deploy only core infrastructure (DB, monitoring)
    security       Deploy with full security hardening
    
ENVIRONMENT OPTIONS:
    development    Local development (default)
    staging        Staging environment
    production     Production deployment

OPTIONS:
    -b, --build                Build images before deployment
    -f, --force-rebuild        Force rebuild of all images
    -e, --environment ENV      Set deployment environment
    -j, --parallel N           Parallel build jobs (default: 4)
    -t, --timeout N            Health check timeout (default: 300s)
    --skip-health             Skip health check validation
    --dry-run                 Show what would be deployed without executing
    -h, --help                Show this help message

EXAMPLES:
    $0 minimal                           # Quick minimal deployment
    $0 --build --environment staging standard  # Build and deploy staging
    $0 --force-rebuild production        # Full production rebuild
    $0 agents-only                       # Deploy only AI agents

ENVIRONMENT VARIABLES:
    COMPOSE_FILE              Override docker-compose file
    SUTAZAI_ENV              Environment setting
    DOCKER_BUILDKIT          Enable BuildKit (recommended: 1)

LOG FILE: $LOG_FILE
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--build)
                BUILD_IMAGES=true
                shift
                ;;
            -f|--force-rebuild)
                BUILD_IMAGES=true
                FORCE_REBUILD=true
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -j|--parallel)
                PARALLEL_BUILDS="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2" 
                shift 2
                ;;
            --skip-health)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [[ -z "${DEPLOYMENT_TYPE:-}" ]]; then
                    DEPLOYMENT_TYPE="$1"
                else
                    log_error "Multiple deployment types specified"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Set default deployment type if not specified
    if [[ -z "${DEPLOYMENT_TYPE:-}" ]]; then
        DEPLOYMENT_TYPE="minimal"
    fi
}

# Validate environment and dependencies
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $available_space -lt 10 ]]; then
        log "Warning: Low disk space (${available_space}GB available)"
    fi
}

# Set up environment files based on deployment type
setup_environment() {
    log "Setting up environment for: $ENVIRONMENT"
    
    local env_file="${PROJECT_ROOT}/.env"
    local env_template
    
    case "$ENVIRONMENT" in
        development)
            env_template="${PROJECT_ROOT}/.env.development"
            ;;
        staging)
            env_template="${PROJECT_ROOT}/.env.staging"
            ;;
        production)
            env_template="${PROJECT_ROOT}/.env.production.secure"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    if [[ -f "$env_template" ]]; then
        cp "$env_template" "$env_file"
        log "Environment configured: $env_template -> $env_file"
    else
        log "Warning: Environment template not found: $env_template"
    fi
}

# Build Docker images with optimization
build_images() {
    if [[ "$BUILD_IMAGES" == "false" ]]; then
        return 0
    fi
    
    log "Building Docker images (parallel: $PARALLEL_BUILDS)..."
    
    # Enable BuildKit for better performance
    export DOCKER_BUILDKIT=1
    export BUILDKIT_PROGRESS=plain
    
    # Build base images first
    log "Building base images..."
    docker build -t sutazai-python-agent-master:latest \
        -f "${PROJECT_ROOT}/docker/base/Dockerfile.python-agent-master" \
        "${PROJECT_ROOT}/docker/base/"
    
    docker build -t sutazai-nodejs-agent-master:latest \
        -f "${PROJECT_ROOT}/docker/base/Dockerfile.nodejs-agent-master" \
        "${PROJECT_ROOT}/docker/base/"
    
    # Generate service Dockerfiles from templates
    log "Generating service Dockerfiles from templates..."
    cd "${PROJECT_ROOT}/docker/templates"
    python3 generate-dockerfile.py --all --output-dir ../generated
    
    # Build service images based on deployment type
    case "$DEPLOYMENT_TYPE" in
        minimal|standard)
            # Build core services
            docker-compose build --parallel backend frontend ollama
            ;;
        full)
            # Build all services
            if [[ "$FORCE_REBUILD" == "true" ]]; then
                docker-compose build --no-cache --parallel
            else
                docker-compose build --parallel
            fi
            ;;
        agents-only)
            # Build only agent services
            docker-compose build --parallel \
                hardware-resource-optimizer \
                jarvis-automation-agent \
                ai-agent-orchestrator
            ;;
    esac
}

# Deploy services based on type
deploy_services() {
    log "Deploying: $DEPLOYMENT_TYPE"
    
    local compose_files=()
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    # Select appropriate compose files
    case "$DEPLOYMENT_TYPE" in
        minimal)
            compose_files+=("-f" "${PROJECT_ROOT}/docker-compose.minimal.yml")
            ;;
        security)
            compose_files+=("-f" "${PROJECT_ROOT}/docker-compose.yml")
            compose_files+=("-f" "${PROJECT_ROOT}/docker-compose.security.yml")
            ;;
        production)
            compose_files+=("-f" "${PROJECT_ROOT}/docker-compose.yml")
            compose_files+=("-f" "${PROJECT_ROOT}/docker-compose.security.yml")
            ;;
        *)
            compose_files+=("-f" "$compose_file")
            ;;
    esac
    
    # Execute deployment
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log "DRY RUN - Would execute:"
        echo "docker-compose ${compose_files[*]} up -d"
        return 0
    fi
    
    log "Starting services..."
    docker-compose "${compose_files[@]}" up -d
    
    # Wait for services to be ready
    if [[ "$SKIP_HEALTH_CHECK" == "false" ]]; then
        wait_for_services
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to be healthy (timeout: ${TIMEOUT}s)..."
    
    local start_time=$(date +%s)
    local healthy_count=0
    local total_services=0
    
    # Get list of services with health checks
    local services=($(docker-compose ps --services))
    total_services=${#services[@]}
    
    while [[ $(($(date +%s) - start_time)) -lt $TIMEOUT ]]; do
        healthy_count=0
        
        for service in "${services[@]}"; do
            local health=$(docker-compose ps -q "$service" | xargs -I {} docker inspect {} --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
            
            if [[ "$health" == "healthy" || "$health" == "unknown" ]]; then
                ((healthy_count++))
            fi
        done
        
        log "Health check: $healthy_count/$total_services services ready"
        
        if [[ $healthy_count -eq $total_services ]]; then
            log "All services are healthy!"
            return 0
        fi
        
        sleep 10
    done
    
    log_error "Health check timeout after ${TIMEOUT}s"
    log "Service status:"
    docker-compose ps
    return 1
}

# Main execution flow
main() {
    log "Starting SutazAI Master Deployment"
    log "Deployment type: $DEPLOYMENT_TYPE"
    log "Environment: $ENVIRONMENT"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Parse arguments and validate
    parse_args "$@"
    validate_environment
    setup_environment
    
    # Execute deployment steps
    cd "$PROJECT_ROOT"
    
    if [[ "$BUILD_IMAGES" == "true" ]]; then
        build_images
    fi
    
    deploy_services
    
    # Show final status
    log "Deployment completed successfully!"
    docker-compose ps
    
    log "Access points:"
    log "  - Backend API: http://localhost:10010"
    log "  - Frontend UI: http://localhost:10011" 
    log "  - Grafana: http://localhost:10201 (admin/admin)"
    log "  - Hardware Optimizer: http://localhost:11110"
}

# Execute main function with all arguments
main "$@"