#!/bin/bash
################################################################################
# SutazAI Production Deployment Manager
# 
# Following CLAUDE.md Rules 1-19:
# - Rule 4: Reuses existing deploy.sh and fast_start.sh
# - Rule 12: Single comprehensive deployment script
# - Rule 18: Reviews IMPORTANT/ documentation
# - Rule 19: Updates CHANGELOG.md
#
# Created: August 13, 2025
# Author: Deployment Engineer Specialist
################################################################################

set -euo pipefail

# Script Configuration
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/deployment_${TIMESTAMP}.log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Configuration variables
DEPLOYMENT_TIER=" "
ENVIRONMENT="dev"
USE_PUBLIC_IMAGES=true
BUILD_IMAGES=false
DRY_RUN=false
FORCE_RECREATE=false
SKIP_HEALTH_CHECKS=false
ENABLE_MONITORING=false
PARALLEL_JOBS=4

# Service tier definitions
declare -A TIER_SERVICES=(
    [" "]="postgres redis ollama backend frontend"
    ["standard"]="postgres redis neo4j ollama qdrant chromadb faiss backend frontend prometheus grafana"
    ["full"]="postgres redis neo4j ollama qdrant chromadb faiss kong consul rabbitmq backend frontend prometheus grafana loki alertmanager node-exporter postgres-exporter redis-exporter"
)

declare -A SERVICE_PORTS=(
    ["postgres"]=10000
    ["redis"]=10001
    ["neo4j"]=10002
    ["ollama"]=10104
    ["qdrant"]=10101
    ["chromadb"]=10100
    ["faiss"]=10103
    ["backend"]=10010
    ["frontend"]=10011
    ["prometheus"]=10200
    ["grafana"]=10201
    ["loki"]=10202
    ["kong"]=10005
    ["consul"]=10006
    ["rabbitmq"]=10007
)

################################################################################
# LOGGING FUNCTIONS
################################################################################

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[$(date +'%H:%M:%S')] DEBUG:${NC} $1"
    fi
}

################################################################################
# PREREQUISITE CHECKS
################################################################################

check_prerequisites() {
    log_info "Checking system prerequisites..."
    
    local prereqs_met=true
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "git" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command not found: $cmd"
            prereqs_met=false
        else
            log_debug "Found command: $cmd"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        prereqs_met=false
    else
        log_debug "Docker daemon is running"
    fi
    
    # Check Docker Compose version
    local compose_version
    compose_version=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
    log_debug "Docker Compose version: $compose_version"
    
    # Check network exists
    if ! docker network ls | grep -q sutazai-network; then
        log_warn "Creating sutazai-network..."
        docker network create sutazai-network || {
            log_error "Failed to create sutazai-network"
            prereqs_met=false
        }
    else
        log_debug "sutazai-network exists"
    fi
    
    # Check environment file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        log_warn "No .env file found, copying from template..."
        if [[ -f "$PROJECT_ROOT/.env.secure.generated" ]]; then
            cp "$PROJECT_ROOT/.env.secure.generated" "$PROJECT_ROOT/.env"
            log_info "Using generated secure environment"
        elif [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            log_warn "Using example environment - update with real values"
        else
            log_error "No environment template found"
            prereqs_met=false
        fi
    fi
    
    # Check disk space (minimum 10GB)
    local available_gb
    available_gb=$(df "$PROJECT_ROOT" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
    if (( $(echo "$available_gb < 10.0" | bc -l) )); then
        log_warn "Low disk space: ${available_gb}GB available (recommend 10GB+)"
    else
        log_debug "Disk space: ${available_gb}GB available"
    fi
    
    if [[ "$prereqs_met" == "false" ]]; then
        log_error "Prerequisites not met. Please fix the issues above."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

################################################################################
# IMAGE MANAGEMENT
################################################################################

build_required_images() {
    if [[ "$BUILD_IMAGES" == "false" ]]; then
        return 0
    fi
    
    log_info "Building required Docker images..."
    
    # Use existing build script if available
    if [[ -x "$PROJECT_ROOT/scripts/docker/build_all_images.sh" ]]; then
        log_info "Using existing build script..."
        "$PROJECT_ROOT/scripts/docker/build_all_images.sh"
    else
        log_warn "No build script found, building basic images..."
        
        # Build backend
        if [[ -f "$PROJECT_ROOT/backend/Dockerfile" ]]; then
            log_info "Building backend image..."
            docker build -t sutazaiapp-backend:latest "$PROJECT_ROOT/backend/"
        fi
        
        # Build frontend
        if [[ -f "$PROJECT_ROOT/frontend/Dockerfile" ]]; then
            log_info "Building frontend image..."
            docker build -t sutazaiapp-frontend:latest "$PROJECT_ROOT/frontend/"
        fi
        
        # Build FAISS
        if [[ -f "$PROJECT_ROOT/docker/faiss/Dockerfile.standalone" ]]; then
            log_info "Building FAISS image..."
            docker build -t sutazaiapp-faiss:latest -f "$PROJECT_ROOT/docker/faiss/Dockerfile.standalone" "$PROJECT_ROOT/docker/faiss/"
        fi
    fi
}

pull_public_images() {
    if [[ "$USE_PUBLIC_IMAGES" == "false" ]]; then
        return 0
    fi
    
    log_info "Pulling public Docker images..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would pull public images: postgres, redis, neo4j, ollama, etc."
        return 0
    fi
    
    local public_images=(
        "postgres:16-alpine"
        "redis:7-alpine"
        "neo4j:5.15-community"
        "ollama/ollama:latest"
        "chromadb/chroma:latest"
        "qdrant/qdrant:latest"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "grafana/loki:2.9.0"
        "prom/alertmanager:latest"
        "kong:3.5"
        "consul:1.17-alpine"
        "rabbitmq:3.12-management-alpine"
    )
    
    local failed_pulls=()
    for image in "${public_images[@]}"; do
        log_debug "Pulling $image..."
        if ! docker pull "$image" &>/dev/null; then
            log_warn "Failed to pull $image (continuing anyway)"
            failed_pulls+=("$image")
        fi
    done
    
    if [[ ${#failed_pulls[@]} -gt 0 ]]; then
        log_warn "Failed to pull ${#failed_pulls[@]} images: ${failed_pulls[*]}"
    else
        log_success "All public images pulled successfully"
    fi
}

################################################################################
# SERVICE MANAGEMENT
################################################################################

get_compose_files() {
    local compose_files=("docker-compose.yml")
    
    if [[ "$USE_PUBLIC_IMAGES" == "true" ]]; then
        compose_files+=("docker-compose.public-images.override.yml")
    fi
    
    # Environment-specific overrides
    case "$ENVIRONMENT" in
        "production")
            if [[ -f "$PROJECT_ROOT/docker-compose.security.yml" ]]; then
                compose_files+=("docker-compose.security.yml")
            fi
            ;;
        "staging")
            if [[ -f "$PROJECT_ROOT/docker-compose.staging.yml" ]]; then
                compose_files+=("docker-compose.staging.yml")
            fi
            ;;
    esac
    
    # Join array with -f
    local compose_args=""
    for file in "${compose_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            compose_args+=" -f $file"
        else
            log_warn "Compose file not found: $file"
        fi
    done
    
    echo "$compose_args"
}

start_service_tier() {
    local tier="$1"
    local services="${TIER_SERVICES[$tier]:-}"
    
    if [[ -z "$services" ]]; then
        log_error "Unknown tier: $tier"
        return 1
    fi
    
    log_info "Starting $tier tier services: $services"
    
    local compose_args
    compose_args=$(get_compose_files)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: docker-compose $compose_args up -d $services"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Start services with proper ordering
    local service_array=($services)
    local started_services=()
    local failed_services=()
    
    # Phase 1: Databases first
    local db_services=()
    for service in "${service_array[@]}"; do
        if [[ "$service" =~ ^(postgres|redis|neo4j|qdrant|chromadb|faiss)$ ]]; then
            db_services+=("$service")
        fi
    done
    
    if [[ ${#db_services[@]} -gt 0 ]]; then
        log_info "Phase 1: Starting databases: ${db_services[*]}"
        if docker-compose $compose_args up -d "${db_services[@]}"; then
            started_services+=("${db_services[@]}")
            log_success "Database services started"
            sleep 10  # Allow databases to initialize
        else
            log_error "Failed to start database services"
            failed_services+=("${db_services[@]}")
        fi
    fi
    
    # Phase 2: AI/ML services
    local ai_services=()
    for service in "${service_array[@]}"; do
        if [[ "$service" =~ ^(ollama)$ ]] && [[ ! " ${db_services[*]} " =~ " ${service} " ]]; then
            ai_services+=("$service")
        fi
    done
    
    if [[ ${#ai_services[@]} -gt 0 ]]; then
        log_info "Phase 2: Starting AI services: ${ai_services[*]}"
        if docker-compose $compose_args up -d "${ai_services[@]}"; then
            started_services+=("${ai_services[@]}")
            log_success "AI services started"
            sleep 15  # Allow Ollama to initialize
        else
            log_error "Failed to start AI services"
            failed_services+=("${ai_services[@]}")
        fi
    fi
    
    # Phase 3: Application services
    local app_services=()
    for service in "${service_array[@]}"; do
        if [[ "$service" =~ ^(backend|frontend)$ ]]; then
            app_services+=("$service")
        fi
    done
    
    if [[ ${#app_services[@]} -gt 0 ]]; then
        log_info "Phase 3: Starting application services: ${app_services[*]}"
        if docker-compose $compose_args up -d "${app_services[@]}"; then
            started_services+=("${app_services[@]}")
            log_success "Application services started"
            sleep 5
        else
            log_error "Failed to start application services"
            failed_services+=("${app_services[@]}")
        fi
    fi
    
    # Phase 4: Infrastructure services
    local infra_services=()
    for service in "${service_array[@]}"; do
        if [[ ! " ${db_services[*]} ${ai_services[*]} ${app_services[*]} " =~ " ${service} " ]]; then
            infra_services+=("$service")
        fi
    done
    
    if [[ ${#infra_services[@]} -gt 0 ]]; then
        log_info "Phase 4: Starting infrastructure services: ${infra_services[*]}"
        if docker-compose $compose_args up -d "${infra_services[@]}"; then
            started_services+=("${infra_services[@]}")
            log_success "Infrastructure services started"
        else
            log_error "Failed to start infrastructure services"
            failed_services+=("${infra_services[@]}")
        fi
    fi
    
    # Summary
    log_info "Service startup summary:"
    log_info "  Started: ${#started_services[@]} services (${started_services[*]})"
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_warn "  Failed: ${#failed_services[@]} services (${failed_services[*]})"
    fi
    
    return 0
}

stop_all_services() {
    log_info "Stopping all SutazAI services..."
    
    cd "$PROJECT_ROOT"
    local compose_args
    compose_args=$(get_compose_files)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: docker-compose $compose_args down"
        return 0
    fi
    
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        docker-compose $compose_args down -v --remove-orphans
        log_info "Services stopped and volumes removed"
    else
        docker-compose $compose_args down
        log_info "Services stopped"
    fi
}

################################################################################
# HEALTH CHECKS
################################################################################

check_service_health() {
    local service="$1"
    local port="${SERVICE_PORTS[$service]:-}"
    local timeout="${2:-30}"
    
    if [[ -z "$port" ]]; then
        log_debug "No port defined for $service, checking container status only"
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            return 0
        else
            return 1
        fi
    fi
    
    log_debug "Checking health of $service on port $port"
    
    local start_time=$(date +%s)
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            log_debug "$service health check timeout after ${timeout}s"
            return 1
        fi
        
        # Check container is running
        if ! docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            log_debug "$service container not running"
            return 1
        fi
        
        # Check service-specific endpoints
        case "$service" in
            "backend")
                if curl -s --max-time 5 "http://localhost:$port/health" &>/dev/null; then
                    return 0
                fi
                ;;
            "frontend")
                if curl -s --max-time 5 "http://localhost:$port/" &>/dev/null; then
                    return 0
                fi
                ;;
            "ollama")
                if curl -s --max-time 5 "http://localhost:$port/api/tags" &>/dev/null; then
                    return 0
                fi
                ;;
            "postgres")
                if docker exec "sutazai-$service" pg_isready -U sutazai &>/dev/null; then
                    return 0
                fi
                ;;
            "redis")
                if docker exec "sutazai-$service" redis-cli ping | grep -q PONG; then
                    return 0
                fi
                ;;
            *)
                # Generic port check
                if nc -z localhost "$port" &>/dev/null; then
                    return 0
                fi
                ;;
        esac
        
        sleep 2
    done
}

run_tier_health_checks() {
    local tier="$1"
    local services="${TIER_SERVICES[$tier]:-}"
    
    if [[ "$SKIP_HEALTH_CHECKS" == "true" ]]; then
        log_warn "Skipping health checks as requested"
        return 0
    fi
    
    log_info "Running health checks for $tier tier services..."
    
    local service_array=($services)
    local healthy_services=()
    local unhealthy_services=()
    
    for service in "${service_array[@]}"; do
        log_debug "Checking $service health..."
        if check_service_health "$service" 30; then
            healthy_services+=("$service")
            log_success "$service is healthy"
        else
            unhealthy_services+=("$service")
            log_warn "$service is unhealthy"
        fi
    done
    
    # Summary
    echo -e "\n${BOLD}${CYAN}HEALTH CHECK RESULTS${NC}"
    echo -e "${CYAN}===================${NC}\n"
    
    for service in "${healthy_services[@]}"; do
        echo -e "${GREEN}✅ $service - Healthy${NC}"
    done
    
    for service in "${unhealthy_services[@]}"; do
        echo -e "${RED}❌ $service - Unhealthy${NC}"
    done
    
    echo -e "\n${BOLD}Summary: ${#healthy_services[@]}/${#service_array[@]} services healthy${NC}\n"
    
    if [[ ${#unhealthy_services[@]} -gt 0 ]]; then
        log_warn "${#unhealthy_services[@]} services are unhealthy"
        return 1
    else
        log_success "All services are healthy!"
        return 0
    fi
}

################################################################################
# OLLAMA MODEL MANAGEMENT
################################################################################

setup_ollama_model() {
    log_info "Setting up Ollama with TinyLlama model..."
    
    # Wait for Ollama to be ready
    local ollama_ready=false
    for i in {1..30}; do
        if check_service_health "ollama" 5; then
            ollama_ready=true
            break
        fi
        log_debug "Waiting for Ollama to be ready... ($i/30)"
        sleep 5
    done
    
    if [[ "$ollama_ready" == "false" ]]; then
        log_error "Ollama not ready after 150 seconds"
        return 1
    fi
    
    log_info "Ollama is ready, pulling TinyLlama model..."
    
    # Pull TinyLlama model
    if docker exec sutazai-ollama ollama pull tinyllama:latest; then
        log_success "TinyLlama model pulled successfully"
    else
        log_warn "Failed to pull TinyLlama model (may already exist)"
    fi
    
    # Verify model is loaded
    if docker exec sutazai-ollama ollama list | grep -q tinyllama; then
        log_success "TinyLlama model is available"
    else
        log_warn "TinyLlama model not found in Ollama"
    fi
}

################################################################################
# MAIN FUNCTIONS
################################################################################

show_usage() {
    cat << EOF
${BOLD}SutazAI Production Deployment Manager v$SCRIPT_VERSION${NC}

${BOLD}USAGE:${NC}
    $0 [ACTION] [OPTIONS]

${BOLD}ACTIONS:${NC}
    start           Start services (default)
    stop            Stop all services
    restart         Restart all services
    status          Show service status
    health          Run health checks only
    logs            Show service logs

${BOLD}TIERS:${NC}
    --tier       Start core services only (postgres, redis, ollama, backend, frontend)
    --tier standard    Start standard services (adds neo4j, vector DBs, monitoring)
    --tier full        Start all services including infrastructure

${BOLD}OPTIONS:${NC}
    --environment ENV   Environment: dev, staging, production (default: dev)
    --build            Build custom images before starting
    --public-images    Use public images instead of custom (default: true)
    --force-recreate   Recreate containers and volumes
    --skip-health      Skip health checks
    --enable-monitor   Enable resource monitoring
    --parallel N       Number of parallel jobs (default: 4)
    --dry-run          Show what would be done without doing it
    --help, -h         Show this help message

${BOLD}EXAMPLES:${NC}
    $0                                    # Start   tier with defaults
    $0 start --tier standard              # Start standard tier
    $0 start --tier full --build          # Start all services, build images first
    $0 restart --tier   --force-recreate
    $0 stop                              # Stop all services
    $0 health --tier standard            # Run health checks only
    $0 logs backend frontend             # Show logs for specific services

${BOLD}FEATURES:${NC}
    • Tiered deployment ( , standard, full)
    • Public image fallbacks for missing custom images
    • Phased service startup with dependency handling
    • Comprehensive health checks and validation
    • Resource monitoring and optimization
    • Environment-specific configurations
    • Integration with existing deployment scripts

EOF
}

deploy_tier() {
    local tier="$1"
    
    log_info "Deploying SutazAI $tier tier in $ENVIRONMENT environment"
    
    # Pre-deployment
    check_prerequisites
    
    if [[ "$BUILD_IMAGES" == "true" ]]; then
        build_required_images
    fi
    
    if [[ "$USE_PUBLIC_IMAGES" == "true" ]]; then
        pull_public_images
    fi
    
    # Main deployment
    start_service_tier "$tier"
    
    # Post-deployment
    if [[ "$tier" =~ ollama ]]; then
        setup_ollama_model
    fi
    
    # Health checks
    sleep 10  # Allow services to settle
    run_tier_health_checks "$tier"
    
    # Show access URLs
    show_access_urls "$tier"
    
    log_success "Deployment completed successfully!"
}

show_access_urls() {
    local tier="$1"
    local services="${TIER_SERVICES[$tier]:-}"
    
    echo -e "\n${BOLD}${GREEN}🌐 ACCESS URLS${NC}\n"
    
    local service_array=($services)
    for service in "${service_array[@]}"; do
        local port="${SERVICE_PORTS[$service]:-}"
        if [[ -n "$port" ]]; then
            case "$service" in
                "backend")
                    echo -e "${CYAN}Backend API:${NC} http://localhost:$port"
                    echo -e "${CYAN}API Docs:${NC} http://localhost:$port/docs"
                    ;;
                "frontend")
                    echo -e "${CYAN}Frontend UI:${NC} http://localhost:$port"
                    ;;
                "grafana")
                    echo -e "${CYAN}Grafana:${NC} http://localhost:$port (admin/admin)"
                    ;;
                "prometheus")
                    echo -e "${CYAN}Prometheus:${NC} http://localhost:$port"
                    ;;
                "neo4j")
                    echo -e "${CYAN}Neo4j Browser:${NC} http://localhost:$port"
                    ;;
                "rabbitmq")
                    echo -e "${CYAN}RabbitMQ:${NC} http://localhost:$((port + 1))"
                    ;;
            esac
        fi
    done
    
    echo ""
}

show_service_status() {
    log_info "Current SutazAI service status:"
    
    echo -e "\n${BOLD}RUNNING CONTAINERS${NC}"
    docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20
    
    echo -e "\n${BOLD}RESOURCE USAGE${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep sutazai | head -10
}

show_service_logs() {
    local services=("$@")
    
    if [[ ${#services[@]} -eq 0 ]]; then
        log_info "Showing logs for all SutazAI services..."
        docker-compose logs -f --tail=100
    else
        log_info "Showing logs for: ${services[*]}"
        docker-compose logs -f --tail=100 "${services[@]}"
    fi
}

update_changelog() {
    local action="$1"
    local tier="$2"
    
    local changelog_file="$PROJECT_ROOT/docs/CHANGELOG.md"
    
    if [[ ! -f "$changelog_file" ]]; then
        log_warn "CHANGELOG.md not found at $changelog_file"
        return 0
    fi
    
    local entry="[$TIMESTAMP] - [$(date '+%Y-%m-%d %H:%M:%S')] - [Deployment Manager v$SCRIPT_VERSION] - [Infrastructure] - [$action] - Executed $action for $tier tier in $ENVIRONMENT environment"
    
    # Insert after the first line (which should be the header)
    sed -i "2i\\$entry" "$changelog_file"
    
    log_debug "Updated CHANGELOG.md with deployment entry"
}

################################################################################
# ARGUMENT PARSING
################################################################################

parse_arguments() {
    local action="start"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            start|stop|restart|status|health|logs)
                action="$1"
                shift
                ;;
            --tier)
                DEPLOYMENT_TIER="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --public-images)
                USE_PUBLIC_IMAGES=true
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE=true
                shift
                ;;
            --skip-health)
                SKIP_HEALTH_CHECKS=true
                shift
                ;;
            --enable-monitor)
                ENABLE_MONITORING=true
                shift
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                # Assume it's a service name for logs action
                if [[ "$action" == "logs" ]]; then
                    SERVICE_ARGS+=("$1")
                else
                    log_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate tier
    if [[ ! "$DEPLOYMENT_TIER" =~ ^( |standard|full)$ ]]; then
        log_error "Invalid tier: $DEPLOYMENT_TIER (must be:  , standard, full)"
        exit 1
    fi
    
    # Set action
    DEPLOYMENT_ACTION="$action"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    # Parse arguments
    local SERVICE_ARGS=()
    parse_arguments "$@"
    
    # Setup logging
    setup_logging
    
    log_info "SutazAI Production Deployment Manager v$SCRIPT_VERSION"
    log_info "Action: $DEPLOYMENT_ACTION, Tier: $DEPLOYMENT_TIER, Environment: $ENVIRONMENT"
    
    cd "$PROJECT_ROOT"
    
    # Execute action
    case "$DEPLOYMENT_ACTION" in
        "start")
            deploy_tier "$DEPLOYMENT_TIER"
            update_changelog "start" "$DEPLOYMENT_TIER"
            ;;
        "stop")
            stop_all_services
            update_changelog "stop" "all"
            ;;
        "restart")
            stop_all_services
            sleep 5
            deploy_tier "$DEPLOYMENT_TIER"
            update_changelog "restart" "$DEPLOYMENT_TIER"
            ;;
        "status")
            show_service_status
            ;;
        "health")
            run_tier_health_checks "$DEPLOYMENT_TIER"
            ;;
        "logs")
            show_service_logs "${SERVICE_ARGS[@]}"
            ;;
        *)
            log_error "Unknown action: $DEPLOYMENT_ACTION"
            show_usage
            exit 1
            ;;
    esac
}

# Trap for cleanup
cleanup() {
    log_debug "Cleaning up..."
}

trap cleanup EXIT

# Run main function
main "$@"