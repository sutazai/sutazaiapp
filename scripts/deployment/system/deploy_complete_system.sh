#!/bin/bash
# SutazAI Multi-Agent Task Automation System Deployment
# Production-ready deployment script for AI-powered task automation
# Deploys PostgreSQL, Redis, backend services, Ollama, and essential AI agents

set -euo pipefail

# ===========================================
# CONFIGURATION
# ===========================================

PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/deployment_${TIMESTAMP}.log"

# Deployment state tracking
DEPLOYMENT_STATE_FILE="$LOG_DIR/deployment_state_${TIMESTAMP}.json"
ERROR_COUNT=0
WARNING_COUNT=0
DEPLOYMENT_ERRORS=()
SUCCESSFUL_SERVICES=()
FAILED_SERVICES=()

# Service definitions
CORE_SERVICES=("postgres" "redis" "ollama")
BACKEND_SERVICES=("backend-agi")
AGENT_SERVICES=("senior-ai-engineer" "deployment-automation-master" "infrastructure-devops-manager" "ollama-integration-specialist" "testing-qa-validator")
OPTIONAL_SERVICES=("chromadb" "neo4j")

# ===========================================
# LOGGING FUNCTIONS
# ===========================================

# Ensure log directory exists
mkdir -p "$LOG_DIR"

log_info() {
    echo "[$(date +'%H:%M:%S')] INFO: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date +'%H:%M:%S')] SUCCESS: $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[$(date +'%H:%M:%S')] WARNING: $1" | tee -a "$LOG_FILE"
    ((WARNING_COUNT++))
}

log_error() {
    echo "[$(date +'%H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
    ((ERROR_COUNT++))
    DEPLOYMENT_ERRORS+=("$1")
}

log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
}

# ===========================================
# SYSTEM VALIDATION
# ===========================================

check_prerequisites() {
    log_header "Checking System Prerequisites"
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_success "$cmd is available"
        else
            log_error "$cmd is not installed"
            return 1
        fi
    done
    
    # Check Docker daemon
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon is not running"
        return 1
    fi
    
    # Check system resources
    local cpu_cores=$(nproc)
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    local disk_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log_info "System Resources:"
    log_info "  CPU Cores: $cpu_cores"
    log_info "  Memory: ${memory_gb}GB"
    log_info "  Available Disk: ${disk_gb}GB"
    
    # Minimum requirements check
    if [ "$cpu_cores" -lt 4 ]; then
        log_warn "CPU cores ($cpu_cores) below recommended minimum (4)"
    fi
    
    if [ "$memory_gb" -lt 8 ]; then
        log_warn "Memory (${memory_gb}GB) below recommended minimum (8GB)"
    fi
    
    if [ "$disk_gb" -lt 20 ]; then
        log_error "Insufficient disk space (${disk_gb}GB). Need at least 20GB"
        return 1
    fi
    
    log_success "System prerequisites validated"
}

check_docker_files() {
    log_header "Validating Docker Configuration Files"
    
    local compose_files=(
        "$PROJECT_ROOT/docker-compose.yml"
        "$PROJECT_ROOT/docker-compose.agents.yml"
        "$PROJECT_ROOT/docker-compose.minimal.yml"
    )
    
    for file in "${compose_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "Found: $(basename "$file")"
            # Validate YAML syntax
            if docker-compose -f "$file" config &> /dev/null; then
                log_success "Valid YAML: $(basename "$file")"
            else
                log_error "Invalid YAML syntax: $(basename "$file")"
                return 1
            fi
        else
            log_warn "Missing: $(basename "$file")"
        fi
    done
    
    # Check environment file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        log_success "Environment file found"
    else
        log_info "Creating default environment file"
        create_default_env_file
    fi
}

create_default_env_file() {
    cat > "$PROJECT_ROOT/.env" << EOF
# SutazAI Configuration
SUTAZAI_ENV=production
TZ=UTC

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
POSTGRES_DB=sutazai

# Redis Configuration
REDIS_PASSWORD=redis_password

# Ollama Configuration
OLLAMA_MODELS=tinyllama
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*

# API Keys (replace with your own)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
EOF
    log_success "Created default .env file"
}

# ===========================================
# SERVICE MANAGEMENT
# ===========================================

wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    log_info "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "$service_name is ready"
            return 0
        fi
        sleep 2
        ((attempt++))
    done
    
    log_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

check_service_health() {
    local service_name=$1
    
    # Check if container is running
    if docker ps --filter "name=$service_name" --format "{{.Names}}" | grep -q "$service_name"; then
        log_success "$service_name container is running"
        
        # Check container health if healthcheck is defined
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "sutazai-$service_name" 2>/dev/null || echo "unknown")
        if [ "$health_status" = "healthy" ]; then
            log_success "$service_name is healthy"
            return 0
        elif [ "$health_status" = "unhealthy" ]; then
            log_error "$service_name is unhealthy"
            return 1
        else
            log_info "$service_name health status: $health_status"
            return 0
        fi
    else
        log_error "$service_name container is not running"
        return 1
    fi
}

deploy_service_group() {
    local group_name=$1
    local compose_file=$2
    shift 2
    local services=("$@")
    
    log_header "Deploying $group_name Services"
    
    for service in "${services[@]}"; do
        log_info "Starting $service..."
        
        if docker-compose -f "$compose_file" up -d "$service"; then
            log_success "$service started successfully"
            SUCCESSFUL_SERVICES+=("$service")
            
            # Wait for specific services to be ready
            case "$service" in
                postgres)
                    wait_for_service "$service" 5432
                    ;;
                redis)
                    wait_for_service "$service" 6379
                    ;;
                ollama)
                    wait_for_service "$service" 11434
                    ;;
                backend-agi)
                    wait_for_service "$service" 8000
                    ;;
            esac
            
            # Verify service health
            if ! check_service_health "$service"; then
                log_error "$service health check failed"
                FAILED_SERVICES+=("$service")
            fi
        else
            log_error "Failed to start $service"
            FAILED_SERVICES+=("$service")
        fi
        
        sleep 2
    done
}

# ===========================================
# OLLAMA MODEL MANAGEMENT
# ===========================================

setup_ollama_models() {
    log_header "Setting up Ollama Models"
    
    # Wait for Ollama to be fully ready
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            log_success "Ollama API is responding"
            break
        fi
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Ollama API not responding after 2 minutes"
        return 1
    fi
    
    # Pull essential models
    local models=("tinyllama")
    
    for model in "${models[@]}"; do
        log_info "Pulling model: $model"
        if docker exec sutazai-ollama ollama pull "$model"; then
            log_success "Successfully pulled $model"
        else
            log_error "Failed to pull $model"
        fi
    done
    
    # List available models
    log_info "Available models:"
    docker exec sutazai-ollama ollama list
}

# ===========================================
# DATABASE INITIALIZATION
# ===========================================

initialize_database() {
    log_header "Initializing Database"
    
    # Wait for PostgreSQL to be ready
    if ! wait_for_service "postgres" 5432 30; then
        log_error "PostgreSQL not ready for initialization"
        return 1
    fi
    
    # Check if database is accessible
    if docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;" &>/dev/null; then
        log_success "Database connection verified"
        
        # Run any initialization scripts
        if [ -f "$PROJECT_ROOT/config/postgres/init.sql" ]; then
            log_info "Running database initialization script"
            docker exec -i sutazai-postgres psql -U sutazai -d sutazai < "$PROJECT_ROOT/config/postgres/init.sql"
            log_success "Database initialization completed"
        fi
    else
        log_error "Cannot connect to database"
        return 1
    fi
}

# ===========================================
# MONITORING SETUP (OPTIONAL)
# ===========================================

setup_monitoring() {
    log_header "Setting up Monitoring (Optional)"
    
    # Only deploy monitoring if explicitly requested or in production
    if [ "${DEPLOY_MONITORING:-false}" = "true" ] || [ "${SUTAZAI_ENV:-}" = "production" ]; then
        log_info "Deploying monitoring services..."
        
        for service in "${OPTIONAL_SERVICES[@]}"; do
            case "$service" in
                chromadb|neo4j)
                    log_info "Starting $service..."
                    if docker-compose -f docker-compose.yml up -d "$service"; then
                        log_success "$service started"
                        SUCCESSFUL_SERVICES+=("$service")
                    else
                        log_warn "Failed to start $service (optional)"
                    fi
                    ;;
            esac
        done
    else
        log_info "Monitoring deployment skipped (set DEPLOY_MONITORING=true to enable)"
    fi
}

# ===========================================
# DEPLOYMENT VALIDATION
# ===========================================

validate_deployment() {
    log_header "Validating Deployment"
    
    local validation_passed=true
    
    # Check core services
    for service in "${CORE_SERVICES[@]}"; do
        if ! check_service_health "$service"; then
            validation_passed=false
        fi
    done
    
    # Check backend services
    for service in "${BACKEND_SERVICES[@]}"; do
        if ! check_service_health "$service"; then
            validation_passed=false
        fi
    done
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Test backend health endpoint
    if curl -s http://localhost:8000/health &>/dev/null; then
        log_success "Backend API is responding"
    else
        log_error "Backend API is not responding"
        validation_passed=false
    fi
    
    # Test Ollama API
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        log_success "Ollama API is responding"
    else
        log_error "Ollama API is not responding"
        validation_passed=false
    fi
    
    if [ "$validation_passed" = true ]; then
        log_success "Deployment validation completed successfully"
        return 0
    else
        log_error "Deployment validation failed"
        return 1
    fi
}

# ===========================================
# CLEANUP AND RECOVERY
# ===========================================

cleanup_failed_deployment() {
    log_header "Cleaning up Failed Deployment"
    
    if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
        log_info "Stopping failed services..."
        for service in "${FAILED_SERVICES[@]}"; do
            docker-compose stop "$service" 2>/dev/null || true
            docker-compose rm -f "$service" 2>/dev/null || true
            docker-compose -f docker-compose.agents.yml stop "$service" 2>/dev/null || true
            docker-compose -f docker-compose.agents.yml rm -f "$service" 2>/dev/null || true
        done
    fi
    
    # Clean up any orphaned containers
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.agents.yml down --remove-orphans 2>/dev/null || true
}

save_deployment_state() {
    local status=$1
    
    cat > "$DEPLOYMENT_STATE_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "status": "$status",
  "successful_services": $(printf '%s\n' "${SUCCESSFUL_SERVICES[@]}" | jq -R . | jq -s .),
  "failed_services": $(printf '%s\n' "${FAILED_SERVICES[@]}" | jq -R . | jq -s .),
  "errors": $(printf '%s\n' "${DEPLOYMENT_ERRORS[@]}" | jq -R . | jq -s .),
  "error_count": $ERROR_COUNT,
  "warning_count": $WARNING_COUNT
}
EOF
}

# ===========================================
# MAIN DEPLOYMENT FUNCTION
# ===========================================

deploy_multi_agent_system() {
    log_header "SutazAI Multi-Agent Task Automation System Deployment"
    log_info "Starting deployment at $(date)"
    log_info "Deployment log: $LOG_FILE"
    
    # Change to project directory
    cd "$PROJECT_ROOT" || {
        log_error "Cannot change to project directory: $PROJECT_ROOT"
        exit 1
    }
    
    # Pre-deployment validation
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        save_deployment_state "failed"
        exit 1
    fi
    
    if ! check_docker_files; then
        log_error "Docker configuration validation failed"
        save_deployment_state "failed"
        exit 1
    fi
    
    # Stop any existing services and clean up networks
    log_info "Stopping existing services..."
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.agents.yml down --remove-orphans 2>/dev/null || true
    
    # Clean up conflicting networks
    log_info "Cleaning up Docker networks..."
    docker network rm sutazaiapp_sutazai-network 2>/dev/null || true
    docker network prune -f 2>/dev/null || true
    
    # Deploy services in dependency order
    deploy_service_group "Core Infrastructure" "docker-compose.yml" "${CORE_SERVICES[@]}"
    
    # Initialize database after core services are up
    if ! initialize_database; then
        log_error "Database initialization failed"
        cleanup_failed_deployment
        save_deployment_state "failed"
        exit 1
    fi
    
    # Setup Ollama models
    if ! setup_ollama_models; then
        log_warn "Ollama model setup failed, but continuing deployment"
    fi
    
    # Deploy backend services
    deploy_service_group "Backend Services" "docker-compose.yml" "${BACKEND_SERVICES[@]}"
    
    # Deploy AI agents
    deploy_service_group "AI Agents" "docker-compose.agents.yml" "${AGENT_SERVICES[@]}"
    
    # Setup optional monitoring
    setup_monitoring
    
    # Validate deployment
    if validate_deployment; then
        log_success "Deployment completed successfully!"
        save_deployment_state "success"
        
        # Display deployment summary
        display_deployment_summary
        
        return 0
    else
        log_error "Deployment validation failed"
        cleanup_failed_deployment
        save_deployment_state "failed"
        return 1
    fi
}

display_deployment_summary() {
    log_header "Deployment Summary"
    
    echo "🎉 SutazAI Multi-Agent Task Automation System Deployed!"
    echo ""
    echo "📊 Service Status:"
    echo "  ✅ Successful: ${#SUCCESSFUL_SERVICES[@]}"
    echo "  ❌ Failed: ${#FAILED_SERVICES[@]}"
    echo "  ⚠️  Warnings: $WARNING_COUNT"
    echo ""
    echo "🔗 Access Points:"
    echo "  Backend API: http://localhost:8000"
    echo "  Ollama API: http://localhost:11434"
    echo "  PostgreSQL: localhost:5432"
    echo "  Redis: localhost:6379"
    echo ""
    
    if [ ${#SUCCESSFUL_SERVICES[@]} -gt 0 ]; then
        echo "✅ Successfully deployed services:"
        for service in "${SUCCESSFUL_SERVICES[@]}"; do
            echo "  - $service"
        done
        echo ""
    fi
    
    if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
        echo "❌ Failed services:"
        for service in "${FAILED_SERVICES[@]}"; do
            echo "  - $service"
        done
        echo ""
    fi
    
    echo "📋 Next Steps:"
    echo "  1. Test the API endpoints above"
    echo "  2. Check agent status: docker-compose ps"
    echo "  3. View logs: docker-compose logs [service-name]"
    echo "  4. Access the web interface (if deployed)"
    echo ""
    echo "📝 Deployment log: $LOG_FILE"
    echo "📊 State file: $DEPLOYMENT_STATE_FILE"
}

# ===========================================
# UTILITY FUNCTIONS
# ===========================================

show_help() {
    echo "SutazAI Multi-Agent Task Automation System Deployment"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy         Deploy the complete system (default)"
    echo "  start          Alias for deploy"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  status         Show service status"
    echo "  health         Run health checks"
    echo "  logs [service] Show logs for all services or specific service"
    echo "  clean          Clean up stopped containers and volumes"
    echo "  help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOY_MONITORING=true   Enable monitoring services"
    echo "  SUTAZAI_ENV=production   Set environment mode"
    echo ""
}

show_status() {
    log_header "Service Status"
    
    log_info "Core and Backend Services:"
    docker-compose ps
    echo ""
    
    log_info "AI Agent Services:"
    docker-compose -f docker-compose.agents.yml ps
    echo ""
    
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "Unable to get stats"
}

run_health_checks() {
    log_header "Health Check Results"
    
    local all_healthy=true
    
    # Check core services
    for service in "${CORE_SERVICES[@]}" "${BACKEND_SERVICES[@]}"; do
        if check_service_health "$service"; then
            echo "✅ $service"
        else
            echo "❌ $service"
            all_healthy=false
        fi
    done
    
    # Test API endpoints
    echo ""
    log_info "API Endpoint Tests:"
    
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend API (http://localhost:8000/health)"
    else
        echo "❌ Backend API (http://localhost:8000/health)"
        all_healthy=false
    fi
    
    if curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama API (http://localhost:11434/api/tags)"
    else
        echo "❌ Ollama API (http://localhost:11434/api/tags)"
        all_healthy=false
    fi
    
    echo ""
    if [ "$all_healthy" = true ]; then
        log_success "All health checks passed!"
        return 0
    else
        log_error "Some health checks failed"
        return 1
    fi
}

clean_system() {
    log_header "Cleaning System"
    
    log_info "Stopping all services..."
    docker-compose down --remove-orphans
    docker-compose -f docker-compose.agents.yml down --remove-orphans
    
    log_info "Removing stopped containers..."
    docker container prune -f
    
    log_info "Removing unused networks..."
    docker network prune -f
    
    log_info "Removing unused images..."
    docker image prune -f
    
    # Only remove volumes if explicitly confirmed
    read -p "Remove all volumes? This will delete all data (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing volumes..."
        docker volume prune -f
        log_warn "All data has been removed"
    fi
    
    log_success "System cleanup completed"
}

# ===========================================
# SCRIPT ENTRY POINT
# ===========================================

main() {
    case "${1:-deploy}" in
        deploy|start)
            deploy_multi_agent_system
            exit $?
            ;;
        stop)
            log_info "Stopping all services..."
            docker-compose down --remove-orphans
            docker-compose -f docker-compose.agents.yml down --remove-orphans
            exit $?
            ;;
        restart)
            log_info "Restarting system..."
            docker-compose down --remove-orphans
            docker-compose -f docker-compose.agents.yml down --remove-orphans
            deploy_multi_agent_system
            exit $?
            ;;
        status)
            show_status
            exit $?
            ;;
        health|check)
            run_health_checks
            exit $?
            ;;
        logs)
            if [ -n "${2:-}" ]; then
                # Try both compose files for the service
                docker-compose logs -f "$2" 2>/dev/null || docker-compose -f docker-compose.agents.yml logs -f "$2"
            else
                docker-compose logs -f &
                docker-compose -f docker-compose.agents.yml logs -f &
                wait
            fi
            exit $?
            ;;
        clean)
            clean_system
            exit $?
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"