#!/bin/bash
# SutazAI Multi-Agent Task Automation System Deployment
# Production-ready deployment script with comprehensive service orchestration
# Deploys all infrastructure, AI agents, and monitoring services

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

# Service definitions - Complete service list
CORE_SERVICES=("postgres" "redis" "ollama")
VECTOR_SERVICES=("chromadb" "qdrant" "neo4j")
BACKEND_SERVICES=("backend" "frontend")
ML_SERVICES=("pytorch" "tensorflow" "jax")
MONITORING_SERVICES=("prometheus" "grafana" "loki" "promtail")
WORKFLOW_SERVICES=("n8n" "langflow" "flowise" "llamaindex")
AGENT_SERVICES=(
    "senior-ai-engineer"
    "deployment-automation-master"
    "infrastructure-devops-manager"
    "ollama-integration-specialist"
    "testing-qa-validator"
    "code-generation-improver"
)
ADDITIONAL_AGENTS=(
    "autogpt" "crewai" "letta" "aider" "gpt-engineer"
    "browser-use" "skyvern" "agentgpt" "privategpt"
    "shellgpt" "pentestgpt" "documind" "litellm"
    "context-framework" "autogen" "opendevin" "finrobot"
    "realtimestt" "code-improver" "service-hub"
    "awesome-code-ai" "fsdp" "agentzero" "dify"
    "mcp-server" "jarvis" "health-monitor"
)

# Deployment profiles
DEPLOYMENT_PROFILE="${DEPLOY_PROFILE:-standard}"

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
    local required_commands=("docker" "docker-compose" "curl" "jq" "nc")
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
NEO4J_PASSWORD=sutazai_neo4j_password

# Redis Configuration
REDIS_PASSWORD=redis_password

# Ollama Configuration
OLLAMA_MODELS=tinyllama
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*

# Monitoring
GRAFANA_PASSWORD=sutazai_grafana
N8N_USER=admin
N8N_PASSWORD=sutazai_n8n

# MCP Server
MCP_LOG_LEVEL=INFO
MCP_MAX_TASKS=10
MCP_TASK_TIMEOUT=300

# API Keys (replace with your own)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
CHROMADB_API_KEY=test-token
LITELLM_KEY=sk-1234
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
    local container_name="sutazai-$service_name"
    
    # For minimal deployment, check with -minimal suffix
    if [ "$DEPLOYMENT_PROFILE" = "minimal" ]; then
        container_name="sutazai-$service_name-minimal"
    fi
    
    # Check if container is running
    if docker ps --filter "name=$container_name" --format "{{.Names}}" | grep -q "$container_name"; then
        log_success "$service_name container is running"
        
        # Check container health if healthcheck is defined
        local health_status=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-health-check{{end}}' "$container_name" 2>/dev/null || echo "unknown")
        
        case "$health_status" in
            healthy)
                log_success "$service_name is healthy"
                return 0
                ;;
            unhealthy)
                # Special handling for Redis with auth issues
                if [ "$service_name" = "redis" ]; then
                    log_warn "$service_name health check failed, checking connectivity..."
                    if docker exec "$container_name" redis-cli -a "${REDIS_PASSWORD:-redis_password}" ping 2>/dev/null | grep -q "PONG"; then
                        log_success "$service_name is responding to ping"
                        return 0
                    fi
                fi
                log_error "$service_name is unhealthy"
                return 1
                ;;
            no-health-check)
                log_info "$service_name has no health check defined"
                return 0
                ;;
            starting)
                log_info "$service_name health status: starting"
                return 0
                ;;
            *)
                log_info "$service_name health status: $health_status"
                return 0
                ;;
        esac
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
        
        if docker-compose -f "$compose_file" up -d "$service" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "$service started successfully"
            SUCCESSFUL_SERVICES+=("$service")
            
            # Wait for specific services to be ready
            case "$service" in
                postgres*)
                    wait_for_service "$service" 5432
                    ;;
                redis*)
                    wait_for_service "$service" 6379
                    ;;
                ollama*)
                    wait_for_service "$service" 11434
                    ;;
                backend*)
                    wait_for_service "$service" 8000
                    ;;
                frontend*)
                    wait_for_service "$service" 8501
                    ;;
                chromadb)
                    wait_for_service "$service" 8001
                    ;;
                qdrant)
                    wait_for_service "$service" 6333
                    ;;
                neo4j)
                    wait_for_service "$service" 7474
                    ;;
                prometheus)
                    wait_for_service "$service" 9090
                    ;;
                grafana)
                    wait_for_service "$service" 3000
                    ;;
                loki)
                    wait_for_service "$service" 3100
                    ;;
            esac
            
            # Verify service health
            if ! check_service_health "$service"; then
                log_warn "$service health check failed but continuing"
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
    
    # Determine container name based on profile
    local ollama_container="sutazai-ollama"
    if [ "$DEPLOYMENT_PROFILE" = "minimal" ]; then
        ollama_container="sutazai-ollama-minimal"
    fi
    
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
    
    # Add additional models based on profile
    if [ "$DEPLOYMENT_PROFILE" = "full" ]; then
        models+=("llama2:7b" "codellama:7b" "mistral:7b")
    fi
    
    for model in "${models[@]}"; do
        log_info "Pulling model: $model"
        if docker exec "$ollama_container" ollama pull "$model" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Successfully pulled $model"
        else
            log_error "Failed to pull $model"
        fi
    done
    
    # List available models
    log_info "Available models:"
    docker exec "$ollama_container" ollama list | tee -a "$LOG_FILE"
}

# ===========================================
# DATABASE INITIALIZATION
# ===========================================

initialize_database() {
    log_header "Initializing Database"
    
    # Determine container name based on profile
    local postgres_container="sutazai-postgres"
    if [ "$DEPLOYMENT_PROFILE" = "minimal" ]; then
        postgres_container="sutazai-postgres-minimal"
    fi
    
    # Wait for PostgreSQL to be ready
    if ! wait_for_service "postgres" 5432 30; then
        log_error "PostgreSQL not ready for initialization"
        return 1
    fi
    
    # Check if database is accessible
    if docker exec "$postgres_container" psql -U sutazai -d sutazai -c "SELECT 1;" &>/dev/null; then
        log_success "Database connection verified"
        
        # Run initialization scripts if they exist
        if [ -f "$PROJECT_ROOT/config/postgres/init.sql" ]; then
            log_info "Running database initialization script"
            docker exec -i "$postgres_container" psql -U sutazai -d sutazai < "$PROJECT_ROOT/config/postgres/init.sql" 2>&1 | tee -a "$LOG_FILE"
            log_success "Database initialization completed"
        fi
    else
        log_error "Cannot connect to database"
        return 1
    fi
}

# ===========================================
# NETWORK CLEANUP
# ===========================================

cleanup_networks() {
    log_info "Cleaning up Docker networks..."
    
    # Remove any existing SutazAI networks
    docker network ls | grep sutazai | awk '{print $2}' | while read network; do
        log_info "Removing network: $network"
        docker network rm "$network" 2>/dev/null || true
    done
    
    # Prune unused networks
    docker network prune -f 2>&1 | tee -a "$LOG_FILE"
}

# ===========================================
# DEPLOYMENT PROFILES
# ===========================================

get_deployment_services() {
    local profile=$1
    local services=()
    
    case "$profile" in
        minimal)
            services=("${CORE_SERVICES[@]}" "${BACKEND_SERVICES[@]}")
            ;;
        standard)
            services=("${CORE_SERVICES[@]}" "${VECTOR_SERVICES[@]}" "${BACKEND_SERVICES[@]}" "${AGENT_SERVICES[@]}")
            ;;
        full)
            services=("${CORE_SERVICES[@]}" "${VECTOR_SERVICES[@]}" "${BACKEND_SERVICES[@]}" 
                     "${ML_SERVICES[@]}" "${MONITORING_SERVICES[@]}" "${WORKFLOW_SERVICES[@]}"
                     "${AGENT_SERVICES[@]}" "${ADDITIONAL_AGENTS[@]}")
            ;;
        *)
            log_error "Unknown deployment profile: $profile"
            exit 1
            ;;
    esac
    
    echo "${services[@]}"
}

# ===========================================
# DEPLOYMENT VALIDATION
# ===========================================

validate_deployment() {
    log_header "Validating Deployment"
    
    local validation_passed=true
    
    # Get services based on profile
    local expected_services=($(get_deployment_services "$DEPLOYMENT_PROFILE"))
    
    # Check services based on what was actually deployed
    for service in "${SUCCESSFUL_SERVICES[@]}"; do
        if ! check_service_health "$service"; then
            validation_passed=false
        fi
    done
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Test backend health endpoint
    if curl -s http://localhost:8000/health &>/dev/null; then
        log_success "Backend API is responding"
        
        # Get detailed health info
        local health_data=$(curl -s http://localhost:8000/health | jq -r '.')
        log_info "Backend health: $(echo "$health_data" | jq -r '.status // "unknown"')"
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
    
    # Test Frontend (if deployed)
    if [[ " ${SUCCESSFUL_SERVICES[@]} " =~ " frontend " ]]; then
        if curl -s http://localhost:8501 &>/dev/null; then
            log_success "Frontend is accessible"
        else
            log_error "Frontend is not accessible"
            validation_passed=false
        fi
    fi
    
    # Test monitoring endpoints (if deployed)
    if [[ " ${SUCCESSFUL_SERVICES[@]} " =~ " prometheus " ]]; then
        if curl -s http://localhost:9090/-/healthy &>/dev/null; then
            log_success "Prometheus is healthy"
        else
            log_warn "Prometheus health check failed"
        fi
    fi
    
    if [[ " ${SUCCESSFUL_SERVICES[@]} " =~ " grafana " ]]; then
        if curl -s http://localhost:3000/api/health &>/dev/null; then
            log_success "Grafana is healthy"
        else
            log_warn "Grafana health check failed"
        fi
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
  "profile": "$DEPLOYMENT_PROFILE",
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
    log_info "Deployment profile: $DEPLOYMENT_PROFILE"
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
    docker-compose -f docker-compose.minimal.yml down --remove-orphans 2>/dev/null || true
    
    # Clean up networks
    cleanup_networks
    
    # Select compose file based on profile
    local compose_file="docker-compose.yml"
    if [ "$DEPLOYMENT_PROFILE" = "minimal" ]; then
        compose_file="docker-compose.minimal.yml"
    fi
    
    # Deploy services based on profile
    case "$DEPLOYMENT_PROFILE" in
        minimal)
            # Deploy minimal setup
            deploy_service_group "Core Infrastructure" "$compose_file" "${CORE_SERVICES[@]}"
            initialize_database
            setup_ollama_models
            deploy_service_group "Backend Services" "$compose_file" "${BACKEND_SERVICES[@]}"
            
            # Deploy basic agents from minimal compose
            local minimal_agents=("code-improver" "qa-validator" "ai-engineer")
            deploy_service_group "Basic AI Agents" "$compose_file" "${minimal_agents[@]}"
            ;;
            
        standard)
            # Deploy standard setup
            deploy_service_group "Core Infrastructure" "$compose_file" "${CORE_SERVICES[@]}"
            deploy_service_group "Vector Databases" "$compose_file" "${VECTOR_SERVICES[@]}"
            initialize_database
            setup_ollama_models
            deploy_service_group "Backend Services" "$compose_file" "${BACKEND_SERVICES[@]}"
            deploy_service_group "AI Agents" "docker-compose.agents.yml" "${AGENT_SERVICES[@]}"
            ;;
            
        full)
            # Deploy everything
            deploy_service_group "Core Infrastructure" "$compose_file" "${CORE_SERVICES[@]}"
            deploy_service_group "Vector Databases" "$compose_file" "${VECTOR_SERVICES[@]}"
            initialize_database
            setup_ollama_models
            deploy_service_group "Backend Services" "$compose_file" "${BACKEND_SERVICES[@]}"
            deploy_service_group "Machine Learning" "$compose_file" "${ML_SERVICES[@]}"
            deploy_service_group "Monitoring Stack" "$compose_file" "${MONITORING_SERVICES[@]}"
            deploy_service_group "Workflow Tools" "$compose_file" "${WORKFLOW_SERVICES[@]}"
            deploy_service_group "Core AI Agents" "docker-compose.agents.yml" "${AGENT_SERVICES[@]}"
            deploy_service_group "Additional AI Agents" "$compose_file" "${ADDITIONAL_AGENTS[@]}"
            ;;
    esac
    
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
    echo "📊 Deployment Profile: $DEPLOYMENT_PROFILE"
    echo ""
    echo "📊 Service Status:"
    echo "  ✅ Successful: ${#SUCCESSFUL_SERVICES[@]}"
    echo "  ❌ Failed: ${#FAILED_SERVICES[@]}"
    echo "  ⚠️  Warnings: $WARNING_COUNT"
    echo ""
    echo "🔗 Access Points:"
    echo "  Backend API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Frontend UI: http://localhost:8501"
    echo "  Ollama API: http://localhost:11434"
    
    if [ "$DEPLOYMENT_PROFILE" != "minimal" ]; then
        echo "  ChromaDB: http://localhost:8001"
        echo "  Qdrant: http://localhost:6333"
        echo "  Neo4j: http://localhost:7474"
    fi
    
    if [[ " ${SUCCESSFUL_SERVICES[@]} " =~ " prometheus " ]]; then
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3000"
    fi
    
    echo ""
    echo "📊 Database Access:"
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
    
    echo "📋 Management Commands:"
    echo "  View logs: ./scripts/live_logs.sh"
    echo "  Check status: docker-compose ps"
    echo "  View specific logs: docker-compose logs [service-name]"
    echo "  Stop all: docker-compose down"
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
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy         Deploy the system (default)"
    echo "  start          Alias for deploy"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  status         Show service status"
    echo "  health         Run health checks"
    echo "  logs [service] Show logs for all services or specific service"
    echo "  clean          Clean up stopped containers and volumes"
    echo "  help           Show this help message"
    echo ""
    echo "Options:"
    echo "  --profile [minimal|standard|full]  Set deployment profile (default: standard)"
    echo "    minimal:  Core services + backend + basic agents"
    echo "    standard: Core + vector DBs + backend + core agents"
    echo "    full:     Everything including ML, monitoring, all agents"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOY_PROFILE=minimal|standard|full   Set deployment profile"
    echo "  SUTAZAI_ENV=production                 Set environment mode"
    echo ""
    echo "Examples:"
    echo "  $0 deploy --profile minimal"
    echo "  $0 deploy --profile full"
    echo "  $0 status"
    echo "  $0 logs backend"
    echo ""
}

show_status() {
    log_header "Service Status"
    
    # Check which compose files have running services
    local has_services=false
    
    log_info "Core and Backend Services:"
    if docker-compose ps 2>/dev/null | grep -q "Up"; then
        docker-compose ps
        has_services=true
    else
        echo "No services running in docker-compose.yml"
    fi
    echo ""
    
    log_info "AI Agent Services:"
    if docker-compose -f docker-compose.agents.yml ps 2>/dev/null | grep -q "Up"; then
        docker-compose -f docker-compose.agents.yml ps
        has_services=true
    else
        echo "No services running in docker-compose.agents.yml"
    fi
    echo ""
    
    log_info "Minimal Services:"
    if docker-compose -f docker-compose.minimal.yml ps 2>/dev/null | grep -q "Up"; then
        docker-compose -f docker-compose.minimal.yml ps
        has_services=true
    else
        echo "No services running in docker-compose.minimal.yml"
    fi
    echo ""
    
    if [ "$has_services" = true ]; then
        log_info "Resource Usage:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || echo "Unable to get stats"
    fi
}

run_health_checks() {
    log_header "Health Check Results"
    
    local all_healthy=true
    
    # Get all running containers
    local running_containers=$(docker ps --format "{{.Names}}" | grep "sutazai-" | sed 's/sutazai-//')
    
    if [ -z "$running_containers" ]; then
        log_warn "No SutazAI containers are running"
        return 1
    fi
    
    # Check each running service
    for service in $running_containers; do
        # Clean up service name (remove -minimal suffix for checking)
        local clean_service=$(echo "$service" | sed 's/-minimal$//')
        
        if check_service_health "$clean_service"; then
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
        # Show health details
        curl -s http://localhost:8000/health | jq -r '. | "   Status: \(.status), Version: \(.version), Agents: \(.services.agents.active_count)"' 2>/dev/null || true
    else
        echo "❌ Backend API (http://localhost:8000/health)"
        all_healthy=false
    fi
    
    if curl -s -f http://localhost:8501 > /dev/null 2>&1; then
        echo "✅ Frontend (http://localhost:8501)"
    else
        echo "❌ Frontend (http://localhost:8501)"
    fi
    
    if curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama API (http://localhost:11434/api/tags)"
        # Show model count
        local model_count=$(curl -s http://localhost:11434/api/tags | jq -r '.models | length' 2>/dev/null || echo "0")
        echo "   Models loaded: $model_count"
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
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.agents.yml down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.minimal.yml down --remove-orphans 2>/dev/null || true
    
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
    # Parse command line arguments
    local command="${1:-deploy}"
    shift || true
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                DEPLOYMENT_PROFILE="$2"
                shift 2
                ;;
            *)
                # Pass through to command
                break
                ;;
        esac
    done
    
    case "$command" in
        deploy|start)
            deploy_multi_agent_system
            exit $?
            ;;
        stop)
            log_info "Stopping all services..."
            docker-compose down --remove-orphans 2>/dev/null || true
            docker-compose -f docker-compose.agents.yml down --remove-orphans 2>/dev/null || true
            docker-compose -f docker-compose.minimal.yml down --remove-orphans 2>/dev/null || true
            log_success "All services stopped"
            exit 0
            ;;
        restart)
            log_info "Restarting system..."
            $0 stop
            sleep 2
            $0 deploy "$@"
            exit $?
            ;;
        status)
            show_status
            exit 0
            ;;
        health|check)
            run_health_checks
            exit $?
            ;;
        logs)
            if [ -n "${1:-}" ]; then
                # Show logs for specific service
                docker-compose logs -f "$1" 2>/dev/null || \
                docker-compose -f docker-compose.agents.yml logs -f "$1" 2>/dev/null || \
                docker-compose -f docker-compose.minimal.yml logs -f "$1" 2>/dev/null || \
                echo "Service '$1' not found"
            else
                # Show all logs
                log_info "Showing logs from all compose files..."
                docker-compose logs -f &
                docker-compose -f docker-compose.agents.yml logs -f &
                docker-compose -f docker-compose.minimal.yml logs -f &
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
            echo "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"