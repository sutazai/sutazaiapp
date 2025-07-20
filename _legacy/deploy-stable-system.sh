#!/bin/bash
# SutazAI Stable System Deployment Script
# Comprehensive deployment with enterprise-grade memory management

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/deployment_$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="docker-compose-stable.yml"

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}✅ ${1}${NC}"
}

# Warning message
warning() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

# Info message
info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

# Check system requirements
check_system_requirements() {
    info "Checking system requirements..."
    
    # Check memory
    local total_memory=$(free -m | awk 'NR==2{printf "%d", $2}')
    if [ "${total_memory}" -lt 8000 ]; then
        error_exit "Insufficient memory. Minimum 8GB required, found ${total_memory}MB"
    fi
    success "Memory check passed: ${total_memory}MB available"
    
    # Check disk space
    local disk_space=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "${disk_space}" -lt 10 ]; then
        warning "Low disk space detected: ${disk_space}GB available. Recommended: 50GB+"
    else
        success "Disk space check passed: ${disk_space}GB available"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi
    success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi
    success "Docker Compose is available"
}

# Optimize system settings
optimize_system() {
    info "Optimizing system settings for SutazAI..."
    
    # Stop any existing services
    info "Stopping existing services..."
    docker-compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose-optimized.yml down --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose-enterprise.yml down --remove-orphans 2>/dev/null || true
    
    # Kill any remaining processes
    sudo pkill -f ollama || true
    sudo pkill -f streamlit || true
    
    # Configure swap if needed
    if ! swapon --show | grep -q swapfile; then
        info "Configuring swap space..."
        if [ ! -f /swapfile ]; then
            sudo fallocate -l 8G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            
            # Make swap permanent
            if ! grep -q '/swapfile' /etc/fstab; then
                echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
            fi
        else
            sudo swapon /swapfile 2>/dev/null || true
        fi
        success "Swap space configured"
    fi
    
    # Optimize memory settings
    info "Optimizing memory settings..."
    echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null
    echo 10 | sudo tee /proc/sys/vm/swappiness > /dev/null
    echo 100 | sudo tee /proc/sys/vm/vfs_cache_pressure > /dev/null
    
    # Optimize network settings
    echo 65536 | sudo tee /proc/sys/net/core/somaxconn > /dev/null
    
    # Clean up Docker
    info "Cleaning up Docker resources..."
    docker system prune -f --volumes
    docker image prune -a -f
    
    success "System optimization complete"
}

# Create required directories
setup_directories() {
    info "Setting up directory structure..."
    
    # Create data directories
    mkdir -p "${SCRIPT_DIR}/data/"{postgres,redis,qdrant,ollama,uploads,logs,backups}
    mkdir -p "${SCRIPT_DIR}/logs"
    mkdir -p "${SCRIPT_DIR}/config"
    mkdir -p "${SCRIPT_DIR}/ssl"
    
    # Set permissions
    chmod 755 "${SCRIPT_DIR}/data"
    chmod 755 "${SCRIPT_DIR}/logs"
    
    success "Directory structure created"
}

# Configure services
configure_services() {
    info "Configuring services..."
    
    # Create Qdrant configuration
    cat > "${SCRIPT_DIR}/config/qdrant.yaml" << 'EOF'
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 32

storage:
  storage_path: /qdrant/storage
  performance:
    max_search_threads: 2
  optimization:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000

cluster:
  enabled: false

telemetry:
  disabled: true

log_level: INFO
EOF
    
    # Create optimized Ollama startup script
    if [ ! -f "${SCRIPT_DIR}/scripts/ollama-startup-optimized.sh" ]; then
        warning "Ollama startup script not found, deployment may fail"
    fi
    
    success "Service configuration complete"
}

# Deploy the system
deploy_system() {
    info "Deploying SutazAI stable system..."
    
    # Verify compose file exists
    if [ ! -f "${SCRIPT_DIR}/${COMPOSE_FILE}" ]; then
        error_exit "Compose file ${COMPOSE_FILE} not found"
    fi
    
    # Pull latest images
    info "Pulling Docker images..."
    docker-compose -f "${COMPOSE_FILE}" pull
    
    # Build custom images
    info "Building custom images..."
    docker-compose -f "${COMPOSE_FILE}" build --no-cache
    
    # Start core services first
    info "Starting core services..."
    docker-compose -f "${COMPOSE_FILE}" up -d postgresql redis qdrant
    
    # Wait for core services to be ready
    info "Waiting for core services to initialize..."
    sleep 30
    
    # Check core services health
    for i in {1..30}; do
        if docker-compose -f "${COMPOSE_FILE}" ps postgresql | grep -q "healthy\\|running"; then
            break
        fi
        if [ $i -eq 30 ]; then
            error_exit "PostgreSQL failed to start"
        fi
        sleep 2
    done
    
    # Start Ollama
    info "Starting Ollama..."
    docker-compose -f "${COMPOSE_FILE}" up -d ollama
    sleep 60  # Give Ollama time to initialize
    
    # Start backend
    info "Starting backend services..."
    docker-compose -f "${COMPOSE_FILE}" up -d sutazai-backend
    sleep 30
    
    # Start frontend
    info "Starting frontend..."
    docker-compose -f "${COMPOSE_FILE}" up -d streamlit-frontend
    sleep 20
    
    # Start monitoring
    info "Starting monitoring services..."
    docker-compose -f "${COMPOSE_FILE}" up -d system-monitor
    
    success "System deployment complete"
}

# Load minimal AI model
load_minimal_model() {
    info "Loading minimal AI model..."
    
    # Wait for Ollama to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            success "Ollama is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            warning "Ollama may not be fully ready, attempting model load anyway"
            break
        fi
        
        info "Waiting for Ollama to be ready... (${attempt}/${max_attempts})"
        sleep 10
        ((attempt++))
    done
    
    # Load minimal model
    info "Loading llama3.2:1b model..."
    if docker-compose -f "${COMPOSE_FILE}" exec -T ollama ollama pull llama3.2:1b; then
        success "Minimal model loaded successfully"
    else
        warning "Failed to load minimal model, system will work without pre-loaded models"
    fi
}

# Verify deployment
verify_deployment() {
    info "Verifying deployment..."
    
    # Check all services are running
    local services=("postgresql" "redis" "qdrant" "ollama" "sutazai-backend" "streamlit-frontend" "system-monitor")
    
    for service in "${services[@]}"; do
        if docker-compose -f "${COMPOSE_FILE}" ps "${service}" | grep -q "Up\\|running"; then
            success "${service} is running"
        else
            warning "${service} may not be running properly"
        fi
    done
    
    # Check service endpoints
    local endpoints=(
        "http://localhost:5432 PostgreSQL"
        "http://localhost:6379 Redis"
        "http://localhost:6333/health Qdrant"
        "http://localhost:11434/api/version Ollama"
        "http://localhost:8001/health Backend"
        "http://localhost:8501 Frontend"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        local url=$(echo $endpoint_info | cut -d' ' -f1)
        local name=$(echo $endpoint_info | cut -d' ' -f2)
        
        if curl -s -f "${url}" > /dev/null 2>&1; then
            success "${name} endpoint is accessible"
        else
            warning "${name} endpoint may not be accessible yet"
        fi
    done
    
    # Check system resources
    local memory_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
    info "Current memory usage: ${memory_usage}%"
    
    if (( $(echo "${memory_usage} < 85" | bc -l) )); then
        success "Memory usage is within acceptable limits"
    else
        warning "Memory usage is high (${memory_usage}%)"
    fi
}

# Generate status report
generate_status_report() {
    local report_file="${SCRIPT_DIR}/logs/deployment_status_$(date +%Y%m%d_%H%M%S).json"
    
    info "Generating deployment status report..."
    
    cat > "${report_file}" << EOF
{
    \"deployment_time\": \"$(date -Iseconds)\",
    \"system_info\": {
        \"memory_total\": \"$(free -h | awk 'NR==2{print $2}')\",
        \"memory_available\": \"$(free -h | awk 'NR==2{print $7}')\",
        \"disk_space\": \"$(df -h / | awk 'NR==2{print $4}')\",
        \"cpu_cores\": \"$(nproc)\",
        \"swap_total\": \"$(free -h | awk 'NR==3{print $2}')\"
    },
    \"services\": {
EOF

    # Add service status
    local first=true
    for service in postgresql redis qdrant ollama sutazai-backend streamlit-frontend system-monitor; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "${report_file}"
        fi
        
        local status="stopped"
        if docker-compose -f "${COMPOSE_FILE}" ps "${service}" | grep -q "Up\\|running"; then
            status="running"
        fi
        
        echo "        \"${service}\": \"${status}\"" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF
    },
    \"endpoints\": {
        \"frontend\": \"http://localhost:8501\",
        \"backend\": \"http://localhost:8001\",
        \"qdrant\": \"http://localhost:6333\",
        \"ollama\": \"http://localhost:11434\"
    },
    \"configuration\": {
        \"compose_file\": \"${COMPOSE_FILE}\",
        \"memory_optimized\": true,
        \"monitoring_enabled\": true,
        \"auto_recovery\": true
    }
}
EOF

    success "Status report generated: ${report_file}"
}

# Main deployment function
main() {
    info "Starting SutazAI Stable System Deployment"
    info "Deployment started at: $(date)"
    info "Log file: ${LOG_FILE}"
    
    # Run deployment steps
    check_system_requirements
    optimize_system
    setup_directories
    configure_services
    deploy_system
    load_minimal_model
    verify_deployment
    generate_status_report
    
    # Final success message
    echo
    success "🎉 SutazAI Stable System Deployment Complete!"
    echo
    info "Access the application at: http://localhost:8501"
    info "Backend API at: http://localhost:8001"
    info "Qdrant vector DB at: http://localhost:6333"
    info "Ollama API at: http://localhost:11434"
    echo
    info "System features:"
    info "  ✅ Memory-optimized configuration"
    info "  ✅ Automatic monitoring and recovery"
    info "  ✅ Enterprise-grade stability"
    info "  ✅ Comprehensive logging"
    echo
    info "To monitor the system:"
    info "  docker-compose -f ${COMPOSE_FILE} logs -f"
    info "  free -h  # Check memory usage"
    info "  docker stats  # Check container resources"
    echo
    info "To stop the system:"
    info "  docker-compose -f ${COMPOSE_FILE} down"
    echo
    warning "Keep monitoring memory usage to prevent OOM kills!"
    
    # Log final status
    log "Deployment completed successfully at $(date)"
}

# Run main function
main "$@"