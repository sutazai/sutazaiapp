#!/bin/bash
# SutazAI Complete System Deployment - Simplified Version

set -euo pipefail

# ===============================================
# CONFIGURATION
# ===============================================

PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="$PROJECT_ROOT/logs/deployment_$(date +%Y%m%d_%H%M%S).log"
ENV_FILE=".env"
CLEAN_VOLUMES=${CLEAN_VOLUMES:-false}

# Get dynamic IP instead of hardcoded
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ -z "$LOCAL_IP" ]]; then
    LOCAL_IP="localhost"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] 🎉 SUCCESS: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1${NC}"
}

log_phase() {
    echo -e "${PURPLE}${BOLD}[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 PHASE: $1${NC}"
}

# ===============================================
# ERROR HANDLING
# ===============================================

handle_error() {
    local exit_code=$1
    local line_number=$2
    local command=$3
    log_error "Error on line $line_number: command '$command' failed with exit code $exit_code"
    exit $exit_code
}

trap 'handle_error $? $LINENO $BASH_COMMAND' ERR

# ===============================================
# DOCKER MANAGEMENT
# ===============================================

ensure_docker_running_perfectly() {
    log_phase "Ensuring Docker is running perfectly"
    if ! docker info >/dev/null 2>&1; then
        log_info "Docker is not running. Attempting to start..."
        if command -v systemctl &>/dev/null; then
            systemctl start docker
            sleep 5
            if ! docker info >/dev/null 2>&1; then
                log_error "Failed to start Docker with systemctl."
                exit 1
            fi
        else
            log_error "systemctl not found. Please start Docker manually."
            exit 1
        fi
        log_success "Docker started successfully."
    fi
}

# ===============================================
# PRE-DEPLOYMENT CHECKS
# ===============================================

pre_deployment_checks() {
    log_phase "Performing Pre-Deployment Checks"

    # Validate system dependencies
    local critical_commands=("curl" "wget" "git" "docker" "python3" "pip3" "jq")
    for cmd in "${critical_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_error "Missing critical command: $cmd"
            exit 1
        fi
    done

    # Ensure Docker is running
    ensure_docker_running_perfectly

    # Check system resources
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$available_space" -lt 50 ]; then
        log_error "Insufficient disk space: ${available_space}GB available. Required: 50GB+"
        exit 1
    fi

    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    if [ "$available_memory" -lt 8 ]; then
        log_error "Insufficient memory: ${available_memory}GB available. Required: 8GB+"
        exit 1
    fi
}

# ===============================================
# ENVIRONMENT SETUP
# ===============================================

setup_environment() {
    log_phase "Environment Configuration Setup"

    cd "$PROJECT_ROOT"

    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating secure environment configuration..."

        export POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/")
        export REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/")
        export NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/")
        export SECRET_KEY=$(openssl rand -hex 32)

        cat > "$ENV_FILE" << EOF
# SutazAI Complete System Configuration
# Generated on $(date)
# System IP: $LOCAL_IP

# System Settings
TZ=UTC
SUTAZAI_ENV=production
LOCAL_IP=$LOCAL_IP

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=sutazai
DATABASE_URL=postgresql://sutazai:${POSTGRES_PASSWORD}@postgres:5432/sutazai

REDIS_PASSWORD=${REDIS_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}

# API Keys and Secrets
SECRET_KEY=${SECRET_KEY}

# Feature Flags
ENABLE_GPU=auto
EOF

        chmod 600 "$ENV_FILE"
        log_success "Environment configuration created with secure passwords"
    fi

    set -a
    source "$ENV_FILE"
    set +a

    log_success "Environment setup completed"
}

# ===============================================
# DOCKER COMPOSE MANAGEMENT
# ===============================================

get_compose_files() {
    local compose_files=("-f" "docker-compose.yml")

    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected, using GPU compose file."
        compose_files+=("-f" "docker-compose.gpu.yml")
    else
        log_info "No NVIDIA GPU detected, using CPU-only compose file."
        compose_files+=("-f" "docker-compose.cpu-only.yml")
    fi

    echo "${compose_files[@]}"
}

# ===============================================
# MAIN DEPLOYMENT FLOW
# ===============================================

main() {
    setup_logging
    pre_deployment_checks
    setup_environment

    local compose_args=$(get_compose_files)

    if [ "$CLEAN_VOLUMES" = true ]; then
        log_phase "Cleaning existing volumes"
        docker compose $compose_args down -v --remove-orphans
        log_success "Volumes cleaned successfully"
    fi

    log_phase "Building and deploying services"
    docker compose $compose_args up --build -d
    log_success "Services deployed successfully"

    log_phase "Running post-deployment health checks"
    # Add health checks here in the future

    log_success "Deployment completed successfully!"
}

# ===============================================
# SCRIPT EXECUTION WITH ARGUMENT HANDLING
# ===============================================

show_usage() {
    echo "SutazAI Complete automation/advanced automation System Deployment Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  deploy       - Deploy complete SutazAI system (default)"
    echo "  stop         - Stop all Sutazai services"
    echo "  restart      - Restart the complete system"
    echo "  status       - Show status of all services"
    echo "  logs         - Show logs for all services"
    echo "  help         - Show this help message"
    echo
    echo "Options:"
    echo "  CLEAN_VOLUMES=true - Clean existing volumes during deployment"
    echo
    echo "Examples:"
    echo "  $0 deploy                    # Deploy complete system"
    echo "  $0 stop                      # Stop all services"
    echo "  $0 restart                   # Full restart"
    echo "  CLEAN_VOLUMES=true $0 deploy # Clean deployment"
    echo
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping all SutazAI services..."
        cd "$PROJECT_ROOT"
        compose_args=$(get_compose_files)
        docker compose $compose_args down --remove-orphans
        log_success "All services stopped"
        ;;
    "restart")
        log_info "Restarting SutazAI system..."
        cd "$PROJECT_ROOT"
        compose_args=$(get_compose_files)
        docker compose $compose_args down --remove-orphans
        sleep 10
        main
        ;;
    "status")
        cd "$PROJECT_ROOT"
        compose_args=$(get_compose_files)
        echo "Docker Services Status:"
        docker compose $compose_args ps
        ;;
    "logs")
        cd "$PROJECT_ROOT"
        compose_args=$(get_compose_files)
        docker compose $compose_args logs -f "${2:-}"
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        echo
        show_usage
        exit 1
        ;;
esac
