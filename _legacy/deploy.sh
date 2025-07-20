#!/bin/bash

# SutazAI AGI/ASI Autonomous System - Master Deployment Script
# ============================================================
# This script automates the complete setup, configuration, and
# deployment of the SutazAI ecosystem.

set -euo pipefail

# --- Configuration ---
readonly LOG_FILE="deployment.log"
readonly ENV_TEMPLATE=".env.template"
readonly ENV_FILE=".env"
readonly COMPOSE_FILE="docker-compose.yml"

# --- Colors for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Logging Functions ---
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOG_FILE"
}
error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}
warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# --- Helper Functions ---
confirm() {
    read -p "$1 [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            true
            ;;
        *)
            false
            ;;
    esac
}

# --- Deployment Phases ---

phase_1_prerequisites() {
    log "Phase 1: Checking System Prerequisites..."
    command -v docker >/dev/null 2>&1 || error "Docker is not installed. Please install Docker and try again."
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed. Please install it and try again."

    if ! docker info >/dev/null 2>&1; then
        error "The Docker daemon is not running. Please start Docker and try again."
    fi

    local available_space
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( available_space < 50 )); then
        warning "Less than 50GB of free disk space detected. This may not be enough for all AI models."
    fi
    log "Prerequisites check passed."
}

phase_2_configure_environment() {
    log "Phase 2: Configuring System Environment..."
    if [ ! -f "$ENV_FILE" ]; then
        log "No .env file found. Creating one from template."
        cp "$ENV_TEMPLATE" "$ENV_FILE" || error "Failed to create .env file. Template missing."
    else
        log ".env file already exists."
    fi
    # Source the .env file to use the variables in this script
    set -a
    source "$ENV_FILE"
    set +a
    log "Environment configured."
}

phase_3_build_and_launch() {
    log "Phase 3: Building and Launching Core System..."
    log "This may take a significant amount of time, especially on the first run."

    log "Building all service images from Dockerfiles..."
    docker-compose -f "$COMPOSE_FILE" build --parallel || error "Docker build failed. Check logs for errors."

    log "Starting all services in detached mode..."
    docker-compose -f "$COMPOSE_FILE" up -d || error "Failed to start services. Check 'docker-compose logs' for details."

    log "Core system services are starting up in the background."
}

phase_4_initialize_models() {
    log "Phase 4: Initializing AI Models..."
    log "Waiting for the Ollama service to become available..."

    local retries=40
    local count=0
    while [ $count -lt $retries ]; do
        if docker-compose ps ollama | grep -q "Up" && curl -s --fail http://localhost:11434/ > /dev/null; then
            log "Ollama service is up and responsive."
            
            local models_to_pull=("deepseek-coder:33b" "llama2" "codellama:7b" "qwen3:8b")
            log "Pulling required models: ${models_to_pull[*]}"
            for model in "${models_to_pull[@]}"; do
                log "Pulling $model..."
                docker-compose exec ollama ollama pull "$model" || warning "Failed to pull model $model. It may not be available."
            done
            log "Model initialization complete."
            return
        fi
        sleep 5
        ((count++))
        log "Waiting for Ollama... attempt ($count/$retries)"
    done
    error "Ollama service failed to start in time. Please check its logs: 'docker-compose logs ollama'"
}

phase_5_verify_deployment() {
    log "Phase 5: Verifying System Deployment..."
    sleep 10 # Give services a moment to stabilize.

    log "Final status of all services:"
    docker-compose ps

    local unhealthy_services
    unhealthy_services=$(docker-compose ps | grep -v "Up")
    if [ -n "$unhealthy_services" ]; then
        warning "The following services may not be running correctly:"
        echo -e "${YELLOW}$unhealthy_services${NC}"
        warning "Please check their logs with 'docker-compose logs <service_name>'."
    else
        log "All services appear to be running correctly."
    fi
}

# --- Main Execution ---
main() {
    clear
    echo -e "${BLUE}=======================================================================${NC}"
    echo -e "${BLUE}      🚀 Welcome to the SutazAI Enterprise Deployment Script 🚀       ${NC}"
    echo -e "${BLUE}=======================================================================${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        rm "$LOG_FILE"
    fi

    trap 'error "Deployment script interrupted by user."' SIGINT

    phase_1_prerequisites
    phase_2_configure_environment
    phase_3_build_and_launch
    phase_4_initialize_models
    phase_5_verify_deployment

    echo -e "\n${GREEN}=======================================================================${NC}"
    echo -e "${GREEN}  ✅ SutazAI Deployment Complete!                                     ${NC}"
    echo -e "${GREEN}-----------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}  - Main UI (Streamlit):  http://localhost:8501                        ${NC}"
    echo -e "${YELLOW}  - Backend API Docs:     http://localhost:8000/docs                   ${NC}"
    echo -e "${YELLOW}  - Monitoring (Grafana): http://localhost:3000 (user/pass: admin/admin)${NC}"
    echo -e "${GREEN}=======================================================================${NC}\n"
    log "To view live logs for all services, run: docker-compose logs -f"
    log "To stop the system, run: docker-compose down"
}

main "$@"