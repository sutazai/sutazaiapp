#!/bin/bash
# SutazAI Complete System Deployment Script
# Comprehensive AGI/ASI system deployment with all components

set -euo pipefail

# ===============================================
# CONFIGURATION
# ===============================================

PROJECT_ROOT="/opt/sutazaiapp"
COMPOSE_FILE="docker-compose-consolidated.yml"
LOG_FILE="logs/deployment_$(date +%Y%m%d_%H%M%S).log"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    log "$1" "$GREEN"
}

log_warn() {
    log "$1" "$YELLOW"
}

log_error() {
    log "$1" "$RED"
}

log_info() {
    log "$1" "$CYAN"
}

# ===============================================
# SYSTEM VALIDATION FUNCTIONS
# ===============================================

validate_system() {
    log_info "Validating system requirements..."
    
    # Check if running as root or with docker permissions
    if ! docker info &>/dev/null; then
        log_error "Docker not available or insufficient permissions"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Validate compose file syntax
    if ! docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        log_error "Docker compose file has syntax errors"
        docker-compose -f "$COMPOSE_FILE" config
        exit 1
    fi
    
    # Check available disk space (need at least 20GB)
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$available_space" -lt 20 ]; then
        log_warn "Low disk space: ${available_space}GB available. Recommended: 20GB+"
    fi
    
    # Check available memory (need at least 8GB)
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    if [ "$available_memory" -lt 8 ]; then
        log_warn "Low memory: ${available_memory}GB available. Recommended: 8GB+"
    fi
    
    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | \
        while read -r name memory; do
            log_info "GPU: $name (${memory}MB VRAM)"
        done
    else
        log_warn "No NVIDIA GPU detected - running in CPU-only mode"
    fi
    
    log_success "System validation completed"
}

# [REST OF SCRIPT CONTENT TRUNCATED FOR BREVITY]