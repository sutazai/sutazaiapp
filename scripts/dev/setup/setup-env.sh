#!/bin/bash
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script: setup-env.sh
# Purpose: Configure development environment with proper security and validation
# Author: Sutazai System
# Date: 2025-09-03
# Version: 1.0.0
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage:
#   ./setup-env.sh [options]
#
# Options:
#   -h, --help          Show this help message
#   -v, --verbose       Enable verbose output
#   -d, --dry-run       Run in simulation mode
#   -e, --env ENV       Environment to configure (dev|staging|prod)
#
# Requirements:
#   - Bash 4.0+
#   - Docker 20.10+
#   - Python 3.8+
#
# Examples:
#   ./setup-env.sh --env dev
#   ./setup-env.sh --dry-run --verbose
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly ENV_FILE="${PROJECT_ROOT}/.env"
readonly ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"
readonly SECRETS_DIR="${PROJECT_ROOT}/.secrets"
readonly LOG_FILE="${PROJECT_ROOT}/logs/setup-env_$(date +%Y%m%d_%H%M%S).log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Default options
VERBOSE=false
DRY_RUN=false
ENVIRONMENT="dev"

# Logging
log() {
    local level="$1"
    local message="$2"
    local color="${3:-$NC}"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
    
    if [[ "$VERBOSE" == true ]] || [[ "$level" == "ERROR" ]]; then
        echo -e "${color}[${level}]${NC} ${message}"
    fi
}

# Error handler
error_handler() {
    local line_no=$1
    log "ERROR" "Error occurred at line ${line_no}" "${RED}"
    exit 1
}
trap 'error_handler ${LINENO}' ERR

# Cleanup
cleanup() {
    local exit_code=$?
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "${TEMP_DIR}" ]]; then
        rm -rf "${TEMP_DIR}"
    fi
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

# Usage
show_usage() {
    head -n 30 "${BASH_SOURCE[0]}" | grep -E '^#( |$)' | sed 's/^#//'
}

# Check dependencies
check_dependencies() {
    local deps=("docker" "python3" "openssl")
    for cmd in "${deps[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Missing dependency: $cmd" "${RED}"
            exit 3
        fi
    done
    log "INFO" "All dependencies satisfied" "${GREEN}"
}

# Generate secure password
generate_password() {
    openssl rand -base64 32 | tr -d '\n'
}

# Setup environment file
setup_env_file() {
    log "INFO" "Setting up environment configuration..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would create environment file from template" "${YELLOW}"
        return 0
    fi
    
    # Create from template if not exists
    if [[ ! -f "${ENV_FILE}" ]]; then
        if [[ -f "${ENV_EXAMPLE}" ]]; then
            cp "${ENV_EXAMPLE}" "${ENV_FILE}"
            log "INFO" "Created .env from .env.example" "${GREEN}"
        else
            log "ERROR" "No .env.example found" "${RED}"
            exit 5
        fi
    fi
    
    # Generate secure passwords for critical services
    local db_pass=$(generate_password)
    local redis_pass=$(generate_password)
    local rabbitmq_pass=$(generate_password)
    local jwt_secret=$(generate_password)
    
    # Update passwords in .env file (remove hardcoded values)
    sed -i.bak \
        -e "s/sutazai_secure_2024/${db_pass}/g" \
        -e "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${db_pass}/" \
        -e "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=${redis_pass}/" \
        -e "s/RABBITMQ_PASSWORD=.*/RABBITMQ_PASSWORD=${rabbitmq_pass}/" \
        -e "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${jwt_secret}/" \
        "${ENV_FILE}"
    
    # Secure the file
    chmod 600 "${ENV_FILE}"
    
    # Store passwords securely
    mkdir -p "${SECRETS_DIR}"
    chmod 700 "${SECRETS_DIR}"
    
    cat > "${SECRETS_DIR}/database.secret" << EOF
POSTGRES_PASSWORD=${db_pass}
REDIS_PASSWORD=${redis_pass}
RABBITMQ_PASSWORD=${rabbitmq_pass}
EOF
    chmod 600 "${SECRETS_DIR}/database.secret"
    
    log "INFO" "Environment configuration completed with secure passwords" "${GREEN}"
}

# Setup Python virtual environment
setup_python_env() {
    log "INFO" "Setting up Python virtual environment..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would setup Python virtual environment" "${YELLOW}"
        return 0
    fi
    
    # Backend venv
    if [[ ! -d "${PROJECT_ROOT}/backend/venv" ]]; then
        python3 -m venv "${PROJECT_ROOT}/backend/venv"
        "${PROJECT_ROOT}/backend/venv/bin/pip" install --upgrade pip
        "${PROJECT_ROOT}/backend/venv/bin/pip" install -r "${PROJECT_ROOT}/backend/requirements.txt"
        log "INFO" "Backend Python environment created" "${GREEN}"
    fi
    
    # Frontend venv
    if [[ ! -d "${PROJECT_ROOT}/frontend/venv" ]]; then
        python3 -m venv "${PROJECT_ROOT}/frontend/venv"
        "${PROJECT_ROOT}/frontend/venv/bin/pip" install --upgrade pip
        "${PROJECT_ROOT}/frontend/venv/bin/pip" install -r "${PROJECT_ROOT}/frontend/requirements.txt"
        log "INFO" "Frontend Python environment created" "${GREEN}"
    fi
}

# Setup Docker networks
setup_docker_networks() {
    log "INFO" "Setting up Docker networks..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would create Docker networks" "${YELLOW}"
        return 0
    fi
    
    # Create network if not exists
    if ! docker network inspect sutazai-network &>/dev/null; then
        docker network create \
            --driver bridge \
            --subnet=172.20.0.0/16 \
            --gateway=172.20.0.1 \
            sutazai-network
        log "INFO" "Docker network created" "${GREEN}"
    else
        log "INFO" "Docker network already exists" "${YELLOW}"
    fi
}

# Validate environment
validate_environment() {
    log "INFO" "Validating environment setup..." "${BLUE}"
    
    local errors=0
    
    # Check .env file
    if [[ ! -f "${ENV_FILE}" ]]; then
        log "ERROR" ".env file not found" "${RED}"
        ((errors++))
    fi
    
    # Check no hardcoded passwords
    if grep -q "sutazai_secure_2024" "${ENV_FILE}" 2>/dev/null; then
        log "ERROR" "Hardcoded passwords found in .env file!" "${RED}"
        ((errors++))
    fi
    
    # Check Docker
    if ! docker info &>/dev/null; then
        log "ERROR" "Docker is not running" "${RED}"
        ((errors++))
    fi
    
    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Environment validation failed with $errors errors" "${RED}"
        exit 1
    fi
    
    log "INFO" "Environment validation passed" "${GREEN}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -e|--env)
            ENVIRONMENT="${2:-dev}"
            shift 2
            ;;
        *)
            log "ERROR" "Unknown option: $1" "${RED}"
            show_usage
            exit 2
            ;;
    esac
done

# Main execution
main() {
    mkdir -p "$(dirname "${LOG_FILE}")"
    
    log "INFO" "Starting environment setup for ${ENVIRONMENT}" "${BLUE}"
    
    check_dependencies
    setup_env_file
    setup_python_env
    setup_docker_networks
    validate_environment
    
    log "INFO" "Environment setup completed successfully!" "${GREEN}"
    echo -e "\n${GREEN}✓${NC} Environment is ready for development"
    echo -e "${YELLOW}Note:${NC} Secure passwords have been generated and stored in ${SECRETS_DIR}"
    echo -e "${YELLOW}Important:${NC} Please update your database connection strings with the new passwords"
}

main