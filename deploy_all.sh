#!/bin/bash
set -euo pipefail

# Enhanced Deployment Script for Production

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function with color support
log() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "ERROR")   color="$RED" ;;
        "WARNING") color="$YELLOW" ;;
        "SUCCESS") color="$GREEN" ;;
        *)         color="$NC" ;;
    esac

    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${color}[$level]${NC} $message"
}

# Pre-deployment security checks
security_checks() {
    log "INFO" "Performing pre-deployment security checks"

    # Check .env file permissions
    if [ "$(stat -c '%a' .env)" != "600" ]; then
        log "WARNING" ".env file permissions are too open. Fixing..."
        chmod 600 .env
    fi

    # Validate required environment variables
    local required_vars=(
        "DB_HOST" "DB_PORT" "DB_USER" "DB_PASS" "DB_NAME"
        "REDIS_HOST" "REDIS_PORT" "SECRET_KEY"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var+x}" ]; then
            log "ERROR" "Required environment variable $var is not set"
            return 1
        fi
    done

    log "SUCCESS" "Security checks passed"
}

# Environment setup with advanced error handling
setup_environment() {
    log "INFO" "Setting up Production Python environment"
    
    # Create a virtual environment with error handling
    python3 -m venv /tmp/sutazai_prod_venv || {
        log "ERROR" "Failed to create production virtual environment"
        return 1
    }

    # Activate virtual environment
    # shellcheck disable=SC1091
    source /tmp/sutazai_prod_venv/bin/activate

    # Upgrade pip with retry mechanism
    for _ in {1..3}; do
        pip install --upgrade pip && break || {
            log "WARNING" "Pip upgrade failed. Retrying..."
            sleep 2
        }
    done

    # Install production dependencies
    local req_files=(
        "requirements-prod.txt"
        "requirements.txt"
    )

    local installed=false
    for req_file in "${req_files[@]}"; do
        if [ -f "$req_file" ]; then
            log "INFO" "Attempting to install dependencies from $req_file"
            if pip install -r "$req_file"; then
                log "SUCCESS" "Dependencies installed from $req_file"
                installed=true
                break
            else
                log "WARNING" "Failed to install dependencies from $req_file"
            fi
        fi
    done

    if [ "$installed" = false ]; then
        log "ERROR" "No valid requirements file found or dependencies could not be installed"
        deactivate
        return 1
    fi

    deactivate
    log "SUCCESS" "Production environment setup completed"
}

# Docker deployment with advanced error handling and logging
docker_deploy() {
    log "INFO" "Starting Production Docker deployment"

    # Validate docker-compose file
    if [ ! -f "docker-compose.yml" ]; then
        log "ERROR" "docker-compose.yml not found"
        return 1
    fi

    # Pull latest images
    docker-compose pull || {
        log "WARNING" "Failed to pull Docker images. Continuing with local images."
    }

    # Deploy using docker-compose with timeout
    timeout 600 docker-compose up -d || {
        log "ERROR" "Docker deployment failed"
        return 1
    }

    # Verify container health
    docker-compose ps | grep -q "Up" || {
        log "ERROR" "Not all containers are running"
        return 1
    }

    log "SUCCESS" "Production deployment completed successfully"
}

# Main deployment function with comprehensive error handling
main() {
    log "INFO" "Starting SutazAI Production Deployment"

    # Perform security checks
    security_checks || {
        log "ERROR" "Security checks failed"
        exit 1
    }

    # Setup production environment
    setup_environment || {
        log "ERROR" "Environment setup failed"
        exit 1
    }

    # Deploy Docker containers
    docker_deploy || {
        log "ERROR" "Docker deployment failed"
        exit 1
    }

    log "SUCCESS" "SutazAI Production Deployment completed successfully"
}

# Execute main deployment
main