#!/bin/bash

# Enhanced Local Deployment Script for SutazAI
# Updated for Python 3.11 compatibility

# Exit on any error
set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/media/ai/SutazAI_Storage/SutazAI/v1"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$HOME/sutazai_deployment_${TIMESTAMP}.log"
ROLLBACK_LOG="$HOME/sutazai_deployment_rollback_${TIMESTAMP}.log"

# Redirect all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Logging function
log_message() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "INFO") color="$BLUE" ;;
        "WARN") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        "SUCCESS") color="$GREEN" ;;
        *) color="$NC" ;;
    esac

    echo -e "[${level}] ${color}$message${NC}"
}

# Error handling function
handle_error() {
    local error_message="$1"
    log_message "ERROR" "$error_message"
    rollback_deployment
    exit 1
}

# Rollback function
rollback_deployment() {
    log_message "WARN" "Initiating deployment rollback..."
    
    {
        echo "Rollback Timestamp: $(date)"
        echo "Deployment Failure Details:"
        
        # Stop and remove Docker containers
        if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps | grep -q "Up"; then
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down
            echo "- Docker containers stopped and removed"
        fi
        
        # Revert to previous environment configuration
        if [ -f "$PROJECT_ROOT/.env.backup" ]; then
            mv "$PROJECT_ROOT/.env.backup" "$PROJECT_ROOT/.env"
            echo "- Environment configuration restored from backup"
        fi
        
        # Deactivate virtual environment if active
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
            echo "- Virtual environment deactivated"
        fi
        
    } >> "$ROLLBACK_LOG"
}

# Trap errors and call rollback function
trap 'handle_error "Deployment failed unexpectedly"' ERR

deploy_system_checks() {
    log_message "INFO" "Running system compatibility checks for Python 3.11..."
    
    # Run pre-deployment checklist
    "$PROJECT_ROOT/pre_deploy_checklist.sh" || handle_error "System compatibility checks failed"
}

setup_virtual_environment() {
    log_message "INFO" "Setting up Python 3.11 virtual environment..."
    
    # Backup current .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.backup"
    fi
    
    # Check if venv exists
    if [ ! -d "$PROJECT_ROOT/venv-3.11" ]; then
        python3.11 -m venv "$PROJECT_ROOT/venv-3.11"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv-3.11/bin/activate"
    
    # Upgrade pip and setuptools
    python3.11 -m pip install --upgrade pip setuptools wheel
    
    # Install production requirements
    python3.11 -m pip install -r "$PROJECT_ROOT/requirements-prod.txt" || handle_error "Failed to install dependencies"
}

configure_environment() {
    log_message "INFO" "Configuring environment..."
    
    # Copy .env if not exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    fi
    
    # Generate or update secrets
    python3.11 "$PROJECT_ROOT/config.py" --generate-secrets || handle_error "Failed to generate secrets"
}

start_services() {
    log_message "INFO" "Starting services..."
    
    # Validate Docker Compose file
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config > /dev/null || handle_error "Invalid Docker Compose configuration"
    
    # Pull latest images
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" pull || handle_error "Failed to pull Docker images"
    
    # Start services
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d || handle_error "Failed to start services"
    
    # Wait for services to be fully up
    sleep 15
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps || handle_error "Services not running correctly"
}

run_system_optimization() {
    log_message "INFO" "Optimizing system..."
    
    python3.11 "$PROJECT_ROOT/system_optimizer.py" || log_message "WARN" "System optimization script encountered issues"
}

run_database_migrations() {
    log_message "INFO" "Running database migrations..."
    
    # Activate virtual environment if not already active
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$PROJECT_ROOT/venv-3.11/bin/activate"
    fi
    
    # Run database migrations
    python3.11 -m alembic upgrade head || handle_error "Database migration failed"
}

main() {
    log_message "INFO" "Starting SutazAI Local Deployment with Python 3.11"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run deployment stages
    deploy_system_checks
    setup_virtual_environment
    configure_environment
    start_services
    run_database_migrations
    run_system_optimization
    
    log_message "SUCCESS" "Deployment Completed Successfully with Python 3.11!"
    
    # Optional: Run performance monitor
    python3.11 "$PROJECT_ROOT/performance_monitor.py" &
}

# Run main deployment function
main
