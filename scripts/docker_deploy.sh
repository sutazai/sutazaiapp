#!/bin/bash

# SutazAI Docker Deployment Script
# Advanced deployment with pre-checks, rollback, and monitoring

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/media/ai/SutazAI_Storage/SutazAI/v1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEPLOYMENT_LOG="/var/log/sutazai/docker_deployment_${TIMESTAMP}.log"
ROLLBACK_LOG="/var/log/sutazai/docker_rollback_${TIMESTAMP}.log"

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

    echo -e "[${level}] ${color}$message${NC}" | tee -a "$DEPLOYMENT_LOG"
}

# Error handling and rollback
handle_error() {
    local error_message="$1"
    log_message "ERROR" "$error_message"
    rollback_deployment
    exit 1
}

rollback_deployment() {
    log_message "WARN" "Initiating Docker deployment rollback..."
    
    {
        echo "Rollback Timestamp: $(date)"
        echo "Deployment Failure Details:"
        
        # Stop and remove current containers
        docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down || true
        
        # Restore previous configuration if backup exists
        if [ -f "$PROJECT_ROOT/.env.backup" ]; then
            mv "$PROJECT_ROOT/.env.backup" "$PROJECT_ROOT/.env"
            echo "- Environment configuration restored from backup"
        fi
        
        # Prune unused Docker resources
        docker system prune -f
        
    } >> "$ROLLBACK_LOG"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_message "INFO" "Running pre-deployment system checks..."
    
    # Run system verification script
    python3 "$PROJECT_ROOT/system_verify.py" || handle_error "System verification failed"
    
    # Check Docker and Docker Compose
    docker --version || handle_error "Docker is not installed"
    docker-compose --version || handle_error "Docker Compose is not installed"
    
    # Validate Docker Compose configuration
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config > /dev/null || handle_error "Invalid Docker Compose configuration"
}

# Pull latest images
pull_images() {
    log_message "INFO" "Pulling latest Docker images..."
    
    # Backup current .env file
    cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.backup"
    
    # Pull images with timeout
    timeout 300 docker-compose -f "$PROJECT_ROOT/docker-compose.yml" pull || handle_error "Failed to pull Docker images"
}

# Deploy services
deploy_services() {
    log_message "INFO" "Deploying SutazAI services..."
    
    # Stop existing containers
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down
    
    # Start new containers
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d || handle_error "Failed to start services"
    
    # Wait for services to stabilize
    sleep 30
    
    # Verify service health
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps || handle_error "Services not running correctly"
}

# Post-deployment checks
post_deployment_checks() {
    log_message "INFO" "Running post-deployment health checks..."
    
    # Check service health endpoints
    services=("backend" "postgres" "redis")
    for service in "${services[@]}"; do
        if ! docker-compose -f "$PROJECT_ROOT/docker-compose.yml" ps "$service" | grep -q "Up"; then
            handle_error "Service $service is not running"
        fi
    done
    
    # Run performance monitoring
    python3 "$PROJECT_ROOT/performance_monitor.py" --duration 5 &
}

# Main deployment function
main() {
    log_message "INFO" "Starting SutazAI Docker Deployment"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Trap errors
    trap 'handle_error "Deployment interrupted"' SIGINT SIGTERM
    
    # Run deployment stages
    pre_deployment_checks
    pull_images
    deploy_services
    post_deployment_checks
    
    log_message "SUCCESS" "Docker Deployment Completed Successfully!"
}

# Execute main deployment function
main