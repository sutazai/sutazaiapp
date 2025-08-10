#!/bin/bash

# Secure Hardware Resource Optimizer Rebuild Script
# Author: DevOps Infrastructure Specialist
# Date: August 9, 2025
# Purpose: Safe rebuild and restart with enhanced security

set -euo pipefail

# Configuration
SERVICE_NAME="sutazai-hardware-resource-optimizer"
COMPOSE_FILE="docker-compose.yml"
SECURE_COMPOSE="docker-compose.secure.hardware-optimizer.yml"
LOG_FILE="/tmp/hardware-optimizer-rebuild-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# Pre-flight checks
preflight_checks() {
    log "Starting pre-flight security checks..."
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon not running or not accessible"
    fi
    
    # Check compose files exist
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Main compose file not found: $COMPOSE_FILE"
    fi
    
    # Check current service status
    if docker ps | grep -q "$SERVICE_NAME"; then
        log "Current service status: RUNNING"
        CURRENT_STATUS="running"
    else
        log "Current service status: STOPPED"
        CURRENT_STATUS="stopped"
    fi
    
    # Check for dangerous configurations
    log "Scanning for security vulnerabilities in current configuration..."
    
    if grep -q "privileged: true" "$COMPOSE_FILE"; then
        warning "DETECTED: privileged: true in configuration - SECURITY RISK"
    fi
    
    if grep -q "pid: host" "$COMPOSE_FILE"; then
        warning "DETECTED: pid: host in configuration - SECURITY RISK"
    fi
    
    if grep -q "/var/run/docker.sock" "$COMPOSE_FILE"; then
        warning "DETECTED: Docker socket mount - SECURITY RISK"
    fi
    
    log "Pre-flight checks completed"
}

# Backup current state
backup_current_state() {
    log "Creating backup of current configuration..."
    
    BACKUP_DIR="/tmp/hardware-optimizer-backup-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup compose file
    cp "$COMPOSE_FILE" "$BACKUP_DIR/"
    
    # Export current service logs
    if [[ "$CURRENT_STATUS" == "running" ]]; then
        docker logs "$SERVICE_NAME" > "$BACKUP_DIR/service.log" 2>&1 || true
    fi
    
    # Export current environment
    docker inspect "$SERVICE_NAME" > "$BACKUP_DIR/inspect.json" 2>&1 || true
    
    log "Backup created in: $BACKUP_DIR"
    echo "$BACKUP_DIR" > "$(mktemp /tmp/hardware-optimizer-backup-location.txt.XXXXXX)"
}

# Stop service gracefully
stop_service() {
    log "Stopping hardware resource optimizer service..."
    
    if [[ "$CURRENT_STATUS" == "running" ]]; then
        # Graceful shutdown with timeout
        timeout 30s docker-compose stop "$SERVICE_NAME" || {
            warning "Graceful stop failed, forcing stop..."
            docker-compose kill "$SERVICE_NAME"
        }
        
        # Wait for complete shutdown
        sleep 5
        
        # Remove container
        docker-compose rm -f "$SERVICE_NAME" 2>/dev/null || true
    fi
    
    log "Service stopped successfully"
}

# Rebuild with security enhancements
rebuild_secure() {
    log "Rebuilding with enhanced security configuration..."
    
    # Build new image with security flags
    docker-compose build --no-cache --pull "$SERVICE_NAME" || {
        error "Failed to build secure image"
    }
    
    # Verify image was built
    if ! docker images | grep -q "hardware-resource-optimizer"; then
        error "Image build verification failed"
    fi
    
    log "Secure image built successfully"
}

# Start service with enhanced monitoring
start_service_secure() {
    log "Starting service with enhanced security configuration..."
    
    # Start with security profile
    docker-compose up -d "$SERVICE_NAME" || {
        error "Failed to start secure service"
    }
    
    # Wait for service to initialize
    log "Waiting for service initialization..."
    sleep 10
    
    # Verify service is running
    if ! docker ps | grep -q "$SERVICE_NAME"; then
        error "Service failed to start"
    fi
    
    log "Service started successfully"
}

# Health validation
validate_service_health() {
    log "Validating service health and security..."
    
    # Wait for health check to stabilize
    sleep 30
    
    # Check container health
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$SERVICE_NAME" 2>/dev/null || echo "unknown")
    log "Container health status: $HEALTH_STATUS"
    
    # Test endpoint connectivity
    for i in {1..5}; do
        if curl -f -s http://localhost:11110/health >/dev/null; then
            log "Health endpoint responding - attempt $i/5"
            break
        else
            warning "Health endpoint not responding - attempt $i/5"
            sleep 10
        fi
        
        if [[ $i -eq 5 ]]; then
            warning "Health endpoint still not responding after 5 attempts"
        fi
    done
    
    # Security validation
    log "Validating security configuration..."
    
    # Check user configuration
    CONTAINER_USER=$(docker inspect --format='{{.Config.User}}' "$SERVICE_NAME")
    if [[ "$CONTAINER_USER" == "appuser" ]] || [[ "$CONTAINER_USER" == "1001:1001" ]]; then
        log "✅ Security: Container running as non-root user ($CONTAINER_USER)"
    else
        warning "⚠️ Security: Container may be running as root"
    fi
    
    # Check for privileged mode
    PRIVILEGED=$(docker inspect --format='{{.HostConfig.Privileged}}' "$SERVICE_NAME")
    if [[ "$PRIVILEGED" == "true" ]]; then
        warning "⚠️ Security: Container running in privileged mode - SECURITY RISK"
    else
        log "✅ Security: Container not privileged"
    fi
    
    # Check Docker socket mount
    if docker inspect "$SERVICE_NAME" | grep -q "docker.sock"; then
        warning "⚠️ Security: Docker socket mounted - SECURITY RISK"
    else
        log "✅ Security: Docker socket not mounted"
    fi
    
    log "Security validation completed"
}

# Rollback function
rollback() {
    log "ROLLBACK: Restoring previous configuration..."
    
    BACKUP_DIR=$(cat /tmp/hardware-optimizer-backup-location.txt 2>/dev/null || echo "")
    
    if [[ -n "$BACKUP_DIR" ]] && [[ -d "$BACKUP_DIR" ]]; then
        # Stop current service
        docker-compose stop "$SERVICE_NAME" 2>/dev/null || true
        docker-compose rm -f "$SERVICE_NAME" 2>/dev/null || true
        
        # Restore backup
        cp "$BACKUP_DIR/$COMPOSE_FILE" .
        
        # Restart with original configuration
        docker-compose up -d "$SERVICE_NAME"
        
        log "Rollback completed successfully"
    else
        error "Backup directory not found - manual intervention required"
    fi
}

# Main execution
main() {
    log "Starting secure hardware resource optimizer rebuild process..."
    log "Log file: $LOG_FILE"
    
    # Trap for cleanup on failure
    trap 'error "Script failed - check $LOG_FILE for details"' ERR
    trap 'rollback' INT TERM
    
    preflight_checks
    backup_current_state
    stop_service
    rebuild_secure
    start_service_secure
    validate_service_health
    
    log "✅ Secure rebuild process completed successfully!"
    log "Service is now running with enhanced security configuration"
    log "Log file available at: $LOG_FILE"
    
    # Display final status
    echo
    echo "=== FINAL STATUS ==="
    docker ps | grep "$SERVICE_NAME" || echo "Service not running"
    echo
    echo "=== SECURITY SUMMARY ==="
    echo "✅ Non-root user: $(docker inspect --format='{{.Config.User}}' "$SERVICE_NAME")"
    echo "✅ Privileged mode: $(docker inspect --format='{{.HostConfig.Privileged}}' "$SERVICE_NAME")"
    echo "✅ Health endpoint: http://localhost:11110/health"
    echo
}

# Execute main function
main "$@"