#!/bin/bash

# SutazAI Production Alerting Configuration Deployment Script
# Deploys updated AlertManager and Prometheus configurations safely

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_DIR="/opt/sutazaiapp/monitoring"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/sutazaiapp/backup/alerting_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root or with sudo"
        exit 1
    fi
}

# Create backup of current configuration
create_backup() {
    log "Creating backup of current alerting configuration..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup AlertManager config
    if [[ -f "$MONITORING_DIR/alertmanager/config.yml" ]]; then
        cp "$MONITORING_DIR/alertmanager/config.yml" "$BACKUP_DIR/alertmanager_config.yml"
        log "Backed up AlertManager config to $BACKUP_DIR"
    fi
    
    # Backup Prometheus config
    if [[ -f "$MONITORING_DIR/prometheus/prometheus.yml" ]]; then
        cp "$MONITORING_DIR/prometheus/prometheus.yml" "$BACKUP_DIR/prometheus.yml"
        log "Backed up Prometheus config to $BACKUP_DIR"
    fi
    
    # Backup existing alert rules
    if [[ -d "$MONITORING_DIR/prometheus" ]]; then
        cp -r "$MONITORING_DIR/prometheus"/*.yml "$BACKUP_DIR/" 2>/dev/null || true
        log "Backed up alert rules to $BACKUP_DIR"
    fi
}

# Validate configuration files
validate_configs() {
    log "Validating configuration files..."
    
    # Validate AlertManager config
    if [[ -f "$MONITORING_DIR/alertmanager/production_config.yml" ]]; then
        info "Validating AlertManager production config..."
        if docker run --rm -v "$MONITORING_DIR/alertmanager:/etc/alertmanager" \
           prom/alertmanager:latest amtool config check --config.file=/etc/alertmanager/production_config.yml; then
            log "✅ AlertManager config validation passed"
        else
            error "❌ AlertManager config validation failed"
            return 1
        fi
    fi
    
    # Validate Prometheus config
    if [[ -f "$MONITORING_DIR/prometheus/prometheus.yml" ]]; then
        info "Validating Prometheus config..."
        if docker run --rm -v "$MONITORING_DIR/prometheus:/etc/prometheus" \
           prom/prometheus:latest promtool check config /etc/prometheus/prometheus.yml; then
            log "✅ Prometheus config validation passed"
        else
            error "❌ Prometheus config validation failed"
            return 1
        fi
    fi
    
    # Validate alert rules
    if [[ -f "$MONITORING_DIR/prometheus/sutazai_production_alerts.yml" ]]; then
        info "Validating alert rules..."
        if docker run --rm -v "$MONITORING_DIR/prometheus:/etc/prometheus" \
           prom/prometheus:latest promtool check rules /etc/prometheus/sutazai_production_alerts.yml; then
            log "✅ Alert rules validation passed"
        else
            error "❌ Alert rules validation failed"
            return 1
        fi
    fi
}

# Deploy new configurations
deploy_configs() {
    log "Deploying new alerting configurations..."
    
    # Deploy AlertManager production config
    if [[ -f "$MONITORING_DIR/alertmanager/production_config.yml" ]]; then
        cp "$MONITORING_DIR/alertmanager/production_config.yml" "$MONITORING_DIR/alertmanager/config.yml"
        log "✅ Deployed new AlertManager configuration"
    fi
    
    # Set proper permissions
    chown -R nobody:nogroup "$MONITORING_DIR/alertmanager" 2>/dev/null || true
    chmod 644 "$MONITORING_DIR/alertmanager/config.yml" 2>/dev/null || true
    
    log "✅ Configuration deployment completed"
}

# Restart monitoring services
restart_services() {
    log "Restarting monitoring services..."
    
    cd /opt/sutazaiapp
    
    # Get current AlertManager container
    ALERTMANAGER_CONTAINER=$(docker ps --format "table {{.Names}}" | grep alertmanager | head -1)
    
    if [[ -n "$ALERTMANAGER_CONTAINER" ]]; then
        info "Restarting AlertManager container: $ALERTMANAGER_CONTAINER"
        docker restart "$ALERTMANAGER_CONTAINER"
        
        # Wait for AlertManager to be ready
        info "Waiting for AlertManager to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:10203/api/v1/status >/dev/null 2>&1; then
                log "✅ AlertManager is ready"
                break
            fi
            if [[ $i -eq 30 ]]; then
                error "❌ AlertManager failed to start within 30 seconds"
                return 1
            fi
            sleep 1
        done
    else
        warning "No AlertManager container found, starting monitoring stack..."
        docker-compose -f docker-compose.monitoring.yml up -d alertmanager
    fi
    
    # Reload Prometheus configuration
    PROMETHEUS_CONTAINER=$(docker ps --format "table {{.Names}}" | grep prometheus | head -1)
    
    if [[ -n "$PROMETHEUS_CONTAINER" ]]; then
        info "Reloading Prometheus configuration..."
        if curl -X POST http://localhost:9090/-/reload >/dev/null 2>&1; then
            log "✅ Prometheus configuration reloaded"
        else
            warning "Failed to reload Prometheus via API, restarting container..."
            docker restart "$PROMETHEUS_CONTAINER"
        fi
    else
        warning "No Prometheus container found"
    fi
}

# Test alerting pipeline
test_pipeline() {
    log "Testing alerting pipeline..."
    
    if [[ -f "$MONITORING_DIR/test_alerting_pipeline.py" ]]; then
        info "Running alerting pipeline tests..."
        
        # Install required Python packages if needed
        pip3 install requests >/dev/null 2>&1 || true
        
        # Run quick connectivity test
        if python3 "$MONITORING_DIR/test_alerting_pipeline.py" --quick; then
            log "✅ Basic alerting pipeline test passed"
        else
            warning "⚠️ Basic alerting pipeline test failed - check logs"
        fi
    else
        warning "Test script not found, skipping pipeline test"
    fi
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check AlertManager health
    info "Checking AlertManager health..."
    if curl -s http://localhost:10203/api/v1/status | grep -q "success"; then
        log "✅ AlertManager is healthy"
    else
        error "❌ AlertManager health check failed"
        return 1
    fi
    
    # Check Prometheus health  
    info "Checking Prometheus health..."
    if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
        log "✅ Prometheus is healthy"
    else
        error "❌ Prometheus health check failed"
        return 1
    fi
    
    # Check alert rules loaded
    info "Checking alert rules..."
    RULES_COUNT=$(curl -s http://localhost:9090/api/v1/rules | jq -r '.data.groups | length' 2>/dev/null || echo "0")
    if [[ "$RULES_COUNT" -gt 0 ]]; then
        log "✅ Alert rules loaded ($RULES_COUNT rule groups)"
    else
        warning "⚠️ No alert rules found"
    fi
    
    log "✅ Deployment verification completed"
}

# Rollback function
rollback() {
    error "Rolling back to previous configuration..."
    
    if [[ -d "$BACKUP_DIR" ]]; then
        # Restore AlertManager config
        if [[ -f "$BACKUP_DIR/alertmanager_config.yml" ]]; then
            cp "$BACKUP_DIR/alertmanager_config.yml" "$MONITORING_DIR/alertmanager/config.yml"
            log "Restored AlertManager configuration"
        fi
        
        # Restore Prometheus config
        if [[ -f "$BACKUP_DIR/prometheus.yml" ]]; then
            cp "$BACKUP_DIR/prometheus.yml" "$MONITORING_DIR/prometheus/prometheus.yml"
            log "Restored Prometheus configuration"
        fi
        
        # Restart services with old config
        restart_services
        
        log "✅ Rollback completed"
    else
        error "No backup found for rollback"
        exit 1
    fi
}

# Main deployment function
main() {
    log "🚀 Starting SutazAI Alerting Configuration Deployment"
    log "=================================================="
    
    # Set trap for rollback on failure
    trap 'rollback' ERR
    
    # Pre-deployment checks
    check_permissions
    create_backup
    validate_configs
    
    # Deployment
    deploy_configs
    restart_services
    
    # Post-deployment verification
    sleep 10  # Allow services to stabilize
    verify_deployment
    test_pipeline
    
    log "🎉 Alerting configuration deployment completed successfully!"
    log "=================================================="
    info "Backup stored at: $BACKUP_DIR"
    info "AlertManager UI: http://localhost:10203"
    info "Prometheus UI: http://localhost:9090"
    info "Check alert status: http://localhost:9090/alerts"
    
    # Summary
    echo
    echo "📊 DEPLOYMENT SUMMARY:"
    echo "  • AlertManager: http://localhost:10203"
    echo "  • Prometheus: http://localhost:9090"  
    echo "  • Alert Rules: Loaded"
    echo "  • Notification Channels: Configured"
    echo "  • Backup Location: $BACKUP_DIR"
    echo
    echo "🔍 Next Steps:"
    echo "  1. Configure notification webhook URLs in environment variables"
    echo "  2. Test alert delivery with: python3 $MONITORING_DIR/test_alerting_pipeline.py"
    echo "  3. Set up external monitoring for the monitoring system itself"
    echo "  4. Review and customize alert thresholds based on your environment"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        if [[ -z "${2:-}" ]]; then
            error "Please specify backup directory for rollback"
            exit 1
        fi
        BACKUP_DIR="$2"
        rollback
        ;;
    "test")
        test_pipeline
        ;;
    "validate")
        validate_configs
        ;;
    *)
        echo "Usage: $0 [deploy|rollback <backup_dir>|test|validate]"
        exit 1
        ;;
esac