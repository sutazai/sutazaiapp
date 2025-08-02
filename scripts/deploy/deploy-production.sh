#!/bin/bash
#
# SutazAI Production Deployment Script
# Version: 2.0.0
# 
# This script is a production-optimized wrapper around the master deploy.sh
# It includes additional safety checks, validation, and production-specific
# configurations required for enterprise deployment.
#

set -euo pipefail

# Production deployment configuration
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_NAME="SutazAI Production Deployment"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MASTER_DEPLOY="$PROJECT_ROOT/deploy.sh"

# Production requirements
declare -A PRODUCTION_REQUIREMENTS=(
    ["memory_gb"]="64"
    ["disk_gb"]="1000"
    ["cpu_cores"]="16"
)

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# ===============================================
# PRODUCTION SAFETY CHECKS
# ===============================================

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

production_safety_banner() {
    echo -e "\n${RED}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}${BOLD}║                      PRODUCTION DEPLOYMENT                  ║${NC}"
    echo -e "${RED}${BOLD}║                                                              ║${NC}"
    echo -e "${RED}${BOLD}║  ⚠️  WARNING: This will deploy to PRODUCTION environment  ⚠️   ║${NC}"
    echo -e "${RED}${BOLD}║                                                              ║${NC}"
    echo -e "${RED}${BOLD}║  This action will affect live systems and users.            ║${NC}"
    echo -e "${RED}${BOLD}║  Ensure you have proper authorization and backups.          ║${NC}"
    echo -e "${RED}${BOLD}║                                                              ║${NC}"
    echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}\n"
}

verify_production_authorization() {
    log_info "Verifying production deployment authorization..."
    
    # Check for production authorization token
    if [[ -z "${PRODUCTION_DEPLOY_TOKEN:-}" ]]; then
        log_error "PRODUCTION_DEPLOY_TOKEN environment variable is required"
        log_error "This token should be provided by your system administrator"
        exit 1
    fi
    
    # Verify token format (should be a secure hash)
    if [[ ! "$PRODUCTION_DEPLOY_TOKEN" =~ ^[a-fA-F0-9]{64}$ ]]; then
        log_error "Invalid PRODUCTION_DEPLOY_TOKEN format"
        exit 1
    fi
    
    log_success "Production authorization verified"
}

verify_production_environment() {
    log_info "Verifying production environment requirements..."
    
    # Check system resources
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || echo "0")
    
    local memory_gb
    if [[ "$(uname -s)" == "Linux" ]]; then
        memory_gb=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    else
        memory_gb="0"
    fi
    
    local disk_gb
    disk_gb=$(df -BG "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    
    # Validate production requirements
    local errors=()
    
    if [[ $cpu_cores -lt ${PRODUCTION_REQUIREMENTS[cpu_cores]} ]]; then
        errors+=("CPU cores: $cpu_cores < ${PRODUCTION_REQUIREMENTS[cpu_cores]} required")
    fi
    
    if [[ $memory_gb -lt ${PRODUCTION_REQUIREMENTS[memory_gb]} ]]; then
        errors+=("Memory: ${memory_gb}GB < ${PRODUCTION_REQUIREMENTS[memory_gb]}GB required")
    fi
    
    if [[ $disk_gb -lt ${PRODUCTION_REQUIREMENTS[disk_gb]} ]]; then
        errors+=("Disk space: ${disk_gb}GB < ${PRODUCTION_REQUIREMENTS[disk_gb]}GB required")
    fi
    
    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "Production environment requirements not met:"
        for error in "${errors[@]}"; do
            log_error "  - $error"
        done
        exit 1
    fi
    
    log_success "Production environment requirements verified"
}

verify_network_connectivity() {
    log_info "Verifying network connectivity and security..."
    
    # Check internet connectivity
    if ! curl -s --max-time 10 https://google.com >/dev/null; then
        log_error "No internet connectivity detected"
        exit 1
    fi
    
    # Check DNS resolution
    if ! nslookup google.com >/dev/null 2>&1; then
        log_error "DNS resolution failed"
        exit 1
    fi
    
    # Verify SSL/TLS capabilities
    if ! openssl version >/dev/null 2>&1; then
        log_error "OpenSSL not available for SSL/TLS operations"
        exit 1
    fi
    
    log_success "Network connectivity verified"
}

verify_security_requirements() {
    log_info "Verifying security requirements..."
    
    # Check for required security tools
    local security_tools=("openssl" "gpg" "ssh-keygen")
    
    for tool in "${security_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required security tool not found: $tool"
            exit 1
        fi
    done
    
    # Verify firewall is available
    if ! command -v ufw >/dev/null 2>&1 && ! command -v firewall-cmd >/dev/null 2>&1; then
        log_warn "No firewall tool detected (ufw or firewalld recommended for production)"
    fi
    
    # Check for SELinux/AppArmor
    if command -v getenforce >/dev/null 2>&1; then
        local selinux_status
        selinux_status=$(getenforce 2>/dev/null || echo "Unknown")
        log_info "SELinux status: $selinux_status"
    fi
    
    log_success "Security requirements verified"
}

create_production_backup() {
    log_info "Creating pre-deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/production"
    local backup_file="$backup_dir/pre_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    mkdir -p "$backup_dir"
    
    # Create comprehensive backup
    tar -czf "$backup_file" \
        --exclude='logs' \
        --exclude='data/postgres' \
        --exclude='data/ollama' \
        --exclude='venv' \
        --exclude='node_modules' \
        -C "$PROJECT_ROOT" \
        . 2>/dev/null || {
        log_error "Failed to create pre-deployment backup"
        exit 1
    }
    
    log_success "Pre-deployment backup created: $backup_file"
    export PRODUCTION_BACKUP_FILE="$backup_file"
}

setup_production_monitoring() {
    log_info "Setting up production monitoring configuration..."
    
    # Ensure monitoring configuration exists
    local monitoring_dir="$PROJECT_ROOT/monitoring"
    mkdir -p "$monitoring_dir/prometheus" "$monitoring_dir/grafana/provisioning" "$monitoring_dir/loki"
    
    # Create production Prometheus configuration
    cat > "$monitoring_dir/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    deployment: 'sutazai'

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'sutazai-services'
    static_configs:
      - targets: 
        - 'ollama:11434'
        - 'postgres:5432'
        - 'redis:6379'
        - 'chromadb:8000'
        - 'qdrant:6333'
EOF
    
    # Create alert rules
    cat > "$monitoring_dir/prometheus/alert_rules.yml" << 'EOF'
groups:
  - name: sutazai_alerts
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} service is down"
EOF
    
    log_success "Production monitoring configured"
}

verify_ssl_certificates() {
    log_info "Verifying SSL certificate configuration..."
    
    local ssl_dir="$PROJECT_ROOT/ssl"
    
    # For production, we should have proper certificates
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        log_warn "SSL certificates not found, will generate self-signed certificates"
        log_warn "For production, consider using proper SSL certificates from a CA"
    else
        # Verify certificate validity
        if openssl x509 -in "$ssl_dir/cert.pem" -noout -checkend 86400 >/dev/null 2>&1; then
            log_success "SSL certificate is valid"
        else
            log_warn "SSL certificate expires within 24 hours"
        fi
    fi
}

configure_production_limits() {
    log_info "Configuring production resource limits..."
    
    # Set production environment variables
    export ENABLE_MONITORING=true
    export ENABLE_HEALTH_CHECKS=true
    export AUTO_ROLLBACK=true
    export LOG_LEVEL=INFO
    
    # Production-specific limits
    export OLLAMA_NUM_PARALLEL=4
    export OLLAMA_MAX_LOADED_MODELS=3
    export OLLAMA_KEEP_ALIVE=10m
    
    # Database connection limits
    export POSTGRES_MAX_CONNECTIONS=200
    export REDIS_MAXMEMORY=8gb
    export REDIS_MAXMEMORY_POLICY=allkeys-lru
    
    log_success "Production limits configured"
}

run_production_tests() {
    log_info "Running production readiness tests..."
    
    # Test Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon not accessible"
        exit 1
    fi
    
    # Test disk space for logs and data
    local available_space
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [[ $available_space -lt 100 ]]; then
        log_error "Insufficient disk space for production deployment"
        exit 1
    fi
    
    # Test memory availability
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    
    if [[ $available_memory -lt 16 ]]; then
        log_error "Insufficient available memory for production deployment"
        exit 1
    fi
    
    # Test network ports availability
    local required_ports=(80 443 5432 6379 8000 8501 11434)
    local used_ports=()
    
    for port in "${required_ports[@]}"; do
        if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            used_ports+=("$port")
        fi
    done
    
    if [[ ${#used_ports[@]} -gt 0 ]]; then
        log_warn "Ports already in use: ${used_ports[*]}"
        log_warn "This may cause conflicts during deployment"
    fi
    
    log_success "Production readiness tests passed"
}

# ===============================================
# DEPLOYMENT EXECUTION
# ===============================================

execute_production_deployment() {
    log_info "Executing production deployment with master script..."
    
    # Set production-specific environment variables
    export DEPLOYMENT_TARGET=production
    export FORCE_DEPLOY=${FORCE_DEPLOY:-false}
    
    # Execute the master deployment script
    if "$MASTER_DEPLOY" deploy production; then
        log_success "Production deployment completed successfully"
        return 0
    else
        log_error "Production deployment failed"
        return 1
    fi
}

post_production_validation() {
    log_info "Running post-deployment production validation..."
    
    # Wait for services to stabilize
    sleep 30
    
    # Run comprehensive health checks
    if "$MASTER_DEPLOY" health; then
        log_success "Production health validation passed"
    else
        log_error "Production health validation failed"
        return 1
    fi
    
    # Test external connectivity
    local external_tests=(
        "http://localhost:8501"
        "http://localhost:8000/health"
        "http://localhost:8000/docs"
    )
    
    for endpoint in "${external_tests[@]}"; do
        if curl -f -s --max-time 10 "$endpoint" >/dev/null; then
            log_success "External connectivity test passed: $endpoint"
        else
            log_error "External connectivity test failed: $endpoint"
            return 1
        fi
    done
    
    log_success "Post-deployment validation completed"
}

setup_production_cron_jobs() {
    log_info "Setting up production cron jobs..."
    
    # Create maintenance script
    cat > "$PROJECT_ROOT/scripts/production_maintenance.sh" << 'EOF'
#!/bin/bash
# Production maintenance script for SutazAI

# Cleanup old logs (keep 30 days)
find /opt/sutazaiapp/logs -name "*.log" -mtime +30 -delete

# Cleanup old backups (keep 7 days)
find /opt/sutazaiapp/backups -name "*.tar.gz" -mtime +7 -delete

# Docker system cleanup
docker system prune -f --filter "until=24h"

# Restart services if needed (health check)
if ! curl -f -s http://localhost:8000/health >/dev/null; then
    systemctl restart sutazai || /opt/sutazaiapp/deploy.sh deploy production
fi
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/production_maintenance.sh"
    
    # Add to crontab (run daily at 3 AM)
    (crontab -l 2>/dev/null; echo "0 3 * * * /opt/sutazaiapp/scripts/production_maintenance.sh") | sort -u | crontab -
    
    log_success "Production cron jobs configured"
}

generate_production_report() {
    log_info "Generating production deployment report..."
    
    local report_file="$PROJECT_ROOT/logs/production_deployment_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
SutazAI Production Deployment Report
===================================

Deployment Time: $(date)
Script Version: $SCRIPT_VERSION
System: $(uname -a)

SYSTEM RESOURCES
---------------
CPU Cores: $(nproc)
Memory: $(free -h | awk 'NR==2{print $2}')
Disk Space: $(df -h "$PROJECT_ROOT" | awk 'NR==2{print $4}') available

DEPLOYED SERVICES
----------------
$(docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")

NETWORK CONFIGURATION
--------------------
Local IP: $(hostname -I | awk '{print $1}')
External Access: https://$(hostname -f 2>/dev/null || hostname)

SECURITY STATUS
--------------
Firewall: $(command -v ufw >/dev/null && ufw status | head -1 || echo "Not configured")
SSL Certificates: $(test -f "$PROJECT_ROOT/ssl/cert.pem" && echo "Present" || echo "Self-signed")

MONITORING
---------
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
Health Check: http://localhost:8000/health

BACKUP INFORMATION
-----------------
Pre-deployment backup: ${PRODUCTION_BACKUP_FILE:-Not created}
Backup schedule: Daily at 3 AM
Retention: 7 days

MAINTENANCE
----------
Cron jobs: Configured for daily maintenance
Log rotation: 30 days retention
Docker cleanup: Daily

CONTACT INFORMATION
------------------
For support, contact: support@sutazai.com
Documentation: https://docs.sutazai.com
Status page: https://status.sutazai.com

EOF
    
    log_success "Production deployment report generated: $report_file"
    
    # Display summary
    echo -e "\n${GREEN}${BOLD}🚀 PRODUCTION DEPLOYMENT SUCCESSFUL! 🚀${NC}\n"
    echo -e "${BLUE}Report: $report_file${NC}"
    echo -e "${BLUE}Access: https://$(hostname -f 2>/dev/null || hostname -I | awk '{print $1}')${NC}"
    echo -e "${BLUE}Health: http://localhost:8000/health${NC}\n"
}

# ===============================================
# MAIN EXECUTION FLOW
# ===============================================

main() {
    # Show production warning
    production_safety_banner
    
    # Confirmation prompt
    echo -e "${YELLOW}Do you want to continue with PRODUCTION deployment? ${NC}"
    echo -e "${YELLOW}Type 'DEPLOY PRODUCTION' to confirm: ${NC}"
    read -r confirmation
    
    if [[ "$confirmation" != "DEPLOY PRODUCTION" ]]; then
        log_error "Production deployment cancelled by user"
        exit 1
    fi
    
    # Pre-deployment checks
    verify_production_authorization
    verify_production_environment
    verify_network_connectivity
    verify_security_requirements
    verify_ssl_certificates
    run_production_tests
    
    # Setup production environment
    create_production_backup
    setup_production_monitoring
    configure_production_limits
    
    # Execute deployment
    if execute_production_deployment; then
        # Post-deployment tasks
        post_production_validation
        setup_production_cron_jobs
        generate_production_report
        
        log_success "Production deployment completed successfully!"
        exit 0
    else
        log_error "Production deployment failed!"
        
        # Offer to restore backup
        echo -e "${YELLOW}Would you like to restore the pre-deployment backup? (y/N): ${NC}"
        read -r restore_backup
        
        if [[ "$restore_backup" =~ ^[Yy]$ ]] && [[ -n "${PRODUCTION_BACKUP_FILE:-}" ]]; then
            log_info "Restoring pre-deployment backup..."
            tar -xzf "$PRODUCTION_BACKUP_FILE" -C "$PROJECT_ROOT" 2>/dev/null || true
            log_success "Backup restored"
        fi
        
        exit 1
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi