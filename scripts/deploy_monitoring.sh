#!/bin/bash

# SutazAI Monitoring Stack Deployment Script
# Deploys comprehensive production monitoring for SutazAI

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/monitoring"
LOG_DIR="$PROJECT_ROOT/logs"
DEPLOYMENT_LOG="$LOG_DIR/monitoring_deployment_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$DEPLOYMENT_LOG"
}

info() { log "INFO" "$@"; }
warn() { log "WARN" "${YELLOW}$@${NC}"; }
error() { log "ERROR" "${RED}$@${NC}"; }
success() { log "SUCCESS" "${GREEN}$@${NC}"; }

print_banner() {
    echo -e "${PURPLE}"
    echo "=============================================================="
    echo "          SutazAI Monitoring Stack Deployment"
    echo "=============================================================="
    echo -e "${NC}"
}

check_dependencies() {
    info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in docker docker-compose curl; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    success "All dependencies satisfied"
}

verify_monitoring_structure() {
    info "Verifying monitoring directory structure..."
    
    local required_dirs=(
        "$MONITORING_DIR/prometheus"
        "$MONITORING_DIR/grafana/dashboards"
        "$MONITORING_DIR/grafana/provisioning"
        "$MONITORING_DIR/loki"
        "$MONITORING_DIR/promtail"
        "$MONITORING_DIR/alertmanager"
        "$MONITORING_DIR/blackbox"
        "$MONITORING_DIR/ai-metrics-exporter"
    )
    
    local required_files=(
        "$MONITORING_DIR/prometheus/prometheus.yml"
        "$MONITORING_DIR/prometheus/alert_rules.yml"
        "$MONITORING_DIR/prometheus/ai_model_rules.yml"
        "$MONITORING_DIR/prometheus/business_rules.yml"
        "$MONITORING_DIR/grafana/provisioning/datasources/prometheus.yml"
        "$MONITORING_DIR/grafana/provisioning/dashboards/dashboards.yml"
        "$MONITORING_DIR/loki/config.yml"
        "$MONITORING_DIR/promtail/config.yml"
        "$MONITORING_DIR/alertmanager/config.yml"
        "$MONITORING_DIR/blackbox/config.yml"
        "$MONITORING_DIR/ai-metrics-exporter/Dockerfile"
        "$MONITORING_DIR/ai-metrics-exporter/ai_metrics_exporter.py"
    )
    
    local missing_items=()
    
    # Check directories
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing_items+=("Directory: $dir")
        fi
    done
    
    # Check files
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_items+=("File: $file")
        fi
    done
    
    if [ ${#missing_items[@]} -ne 0 ]; then
        error "Missing monitoring components:"
        for item in "${missing_items[@]}"; do
            error "  - $item"
        done
        exit 1
    fi
    
    success "Monitoring structure verified"
}

create_environment_file() {
    info "Creating monitoring environment configuration..."
    
    local env_file="$PROJECT_ROOT/.env.monitoring"
    
    if [ ! -f "$env_file" ]; then
        cat > "$env_file" << EOF
# SutazAI Monitoring Configuration
# Generated on $(date)

# Grafana Configuration
GRAFANA_PASSWORD=sutazai_grafana_$(openssl rand -hex 8)

# Alerting Configuration
SLACK_WEBHOOK_URL=
SLACK_AI_WEBHOOK_URL=
SLACK_SECURITY_WEBHOOK_URL=
PAGERDUTY_SERVICE_KEY=

# Monitoring Retention
PROMETHEUS_RETENTION=30d
LOKI_RETENTION=720h

# Resource Limits
PROMETHEUS_MEMORY_LIMIT=2G
GRAFANA_MEMORY_LIMIT=512M
LOKI_MEMORY_LIMIT=1G

# Custom Metrics
AI_METRICS_COLLECTION_INTERVAL=30
ENABLE_AI_METRICS=true
ENABLE_BUSINESS_METRICS=true
ENABLE_SECURITY_METRICS=true
EOF
        success "Created monitoring environment file: $env_file"
    else
        info "Using existing monitoring environment file: $env_file"
    fi
}

validate_docker_compose() {
    info "Validating Docker Compose configuration..."
    
    cd "$PROJECT_ROOT"
    
    if docker-compose config &> /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration is invalid"
        docker-compose config
        exit 1
    fi
}

build_custom_images() {
    info "Building custom monitoring images..."
    
    cd "$PROJECT_ROOT"
    
    # Build AI metrics exporter
    info "Building AI metrics exporter..."
    if docker-compose build ai-metrics-exporter; then
        success "AI metrics exporter built successfully"
    else
        error "Failed to build AI metrics exporter"
        exit 1
    fi
}

deploy_monitoring_stack() {
    info "Deploying monitoring stack..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy monitoring services
    local monitoring_services=(
        "prometheus"
        "grafana"
        "loki"
        "promtail"
        "alertmanager"
        "blackbox-exporter"
        "node-exporter"
        "cadvisor"
        "postgres-exporter"
        "redis-exporter"
        "neo4j-exporter"
        "ai-metrics-exporter"
    )
    
    info "Starting monitoring services..."
    if docker-compose up -d "${monitoring_services[@]}"; then
        success "Monitoring services started successfully"
    else
        error "Failed to start monitoring services"
        exit 1
    fi
    
    info "Waiting for services to be ready..."
    sleep 30
}

verify_deployment() {
    info "Verifying monitoring deployment..."
    
    local endpoints=(
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:3100/ready:Loki"
        "http://localhost:9093/-/healthy:Alertmanager"
        "http://localhost:9100/metrics:Node Exporter"
        "http://localhost:8080/healthz:cAdvisor"
        "http://localhost:9187/metrics:Postgres Exporter"
        "http://localhost:9121/metrics:Redis Exporter"
        "http://localhost:9200/health:AI Metrics Exporter"
    )
    
    local failed_services=()
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint service <<< "$endpoint_info"
        
        info "Checking $service at $endpoint..."
        
        if curl -sf "$endpoint" &> /dev/null; then
            success "$service is healthy"
        else
            warn "$service is not responding at $endpoint"
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        success "All monitoring services are healthy"
    else
        warn "Some services are not responding: ${failed_services[*]}"
        warn "This may be normal if services are still starting up"
    fi
}

configure_dashboards() {
    info "Configuring Grafana dashboards..."
    
    # Wait for Grafana to be fully ready
    local grafana_url="http://localhost:3000"
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$grafana_url/api/health" &> /dev/null; then
            break
        fi
        info "Waiting for Grafana to be ready... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        warn "Grafana did not become ready within expected time"
        return 1
    fi
    
    success "Grafana is ready and dashboards should be automatically provisioned"
}

print_access_info() {
    echo -e "${CYAN}"
    echo "=============================================================="
    echo "          SutazAI Monitoring Access Information"
    echo "=============================================================="
    echo -e "${NC}"
    
    echo -e "${GREEN}✓ Prometheus:${NC} http://localhost:9090"
    echo "  - Metrics collection and alerting rules"
    echo "  - Query interface for metrics exploration"
    echo
    
    echo -e "${GREEN}✓ Grafana:${NC} http://localhost:3000"
    echo "  - Username: admin"
    echo "  - Password: Check .env.monitoring file"
    echo "  - Pre-configured dashboards for SutazAI monitoring"
    echo
    
    echo -e "${GREEN}✓ Alertmanager:${NC} http://localhost:9093"
    echo "  - Alert management and routing"
    echo "  - Configure Slack/PagerDuty in alertmanager config"
    echo
    
    echo -e "${GREEN}✓ Loki:${NC} http://localhost:3100"
    echo "  - Log aggregation and querying"
    echo "  - Integrated with Grafana for log exploration"
    echo
    
    echo -e "${YELLOW}📊 Available Dashboards:${NC}"
    echo "  - SutazAI System Overview"
    echo "  - AI Models Performance"
    echo "  - Infrastructure Overview"
    echo "  - Business Metrics"
    echo
    
    echo -e "${YELLOW}⚠️  Configuration Notes:${NC}"
    echo "  1. Update Slack webhook URLs in .env.monitoring"
    echo "  2. Configure PagerDuty service key for critical alerts"
    echo "  3. Review alert thresholds in Prometheus rules"
    echo "  4. Customize dashboards based on your specific needs"
    echo
}

cleanup_on_error() {
    error "Deployment failed, cleaning up..."
    cd "$PROJECT_ROOT"
    docker-compose down prometheus grafana loki promtail alertmanager blackbox-exporter node-exporter cadvisor postgres-exporter redis-exporter neo4j-exporter ai-metrics-exporter || true
}

main() {
    print_banner
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    check_dependencies
    verify_monitoring_structure
    create_environment_file
    validate_docker_compose
    build_custom_images
    deploy_monitoring_stack
    verify_deployment
    configure_dashboards
    
    success "✅ SutazAI monitoring stack deployed successfully!"
    print_access_info
    
    info "Deployment completed. Check the logs at: $DEPLOYMENT_LOG"
}

# Run main function
main "$@"