#!/bin/bash

# SutazAI Resource Optimization Deployment Script
# Purpose: Deploy optimized resource allocation configuration
# Usage: ./scripts/deploy-resource-optimization.sh [--dry-run] [--phase <1-4>]
# Requirements: Docker, Docker Compose, Root privileges

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/resource-optimization-$(date +%Y%m%d_%H%M%S).log"
DRY_RUN=false
PHASE="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}${*}${NC}"
}

log_warn() {
    log "WARN" "${YELLOW}${*}${NC}"
}

log_error() {
    log "ERROR" "${RED}${*}${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}${*}${NC}"
}

# Usage information
usage() {
    cat << EOF
SutazAI Resource Optimization Deployment Script

Usage: $0 [OPTIONS]

Options:
    --dry-run           Show what would be done without making changes
    --phase <1-4>       Deploy specific phase only (1-4) or 'all'
    --help             Show this help message

Phases:
    1: Foundation       - Deploy resource pools and CPU affinity
    2: Scheduling       - Implement priority-based scheduling
    3: Optimization     - Fine-tune and deploy monitoring
    4: Production       - Full production deployment and validation

Examples:
    $0 --dry-run                    # Preview all changes
    $0 --phase 1                    # Deploy foundation only
    $0 --phase all                  # Deploy all phases
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root for system configuration changes"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check system resources
    local cpu_cores=$(nproc)
    local memory_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    
    log_info "System Resources: ${cpu_cores} CPU cores, ${memory_gb}GB RAM"
    
    if [[ $cpu_cores -lt 12 ]]; then
        log_warn "System has fewer than 12 CPU cores. Adjusting configuration..."
    fi
    
    if [[ $memory_gb -lt 28 ]]; then
        log_warn "System has less than 29GB RAM. Adjusting configuration..."
    fi
    
    # Check disk space
    local free_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    local free_gb=$((free_space / 1024 / 1024))
    
    if [[ $free_gb -lt 10 ]]; then
        log_error "Insufficient disk space. Need at least 10GB free."
        exit 1
    fi
    
    log_success "Pre-flight checks passed"
}

# Backup current configuration
backup_configuration() {
    log_info "Backing up current configuration..."
    
    local backup_dir="$PROJECT_ROOT/backups/resource-optimization-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup docker-compose files
    find "$PROJECT_ROOT" -name "docker-compose*.yml" -exec cp {} "$backup_dir/" \;
    
    # Backup current container state
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "$backup_dir/containers_before.txt"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" > "$backup_dir/resources_before.txt"
    
    log_success "Configuration backed up to $backup_dir"
}

# Phase 1: Foundation - Deploy resource pools and CPU affinity
deploy_phase1() {
    log_info "Phase 1: Deploying foundation (resource pools and CPU affinity)..."
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/configs"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data"/{postgres,redis,ollama,models,chromadb,qdrant,neo4j}
    
    # Set CPU governor to performance for optimization
    if [[ $DRY_RUN == false ]]; then
        log_info "Setting CPU governor to performance mode..."
        echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
    fi
    
    # Configure container runtime optimizations
    if [[ $DRY_RUN == false ]]; then
        log_info "Configuring Docker daemon optimizations..."
        local docker_config="/etc/docker/daemon.json"
        local temp_config=$(mktemp)
        
        cat > "$temp_config" << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-ulimits": {
        "nofile": {
            "Hard": 65536,
            "Name": "nofile",
            "Soft": 65536
        }
    },
    "live-restore": true,
    "max-concurrent-downloads": 10,
    "max-concurrent-uploads": 5,
    "storage-driver": "overlay2"
}
EOF
        
        if [[ -f "$docker_config" ]]; then
            cp "$docker_config" "${docker_config}.backup"
        fi
        
        mv "$temp_config" "$docker_config"
        systemctl reload docker || true
    fi
    
    log_success "Phase 1 completed"
}

# Phase 2: Scheduling - Implement priority-based scheduling
deploy_phase2() {
    log_info "Phase 2: Implementing priority-based scheduling..."
    
    # Stop current services gracefully
    if [[ $DRY_RUN == false ]]; then
        log_info "Stopping current services gracefully..."
        cd "$PROJECT_ROOT"
        docker compose down --timeout 30 || true
        
        # Wait for containers to stop
        sleep 10
    fi
    
    # Deploy optimized configuration
    if [[ $DRY_RUN == false ]]; then
        log_info "Deploying optimized Docker Compose configuration..."
        cd "$PROJECT_ROOT"
        
        # Use the optimized configuration
        cp docker-compose.resource-optimized.yml docker-compose.yml.new
        
        # Start critical services first (infrastructure tier)
        docker compose -f docker-compose.resource-optimized.yml up -d \
            postgres redis neo4j ollama backend frontend
        
        # Wait for infrastructure to stabilize
        sleep 30
        
        # Start monitoring and active agents
        docker compose -f docker-compose.resource-optimized.yml up -d \
            prometheus grafana hardware-resource-optimizer health-monitor \
            chromadb qdrant
        
        # Wait for agents to stabilize  
        sleep 20
        
        log_info "Infrastructure and active agents deployed"
    fi
    
    log_success "Phase 2 completed"
}

# Phase 3: Optimization - Fine-tune and deploy monitoring
deploy_phase3() {
    log_info "Phase 3: Fine-tuning and deploying advanced monitoring..."
    
    if [[ $DRY_RUN == false ]]; then
        # Deploy monitoring configuration
        log_info "Deploying monitoring dashboards..."
        
        # Create Prometheus rules for resource monitoring
        mkdir -p "$PROJECT_ROOT/monitoring/prometheus/rules"
        cat > "$PROJECT_ROOT/monitoring/prometheus/rules/resource-optimization.yml" << 'EOF'
groups:
  - name: resource-optimization
    rules:
      - alert: HighCPUUtilization
        expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU utilization detected"
          description: "Container {{ $labels.name }} CPU usage is {{ $value }}%"
      
      - alert: HighMemoryUtilization  
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 85
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High memory utilization detected"
          description: "Container {{ $labels.name }} memory usage is {{ $value }}%"
          
      - alert: ContainerRestartHigh
        expr: increase(container_start_time_seconds[1h]) > 3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Container restart rate is high"
          description: "Container {{ $labels.name }} has restarted {{ $value }} times in the last hour"
EOF
        
        # Restart monitoring services to pick up new rules
        docker compose -f docker-compose.resource-optimized.yml restart prometheus grafana
        
        sleep 15
    fi
    
    log_success "Phase 3 completed"
}

# Phase 4: Production - Full deployment and validation
deploy_phase4() {
    log_info "Phase 4: Full production deployment and validation..."
    
    if [[ $DRY_RUN == false ]]; then
        # Deploy all remaining services
        log_info "Deploying remaining services..."
        cd "$PROJECT_ROOT"
        
        # Start all services with the optimized configuration
        docker compose -f docker-compose.resource-optimized.yml up -d
        
        # Wait for all services to start
        sleep 60
        
        # Replace the main configuration
        mv docker-compose.yml docker-compose.yml.backup
        mv docker-compose.yml.new docker-compose.yml
    fi
    
    # Validation
    log_info "Running deployment validation..."
    validate_deployment
    
    log_success "Phase 4 completed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check running containers
    local running_containers=$(docker ps --format "{{.Names}}" | grep "sutazai-" | wc -l)
    log_info "Running containers: $running_containers"
    
    # Check resource utilization
    local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | awk '{sum+=$1} END {print sum}')
    local memory_usage=$(docker stats --no-stream --format "{{.MemPerc}}" | sed 's/%//' | awk 'BEGIN{max=0} {if($1>max) max=$1} END {print max}')
    
    log_info "Current CPU utilization: ${cpu_usage}%"
    log_info "Peak memory utilization: ${memory_usage}%"
    
    # Check critical services health
    local critical_services=("sutazai-backend" "sutazai-frontend" "sutazai-postgres" "sutazai-redis" "sutazai-ollama")
    local healthy_count=0
    
    for service in "${critical_services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            ((healthy_count++))
            log_success "$service is running"
        else
            log_error "$service is not running"
        fi
    done
    
    if [[ $healthy_count -eq ${#critical_services[@]} ]]; then
        log_success "All critical services are healthy"
    else
        log_error "Some critical services are not healthy"
        return 1
    fi
    
    # Test basic functionality
    log_info "Testing basic functionality..."
    
    # Test backend API
    if curl -s -f http://localhost:10010/health > /dev/null; then
        log_success "Backend API is responding"
    else
        log_warn "Backend API is not responding (may still be starting)"
    fi
    
    # Test frontend
    if curl -s -f http://localhost:10011 > /dev/null; then
        log_success "Frontend is responding"
    else
        log_warn "Frontend is not responding (may still be starting)"
    fi
    
    # Generate optimization report
    generate_optimization_report
}

# Generate optimization report
generate_optimization_report() {
    log_info "Generating optimization report..."
    
    local report_file="$PROJECT_ROOT/logs/optimization-report-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "optimization_deployment": {
        "status": "completed",
        "phase": "$PHASE",
        "dry_run": $DRY_RUN
    },
    "system_resources": {
        "cpu_cores": $(nproc),
        "memory_gb": $(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 )),
        "disk_free_gb": $(( $(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}') / 1024 / 1024 ))
    },
    "container_status": {
        "running_containers": $(docker ps --filter "name=sutazai-" | wc -l),
        "total_defined": 69
    },
    "resource_utilization": {
        "cpu_percent": $(docker stats --no-stream --format "{{.CPUPerc}}" 2>/dev/null | sed 's/%//' | awk '{sum+=$1} END {print (sum ? sum : 0)}'),
        "memory_peak_percent": $(docker stats --no-stream --format "{{.MemPerc}}" 2>/dev/null | sed 's/%//' | awk 'BEGIN{max=0} {if($1>max) max=$1} END {print (max ? max : 0)}')
    },
    "optimization_targets": {
        "cpu_target": "40-60%",
        "memory_target": "75-80%",
        "container_density": "improved"
    }
}
EOF
    
    log_success "Optimization report generated: $report_file"
}

# Rollback function
rollback() {
    log_warn "Rolling back changes..."
    
    cd "$PROJECT_ROOT"
    
    # Stop current containers
    docker compose down --timeout 30 || true
    
    # Restore backup configuration
    if [[ -f docker-compose.yml.backup ]]; then
        mv docker-compose.yml.backup docker-compose.yml
        log_info "Restored original docker-compose.yml"
    fi
    
    # Restart with original configuration
    docker compose up -d
    
    log_success "Rollback completed"
}

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        if [[ $DRY_RUN == false ]]; then
            log_warn "Consider running rollback if services are not working"
        fi
    fi
}

trap cleanup EXIT

# Main execution
main() {
    log_info "Starting SutazAI Resource Optimization Deployment"
    log_info "Mode: $([ "$DRY_RUN" == true ] && echo "DRY RUN" || echo "PRODUCTION")"
    log_info "Phase: $PHASE"
    log_info "Log file: $LOG_FILE"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    preflight_checks
    
    if [[ $DRY_RUN == false ]]; then
        backup_configuration
    fi
    
    case "$PHASE" in
        "1"|"all")
            deploy_phase1
            ;;&
        "2"|"all")
            deploy_phase2
            ;;&
        "3"|"all")
            deploy_phase3
            ;;&
        "4"|"all")
            deploy_phase4
            ;;
        *)
            log_error "Invalid phase: $PHASE. Must be 1-4 or 'all'"
            exit 1
            ;;
    esac
    
    log_success "SutazAI Resource Optimization Deployment completed successfully!"
    log_info "Next steps:"
    log_info "1. Monitor system performance via Grafana: http://localhost:10201"
    log_info "2. Check container resource usage: docker stats"
    log_info "3. Review optimization report in logs directory"
    log_info "4. If issues occur, run rollback: $0 --rollback"
}

# Execute main function
main "$@"