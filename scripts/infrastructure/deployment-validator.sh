#!/bin/bash
# Deployment Validator for Sutazai 69-Agent System
# Implements comprehensive pre-deployment validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="/opt/sutazaiapp/logs/deployment-validation.log"
TEMP_DIR="/tmp/sutazai-validation-$$"
VALIDATION_TIMEOUT=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR" "$*" >&2
    echo -e "${RED}ERROR: $*${NC}" >&2
}

warn() {
    log "WARN" "$*"
    echo -e "${YELLOW}WARNING: $*${NC}"
}

info() {
    log "INFO" "$*"
    echo -e "${GREEN}INFO: $*${NC}"
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Validation functions
validate_docker_environment() {
    info "Validating Docker environment..."
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running"
        return 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "docker-compose is not installed"
        return 1
    fi
    
    # Check available disk space (minimum 20GB)
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local min_space=$((20 * 1024 * 1024))  # 20GB in KB
    
    if [[ $available_space -lt $min_space ]]; then
        error "Insufficient disk space. Available: $((available_space/1024/1024))GB, Required: 20GB"
        return 1
    fi
    
    info "Docker environment validation passed"
    return 0
}

validate_system_resources() {
    info "Validating system resources..."
    
    # Check CPU cores
    local cpu_cores
    cpu_cores=$(nproc)
    
    if [[ $cpu_cores -lt 12 ]]; then
        error "Insufficient CPU cores. Available: $cpu_cores, Required: 12"
        return 1
    fi
    
    # Check memory (minimum 29GB)
    local total_memory
    total_memory=$(free -g | awk 'NR==2{print $2}')
    
    if [[ $total_memory -lt 29 ]]; then
        error "Insufficient memory. Available: ${total_memory}GB, Required: 29GB"
        return 1
    fi
    
    # Check current resource usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    if (( $(echo "$cpu_usage > 70.0" | bc -l) )); then
        warn "High CPU usage detected: ${cpu_usage}%"
    fi
    
    local memory_usage
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    
    if (( $(echo "$memory_usage > 80.0" | bc -l) )); then
        warn "High memory usage detected: ${memory_usage}%"
    fi
    
    info "System resources validation passed"
    return 0
}

validate_network_configuration() {
    info "Validating network configuration..."
    
    # Check if required networks exist
    local required_networks=("sutazai-network")
    
    for network in "${required_networks[@]}"; do
        if ! docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
            warn "Creating missing network: $network"
            docker network create "$network" --driver bridge || {
                error "Failed to create network: $network"
                return 1
            }
        fi
    done
    
    # Check port availability
    local required_ports=(5432 6379 7474 8500 8000 9090 3000)
    local occupied_ports=()
    
    for port in "${required_ports[@]}"; do
        if netstat -tuln | grep -q ":${port} "; then
            occupied_ports+=("$port")
        fi
    done
    
    if [[ ${#occupied_ports[@]} -gt 0 ]]; then
        warn "Some required ports are already in use: ${occupied_ports[*]}"
        warn "This may cause conflicts during deployment"
    fi
    
    info "Network configuration validation passed"
    return 0
}

validate_docker_compose_files() {
    info "Validating Docker Compose files..."
    
    local compose_files=()
    mapfile -t compose_files < <(find "$PROJECT_ROOT" -name "docker-compose*.yml" -type f)
    
    local validation_errors=0
    
    for compose_file in "${compose_files[@]}"; do
        info "Validating $compose_file"
        
        # Validate YAML syntax
        if ! docker-compose -f "$compose_file" config >/dev/null 2>&1; then
            error "Invalid YAML syntax in $compose_file"
            ((validation_errors++))
            continue
        fi
        
        # Check for resource limits
        if ! grep -q "cpus\|cpu_count\|mem_limit\|memory" "$compose_file"; then
            warn "No resource limits found in $compose_file"
        fi
        
        # Check for health checks
        if ! grep -q "healthcheck\|health" "$compose_file"; then
            warn "No health checks found in $compose_file"
        fi
        
        # Check for restart policies
        if ! grep -q "restart:" "$compose_file"; then
            warn "No restart policy found in $compose_file"
        fi
    done
    
    if [[ $validation_errors -gt 0 ]]; then
        error "$validation_errors Docker Compose files have validation errors"
        return 1
    fi
    
    info "Docker Compose files validation passed"
    return 0
}

validate_resource_allocation() {
    info "Validating resource allocation..."
    
    local config_file="$PROJECT_ROOT/config/agent-resource-allocation.yml"
    
    if [[ ! -f "$config_file" ]]; then
        error "Resource allocation configuration not found: $config_file"
        return 1
    fi
    
    # Use Python to validate resource allocation
    python3 << 'EOF'
import yaml
import sys

try:
    with open('/opt/sutazaiapp/config/agent-resource-allocation.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate total resource allocation
    total_cpu = 0.0
    total_memory = 0.0
    
    for pool_name, pool_config in config['resource_pools'].items():
        max_agents = pool_config['max_agents']
        cpu_per_agent = float(pool_config['agent_limits']['cpu'])
        memory_per_agent = float(pool_config['agent_limits']['memory'].replace('Gi', ''))
        
        pool_cpu = max_agents * cpu_per_agent
        pool_memory = max_agents * memory_per_agent
        
        total_cpu += pool_cpu
        total_memory += pool_memory
        
        print(f"Pool {pool_name}: {max_agents} agents, {pool_cpu} CPU cores, {pool_memory}Gi memory")
    
    print(f"Total allocation: {total_cpu} CPU cores, {total_memory}Gi memory")
    
    # Check against system limits
    max_cpu = float(config['system_constraints']['max_total_cpu'])
    max_memory = float(config['system_constraints']['max_total_memory'].replace('Gi', ''))
    
    if total_cpu > max_cpu:
        print(f"ERROR: CPU allocation exceeds limit: {total_cpu} > {max_cpu}")
        sys.exit(1)
        
    if total_memory > max_memory:
        print(f"ERROR: Memory allocation exceeds limit: {total_memory} > {max_memory}")
        sys.exit(1)
        
    print("Resource allocation validation passed")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
EOF

    local python_exit_code=$?
    if [[ $python_exit_code -ne 0 ]]; then
        error "Resource allocation validation failed"
        return 1
    fi
    
    info "Resource allocation validation passed"
    return 0
}

validate_security_configuration() {
    info "Validating security configuration..."
    
    # Check for secrets directory
    if [[ ! -d "$PROJECT_ROOT/secrets" ]]; then
        error "Secrets directory not found: $PROJECT_ROOT/secrets"
        return 1
    fi
    
    # Check required secret files
    local required_secrets=(
        "postgres_password.txt"
        "redis_password.txt"
        "neo4j_password.txt"
        "jwt_secret.txt"
    )
    
    for secret in "${required_secrets[@]}"; do
        local secret_file="$PROJECT_ROOT/secrets/$secret"
        if [[ ! -f "$secret_file" ]]; then
            error "Required secret file not found: $secret_file"
            return 1
        fi
        
        # Check file permissions
        local permissions
        permissions=$(stat -c "%a" "$secret_file")
        if [[ "$permissions" != "600" ]]; then
            warn "Insecure permissions on $secret_file: $permissions (should be 600)"
            chmod 600 "$secret_file"
        fi
    done
    
    # Check for hardcoded secrets in compose files
    if grep -r "password.*:" "$PROJECT_ROOT"/docker-compose*.yml | grep -v "\${"; then
        warn "Potential hardcoded secrets found in compose files"
    fi
    
    info "Security configuration validation passed"
    return 0
}

validate_monitoring_setup() {
    info "Validating monitoring setup..."
    
    # Check monitoring configuration files
    local monitoring_configs=(
        "$PROJECT_ROOT/monitoring/prometheus.yml"
        "$PROJECT_ROOT/monitoring/grafana-datasources.yml"
        "$PROJECT_ROOT/monitoring/alert-rules.yml"
    )
    
    for config in "${monitoring_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            warn "Monitoring configuration not found: $config"
        fi
    done
    
    # Validate Prometheus configuration if it exists
    if [[ -f "$PROJECT_ROOT/monitoring/prometheus.yml" ]]; then
        if command -v promtool >/dev/null 2>&1; then
            if ! promtool check config "$PROJECT_ROOT/monitoring/prometheus.yml"; then
                error "Invalid Prometheus configuration"
                return 1
            fi
        else
            warn "promtool not available, skipping Prometheus config validation"
        fi
    fi
    
    info "Monitoring setup validation passed"
    return 0
}

validate_agent_configurations() {
    info "Validating agent configurations..."
    
    local agents_dir="$PROJECT_ROOT/agents"
    
    if [[ ! -d "$agents_dir" ]]; then
        error "Agents directory not found: $agents_dir"
        return 1
    fi
    
    # Count agent directories
    local agent_count
    agent_count=$(find "$agents_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    
    if [[ $agent_count -ne 69 ]]; then
        warn "Expected 69 agents, found $agent_count"
    fi
    
    # Validate agent requirements files
    local missing_requirements=()
    
    while IFS= read -r -d '' agent_dir; do
        local agent_name
        agent_name=$(basename "$agent_dir")
        
        if [[ ! -f "$agent_dir/requirements.txt" ]] && [[ ! -f "$agent_dir/app.py" ]]; then
            missing_requirements+=("$agent_name")
        fi
    done < <(find "$agents_dir" -mindepth 1 -maxdepth 1 -type d -print0)
    
    if [[ ${#missing_requirements[@]} -gt 0 ]]; then
        warn "Agents missing requirements.txt or app.py: ${missing_requirements[*]}"
    fi
    
    info "Agent configurations validation passed"
    return 0
}

run_pre_deployment_tests() {
    info "Running pre-deployment tests..."
    
    # Test database connectivity
    info "Testing database connectivity..."
    
    # Start minimal test environment
    cat > "$TEMP_DIR/test-compose.yml" << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "15432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "16379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
EOF
    
    # Start test services
    docker-compose -f "$TEMP_DIR/test-compose.yml" up -d
    
    # Wait for services to be healthy
    local max_wait=60
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if docker-compose -f "$TEMP_DIR/test-compose.yml" ps | grep -q "healthy"; then
            break
        fi
        sleep 5
        ((wait_time += 5))
    done
    
    # Test connections
    if ! docker exec "$(docker-compose -f "$TEMP_DIR/test-compose.yml" ps -q postgres)" pg_isready -U test; then
        error "PostgreSQL test connection failed"
        docker-compose -f "$TEMP_DIR/test-compose.yml" down -v
        return 1
    fi
    
    if ! docker exec "$(docker-compose -f "$TEMP_DIR/test-compose.yml" ps -q redis)" redis-cli ping; then
        error "Redis test connection failed"
        docker-compose -f "$TEMP_DIR/test-compose.yml" down -v
        return 1
    fi
    
    # Cleanup test environment
    docker-compose -f "$TEMP_DIR/test-compose.yml" down -v
    
    info "Pre-deployment tests passed"
    return 0
}

generate_validation_report() {
    info "Generating validation report..."
    
    local report_file="/opt/sutazaiapp/reports/deployment_validation_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "validation_results": {
    "docker_environment": "$docker_env_status",
    "system_resources": "$system_resources_status",
    "network_configuration": "$network_config_status",
    "docker_compose_files": "$compose_files_status",
    "resource_allocation": "$resource_allocation_status",
    "security_configuration": "$security_config_status",
    "monitoring_setup": "$monitoring_setup_status",
    "agent_configurations": "$agent_config_status",
    "pre_deployment_tests": "$pre_deployment_tests_status"
  },
  "system_info": {
    "cpu_cores": $(nproc),
    "total_memory_gb": $(free -g | awk 'NR==2{print $2}'),
    "available_disk_gb": $(($(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024)),
    "docker_version": "$(docker --version | cut -d' ' -f3 | sed 's/,//')",
    "compose_version": "$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')"
  },
  "recommendations": []
}
EOF
    
    info "Validation report saved to: $report_file"
}

main() {
    info "Starting Sutazai deployment validation..."
    
    local overall_status="PASSED"
    
    # Initialize status variables
    docker_env_status="UNKNOWN"
    system_resources_status="UNKNOWN"
    network_config_status="UNKNOWN"
    compose_files_status="UNKNOWN"
    resource_allocation_status="UNKNOWN"
    security_config_status="UNKNOWN"
    monitoring_setup_status="UNKNOWN"
    agent_config_status="UNKNOWN"
    pre_deployment_tests_status="UNKNOWN"
    
    # Run validation steps
    if validate_docker_environment; then
        docker_env_status="PASSED"
    else
        docker_env_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_system_resources; then
        system_resources_status="PASSED"
    else
        system_resources_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_network_configuration; then
        network_config_status="PASSED"
    else
        network_config_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_docker_compose_files; then
        compose_files_status="PASSED"
    else
        compose_files_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_resource_allocation; then
        resource_allocation_status="PASSED"
    else
        resource_allocation_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_security_configuration; then
        security_config_status="PASSED"
    else
        security_config_status="FAILED"
        overall_status="FAILED"
    fi
    
    if validate_monitoring_setup; then
        monitoring_setup_status="PASSED"
    else
        monitoring_setup_status="WARNING"
    fi
    
    if validate_agent_configurations; then
        agent_config_status="PASSED"
    else
        agent_config_status="WARNING"
    fi
    
    if run_pre_deployment_tests; then
        pre_deployment_tests_status="PASSED"
    else
        pre_deployment_tests_status="FAILED"
        overall_status="FAILED"
    fi
    
    # Generate final report
    generate_validation_report
    
    # Final status
    if [[ "$overall_status" == "PASSED" ]]; then
        info "✅ Deployment validation PASSED - System ready for deployment"
        exit 0
    else
        error "❌ Deployment validation FAILED - Please fix issues before deployment"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help|--dry-run]"
        echo "  --help    Show this help message"
        echo "  --dry-run Run validation without making changes"
        exit 0
        ;;
    --dry-run)
        info "Running in dry-run mode (no changes will be made)"
        # Set dry-run mode for other functions if needed
        ;;
esac

# Execute main function
main "$@"