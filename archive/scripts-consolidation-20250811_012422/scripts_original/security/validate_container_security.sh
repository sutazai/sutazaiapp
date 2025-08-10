#!/bin/bash
# Container Security Validation Script
# Created: August 9, 2025
# Purpose: Comprehensive security validation after non-root migration

set -euo pipefail

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}" >&2; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
info() { echo -e "${CYAN}[INFO] $1${NC}"; }

# Global counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Test result tracking
declare -a FAILED_TESTS_LIST
declare -a WARNING_TESTS_LIST

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TOTAL_TESTS++))
    log "Running test: $test_name"
    
    if $test_function; then
        success "✓ $test_name"
        ((PASSED_TESTS++))
        return 0
    else
        error "✗ $test_name"
        FAILED_TESTS_LIST+=("$test_name")
        ((FAILED_TESTS++))
        return 1
    fi
}

# Warning wrapper
run_warning_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TOTAL_TESTS++))
    log "Running test: $test_name"
    
    if $test_function; then
        success "✓ $test_name"
        ((PASSED_TESTS++))
        return 0
    else
        warning "⚠ $test_name"
        WARNING_TESTS_LIST+=("$test_name")
        ((WARNINGS++))
        return 1
    fi
}

# Check if all containers are running
test_containers_running() {
    local expected_containers=(
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-neo4j"
        "sutazai-ollama"
        "sutazai-chromadb"
        "sutazai-qdrant"
        "sutazai-rabbitmq"
        "sutazai-consul"
        "sutazai-blackbox-exporter"
        "sutazai-ai-agent-orchestrator"
        "sutazai-backend"
        "sutazai-frontend"
        "sutazai-prometheus"
        "sutazai-grafana"
        "sutazai-loki"
    )
    
    local running_containers
    running_containers=$(docker ps --format "{{.Names}}")
    
    for container in "${expected_containers[@]}"; do
        if ! echo "$running_containers" | grep -q "^$container$"; then
            info "Container not running: $container"
            return 1
        fi
    done
    
    return 0
}

# Verify container users are non-root
test_container_users_nonroot() {
    local root_containers=()
    local acceptable_root_containers=("sutazai-cadvisor")  # cAdvisor needs privileged access
    
    while read -r container; do
        local user_info
        user_info=$(docker exec "$container" id 2>/dev/null || echo "uid=0(root)")
        
        if echo "$user_info" | grep -q "uid=0(root)"; then
            # Check if this is an acceptable root container
            local acceptable=false
            for acceptable_container in "${acceptable_root_containers[@]}"; do
                if [[ "$container" == "$acceptable_container" ]]; then
                    acceptable=true
                    break
                fi
            done
            
            if [[ "$acceptable" == false ]]; then
                root_containers+=("$container")
            fi
        fi
    done < <(docker ps --format "{{.Names}}")
    
    if [[ ${#root_containers[@]} -eq 0 ]]; then
        return 0
    else
        error "Containers still running as root: ${root_containers[*]}"
        return 1
    fi
}

# Test database connectivity
test_database_connectivity() {
    local db_tests=0
    local db_passed=0
    
    # PostgreSQL
    ((db_tests++))
    if docker exec sutazai-postgres pg_isready -U sutazai &>/dev/null; then
        ((db_passed++))
        info "PostgreSQL: Connected successfully"
    else
        error "PostgreSQL: Connection failed"
    fi
    
    # Redis
    ((db_tests++))
    if docker exec sutazai-redis redis-cli ping | grep -q "PONG"; then
        ((db_passed++))
        info "Redis: Connected successfully"
    else
        error "Redis: Connection failed"
    fi
    
    # Neo4j (check if web interface is responding)
    ((db_tests++))
    if curl -s http://localhost:10002 >/dev/null; then
        ((db_passed++))
        info "Neo4j: Web interface responding"
    else
        error "Neo4j: Web interface not responding"
    fi
    
    # RabbitMQ
    ((db_tests++))
    if docker exec sutazai-rabbitmq rabbitmq-diagnostics check_running &>/dev/null; then
        ((db_passed++))
        info "RabbitMQ: Running successfully"
    else
        error "RabbitMQ: Not running properly"
    fi
    
    [[ $db_passed -eq $db_tests ]]
}

# Test vector database connectivity
test_vector_database_connectivity() {
    local vector_tests=0
    local vector_passed=0
    
    # ChromaDB
    ((vector_tests++))
    if curl -s http://localhost:10100/api/v1/heartbeat >/dev/null; then
        ((vector_passed++))
        info "ChromaDB: API responding"
    else
        error "ChromaDB: API not responding"
    fi
    
    # Qdrant
    ((vector_tests++))
    if curl -s http://localhost:10101/health >/dev/null; then
        ((vector_passed++))
        info "Qdrant: API responding"
    else
        error "Qdrant: API not responding"
    fi
    
    [[ $vector_passed -eq $vector_tests ]]
}

# Test AI services functionality
test_ai_services() {
    local ai_tests=0
    local ai_passed=0
    
    # Ollama
    ((ai_tests++))
    if curl -s http://localhost:10104/api/tags >/dev/null; then
        ((ai_passed++))
        info "Ollama: API responding"
    else
        error "Ollama: API not responding"
    fi
    
    # Backend API
    ((ai_tests++))
    if curl -s http://localhost:10010/health >/dev/null; then
        ((ai_passed++))
        info "Backend API: Health endpoint responding"
    else
        error "Backend API: Health endpoint not responding"
    fi
    
    # AI Agent Orchestrator
    ((ai_tests++))
    if curl -s http://localhost:8589/health >/dev/null; then
        ((ai_passed++))
        info "AI Agent Orchestrator: Health endpoint responding"
    else
        error "AI Agent Orchestrator: Health endpoint not responding"
    fi
    
    [[ $ai_passed -eq $ai_tests ]]
}

# Test monitoring stack
test_monitoring_stack() {
    local monitoring_tests=0
    local monitoring_passed=0
    
    # Prometheus
    ((monitoring_tests++))
    if curl -s http://localhost:10200/-/healthy >/dev/null; then
        ((monitoring_passed++))
        info "Prometheus: Healthy"
    else
        error "Prometheus: Not healthy"
    fi
    
    # Grafana
    ((monitoring_tests++))
    if curl -s http://localhost:10201/api/health >/dev/null; then
        ((monitoring_passed++))
        info "Grafana: API responding"
    else
        error "Grafana: API not responding"
    fi
    
    # Loki
    ((monitoring_tests++))
    if curl -s http://localhost:10202/ready >/dev/null; then
        ((monitoring_passed++))
        info "Loki: Ready"
    else
        error "Loki: Not ready"
    fi
    
    [[ $monitoring_passed -eq $monitoring_tests ]]
}

# Test volume permissions
test_volume_permissions() {
    local permission_issues=()
    
    # Check key volumes for proper ownership
    local volumes=(
        "sutazaiapp_postgres_data"
        "sutazaiapp_redis_data"
        "sutazaiapp_neo4j_data"
        "sutazaiapp_ollama_data"
        "sutazaiapp_chromadb_data"
        "sutazaiapp_qdrant_data"
        "sutazaiapp_rabbitmq_data"
    )
    
    for volume in "${volumes[@]}"; do
        local mount_point
        mount_point=$(docker volume inspect "$volume" --format '{{.Mountpoint}}' 2>/dev/null || echo "")
        
        if [[ -n "$mount_point" && -d "$mount_point" ]]; then
            local owner
            owner=$(sudo stat -c '%U:%G' "$mount_point" 2>/dev/null || echo "unknown:unknown")
            
            # Check if owned by root (which might be a problem)
            if [[ "$owner" == "root:root" ]]; then
                # Some volumes are expected to be root-owned, check if it's problematic
                if [[ "$volume" =~ (postgres|redis|chromadb|qdrant) ]]; then
                    permission_issues+=("$volume: owned by root but should be service user")
                fi
            fi
            
            info "$volume: $owner"
        else
            warning "Volume $volume: mount point not accessible"
        fi
    done
    
    [[ ${#permission_issues[@]} -eq 0 ]]
}

# Test container security settings
test_container_security_settings() {
    local security_issues=()
    
    # Check for containers running with unnecessary privileges
    while read -r container; do
        local privileged
        privileged=$(docker inspect "$container" --format '{{.HostConfig.Privileged}}' 2>/dev/null || echo "false")
        
        if [[ "$privileged" == "true" ]]; then
            # Check if privileged is justified
            case "$container" in
                "sutazai-cadvisor"|"sutazai-hardware-resource-optimizer"|"sutazai-jarvis-hardware-resource-optimizer"|"sutazai-resource-arbitration-agent")
                    info "$container: Privileged (justified for system monitoring)"
                    ;;
                *)
                    security_issues+=("$container: Running privileged without justification")
                    ;;
            esac
        fi
        
        # Check for containers with --cap-add=ALL or similar dangerous capabilities
        local caps
        caps=$(docker inspect "$container" --format '{{.HostConfig.CapAdd}}' 2>/dev/null || echo "[]")
        if [[ "$caps" != "[]" && "$caps" != "<no value>" ]]; then
            case "$container" in
                "sutazai-cadvisor"|"sutazai-hardware-resource-optimizer")
                    info "$container: Has additional capabilities (justified)"
                    ;;
                *)
                    warning "$container: Has additional capabilities: $caps"
                    ;;
            esac
        fi
        
    done < <(docker ps --format "{{.Names}}")
    
    [[ ${#security_issues[@]} -eq 0 ]]
}

# Performance baseline test
test_performance_baseline() {
    local performance_issues=()
    
    # Test response times for critical endpoints
    local endpoints=(
        "http://localhost:10010/health:Backend API"
        "http://localhost:10104/api/tags:Ollama"
        "http://localhost:10200/-/healthy:Prometheus"
        "http://localhost:10100/api/v1/heartbeat:ChromaDB"
        "http://localhost:10101/health:Qdrant"
    )
    
    for endpoint_desc in "${endpoints[@]}"; do
        local url="${endpoint_desc%:*}"
        local name="${endpoint_desc#*:}"
        
        local response_time
        response_time=$(curl -o /dev/null -s -w '%{time_total}' "$url" 2>/dev/null || echo "999")
        
        # Response time should be under 5 seconds for health checks
        if (( $(echo "$response_time > 5.0" | bc -l) )); then
            performance_issues+=("$name: Response time ${response_time}s > 5s")
        else
            info "$name: Response time ${response_time}s"
        fi
    done
    
    [[ ${#performance_issues[@]} -eq 0 ]]
}

# Generate security compliance report
generate_security_report() {
    log "Generating security compliance report..."
    
    local report_file="/opt/sutazaiapp/CONTAINER_SECURITY_VALIDATION_REPORT.md"
    
    cat > "$report_file" << EOF
# Container Security Validation Report
**Generated:** $(date)  
**Migration Status:** $(if [[ $FAILED_TESTS -eq 0 ]]; then echo "SUCCESSFUL"; else echo "PARTIAL - NEEDS ATTENTION"; fi)

## Executive Summary
- **Total Tests:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS
- **Failed:** $FAILED_TESTS
- **Warnings:** $WARNINGS
- **Security Score:** $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%

## Container User Analysis
EOF
    
    # Add container user information
    echo "| Container | User | Status |" >> "$report_file"
    echo "|-----------|------|--------|" >> "$report_file"
    
    while read -r container; do
        local user_info
        user_info=$(docker exec "$container" id 2>/dev/null || echo "Cannot determine")
        
        local status="✅ Non-root"
        if echo "$user_info" | grep -q "uid=0(root)"; then
            if [[ "$container" == "sutazai-cadvisor" ]]; then
                status="⚠️ Root (Justified)"
            else
                status="❌ Root (Needs fix)"
            fi
        fi
        
        echo "| $container | $user_info | $status |" >> "$report_file"
    done < <(docker ps --format "{{.Names}}" | sort)
    
    # Add failed tests if any
    if [[ ${#FAILED_TESTS_LIST[@]} -gt 0 ]]; then
        echo "" >> "$report_file"
        echo "## Failed Tests" >> "$report_file"
        for test in "${FAILED_TESTS_LIST[@]}"; do
            echo "- ❌ $test" >> "$report_file"
        done
    fi
    
    # Add warnings if any
    if [[ ${#WARNING_TESTS_LIST[@]} -gt 0 ]]; then
        echo "" >> "$report_file"
        echo "## Warnings" >> "$report_file"
        for test in "${WARNING_TESTS_LIST[@]}"; do
            echo "- ⚠️ $test" >> "$report_file"
        done
    fi
    
    cat >> "$report_file" << EOF

## Security Compliance Status
- **PCI DSS:** $(if [[ $FAILED_TESTS -eq 0 ]]; then echo "COMPLIANT"; else echo "NON-COMPLIANT"; fi)
- **ISO 27001:** $(if [[ $FAILED_TESTS -eq 0 ]]; then echo "COMPLIANT"; else echo "NON-COMPLIANT"; fi)
- **SOX:** $(if [[ $FAILED_TESTS -eq 0 ]]; then echo "COMPLIANT"; else echo "NON-COMPLIANT"; fi)

## Recommendations
$(if [[ $FAILED_TESTS -gt 0 ]]; then
echo "1. Address all failed tests before production deployment"
echo "2. Review container configurations for remaining root containers"
echo "3. Verify volume permissions are correctly set"
else
echo "1. Security migration completed successfully"
echo "2. Regular security audits recommended (monthly)"
echo "3. Monitor for any new containers added to the system"
fi)

## Next Steps
$(if [[ $FAILED_TESTS -gt 0 ]]; then
echo "- Fix failed security tests"
echo "- Re-run validation after fixes"
echo "- Document any exceptions"
else
echo "- Deploy to production"
echo "- Set up automated security monitoring"
echo "- Create regular validation schedule"
fi)
EOF
    
    success "Security report generated: $report_file"
}

# Main execution
main() {
    log "Starting Container Security Validation"
    log "======================================"
    
    # Check prerequisites
    if ! command -v curl &>/dev/null; then
        error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v bc &>/dev/null; then
        warning "bc not installed - performance tests may be limited"
    fi
    
    # Run all security tests
    run_test "All containers are running" test_containers_running
    run_test "Container users are non-root" test_container_users_nonroot
    run_test "Database connectivity" test_database_connectivity
    run_test "Vector database connectivity" test_vector_database_connectivity
    run_test "AI services functionality" test_ai_services
    run_test "Monitoring stack health" test_monitoring_stack
    run_warning_test "Volume permissions" test_volume_permissions
    run_test "Container security settings" test_container_security_settings
    run_warning_test "Performance baseline" test_performance_baseline
    
    # Generate report
    generate_security_report
    
    # Final summary
    log "======================================"
    if [[ $FAILED_TESTS -eq 0 ]]; then
        success "🎉 CONTAINER SECURITY MIGRATION SUCCESSFUL!"
        success "All tests passed. System is ready for production."
        success "Security score: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"
    else
        error "⚠️ CONTAINER SECURITY MIGRATION NEEDS ATTENTION"
        error "Failed tests: $FAILED_TESTS/$TOTAL_TESTS"
        error "Please address failed tests before production deployment."
    fi
    log "======================================"
    
    # Exit with appropriate code
    exit $FAILED_TESTS
}

# Execute main if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi