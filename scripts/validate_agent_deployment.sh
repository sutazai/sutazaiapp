#!/bin/bash
# 🔍 SutazAI Agent Deployment Validation
# Comprehensive validation of all 38 AI agents and infrastructure

set -euo pipefail

# Configuration
PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VALIDATION_LOG="$LOG_DIR/deployment_validation_$TIMESTAMP.log"
REPORT_FILE="$LOG_DIR/validation_report_$TIMESTAMP.json"

# Expected agents and their ports
declare -A EXPECTED_AGENTS=(
    ["agent-message-bus"]="8299"
    ["agent-registry"]="8300"
    ["agi-system-architect"]="8201"
    ["autonomous-system-controller"]="8202"
    ["ai-agent-orchestrator"]="8203"
    ["task-assignment-coordinator"]="8204"
    ["ai-agent-creator"]="8205"
    ["senior-ai-engineer"]="8206"
    ["senior-backend-developer"]="8207"
    ["senior-frontend-developer"]="8208"
    ["opendevin-code-generator"]="8209"
    ["code-generation-improver"]="8210"
    ["localagi-orchestration-manager"]="8211"
    ["agentzero-coordinator"]="8212"
    ["agentgpt-autonomous-executor"]="8213"
    ["bigagi-system-manager"]="8214"
    ["langflow-workflow-designer"]="8215"
    ["flowiseai-flow-manager"]="8216"
    ["dify-automation-specialist"]="8217"
    ["infrastructure-devops-manager"]="8218"
    ["deployment-automation-master"]="8219"
    ["hardware-resource-optimizer"]="8220"
    ["system-optimizer-reorganizer"]="8221"
    ["semgrep-security-analyzer"]="8222"
    ["security-pentesting-specialist"]="8223"
    ["kali-security-specialist"]="8224"
    ["private-data-analyst"]="8225"
    ["ollama-integration-specialist"]="8226"
    ["litellm-proxy-manager"]="8227"
    ["context-optimization-engineer"]="8228"
    ["browser-automation-orchestrator"]="8229"
    ["shell-automation-specialist"]="8230"
    ["financial-analysis-specialist"]="8231"
    ["document-knowledge-manager"]="8232"
    ["deep-learning-coordinator-manager"]="8233"
    ["complex-problem-solver"]="8234"
    ["jarvis-voice-interface"]="8235"
    ["ai-product-manager"]="8236"
    ["ai-scrum-master"]="8237"
    ["testing-qa-validator"]="8238"
)

# Core infrastructure services
CORE_SERVICES=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "prometheus" "grafana" "loki")

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Initialize logs
mkdir -p "$LOG_DIR"

# Validation results
declare -A VALIDATION_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0

# ===============================================
# 🚀 LOGGING FUNCTIONS
# ===============================================

log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${BLUE}ℹ️  [$timestamp] $message${NC}" | tee -a "$VALIDATION_LOG"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}✅ [$timestamp] $message${NC}" | tee -a "$VALIDATION_LOG"
    ((PASSED_TESTS++))
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}⚠️  [$timestamp] WARNING: $message${NC}" | tee -a "$VALIDATION_LOG"
    ((WARNING_TESTS++))
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}❌ [$timestamp] ERROR: $message${NC}" | tee -a "$VALIDATION_LOG"
    ((FAILED_TESTS++))
}

log_header() {
    local message="$1"
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$VALIDATION_LOG"
    echo -e "${CYAN}║ $message${NC}" | tee -a "$VALIDATION_LOG"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}" | tee -a "$VALIDATION_LOG"
}

record_test_result() {
    local test_name="$1"
    local result="$2"  # pass, fail, warn
    local message="$3"
    
    VALIDATION_RESULTS["$test_name"]="$result:$message"
    ((TOTAL_TESTS++))
}

# ===============================================
# 🚀 CORE INFRASTRUCTURE VALIDATION
# ===============================================

validate_docker_environment() {
    log_header "🐳 DOCKER ENVIRONMENT VALIDATION"
    
    # Check Docker daemon
    if docker info >/dev/null 2>&1; then
        log_success "Docker daemon is running"
        record_test_result "docker_daemon" "pass" "Docker daemon accessible"
    else
        log_error "Docker daemon is not accessible"
        record_test_result "docker_daemon" "fail" "Docker daemon not accessible"
        return 1
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        log_success "Docker Compose is available: $compose_version"
        record_test_result "docker_compose" "pass" "Docker Compose v$compose_version"
    else
        log_error "Docker Compose is not available"
        record_test_result "docker_compose" "fail" "Docker Compose not available"
        return 1
    fi
    
    # Check system resources
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    local available_disk=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log_info "Available memory: ${available_memory}MB"
    log_info "Available disk space: ${available_disk}GB"
    
    if [[ $available_memory -lt 8192 ]]; then
        log_warn "Low memory: ${available_memory}MB (recommended: 32GB)"
        record_test_result "system_memory" "warn" "Low memory: ${available_memory}MB"
    else
        log_success "Sufficient memory: ${available_memory}MB"
        record_test_result "system_memory" "pass" "Sufficient memory: ${available_memory}MB"
    fi
    
    if [[ $available_disk -lt 50 ]]; then
        log_warn "Low disk space: ${available_disk}GB (recommended: 100GB)"
        record_test_result "system_disk" "warn" "Low disk space: ${available_disk}GB"
    else
        log_success "Sufficient disk space: ${available_disk}GB"
        record_test_result "system_disk" "pass" "Sufficient disk space: ${available_disk}GB"
    fi
}

validate_core_services() {
    log_header "🏗️ CORE INFRASTRUCTURE SERVICES"
    
    for service in "${CORE_SERVICES[@]}"; do
        local container_name="sutazai-$service"
        
        # Check if container exists and is running
        if docker ps --format "{{.Names}}" | grep -q "^$container_name$"; then
            log_success "Service $service: Container running"
            record_test_result "core_service_$service" "pass" "Container running"
            
            # Check health status if available
            local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check")
            
            case "$health_status" in
                "healthy")
                    log_success "Service $service: Health check passed"
                    ;;
                "unhealthy")
                    log_error "Service $service: Health check failed"
                    record_test_result "core_service_${service}_health" "fail" "Health check failed"
                    ;;
                "starting")
                    log_warn "Service $service: Still starting up"
                    record_test_result "core_service_${service}_health" "warn" "Still starting"
                    ;;
                *)
                    log_info "Service $service: No health check configured"
                    ;;
            esac
            
        else
            log_error "Service $service: Container not running"
            record_test_result "core_service_$service" "fail" "Container not running"
        fi
    done
}

validate_networking() {
    log_header "🌐 NETWORK CONNECTIVITY"
    
    # Check Docker network
    if docker network ls | grep -q "sutazai-network"; then
        log_success "Docker network 'sutazai-network' exists"
        record_test_result "docker_network" "pass" "Network exists"
    else
        log_error "Docker network 'sutazai-network' not found"
        record_test_result "docker_network" "fail" "Network missing"
    fi
    
    # Test core service connectivity
    local connectivity_tests=(
        "redis:6379"
        "postgres:5432"
        "neo4j:7474"
        "chromadb:8000"
        "qdrant:6333"
        "ollama:11434"
    )
    
    for test in "${connectivity_tests[@]}"; do
        local service="${test%:*}"
        local port="${test#*:}"
        
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "Connectivity test: $service:$port"
            record_test_result "connectivity_$service" "pass" "Port $port accessible"
        else
            log_error "Connectivity test failed: $service:$port"
            record_test_result "connectivity_$service" "fail" "Port $port not accessible"
        fi
    done
}

# ===============================================
# 🚀 AGENT VALIDATION
# ===============================================

validate_agent_containers() {
    log_header "🤖 AI AGENT CONTAINERS"
    
    local running_agents=0
    local total_agents=${#EXPECTED_AGENTS[@]}
    
    for agent in "${!EXPECTED_AGENTS[@]}"; do
        local container_name="sutazai-$agent"
        
        if docker ps --format "{{.Names}}" | grep -q "^$container_name$"; then
            log_success "Agent $agent: Container running"
            record_test_result "agent_container_$agent" "pass" "Container running"
            ((running_agents++))
            
            # Check container logs for errors
            local error_count=$(docker logs "$container_name" --since 5m 2>&1 | grep -i "error\|exception\|failed" | wc -l)
            if [[ $error_count -gt 0 ]]; then
                log_warn "Agent $agent: $error_count errors in recent logs"
                record_test_result "agent_logs_$agent" "warn" "$error_count errors in logs"
            else
                log_success "Agent $agent: No recent errors in logs"
                record_test_result "agent_logs_$agent" "pass" "No recent errors"
            fi
            
        else
            log_error "Agent $agent: Container not running"
            record_test_result "agent_container_$agent" "fail" "Container not running"
        fi
    done
    
    log_info "Agent containers: $running_agents/$total_agents running"
    
    if [[ $running_agents -eq $total_agents ]]; then
        log_success "All expected agent containers are running"
        record_test_result "all_agents_running" "pass" "$running_agents/$total_agents agents running"
    elif [[ $running_agents -gt $((total_agents / 2)) ]]; then
        log_warn "Most agent containers are running ($running_agents/$total_agents)"
        record_test_result "all_agents_running" "warn" "$running_agents/$total_agents agents running"
    else
        log_error "Many agent containers are not running ($running_agents/$total_agents)"
        record_test_result "all_agents_running" "fail" "$running_agents/$total_agents agents running"
    fi
}

validate_agent_apis() {
    log_header "🔌 AGENT API ENDPOINTS"
    
    local working_apis=0
    local total_apis=${#EXPECTED_AGENTS[@]}
    
    for agent in "${!EXPECTED_AGENTS[@]}"; do
        local port="${EXPECTED_AGENTS[$agent]}"
        local endpoint="http://localhost:$port/health"
        
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$endpoint" 2>/dev/null || echo "000")
        
        if [[ "$response_code" == "200" ]]; then
            log_success "Agent $agent: API endpoint healthy ($port)"
            record_test_result "agent_api_$agent" "pass" "API healthy on port $port"
            ((working_apis++))
        elif [[ "$response_code" == "000" ]]; then
            log_error "Agent $agent: API endpoint not reachable ($port)"
            record_test_result "agent_api_$agent" "fail" "API not reachable on port $port"
        else
            log_warn "Agent $agent: API endpoint returned HTTP $response_code ($port)"
            record_test_result "agent_api_$agent" "warn" "API returned HTTP $response_code"
        fi
    done
    
    log_info "API endpoints: $working_apis/$total_apis working"
    
    local api_percentage=$(( (working_apis * 100) / total_apis ))
    if [[ $api_percentage -ge 90 ]]; then
        log_success "Excellent API availability: ${api_percentage}%"
    elif [[ $api_percentage -ge 70 ]]; then
        log_warn "Good API availability: ${api_percentage}%"
    else
        log_error "Poor API availability: ${api_percentage}%"
    fi
}

validate_agent_communication() {
    log_header "📡 AGENT COMMUNICATION INFRASTRUCTURE"
    
    # Test message bus
    local message_bus_health=$(curl -s "http://localhost:8299/health" 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null || echo "unreachable")
    
    if [[ "$message_bus_health" == "healthy" ]]; then
        log_success "Agent message bus: Healthy"
        record_test_result "message_bus" "pass" "Message bus healthy"
        
        # Get message bus stats
        local stats=$(curl -s "http://localhost:8299/stats" 2>/dev/null || echo "{}")
        local active_connections=$(echo "$stats" | jq -r '.active_connections // "unknown"' 2>/dev/null || echo "unknown")
        local total_messages=$(echo "$stats" | jq -r '.total_messages // "unknown"' 2>/dev/null || echo "unknown")
        
        log_info "Message bus: $active_connections active connections, $total_messages total messages"
        
    else
        log_error "Agent message bus: Not healthy ($message_bus_health)"
        record_test_result "message_bus" "fail" "Message bus not healthy"
    fi
    
    # Test agent registry
    local registry_health=$(curl -s "http://localhost:8300/health" 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null || echo "unreachable")
    
    if [[ "$registry_health" == "healthy" ]]; then
        log_success "Agent registry: Healthy"
        record_test_result "agent_registry" "pass" "Registry healthy"
        
        # Get registry stats
        local registry_stats=$(curl -s "http://localhost:8300/stats" 2>/dev/null || echo "{}")
        local registered_agents=$(echo "$registry_stats" | jq -r '.total_agents // "unknown"' 2>/dev/null || echo "unknown")
        local online_agents=$(echo "$registry_stats" | jq -r '.online_agents // "unknown"' 2>/dev/null || echo "unknown")
        
        log_info "Agent registry: $online_agents/$registered_agents agents online"
        
        if [[ "$online_agents" != "unknown" ]] && [[ "$registered_agents" != "unknown" ]] && [[ $online_agents -gt 0 ]]; then
            log_success "Agents are registering and communicating"
            record_test_result "agent_registration" "pass" "$online_agents agents registered"
        else
            log_warn "No agents have registered yet"
            record_test_result "agent_registration" "warn" "No agents registered"
        fi
        
    else
        log_error "Agent registry: Not healthy ($registry_health)"
        record_test_result "agent_registry" "fail" "Registry not healthy"
    fi
}

# ===============================================
# 🚀 MONITORING AND OBSERVABILITY
# ===============================================

validate_monitoring_stack() {
    log_header "📊 MONITORING AND OBSERVABILITY"
    
    # Prometheus
    if curl -s -f "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
        log_success "Prometheus: Healthy"
        record_test_result "prometheus" "pass" "Prometheus healthy"
        
        # Check targets
        local targets_up=$(curl -s "http://localhost:9090/api/v1/query?query=up" 2>/dev/null | jq -r '.data.result | length' 2>/dev/null || echo "0")
        log_info "Prometheus: $targets_up targets being monitored"
        
    else
        log_error "Prometheus: Not healthy"
        record_test_result "prometheus" "fail" "Prometheus not healthy"
    fi
    
    # Grafana
    if curl -s -f "http://localhost:3000/api/health" >/dev/null 2>&1; then
        log_success "Grafana: Healthy"
        record_test_result "grafana" "pass" "Grafana healthy"
    else
        log_error "Grafana: Not healthy"
        record_test_result "grafana" "fail" "Grafana not healthy"
    fi
    
    # Loki
    if curl -s -f "http://localhost:3100/ready" >/dev/null 2>&1; then
        log_success "Loki: Healthy"
        record_test_result "loki" "pass" "Loki healthy"
    else
        log_error "Loki: Not healthy"
        record_test_result "loki" "fail" "Loki not healthy"
    fi
}

# ===============================================
# 🚀 PERFORMANCE VALIDATION
# ===============================================

validate_system_performance() {
    log_header "⚡ SYSTEM PERFORMANCE"
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' | sed 's/[^0-9.]//g')
    if [[ -n "$cpu_usage" ]] && (( $(echo "$cpu_usage < 80" | bc -l 2>/dev/null || echo 0) )); then
        log_success "CPU usage: ${cpu_usage}% (good)"
        record_test_result "cpu_usage" "pass" "CPU usage: ${cpu_usage}%"
    elif [[ -n "$cpu_usage" ]]; then
        log_warn "CPU usage: ${cpu_usage}% (high)"
        record_test_result "cpu_usage" "warn" "High CPU usage: ${cpu_usage}%"
    else
        log_warn "CPU usage: Unable to determine"
        record_test_result "cpu_usage" "warn" "Unable to determine CPU usage"
    fi
    
    # Memory usage
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if [[ -n "$memory_usage" ]] && (( $(echo "$memory_usage < 85" | bc -l 2>/dev/null || echo 0) )); then
        log_success "Memory usage: ${memory_usage}% (good)"
        record_test_result "memory_usage" "pass" "Memory usage: ${memory_usage}%"
    elif [[ -n "$memory_usage" ]]; then
        log_warn "Memory usage: ${memory_usage}% (high)"
        record_test_result "memory_usage" "warn" "High memory usage: ${memory_usage}%"
    else
        log_warn "Memory usage: Unable to determine"
        record_test_result "memory_usage" "warn" "Unable to determine memory usage"
    fi
    
    # Docker system resource usage
    local docker_containers=$(docker ps --format "{{.Names}}" | wc -l)
    local docker_images=$(docker images -q | wc -l)
    local docker_volumes=$(docker volume ls -q | wc -l)
    
    log_info "Docker resources: $docker_containers containers, $docker_images images, $docker_volumes volumes"
    
    if [[ $docker_containers -ge 35 ]]; then
        log_success "Good container count: $docker_containers"
        record_test_result "container_count" "pass" "$docker_containers containers running"
    elif [[ $docker_containers -ge 20 ]]; then
        log_warn "Moderate container count: $docker_containers"
        record_test_result "container_count" "warn" "$docker_containers containers running"
    else
        log_error "Low container count: $docker_containers"
        record_test_result "container_count" "fail" "Only $docker_containers containers running"
    fi
}

# ===============================================
# 🚀 REPORT GENERATION
# ===============================================

generate_validation_report() {
    log_header "📋 GENERATING VALIDATION REPORT"
    
    # Calculate success rate
    local success_rate=0
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    fi
    
    # Create JSON report
    cat > "$REPORT_FILE" << EOF
{
  "validation_timestamp": "$TIMESTAMP",
  "validation_date": "$(date -Iseconds)",
  "summary": {
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "warning_tests": $WARNING_TESTS,
    "success_rate": $success_rate
  },
  "test_results": {
EOF
    
    # Add test results
    local first=true
    for test_name in "${!VALIDATION_RESULTS[@]}"; do
        local result_data="${VALIDATION_RESULTS[$test_name]}"
        local status="${result_data%:*}"
        local message="${result_data#*:}"
        
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$REPORT_FILE"
        fi
        
        echo "    \"$test_name\": {" >> "$REPORT_FILE"
        echo "      \"status\": \"$status\"," >> "$REPORT_FILE"
        echo "      \"message\": \"$message\"" >> "$REPORT_FILE"
        echo -n "    }" >> "$REPORT_FILE"
    done
    
    echo "" >> "$REPORT_FILE"
    echo "  }" >> "$REPORT_FILE"
    echo "}" >> "$REPORT_FILE"
    
    # Generate summary
    log_info "Validation Summary:"
    log_info "  Total tests: $TOTAL_TESTS"
    log_info "  Passed: $PASSED_TESTS"
    log_info "  Failed: $FAILED_TESTS"
    log_info "  Warnings: $WARNING_TESTS"
    log_info "  Success rate: ${success_rate}%"
    
    # Overall assessment
    if [[ $success_rate -ge 95 ]]; then
        log_success "EXCELLENT: System is fully operational"
    elif [[ $success_rate -ge 85 ]]; then
        log_success "GOOD: System is mostly operational with minor issues"
    elif [[ $success_rate -ge 70 ]]; then
        log_warn "DEGRADED: System has some issues that should be addressed"
    else
        log_error "CRITICAL: System has significant issues requiring immediate attention"
    fi
    
    log_info "Detailed report: $REPORT_FILE"
    log_info "Validation log: $VALIDATION_LOG"
}

# ===============================================
# 🚀 MAIN VALIDATION ORCHESTRATION
# ===============================================

main() {
    local validation_start=$(date +%s)
    
    log_header "🔍 SUTAZAI AI AGENT DEPLOYMENT VALIDATION"
    log_info "Starting validation at $(date)"
    log_info "Validation log: $VALIDATION_LOG"
    
    # Phase 1: Environment validation
    validate_docker_environment
    
    # Phase 2: Core infrastructure
    validate_core_services
    validate_networking
    
    # Phase 3: Agent validation
    validate_agent_containers
    validate_agent_apis
    validate_agent_communication
    
    # Phase 4: Monitoring
    validate_monitoring_stack
    
    # Phase 5: Performance
    validate_system_performance
    
    # Phase 6: Generate report
    generate_validation_report
    
    # Final summary
    local validation_end=$(date +%s)
    local validation_duration=$((validation_end - validation_start))
    
    log_header "🎯 VALIDATION COMPLETED"
    log_success "Validation duration: ${validation_duration}s"
    
    # Return appropriate exit code
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log_success "All critical tests passed!"
        return 0
    else
        log_error "$FAILED_TESTS critical tests failed"
        return 1
    fi
}

# ===============================================
# 🚀 SCRIPT EXECUTION
# ===============================================

# Ensure we're in the project directory
if [[ ! -f "docker-compose.yml" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Command line options
case "${1:-validate}" in
    "validate"|"")
        main
        ;;
    "report")
        if [[ -f "$REPORT_FILE" ]]; then
            cat "$REPORT_FILE" | jq .
        else
            echo "No validation report found. Run validation first."
            exit 1
        fi
        ;;
    "help"|"-h"|"--help")
        echo "SutazAI Agent Deployment Validation"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  validate    Run full validation (default)"
        echo "  report      Show last validation report"
        echo "  help        Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac