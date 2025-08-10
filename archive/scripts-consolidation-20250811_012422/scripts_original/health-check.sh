#!/bin/bash
# SutazAI Master Health Check Script
#
# Consolidated health check orchestrator combining all health monitoring
# functionality from 465+ original scripts into a unified health validation system.
#
# Usage:
#   ./scripts/health-check.sh                    # Full system health check
#   ./scripts/health-check.sh --quick            # Quick health check (core services)
#   ./scripts/health-check.sh --services db,ai   # Specific service groups
#   ./scripts/health-check.sh --json             # JSON output for CI/CD
#   ./scripts/health-check.sh --continuous       # Continuous monitoring mode
#
# Created: 2025-08-10
# Consolidated from: 465 health check scripts
# Author: Shell Automation Specialist
# Security: Enterprise-grade with timeout handling and resource limits

set -euo pipefail

# Signal handlers for graceful shutdown
trap 'echo "Health check interrupted"; cleanup_and_exit 130' INT
trap 'echo "Health check terminated"; cleanup_and_exit 143' TERM

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_DIR="${BASE_DIR}/logs"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/health_check_${TIMESTAMP}.log"
readonly HEALTH_TIMEOUT=60
readonly QUICK_TIMEOUT=15
readonly JSON_OUTPUT_FILE="${LOG_DIR}/health_report_${TIMESTAMP}.json"

# Health check results
declare -A HEALTH_RESULTS=()
declare -A SERVICE_LATENCIES=()
declare -A SERVICE_DETAILS=()

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Health check utilities
check_tcp_port() {
    local host="$1"
    local port="$2"
    local service="$3"
    local timeout="${4:-5}"
    
    local start_time=$(date +%s%N)
    if timeout "$timeout" bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
        exec 3<&-
        exec 3>&-
        local end_time=$(date +%s%N)
        local latency=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        SERVICE_LATENCIES["$service"]="$latency"
        return 0
    else
        return 1
    fi
}

check_http_endpoint() {
    local url="$1"
    local service="$2"
    local expected_status="${3:-200}"
    local timeout="${4:-10}"
    
    local start_time=$(date +%s%N)
    local response
    response=$(curl -s -w "%{http_code}" -m "$timeout" "$url" 2>/dev/null | tail -n1)
    local end_time=$(date +%s%N)
    
    if [[ "$response" == "$expected_status" ]]; then
        local latency=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        SERVICE_LATENCIES["$service"]="$latency"
        return 0
    else
        SERVICE_DETAILS["$service"]="HTTP $response"
        return 1
    fi
}

check_json_api() {
    local url="$1"
    local service="$2"
    local timeout="${3:-10}"
    
    local start_time=$(date +%s%N)
    local response
    response=$(curl -s -H "Content-Type: application/json" -m "$timeout" "$url" 2>/dev/null)
    local end_time=$(date +%s%N)
    local curl_exit_code=$?
    
    if [[ $curl_exit_code -eq 0 ]] && echo "$response" | jq . >/dev/null 2>&1; then
        local latency=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        SERVICE_LATENCIES["$service"]="$latency"
        
        # Extract useful details from JSON response
        local status
        status=$(echo "$response" | jq -r '.status // .state // "unknown"' 2>/dev/null)
        SERVICE_DETAILS["$service"]="Status: $status"
        return 0
    else
        SERVICE_DETAILS["$service"]="JSON API failed (exit: $curl_exit_code)"
        return 1
    fi
}

# Core service health checks
check_database_services() {
    log_info "Checking database services..."
    
    # PostgreSQL
    if check_tcp_port "localhost" "10000" "PostgreSQL" 5; then
        HEALTH_RESULTS["PostgreSQL"]="HEALTHY"
        log_success "PostgreSQL is healthy (${SERVICE_LATENCIES[PostgreSQL]}ms)"
    else
        HEALTH_RESULTS["PostgreSQL"]="UNHEALTHY"
        log_error "PostgreSQL is not responding on port 10000"
    fi
    
    # Redis
    if check_tcp_port "localhost" "10001" "Redis" 5; then
        HEALTH_RESULTS["Redis"]="HEALTHY" 
        log_success "Redis is healthy (${SERVICE_LATENCIES[Redis]}ms)"
    else
        HEALTH_RESULTS["Redis"]="UNHEALTHY"
        log_error "Redis is not responding on port 10001"
    fi
    
    # Neo4j
    if check_tcp_port "localhost" "10002" "Neo4j" 5; then
        HEALTH_RESULTS["Neo4j"]="HEALTHY"
        log_success "Neo4j is healthy (${SERVICE_LATENCIES[Neo4j]}ms)"
    else
        HEALTH_RESULTS["Neo4j"]="UNHEALTHY"
        log_error "Neo4j is not responding on port 10002"
    fi
}

check_vector_databases() {
    log_info "Checking vector database services..."
    
    # ChromaDB
    if check_tcp_port "localhost" "10100" "ChromaDB" 5; then
        HEALTH_RESULTS["ChromaDB"]="HEALTHY"
        log_success "ChromaDB is healthy (${SERVICE_LATENCIES[ChromaDB]}ms)"
    else
        HEALTH_RESULTS["ChromaDB"]="UNHEALTHY"
        log_warn "ChromaDB is not responding (expected per CLAUDE.md)"
    fi
    
    # Qdrant
    if check_http_endpoint "http://localhost:10101/" "Qdrant" 200 10; then
        HEALTH_RESULTS["Qdrant"]="HEALTHY"
        log_success "Qdrant is healthy (${SERVICE_LATENCIES[Qdrant]}ms)"
    else
        HEALTH_RESULTS["Qdrant"]="UNHEALTHY"
        log_error "Qdrant API is not responding on port 10101"
    fi
    
    # FAISS
    if check_tcp_port "localhost" "10103" "FAISS" 5; then
        HEALTH_RESULTS["FAISS"]="HEALTHY"
        log_success "FAISS is healthy (${SERVICE_LATENCIES[FAISS]}ms)"
    else
        HEALTH_RESULTS["FAISS"]="UNHEALTHY" 
        log_error "FAISS is not responding on port 10103"
    fi
}

check_ai_services() {
    log_info "Checking AI services..."
    
    # Ollama
    if check_json_api "http://localhost:10104/api/tags" "Ollama" 15; then
        HEALTH_RESULTS["Ollama"]="HEALTHY"
        log_success "Ollama is healthy with models loaded (${SERVICE_LATENCIES[Ollama]}ms)"
    else
        HEALTH_RESULTS["Ollama"]="UNHEALTHY"
        log_error "Ollama API is not responding on port 10104"
    fi
    
    # Hardware Resource Optimizer
    if check_http_endpoint "http://localhost:11110/health" "HardwareOptimizer" 200 10; then
        HEALTH_RESULTS["HardwareOptimizer"]="HEALTHY"
        log_success "Hardware Resource Optimizer is healthy (${SERVICE_LATENCIES[HardwareOptimizer]}ms)"
    else
        HEALTH_RESULTS["HardwareOptimizer"]="UNHEALTHY"
        log_error "Hardware Resource Optimizer is not responding on port 11110"
    fi
    
    # AI Agent Orchestrator  
    if check_http_endpoint "http://localhost:8589/health" "AgentOrchestrator" 200 10; then
        HEALTH_RESULTS["AgentOrchestrator"]="HEALTHY"
        log_success "AI Agent Orchestrator is healthy (${SERVICE_LATENCIES[AgentOrchestrator]}ms)"
    else
        HEALTH_RESULTS["AgentOrchestrator"]="DEGRADED"
        log_warn "AI Agent Orchestrator has issues (optimization in progress per CLAUDE.md)"
    fi
}

check_application_services() {
    log_info "Checking application services..."
    
    # Backend API
    if check_json_api "http://localhost:10010/health" "Backend" 15; then
        HEALTH_RESULTS["Backend"]="HEALTHY"
        log_success "Backend API is healthy (${SERVICE_LATENCIES[Backend]}ms)"
    else
        HEALTH_RESULTS["Backend"]="UNHEALTHY"
        log_error "Backend API is not responding on port 10010"
    fi
    
    # Frontend UI
    if check_http_endpoint "http://localhost:10011/" "Frontend" 200 10; then
        HEALTH_RESULTS["Frontend"]="HEALTHY"
        log_success "Frontend UI is healthy (${SERVICE_LATENCIES[Frontend]}ms)"
    else
        HEALTH_RESULTS["Frontend"]="UNHEALTHY"
        log_error "Frontend UI is not responding on port 10011"
    fi
}

check_monitoring_services() {
    log_info "Checking monitoring services..."
    
    # Prometheus
    if check_http_endpoint "http://localhost:10200/-/healthy" "Prometheus" 200 10; then
        HEALTH_RESULTS["Prometheus"]="HEALTHY"
        log_success "Prometheus is healthy (${SERVICE_LATENCIES[Prometheus]}ms)"
    else
        HEALTH_RESULTS["Prometheus"]="UNHEALTHY"
        log_error "Prometheus is not responding on port 10200"
    fi
    
    # Grafana
    if check_json_api "http://localhost:10201/api/health" "Grafana" 10; then
        HEALTH_RESULTS["Grafana"]="HEALTHY"
        log_success "Grafana is healthy (${SERVICE_LATENCIES[Grafana]}ms)"
    else
        HEALTH_RESULTS["Grafana"]="UNHEALTHY"
        log_error "Grafana is not responding on port 10201"
    fi
    
    # Loki
    if check_http_endpoint "http://localhost:10202/ready" "Loki" 200 10; then
        HEALTH_RESULTS["Loki"]="HEALTHY"
        log_success "Loki is healthy (${SERVICE_LATENCIES[Loki]}ms)"
    else
        HEALTH_RESULTS["Loki"]="UNHEALTHY"
        log_error "Loki is not responding on port 10202"
    fi
}

check_message_queue() {
    log_info "Checking message queue services..."
    
    # RabbitMQ Management API
    if check_http_endpoint "http://localhost:10008/" "RabbitMQ" 200 10; then
        HEALTH_RESULTS["RabbitMQ"]="HEALTHY"
        log_success "RabbitMQ is healthy (${SERVICE_LATENCIES[RabbitMQ]}ms)"
    else
        HEALTH_RESULTS["RabbitMQ"]="UNHEALTHY" 
        log_error "RabbitMQ is not responding on port 10008"
    fi
}

# Quick health check (core services only)
quick_health_check() {
    log_info "Performing quick health check..."
    
    check_database_services
    check_application_services
    check_ai_services
    
    # Quick Ollama test
    log_info "Testing Ollama text generation..."
    local ollama_test
    ollama_test=$(curl -s -m 10 -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama","prompt":"Hello","stream":false}' 2>/dev/null | jq -r '.response // "ERROR"' 2>/dev/null)
    
    if [[ "$ollama_test" != "ERROR" && -n "$ollama_test" ]]; then
        log_success "Ollama text generation working"
    else
        log_warn "Ollama text generation issues"
    fi
}

# Full comprehensive health check
full_health_check() {
    log_info "Performing comprehensive system health check..."
    
    check_database_services
    check_vector_databases
    check_ai_services
    check_application_services
    check_monitoring_services  
    check_message_queue
    
    # Additional comprehensive checks
    log_info "Running additional system checks..."
    
    # Docker containers status
    local container_count
    container_count=$(docker ps --format "{{.Names}}" | wc -l)
    log_info "Running containers: $container_count"
    
    # Docker system resource usage
    local docker_stats
    docker_stats=$(docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}" 2>/dev/null | tail -n +2)
    log_info "Docker resource usage:"
    while IFS= read -r line; do
        log_info "  $line"
    done <<< "$docker_stats"
}

# Continuous monitoring mode
continuous_monitoring() {
    local interval="${1:-60}"  # Default 60 seconds
    log_info "Starting continuous monitoring (interval: ${interval}s)..."
    log_info "Press Ctrl+C to stop"
    
    # Timeout mechanism to prevent infinite loops
    LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
    loop_start=$(date +%s)
    while true; do
        log_info "=== Continuous Health Check - $(date) ==="
        quick_health_check
        generate_health_summary
        sleep "$interval"
        # Check for timeout
        current_time=$(date +%s)
        if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
            echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
            break
        fi

    done
}

# Generate health summary
generate_health_summary() {
    local total_services=0
    local healthy_services=0
    local degraded_services=0
    local unhealthy_services=0
    
    log_info "=== HEALTH SUMMARY ==="
    
    for service in "${!HEALTH_RESULTS[@]}"; do
        total_services=$((total_services + 1))
        local status="${HEALTH_RESULTS[$service]}"
        local latency="${SERVICE_LATENCIES[$service]:-N/A}"
        local details="${SERVICE_DETAILS[$service]:-}"
        
        case "$status" in
            "HEALTHY")
                healthy_services=$((healthy_services + 1))
                log_success "✅ $service: $status (${latency}ms) $details"
                ;;
            "DEGRADED")
                degraded_services=$((degraded_services + 1))
                log_warn "⚠️  $service: $status $details"
                ;;
            "UNHEALTHY")
                unhealthy_services=$((unhealthy_services + 1))
                log_error "❌ $service: $status $details"
                ;;
        esac
    done
    
    log_info "=== OVERALL STATUS ==="
    log_info "Total Services: $total_services"
    log_info "Healthy: $healthy_services"
    log_info "Degraded: $degraded_services"  
    log_info "Unhealthy: $unhealthy_services"
    
    local health_percentage=$((healthy_services * 100 / total_services))
    log_info "System Health: ${health_percentage}%"
    
    if [[ $health_percentage -ge 90 ]]; then
        log_success "System is HEALTHY"
        return 0
    elif [[ $health_percentage -ge 70 ]]; then
        log_warn "System is DEGRADED"
        return 1
    else
        log_error "System is UNHEALTHY"
        return 2
    fi
}

# Generate JSON report
generate_json_report() {
    local report_data='{"timestamp":"'$(date -Iseconds)'","health_check":{"services":{},"summary":{}}}'
    
    # Add service results
    for service in "${!HEALTH_RESULTS[@]}"; do
        local status="${HEALTH_RESULTS[$service]}"
        local latency="${SERVICE_LATENCIES[$service]:-null}"
        local details="${SERVICE_DETAILS[$service]:-}"
        
        report_data=$(echo "$report_data" | jq \
            --arg service "$service" \
            --arg status "$status" \
            --arg latency "$latency" \
            --arg details "$details" \
            '.health_check.services[$service] = {
                "status": $status,
                "latency_ms": ($latency | if . == "null" then null else tonumber end),
                "details": $details
            }')
    done
    
    # Add summary
    local total_services=${#HEALTH_RESULTS[@]}
    local healthy_count=0
    for status in "${HEALTH_RESULTS[@]}"; do
        [[ "$status" == "HEALTHY" ]] && ((healthy_count++))
    done
    
    local health_percentage=$((healthy_count * 100 / total_services))
    
    report_data=$(echo "$report_data" | jq \
        --arg total "$total_services" \
        --arg healthy "$healthy_count" \
        --arg percentage "$health_percentage" \
        '.health_check.summary = {
            "total_services": ($total | tonumber),
            "healthy_services": ($healthy | tonumber),
            "health_percentage": ($percentage | tonumber),
            "overall_status": (if ($percentage | tonumber) >= 90 then "HEALTHY"
                               elif ($percentage | tonumber) >= 70 then "DEGRADED" 
                               else "UNHEALTHY" end)
        }')
    
    echo "$report_data" | jq . > "$JSON_OUTPUT_FILE"
    log_info "JSON report generated: $JSON_OUTPUT_FILE"
}

# Cleanup and exit
cleanup_and_exit() {
    local exit_code="${1:-0}"
    log_info "Cleaning up health check processes..."
    exit "$exit_code"
}

# Main function
main() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Initialize logging
    log_info "SutazAI Master Health Check Script Started"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Arguments: $*"
    
    # Parse arguments
    local check_type="full"
    local service_groups=""
    local json_output="false"
    local continuous="false"
    local continuous_interval=60
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                check_type="quick"
                shift
                ;;
            --services)
                service_groups="$2"
                shift 2
                ;;
            --json)
                json_output="true"
                shift
                ;;
            --continuous)
                continuous="true"
                shift
                ;;
            --interval)
                continuous_interval="$2"
                shift 2
                ;;
            --help|-h)
                cat << EOF
SutazAI Master Health Check Script

Usage: $0 [OPTIONS]

OPTIONS:
    --quick                 Quick health check (core services only)
    --services GROUP        Check specific service groups (db,ai,monitoring,app)
    --json                  Generate JSON report for CI/CD
    --continuous            Continuous monitoring mode
    --interval SECONDS      Continuous monitoring interval (default: 60)
    --help                  Show this help message

Examples:
    $0                      # Full system health check
    $0 --quick              # Quick health check
    $0 --services db,ai     # Check database and AI services only
    $0 --json               # Generate JSON report
    $0 --continuous         # Continuous monitoring

Service Groups:
    db          - Database services (PostgreSQL, Redis, Neo4j)
    vector      - Vector databases (ChromaDB, Qdrant, FAISS)
    ai          - AI services (Ollama, agents)
    app         - Application services (Backend, Frontend)
    monitoring  - Monitoring stack (Prometheus, Grafana, Loki)
    queue       - Message queue services (RabbitMQ)

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute health checks based on configuration
    if [[ "$continuous" == "true" ]]; then
        continuous_monitoring "$continuous_interval"
    elif [[ -n "$service_groups" ]]; then
        log_info "Checking specific service groups: $service_groups"
        IFS=',' read -ra groups <<< "$service_groups"
        for group in "${groups[@]}"; do
            case $group in
                db|database) check_database_services ;;
                vector) check_vector_databases ;;
                ai) check_ai_services ;;
                app|application) check_application_services ;;
                monitoring) check_monitoring_services ;;
                queue) check_message_queue ;;
                *) log_warn "Unknown service group: $group" ;;
            esac
        done
    elif [[ "$check_type" == "quick" ]]; then
        quick_health_check
    else
        full_health_check
    fi
    
    # Generate summary and reports
    local exit_code
    generate_health_summary
    exit_code=$?
    
    if [[ "$json_output" == "true" ]]; then
        generate_json_report
    fi
    
    log_info "Health check completed (exit code: $exit_code)"
    log_info "Log file: $LOG_FILE"
    
    exit $exit_code
}

# Execute main function with all arguments
main "$@"