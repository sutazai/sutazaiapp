#!/bin/bash
#
# SutazAI Master Monitoring Script - CONSOLIDATED VERSION
# Consolidates 25+ monitoring scripts into ONE unified monitoring controller
#
# Author: Shell Automation Specialist
# Version: 1.0.0 - Consolidation Release
# Date: 2025-08-10
#
# CONSOLIDATION SUMMARY:
# This script replaces the following 25+ monitoring scripts:
# - All scripts/monitoring/*.sh (15+ scripts)
# - All health-check*.sh variations (11+ scripts)  
# - All performance monitoring scripts
# - All container health monitoring scripts
#
# DESCRIPTION:
# Single, comprehensive monitoring controller for SutazAI platform.
# Handles health checks, performance monitoring, alerting, and continuous
# monitoring with proper logging and reporting capabilities.
#

set -euo pipefail

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    log_error "Monitoring interrupted, cleaning up..."
    # Stop background monitoring processes
    jobs -p | xargs -r kill 2>/dev/null || true
    # Clean up temporary files
    [[ -f "$TEMP_REPORT" ]] && rm -f "$TEMP_REPORT" || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly LOG_DIR="${PROJECT_ROOT}/logs/monitoring"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/monitoring_${TIMESTAMP}.log"
readonly REPORTS_DIR="${PROJECT_ROOT}/reports/monitoring"
readonly TEMP_REPORT="/tmp/sutazai_monitoring_${TIMESTAMP}.json"

# Create required directories
mkdir -p "$LOG_DIR" "$REPORTS_DIR"

# Monitoring configuration
MONITORING_MODE="${MONITORING_MODE:-health}"
CONTINUOUS_MODE="${CONTINUOUS_MODE:-false}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
MONITORING_INTERVAL="${MONITORING_INTERVAL:-30}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"
PERFORMANCE_SAMPLES="${PERFORMANCE_SAMPLES:-5}"
JSON_OUTPUT="${JSON_OUTPUT:-false}"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90

# Service definitions
CORE_SERVICES=(
    "sutazai-postgres:10000"
    "sutazai-redis:10001"
    "sutazai-neo4j:10002"
    "sutazai-ollama:10104"
    "sutazai-backend:10010"
    "sutazai-frontend:10011"
)

MONITORING_SERVICES=(
    "sutazai-prometheus:10200"
    "sutazai-grafana:10201"
    "sutazai-loki:10202"
)

AGENT_SERVICES=(
    "sutazai-ai-agent-orchestrator:8589"
    "sutazai-hardware-resource-optimizer:11110"
    "sutazai-task-assignment-coordinator:8551"
    "sutazai-resource-arbitration-agent:8588"
)

# Health check results
declare -A HEALTH_RESULTS=()
declare -A SERVICE_LATENCIES=()
declare -A SERVICE_DETAILS=()
declare -A PERFORMANCE_METRICS=()

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

# Usage information
show_usage() {
    cat << 'EOF'
SutazAI Master Monitoring Script - Consolidated Edition

USAGE:
    ./master-monitor.sh [MODE] [OPTIONS]

MONITORING MODES:
    health          Quick health check of all services
    performance     Performance monitoring and metrics collection
    continuous      Continuous monitoring with alerts
    deep            Deep diagnostic monitoring
    summary         Generate monitoring summary report

SERVICE GROUPS:
    --core          Monitor only core services (postgres, redis, neo4j, etc.)
    --agents        Monitor only agent services 
    --monitoring    Monitor only monitoring stack services
    --all           Monitor all services (default)

OPTIONS:
    --interval SEC  Monitoring interval in seconds (default: 30)
    --timeout SEC   Health check timeout per service (default: 10)
    --json          Output results in JSON format
    --continuous    Run in continuous monitoring mode
    --alert-webhook Webhook URL for alerts
    --samples N     Number of performance samples (default: 5)
    --debug         Enable debug logging

ALERT THRESHOLDS:
    --cpu-threshold     CPU usage alert threshold (default: 80%)
    --memory-threshold  Memory usage alert threshold (default: 85%)
    --disk-threshold    Disk usage alert threshold (default: 90%)

EXAMPLES:
    ./master-monitor.sh health --core --json
    ./master-monitor.sh performance --agents --samples 10
    ./master-monitor.sh continuous --interval 60 --alert-webhook http://alerts.company.com/webhook
    ./master-monitor.sh deep --all --debug

CONSOLIDATION NOTE:
This script consolidates the functionality of 25+ monitoring scripts:
- scripts/monitoring/* files (15+ scripts)
- scripts/health-check.sh variants (11+ scripts)
- All performance and container monitoring scripts
EOF
}

# Check if service is running
is_service_running() {
    local service_name="$1"
    docker ps --filter "name=$service_name" --filter "status=running" --format "{{.Names}}" | grep -q "^${service_name}$" 2>/dev/null
}

# Check service health endpoint
check_service_health() {
    local service_name="$1"
    local port="$2"
    local endpoint="${3:-/health}"
    
    local start_time=$(date +%s.%N)
    
    # Try health endpoint
    if curl -s --max-time "$HEALTH_TIMEOUT" "http://localhost:${port}${endpoint}" >/dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local latency=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
        SERVICE_LATENCIES["$service_name"]="$latency"
        return 0
    fi
    
    # Try basic port check
    if nc -z localhost "$port" 2>/dev/null; then
        local end_time=$(date +%s.%N)
        local latency=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
        SERVICE_LATENCIES["$service_name"]="$latency"
        return 0
    fi
    
    return 1
}

# Get container resource usage
get_container_metrics() {
    local service_name="$1"
    
    # Get container stats
    local stats=$(docker stats "$service_name" --no-stream --format "table {{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" 2>/dev/null | tail -n1)
    
    if [[ -n "$stats" ]]; then
        local cpu=$(echo "$stats" | cut -d',' -f1 | tr -d '%')
        local mem_usage=$(echo "$stats" | cut -d',' -f2 | cut -d'/' -f1 | tr -d ' ')
        local mem_percent=$(echo "$stats" | cut -d',' -f3 | tr -d '%')
        local net_io=$(echo "$stats" | cut -d',' -f4)
        local block_io=$(echo "$stats" | cut -d',' -f5)
        
        PERFORMANCE_METRICS["${service_name}_cpu"]="${cpu:-0}"
        PERFORMANCE_METRICS["${service_name}_mem_percent"]="${mem_percent:-0}"
        PERFORMANCE_METRICS["${service_name}_mem_usage"]="$mem_usage"
        PERFORMANCE_METRICS["${service_name}_net_io"]="$net_io"
        PERFORMANCE_METRICS["${service_name}_block_io"]="$block_io"
    fi
}

# Perform health check
perform_health_check() {
    local services=("$@")
    local total_services=0
    local healthy_services=0
    local unhealthy_services=()
    
    log_info "Starting health check for ${#services[@]} services..."
    
    for service_spec in "${services[@]}"; do
        local service_name=$(echo "$service_spec" | cut -d':' -f1)
        local service_port=$(echo "$service_spec" | cut -d':' -f2)
        
        total_services=$((total_services + 1))
        
        log_info "Checking health: $service_name"
        
        if is_service_running "$service_name"; then
            if check_service_health "$service_name" "$service_port"; then
                HEALTH_RESULTS["$service_name"]="healthy"
                healthy_services=$((healthy_services + 1))
                local latency="${SERVICE_LATENCIES[$service_name]:-0}"
                log_success "✓ $service_name is healthy (${latency}s)"
            else
                HEALTH_RESULTS["$service_name"]="degraded"
                unhealthy_services+=("$service_name")
                log_warn "⚠ $service_name is running but health check failed"
            fi
        else
            HEALTH_RESULTS["$service_name"]="down"
            unhealthy_services+=("$service_name")
            log_error "✗ $service_name is not running"
        fi
    done
    
    # Health summary
    local health_percentage=$((healthy_services * 100 / total_services))
    log_info "Health Summary: $healthy_services/$total_services services healthy (${health_percentage}%)"
    
    if [[ ${#unhealthy_services[@]} -gt 0 ]]; then
        log_warn "Unhealthy services: ${unhealthy_services[*]}"
    fi
    
    # Generate JSON report if requested
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        generate_health_report "$total_services" "$healthy_services" "${unhealthy_services[@]}"
    fi
    
    return $(( total_services - healthy_services ))
}

# Perform performance monitoring
perform_performance_monitoring() {
    local services=("$@")
    
    log_info "Starting performance monitoring for ${#services[@]} services..."
    log_info "Collecting $PERFORMANCE_SAMPLES samples..."
    
    for ((i=1; i<=PERFORMANCE_SAMPLES; i++)); do
        log_info "Sample $i/$PERFORMANCE_SAMPLES"
        
        for service_spec in "${services[@]}"; do
            local service_name=$(echo "$service_spec" | cut -d':' -f1)
            
            if is_service_running "$service_name"; then
                get_container_metrics "$service_name"
            fi
        done
        
        [[ $i -lt $PERFORMANCE_SAMPLES ]] && sleep 2
    done
    
    # Analyze performance metrics
    analyze_performance_metrics "${services[@]}"
}

# Analyze performance metrics
analyze_performance_metrics() {
    local services=("$@")
    
    log_info "Analyzing performance metrics..."
    
    for service_spec in "${services[@]}"; do
        local service_name=$(echo "$service_spec" | cut -d':' -f1)
        
        if [[ -n "${PERFORMANCE_METRICS[${service_name}_cpu]:-}" ]]; then
            local cpu="${PERFORMANCE_METRICS[${service_name}_cpu]}"
            local mem="${PERFORMANCE_METRICS[${service_name}_mem_percent]}"
            
            log_info "Performance - $service_name: CPU=${cpu}%, Memory=${mem}%"
            
            # Check alert thresholds
            if (( $(echo "$cpu > $ALERT_THRESHOLD_CPU" | bc -l) )); then
                log_warn "⚠ HIGH CPU: $service_name using ${cpu}% CPU (threshold: ${ALERT_THRESHOLD_CPU}%)"
                send_alert "HIGH_CPU" "$service_name" "CPU: ${cpu}%"
            fi
            
            if (( $(echo "$mem > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
                log_warn "⚠ HIGH MEMORY: $service_name using ${mem}% memory (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
                send_alert "HIGH_MEMORY" "$service_name" "Memory: ${mem}%"
            fi
        fi
    done
}

# Send alert
send_alert() {
    local alert_type="$1"
    local service_name="$2"
    local details="$3"
    
    if [[ -n "$ALERT_WEBHOOK" ]]; then
        local payload="{\"alert_type\":\"$alert_type\",\"service\":\"$service_name\",\"details\":\"$details\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
        
        if curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$ALERT_WEBHOOK" >/dev/null; then
            log_info "Alert sent: $alert_type for $service_name"
        else
            log_error "Failed to send alert to webhook"
        fi
    fi
}

# Generate health report
generate_health_report() {
    local total="$1"
    local healthy="$2"
    shift 2
    local unhealthy=("$@")
    
    local report_file="${REPORTS_DIR}/health_report_${TIMESTAMP}.json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "summary": {
        "total_services": $total,
        "healthy_services": $healthy,
        "unhealthy_services": $((total - healthy)),
        "health_percentage": $((healthy * 100 / total))
    },
    "services": {
EOF

    local first=true
    for service in "${!HEALTH_RESULTS[@]}"; do
        [[ "$first" == "false" ]] && echo "," >> "$report_file"
        first=false
        
        local status="${HEALTH_RESULTS[$service]}"
        local latency="${SERVICE_LATENCIES[$service]:-0}"
        
        cat >> "$report_file" << EOF
        "$service": {
            "status": "$status",
            "latency": $latency
        }
EOF
    done
    
    cat >> "$report_file" << EOF
    },
    "unhealthy_services": [
EOF
    
    for ((i=0; i<${#unhealthy[@]}; i++)); do
        [[ $i -gt 0 ]] && echo "," >> "$report_file"
        echo "        \"${unhealthy[$i]}\"" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    ]
}
EOF
    
    log_info "Health report generated: $report_file"
    
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        cat "$report_file"
    fi
}

# Continuous monitoring
run_continuous_monitoring() {
    local services=("$@")
    
    log_info "Starting continuous monitoring (interval: ${MONITORING_INTERVAL}s)..."
    log_info "Press Ctrl+C to stop monitoring"
    
    while true; do
        log_info "=== Monitoring Cycle $(date) ==="
        
        # Health check
        perform_health_check "${services[@]}"
        
        # Performance monitoring
        perform_performance_monitoring "${services[@]}"
        
        log_info "Waiting ${MONITORING_INTERVAL} seconds until next cycle..."
        sleep "$MONITORING_INTERVAL"
    done
}

# Deep diagnostic monitoring
run_deep_monitoring() {
    local services=("$@")
    
    log_info "Starting deep diagnostic monitoring..."
    
    # System resource check
    log_info "=== System Resources ==="
    local disk_usage=$(df -h "${PROJECT_ROOT}" | awk 'NR==2 {print $5}' | tr -d '%')
    local memory_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    local load_avg=$(uptime | grep -o 'load average.*' | cut -d' ' -f3 | tr -d ',')
    
    log_info "Disk usage: ${disk_usage}%"
    log_info "Memory usage: ${memory_usage}%"
    log_info "Load average: $load_avg"
    
    # Check alert thresholds
    if (( $(echo "$disk_usage > $ALERT_THRESHOLD_DISK" | bc -l) )); then
        log_warn "⚠ HIGH DISK USAGE: ${disk_usage}% (threshold: ${ALERT_THRESHOLD_DISK}%)"
        send_alert "HIGH_DISK" "system" "Disk: ${disk_usage}%"
    fi
    
    # Docker system check
    log_info "=== Docker System ==="
    local docker_info=$(docker info --format "{{.Containers}} containers, {{.Images}} images")
    log_info "Docker: $docker_info"
    
    # Service logs check
    log_info "=== Service Logs Analysis ==="
    for service_spec in "${services[@]}"; do
        local service_name=$(echo "$service_spec" | cut -d':' -f1)
        
        if is_service_running "$service_name"; then
            local error_count=$(docker logs "$service_name" --since 1h 2>&1 | grep -ic error || echo 0)
            local warn_count=$(docker logs "$service_name" --since 1h 2>&1 | grep -ic warn || echo 0)
            
            if [[ $error_count -gt 0 ]] || [[ $warn_count -gt 0 ]]; then
                log_warn "$service_name: $error_count errors, $warn_count warnings in last hour"
            else
                log_info "$service_name: No errors or warnings in last hour"
            fi
        fi
    done
    
    # Network connectivity check
    log_info "=== Network Connectivity ==="
    for service_spec in "${services[@]}"; do
        local service_name=$(echo "$service_spec" | cut -d':' -f1)
        local service_port=$(echo "$service_spec" | cut -d':' -f2)
        
        if nc -z localhost "$service_port" 2>/dev/null; then
            log_info "✓ $service_name port $service_port is reachable"
        else
            log_warn "✗ $service_name port $service_port is not reachable"
        fi
    done
}

# Main execution
main() {
    local mode="${1:-health}"
    local service_group="all"
    
    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --core)
                service_group="core"
                shift
                ;;
            --agents)
                service_group="agents"
                shift
                ;;
            --monitoring)
                service_group="monitoring"
                shift
                ;;
            --all)
                service_group="all"
                shift
                ;;
            --interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --timeout)
                HEALTH_TIMEOUT="$2"
                shift 2
                ;;
            --json)
                JSON_OUTPUT="true"
                shift
                ;;
            --continuous)
                CONTINUOUS_MODE="true"
                shift
                ;;
            --alert-webhook)
                ALERT_WEBHOOK="$2"
                shift 2
                ;;
            --samples)
                PERFORMANCE_SAMPLES="$2"
                shift 2
                ;;
            --cpu-threshold)
                ALERT_THRESHOLD_CPU="$2"
                shift 2
                ;;
            --memory-threshold)
                ALERT_THRESHOLD_MEMORY="$2"
                shift 2
                ;;
            --disk-threshold)
                ALERT_THRESHOLD_DISK="$2"
                shift 2
                ;;
            --debug)
                set -x
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            health|performance|continuous|deep|summary)
                mode="$1"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Select services based on group
    local services=()
    case "$service_group" in
        core)
            services=("${CORE_SERVICES[@]}")
            ;;
        agents)
            services=("${AGENT_SERVICES[@]}")
            ;;
        monitoring)
            services=("${MONITORING_SERVICES[@]}")
            ;;
        all)
            services=("${CORE_SERVICES[@]}" "${MONITORING_SERVICES[@]}" "${AGENT_SERVICES[@]}")
            ;;
        *)
            log_error "Unknown service group: $service_group"
            exit 1
            ;;
    esac
    
    log_info "SutazAI Master Monitoring Script - Consolidation Edition"
    log_info "Mode: $mode, Service Group: $service_group (${#services[@]} services)"
    
    # Execute monitoring mode
    case "$mode" in
        health)
            perform_health_check "${services[@]}"
            ;;
        performance)
            perform_performance_monitoring "${services[@]}"
            ;;
        continuous)
            run_continuous_monitoring "${services[@]}"
            ;;
        deep)
            run_deep_monitoring "${services[@]}"
            ;;
        summary)
            log_info "Generating monitoring summary..."
            perform_health_check "${services[@]}"
            perform_performance_monitoring "${services[@]}"
            ;;
        *)
            log_error "Unknown monitoring mode: $mode"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"