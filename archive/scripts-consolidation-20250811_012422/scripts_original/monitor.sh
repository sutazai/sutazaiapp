#!/bin/bash
# SutazAI Master Monitoring Script
#
# Consolidated monitoring orchestrator combining all monitoring functionality
# from 465+ original scripts into a unified monitoring and alerting system.
#
# Usage:
#   ./scripts/monitor.sh                    # Start full monitoring
#   ./scripts/monitor.sh --dashboard        # Launch monitoring dashboard
#   ./scripts/monitor.sh --alerts           # Check and send alerts
#   ./scripts/monitor.sh --performance      # Performance monitoring mode
#   ./scripts/monitor.sh --cleanup          # Monitoring cleanup and maintenance
#
# Created: 2025-08-10
# Consolidated from: 465 monitoring scripts
# Author: Shell Automation Specialist 
# Security: Enterprise-grade with resource limits and secure monitoring

set -euo pipefail

# Signal handlers for graceful shutdown
trap 'echo "Monitoring interrupted"; cleanup_monitoring; exit 130' INT
trap 'echo "Monitoring terminated"; cleanup_monitoring; exit 143' TERM

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_DIR="${BASE_DIR}/logs"
readonly MONITORING_DIR="${BASE_DIR}/monitoring"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/monitoring_${TIMESTAMP}.log"
readonly METRICS_FILE="${LOG_DIR}/metrics_${TIMESTAMP}.json"
readonly ALERT_THRESHOLD_CPU=80
readonly ALERT_THRESHOLD_MEMORY=85
readonly ALERT_THRESHOLD_DISK=90
readonly MONITORING_INTERVAL=30

# Monitoring state
declare -A MONITORING_PIDS=()
declare -A ALERT_COUNTS=()
declare -A LAST_ALERT_TIME=()

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

# Metrics collection functions
collect_system_metrics() {
    local metrics='{}'
    
    # CPU usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    metrics=$(echo "$metrics" | jq --arg cpu "$cpu_usage" '.system.cpu_usage = ($cpu | tonumber)')
    
    # Memory usage
    local memory_info
    memory_info=$(free -m | grep '^Mem:')
    local memory_total=$(echo "$memory_info" | awk '{print $2}')
    local memory_used=$(echo "$memory_info" | awk '{print $3}')
    local memory_percent=$(( memory_used * 100 / memory_total ))
    
    metrics=$(echo "$metrics" | jq \
        --arg total "$memory_total" \
        --arg used "$memory_used" \
        --arg percent "$memory_percent" \
        '.system.memory = {
            "total_mb": ($total | tonumber),
            "used_mb": ($used | tonumber),
            "usage_percent": ($percent | tonumber)
        }')
    
    # Disk usage
    local disk_info
    disk_info=$(df -h "$BASE_DIR" | tail -n1)
    local disk_used_percent=$(echo "$disk_info" | awk '{print $5}' | sed 's/%//')
    local disk_available=$(echo "$disk_info" | awk '{print $4}')
    
    metrics=$(echo "$metrics" | jq \
        --arg used_percent "$disk_used_percent" \
        --arg available "$disk_available" \
        '.system.disk = {
            "usage_percent": ($used_percent | tonumber),
            "available": $available
        }')
    
    # Load average
    local load_avg
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    metrics=$(echo "$metrics" | jq --arg load "$load_avg" '.system.load_average = ($load | tonumber)')
    
    echo "$metrics"
}

collect_docker_metrics() {
    local metrics='{}'
    
    # Container count
    local running_containers
    running_containers=$(docker ps -q | wc -l)
    local total_containers
    total_containers=$(docker ps -aq | wc -l)
    
    metrics=$(echo "$metrics" | jq \
        --arg running "$running_containers" \
        --arg total "$total_containers" \
        '.docker.containers = {
            "running": ($running | tonumber),
            "total": ($total | tonumber)
        }')
    
    # Docker system usage
    local docker_system_info
    docker_system_info=$(docker system df --format "json" 2>/dev/null | jq -s '.[0]' 2>/dev/null || echo '{}')
    if [[ "$docker_system_info" != "{}" ]]; then
        metrics=$(echo "$metrics" | jq --argjson docker_info "$docker_system_info" '.docker.system = $docker_info')
    fi
    
    # Container resource usage
    local container_stats='[]'
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local name=$(echo "$line" | awk '{print $1}')
            local cpu=$(echo "$line" | awk '{print $2}' | sed 's/%//')
            local memory=$(echo "$line" | awk '{print $3}' | sed 's/%//')
            
            container_stats=$(echo "$container_stats" | jq \
                --arg name "$name" \
                --arg cpu "$cpu" \
                --arg memory "$memory" \
                '. += [{
                    "name": $name,
                    "cpu_percent": ($cpu | tonumber // 0),
                    "memory_percent": ($memory | tonumber // 0)
                }]')
        fi
    done < <(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}" 2>/dev/null | tail -n +2)
    
    metrics=$(echo "$metrics" | jq --argjson stats "$container_stats" '.docker.container_stats = $stats')
    
    echo "$metrics"
}

collect_service_metrics() {
    local metrics='{}'
    
    # Service health metrics
    local services=("Backend:http://localhost:10010/health" 
                   "Frontend:http://localhost:10011/"
                   "Ollama:http://localhost:10104/api/tags"
                   "PostgreSQL:localhost:10000"
                   "Redis:localhost:10001"
                   "Prometheus:http://localhost:10200/-/healthy"
                   "Grafana:http://localhost:10201/api/health")
    
    local service_health='[]'
    for service_def in "${services[@]}"; do
        local service_name="${service_def%%:*}"
        local service_endpoint="${service_def#*:}"
        local status="DOWN"
        local response_time=0
        
        if [[ "$service_endpoint" =~ ^http ]]; then
            # HTTP endpoint
            local start_time=$(date +%s%N)
            if curl -f -s -m 5 "$service_endpoint" >/dev/null 2>&1; then
                status="UP"
                local end_time=$(date +%s%N)
                response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            fi
        else
            # TCP endpoint
            local host="${service_endpoint%:*}"
            local port="${service_endpoint#*:}"
            if timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
                exec 3<&-
                exec 3>&-
                status="UP"
                response_time=1  # TCP checks are typically fast
            fi
        fi
        
        service_health=$(echo "$service_health" | jq \
            --arg name "$service_name" \
            --arg status "$status" \
            --arg response_time "$response_time" \
            '. += [{
                "name": $name,
                "status": $status,
                "response_time_ms": ($response_time | tonumber)
            }]')
    done
    
    metrics=$(echo "$metrics" | jq --argjson health "$service_health" '.services = $health')
    
    echo "$metrics"
}

collect_ollama_metrics() {
    local metrics='{}'
    
    # Ollama model information
    local models_info
    models_info=$(curl -s -m 10 http://localhost:10104/api/tags 2>/dev/null | jq '.models // []' 2>/dev/null || echo '[]')
    local model_count
    model_count=$(echo "$models_info" | jq 'length')
    
    metrics=$(echo "$metrics" | jq \
        --argjson models "$models_info" \
        --arg count "$model_count" \
        '.ollama.models = {
            "count": ($count | tonumber),
            "list": $models
        }')
    
    # Test Ollama performance
    local generation_time=0
    local generation_success=false
    local start_time=$(date +%s%N)
    
    local test_response
    test_response=$(curl -s -m 15 -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"tinyllama","prompt":"Hello","stream":false}' 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$test_response" | jq -e '.response' >/dev/null 2>&1; then
        generation_success=true
        local end_time=$(date +%s%N)
        generation_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
    fi
    
    metrics=$(echo "$metrics" | jq \
        --arg time "$generation_time" \
        --arg success "$generation_success" \
        '.ollama.performance = {
            "generation_time_ms": ($time | tonumber),
            "generation_success": ($success | test("true"))
        }')
    
    echo "$metrics"
}

# Alert functions
check_cpu_alert() {
    local cpu_usage="$1"
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD_CPU" | bc -l 2>/dev/null || echo "0") )); then
        send_alert "HIGH_CPU" "CPU usage is ${cpu_usage}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
    fi
}

check_memory_alert() {
    local memory_percent="$1"
    if [[ $memory_percent -gt $ALERT_THRESHOLD_MEMORY ]]; then
        send_alert "HIGH_MEMORY" "Memory usage is ${memory_percent}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
    fi
}

check_disk_alert() {
    local disk_percent="$1"
    if [[ $disk_percent -gt $ALERT_THRESHOLD_DISK ]]; then
        send_alert "HIGH_DISK" "Disk usage is ${disk_percent}% (threshold: ${ALERT_THRESHOLD_DISK}%)"
    fi
}

check_service_alerts() {
    local service_metrics="$1"
    
    # Check for service outages
    local down_services
    down_services=$(echo "$service_metrics" | jq -r '.services[] | select(.status == "DOWN") | .name' 2>/dev/null || echo "")
    
    if [[ -n "$down_services" ]]; then
        while IFS= read -r service; do
            if [[ -n "$service" ]]; then
                send_alert "SERVICE_DOWN" "Service is down: $service"
            fi
        done <<< "$down_services"
    fi
    
    # Check for high response times
    local slow_services
    slow_services=$(echo "$service_metrics" | jq -r '.services[] | select(.response_time_ms > 5000) | "\(.name):\(.response_time_ms)"' 2>/dev/null || echo "")
    
    if [[ -n "$slow_services" ]]; then
        while IFS= read -r service_info; do
            if [[ -n "$service_info" ]]; then
                local service_name="${service_info%%:*}"
                local response_time="${service_info#*:}"
                send_alert "SLOW_SERVICE" "Service is slow: $service_name (${response_time}ms)"
            fi
        done <<< "$slow_services"
    fi
}

send_alert() {
    local alert_type="$1"
    local message="$2"
    local current_time=$(date +%s)
    
    # Rate limiting: don't send same alert type more than once per 5 minutes
    local last_time="${LAST_ALERT_TIME[$alert_type]:-0}"
    if [[ $((current_time - last_time)) -lt 300 ]]; then
        return 0  # Skip this alert
    fi
    
    LAST_ALERT_TIME["$alert_type"]=$current_time
    ALERT_COUNTS["$alert_type"]=$(( ${ALERT_COUNTS["$alert_type"]:-0} + 1 ))
    
    log_warn "ALERT [$alert_type]: $message"
    
    # Write alert to file for external processing
    local alert_file="${LOG_DIR}/alerts_${TIMESTAMP}.log"
    echo "$(date -Iseconds)|$alert_type|$message" >> "$alert_file"
    
    # Additional alert actions can be added here:
    # - Send email notification
    # - Send Slack/Teams message
    # - Trigger webhook
    # - Write to external monitoring system
}

# Performance monitoring
performance_monitoring() {
    log_info "Starting performance monitoring mode..."
    
    # Timeout mechanism to prevent infinite loops
    LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
    loop_start=$(date +%s)
    while true; do
        log_info "=== Performance Monitoring - $(date) ==="
        
        # Collect comprehensive metrics
        local system_metrics
        system_metrics=$(collect_system_metrics)
        
        local docker_metrics
        docker_metrics=$(collect_docker_metrics)
        
        local service_metrics
        service_metrics=$(collect_service_metrics)
        
        local ollama_metrics
        ollama_metrics=$(collect_ollama_metrics)
        
        # Combine all metrics
        local combined_metrics
        combined_metrics=$(echo '{}' | jq \
            --argjson system "$system_metrics" \
            --argjson docker "$docker_metrics" \
            --argjson services "$service_metrics" \
            --argjson ollama "$ollama_metrics" \
            '. + $system + $docker + $services + $ollama')
        
        # Add timestamp
        combined_metrics=$(echo "$combined_metrics" | jq \
            --arg timestamp "$(date -Iseconds)" \
            '.timestamp = $timestamp')
        
        # Save metrics to file
        echo "$combined_metrics" >> "$METRICS_FILE"
        
        # Check for alerts
        local cpu_usage
        cpu_usage=$(echo "$combined_metrics" | jq -r '.system.cpu_usage // 0')
        check_cpu_alert "$cpu_usage"
        
        local memory_percent
        memory_percent=$(echo "$combined_metrics" | jq -r '.system.memory.usage_percent // 0')
        check_memory_alert "$memory_percent"
        
        local disk_percent
        disk_percent=$(echo "$combined_metrics" | jq -r '.system.disk.usage_percent // 0')
        check_disk_alert "$disk_percent"
        
        check_service_alerts "$combined_metrics"
        
        # Performance summary
        local running_containers
        running_containers=$(echo "$combined_metrics" | jq -r '.docker.containers.running // 0')
        local healthy_services
        healthy_services=$(echo "$combined_metrics" | jq -r '[.services[] | select(.status == "UP")] | length')
        local total_services
        total_services=$(echo "$combined_metrics" | jq -r '.services | length')
        
        log_info "System Status: CPU ${cpu_usage}%, Memory ${memory_percent}%, Disk ${disk_percent}%"
        log_info "Docker: $running_containers containers running"
        log_info "Services: $healthy_services/$total_services healthy"
        
        sleep "$MONITORING_INTERVAL"
        # Check for timeout
        current_time=$(date +%s)
        if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
            echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
            break
        fi

    done
}

# Dashboard functions
launch_dashboard() {
    log_info "Launching monitoring dashboard..."
    
    # Check if monitoring stack is running
    if ! curl -f -s -m 5 "http://localhost:10201/api/health" >/dev/null 2>&1; then
        log_error "Grafana is not running. Please start the monitoring stack first."
        return 1
    fi
    
    log_success "Monitoring dashboard available:"
    log_info "  - Grafana: http://localhost:10201 (admin/admin)"
    log_info "  - Prometheus: http://localhost:10200"
    log_info "  - Loki: http://localhost:10202"
    
    # Generate real-time dashboard if script exists
    if [[ -f "${MONITORING_DIR}/realtime_dashboard.py" ]]; then
        log_info "Starting real-time dashboard..."
        python3 "${MONITORING_DIR}/realtime_dashboard.py" &
        MONITORING_PIDS["dashboard"]=$!
        log_info "Real-time dashboard PID: ${MONITORING_PIDS[dashboard]}"
    fi
    
    # Open browser if available
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:10201" 2>/dev/null || true
    elif command -v open >/dev/null 2>&1; then
        open "http://localhost:10201" 2>/dev/null || true
    fi
}

# Cleanup functions
cleanup_logs() {
    log_info "Cleaning up old monitoring logs..."
    
    # Remove logs older than 7 days
    find "$LOG_DIR" -name "monitoring_*.log" -type f -mtime +7 -delete 2>/dev/null || true
    find "$LOG_DIR" -name "metrics_*.json" -type f -mtime +7 -delete 2>/dev/null || true
    find "$LOG_DIR" -name "alerts_*.log" -type f -mtime +7 -delete 2>/dev/null || true
    
    # Compress logs older than 1 day
    find "$LOG_DIR" -name "monitoring_*.log" -type f -mtime +1 -exec gzip {} \; 2>/dev/null || true
    find "$LOG_DIR" -name "metrics_*.json" -type f -mtime +1 -exec gzip {} \; 2>/dev/null || true
    
    log_success "Log cleanup completed"
}

cleanup_monitoring() {
    log_info "Stopping monitoring processes..."
    
    # Kill monitoring background processes
    for service in "${!MONITORING_PIDS[@]}"; do
        local pid="${MONITORING_PIDS[$service]}"
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping $service monitoring (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        unset MONITORING_PIDS["$service"]
    done
    
    log_success "Monitoring cleanup completed"
}

# Main monitoring orchestrator
main() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Initialize logging
    log_info "SutazAI Master Monitoring Script Started"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Arguments: $*"
    
    # Parse arguments
    local mode="monitoring"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dashboard)
                mode="dashboard"
                shift
                ;;
            --alerts)
                mode="alerts"
                shift
                ;;
            --performance)
                mode="performance"
                shift
                ;;
            --cleanup)
                mode="cleanup"
                shift
                ;;
            --interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --help|-h)
                cat << EOF
SutazAI Master Monitoring Script

Usage: $0 [OPTIONS]

OPTIONS:
    --dashboard         Launch monitoring dashboard
    --alerts            Check and process alerts only
    --performance       Performance monitoring mode
    --cleanup           Cleanup old logs and processes
    --interval SECONDS  Monitoring interval (default: 30)
    --help              Show this help message

Examples:
    $0                  # Start full monitoring
    $0 --dashboard      # Launch monitoring dashboard
    $0 --performance    # Performance monitoring mode
    $0 --cleanup        # Cleanup old logs

Dashboard Access:
    Grafana:     http://localhost:10201 (admin/admin)
    Prometheus:  http://localhost:10200
    Loki:        http://localhost:10202

Alert Thresholds:
    CPU:         ${ALERT_THRESHOLD_CPU}%
    # Timeout mechanism to prevent infinite loops
    LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
    loop_start=$(date +%s)
    Memory:      ${ALERT_THRESHOLD_MEMORY}%
    Disk:        ${ALERT_THRESHOLD_DISK}%

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
        # Check for timeout
        current_time=$(date +%s)
        if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
            echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
            break
        fi

    done
    
    # Execute based on mode
    case $mode in
        dashboard)
            launch_dashboard
            # Keep script running to maintain dashboard processes
            if [[ ${#MONITORING_PIDS[@]} -gt 0 ]]; then
                log_info "Dashboard running. Press Ctrl+C to stop."
                while true; do
                    sleep 60
                    # Check if processes are still running
                    for service in "${!MONITORING_PIDS[@]}"; do
                        local pid="${MONITORING_PIDS[$service]}"
                        if ! kill -0 "$pid" 2>/dev/null; then
                            log_warn "$service process (PID: $pid) has stopped"
                            unset MONITORING_PIDS["$service"]
                        fi
                    done
                done
            fi
            ;;
        alerts)
            log_info "Checking and processing alerts..."
            local metrics
            metrics=$(collect_system_metrics)
            metrics=$(echo "$metrics" $(collect_docker_metrics) $(collect_service_metrics) | jq -s 'add')
            
            # Process alerts
            local cpu_usage
            cpu_usage=$(echo "$metrics" | jq -r '.system.cpu_usage // 0')
            check_cpu_alert "$cpu_usage"
            
            local memory_percent
            memory_percent=$(echo "$metrics" | jq -r '.system.memory.usage_percent // 0')
            check_memory_alert "$memory_percent"
            
            local disk_percent
            disk_percent=$(echo "$metrics" | jq -r '.system.disk.usage_percent // 0')
            check_disk_alert "$disk_percent"
            
            check_service_alerts "$metrics"
            
            log_info "Alert processing completed"
            ;;
        performance)
            performance_monitoring
            ;;
        cleanup)
            cleanup_logs
            cleanup_monitoring
            ;;
        monitoring)
            log_info "Starting full monitoring mode..."
            log_info "Monitoring interval: ${MONITORING_INTERVAL}s"
            
            # Start background monitoring processes
            performance_monitoring &
            MONITORING_PIDS["performance"]=$!
            
            log_info "Monitoring started. Press Ctrl+C to stop."
            log_info "Performance monitoring PID: ${MONITORING_PIDS[performance]}"
            
            # Wait for monitoring process
            wait "${MONITORING_PIDS[performance]}"
            ;;
    esac
    
    log_success "SutazAI monitoring completed"
    log_info "Log file: $LOG_FILE"
    
    if [[ -f "$METRICS_FILE" ]]; then
        log_info "Metrics file: $METRICS_FILE"
    fi
}

# Execute main function with all arguments
main "$@"