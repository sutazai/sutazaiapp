#!/bin/bash
# 🔍 SutazAI Agent Infrastructure Monitoring
# Real-time monitoring and health checking for all 38 AI agents

set -euo pipefail

# Configuration
PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MONITOR_LOG="$LOG_DIR/agent_monitoring_$TIMESTAMP.log"
ALERT_LOG="$LOG_DIR/agent_alerts_$TIMESTAMP.log"

# Monitoring intervals
HEALTH_CHECK_INTERVAL=30
PERFORMANCE_CHECK_INTERVAL=60
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_RESPONSE_TIME=5000  # 5 seconds

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# ===============================================
# 🚀 LOGGING FUNCTIONS
# ===============================================

log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${BLUE}ℹ️  [$timestamp] $message${NC}" | tee -a "$MONITOR_LOG"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}✅ [$timestamp] $message${NC}" | tee -a "$MONITOR_LOG"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}⚠️  [$timestamp] WARNING: $message${NC}" | tee -a "$MONITOR_LOG"
    echo -e "[$timestamp] WARNING: $message" >> "$ALERT_LOG"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}❌ [$timestamp] ERROR: $message${NC}" | tee -a "$MONITOR_LOG"
    echo -e "[$timestamp] ERROR: $message" >> "$ALERT_LOG"
}

log_header() {
    local message="$1"
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}" | tee -a "$MONITOR_LOG"
    echo -e "${CYAN}║ $message${NC}" | tee -a "$MONITOR_LOG"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}" | tee -a "$MONITOR_LOG"
}

# ===============================================
# 🚀 CORE MONITORING FUNCTIONS
# ===============================================

check_agent_health() {
    local agent_name="$1"
    local container_name="sutazai-$agent_name"
    
    # Check if container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "^$container_name$"; then
        log_error "Agent $agent_name: Container not running"
        return 1
    fi
    
    # Check container health status
    local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "unknown")
    
    case "$health_status" in
        "healthy")
            log_success "Agent $agent_name: Healthy"
            return 0
            ;;
        "unhealthy")
            log_error "Agent $agent_name: Unhealthy"
            return 1
            ;;
        "starting")
            log_warn "Agent $agent_name: Still starting"
            return 1
            ;;
        *)
            log_warn "Agent $agent_name: Health status unknown"
            return 1
            ;;
    esac
}

check_agent_performance() {
    local agent_name="$1"
    local container_name="sutazai-$agent_name"
    
    # Get container stats
    local stats_output=$(docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" "$container_name" 2>/dev/null || echo "N/A N/A N/A N/A")
    
    if [[ "$stats_output" == "N/A N/A N/A N/A" ]]; then
        log_error "Agent $agent_name: Cannot retrieve performance stats"
        return 1
    fi
    
    # Parse stats (skip header line)
    local stats_line=$(echo "$stats_output" | tail -n 1)
    local cpu_percent=$(echo "$stats_line" | awk '{print $1}' | sed 's/%//')
    local mem_percent=$(echo "$stats_line" | awk '{print $2}' | sed 's/%//')
    
    # Check CPU threshold
    if (( $(echo "$cpu_percent > $ALERT_THRESHOLD_CPU" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "Agent $agent_name: High CPU usage: ${cpu_percent}%"
    fi
    
    # Check memory threshold
    if (( $(echo "$mem_percent > $ALERT_THRESHOLD_MEMORY" | bc -l 2>/dev/null || echo 0) )); then
        log_warn "Agent $agent_name: High memory usage: ${mem_percent}%"
    fi
    
    log_info "Agent $agent_name: CPU: ${cpu_percent}%, Memory: ${mem_percent}%"
}

check_agent_api_endpoint() {
    local agent_name="$1"
    local port="$2"
    local endpoint="http://localhost:${port}/health"
    
    local start_time=$(date +%s%3N)
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$endpoint" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    if [[ "$response_code" == "200" ]]; then
        if [[ $response_time -gt $ALERT_THRESHOLD_RESPONSE_TIME ]]; then
            log_warn "Agent $agent_name: Slow response time: ${response_time}ms"
        else
            log_success "Agent $agent_name: API healthy (${response_time}ms)"
        fi
        return 0
    else
        log_error "Agent $agent_name: API unhealthy (HTTP $response_code)"
        return 1
    fi
}

check_message_bus_health() {
    log_info "Checking agent message bus..."
    
    if check_agent_api_endpoint "agent-message-bus" "8299"; then
        # Check message bus stats
        local stats=$(curl -s "http://localhost:8299/stats" 2>/dev/null || echo "{}")
        local active_connections=$(echo "$stats" | jq -r '.active_connections // "unknown"' 2>/dev/null || echo "unknown")
        local total_messages=$(echo "$stats" | jq -r '.total_messages // "unknown"' 2>/dev/null || echo "unknown")
        
        log_info "Message Bus: $active_connections active connections, $total_messages total messages"
    fi
}

check_agent_registry_health() {
    log_info "Checking agent registry..."
    
    if check_agent_api_endpoint "agent-registry" "8300"; then
        # Check registry stats
        local stats=$(curl -s "http://localhost:8300/stats" 2>/dev/null || echo "{}")
        local total_agents=$(echo "$stats" | jq -r '.total_agents // "unknown"' 2>/dev/null || echo "unknown")
        local online_agents=$(echo "$stats" | jq -r '.online_agents // "unknown"' 2>/dev/null || echo "unknown")
        
        log_info "Agent Registry: $online_agents/$total_agents agents online"
    fi
}

# ===============================================
# 🚀 AGENT DISCOVERY AND MONITORING
# ===============================================

discover_agents() {
    log_info "Discovering running agents..."
    
    # Get all running SutazAI containers
    local running_agents=()
    while IFS= read -r container_name; do
        if [[ "$container_name" =~ ^sutazai- ]]; then
            local agent_name="${container_name#sutazai-}"
            running_agents+=("$agent_name")
        fi
    done < <(docker ps --format "{{.Names}}")
    
    echo "${running_agents[@]}"
}

monitor_all_agents() {
    local agents=($1)
    local healthy_count=0
    local unhealthy_count=0
    local total_count=${#agents[@]}
    
    log_header "MONITORING ${total_count} AGENTS"
    
    for agent in "${agents[@]}"; do
        if check_agent_health "$agent"; then
            ((healthy_count++))
            
            # Check performance for healthy agents
            check_agent_performance "$agent" || true
            
            # Check API endpoints for known agents with API ports
            case "$agent" in
                "agent-message-bus")
                    check_agent_api_endpoint "$agent" "8299" || true
                    ;;
                "agent-registry")
                    check_agent_api_endpoint "$agent" "8300" || true
                    ;;
                "agi-system-architect")
                    check_agent_api_endpoint "$agent" "8201" || true
                    ;;
                "autonomous-system-controller")
                    check_agent_api_endpoint "$agent" "8202" || true
                    ;;
                # Add more agents as needed
            esac
        else
            ((unhealthy_count++))
        fi
    done
    
    log_info "Health Summary: $healthy_count healthy, $unhealthy_count unhealthy, $total_count total"
    
    # Overall system health percentage
    local health_percentage=$(( (healthy_count * 100) / total_count ))
    
    if [[ $health_percentage -ge 90 ]]; then
        log_success "System Health: ${health_percentage}% - Excellent"
    elif [[ $health_percentage -ge 75 ]]; then
        log_info "System Health: ${health_percentage}% - Good"
    elif [[ $health_percentage -ge 50 ]]; then
        log_warn "System Health: ${health_percentage}% - Degraded"
    else
        log_error "System Health: ${health_percentage}% - Critical"
    fi
}

# ===============================================
# 🚀 SYSTEM METRICS AND REPORTING
# ===============================================

generate_system_report() {
    log_header "SYSTEM METRICS REPORT"
    
    # Docker system info
    log_info "Docker System Overview:"
    docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}\t{{.Reclaimable}}" | while read -r line; do
        log_info "  $line"
    done
    
    # System resources
    log_info "System Resources:"
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local memory_info=$(free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)", $3,$2,$3*100/$2}')
    local disk_usage=$(df -h . | awk 'NR==2{printf "Used: %s/%s (%s)", $3,$2,$5}')
    
    log_info "  CPU Usage: ${cpu_usage:-unknown}"
    log_info "  Memory: ${memory_info:-unknown}"
    log_info "  Disk: ${disk_usage:-unknown}"
    
    # Network connectivity
    log_info "Network Connectivity:"
    local redis_status=$(docker compose exec redis redis-cli ping 2>/dev/null || echo "FAILED")
    local postgres_status=$(docker compose exec postgres pg_isready -U sutazai 2>/dev/null && echo "READY" || echo "NOT READY")
    
    log_info "  Redis: $redis_status"
    log_info "  PostgreSQL: $postgres_status"
}

# ===============================================
# 🚀 AUTOMATED RECOVERY ACTIONS
# ===============================================

attempt_agent_recovery() {
    local agent_name="$1"
    local container_name="sutazai-$agent_name"
    
    log_warn "Attempting recovery for agent: $agent_name"
    
    # Try restarting the container
    if docker restart "$container_name" >/dev/null 2>&1; then
        log_info "Restarted container for agent: $agent_name"
        
        # Wait for recovery
        sleep 30
        
        # Check if recovery was successful
        if check_agent_health "$agent_name"; then
            log_success "Recovery successful for agent: $agent_name"
            return 0
        else
            log_error "Recovery failed for agent: $agent_name"
            return 1
        fi
    else
        log_error "Failed to restart container for agent: $agent_name"
        return 1
    fi
}

# ===============================================
# 🚀 MAIN MONITORING LOOP
# ===============================================

continuous_monitoring() {
    log_header "STARTING CONTINUOUS MONITORING"
    log_info "Health check interval: ${HEALTH_CHECK_INTERVAL}s"
    log_info "Performance check interval: ${PERFORMANCE_CHECK_INTERVAL}s"
    log_info "Monitor log: $MONITOR_LOG"
    log_info "Alert log: $ALERT_LOG"
    
    local cycle_count=0
    
    while true; do
        ((cycle_count++))
        log_info "Starting monitoring cycle #$cycle_count"
        
        # Discover running agents
        local agents_list=$(discover_agents)
        local agents_array=($agents_list)
        
        if [[ ${#agents_array[@]} -eq 0 ]]; then
            log_error "No SutazAI agents found running!"
        else
            # Monitor all agents
            monitor_all_agents "$agents_list"
            
            # Check core infrastructure
            check_message_bus_health
            check_agent_registry_health
            
            # Generate system report every 5 cycles
            if (( cycle_count % 5 == 0 )); then
                generate_system_report
            fi
        fi
        
        log_info "Monitoring cycle #$cycle_count completed. Next check in ${HEALTH_CHECK_INTERVAL}s"
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

# ===============================================
# 🚀 COMMAND LINE INTERFACE
# ===============================================

show_help() {
    echo "SutazAI Agent Infrastructure Monitoring"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  monitor     Start continuous monitoring (default)"
    echo "  check       Run single health check"
    echo "  report      Generate system report"
    echo "  agents      List all discovered agents"
    echo "  recovery    Attempt recovery for specific agent"
    echo "  help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0                              # Start continuous monitoring"
    echo "  $0 check                        # Single health check"  
    echo "  $0 recovery agent-name          # Recover specific agent"
    echo
}

# ===============================================
# 🚀 SCRIPT EXECUTION
# ===============================================

main() {
    local command="${1:-monitor}"
    
    case "$command" in
        "monitor")
            continuous_monitoring
            ;;
        "check")
            log_header "SINGLE HEALTH CHECK"
            local agents_list=$(discover_agents)
            monitor_all_agents "$agents_list"
            ;;
        "report")
            generate_system_report
            ;;
        "agents")
            log_header "DISCOVERED AGENTS"
            local agents_list=$(discover_agents)
            local agents_array=($agents_list)
            for agent in "${agents_array[@]}"; do
                log_info "Agent: $agent"
            done
            log_info "Total agents discovered: ${#agents_array[@]}"
            ;;
        "recovery")
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 recovery <agent-name>"
                exit 1
            fi
            attempt_agent_recovery "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'log_info "Monitoring stopped by user"; exit 0' INT

# Run main function
main "$@"