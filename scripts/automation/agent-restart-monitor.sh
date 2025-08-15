#!/bin/bash
# Purpose: Monitor AI agents and restart failed ones automatically
# Usage: ./agent-restart-monitor.sh [--dry-run] [--max-restarts N]
# Requires: Docker, jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"
STATE_DIR="$BASE_DIR/data/agent-monitor"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
DRY_RUN=false
MAX_RESTARTS_PER_HOUR=3    # Maximum restarts per agent per hour
HEALTH_CHECK_INTERVAL=60   # Seconds between health checks
MAX_RESPONSE_TIME=30       # Maximum response time for health checks (seconds)
CONSECUTIVE_FAILURES=3     # Number of consecutive failures before restart

# Expected agents configuration
EXPECTED_AGENTS=(
    "sutazai-senior-ai-engineer:8001"
    "sutazai-infrastructure-devops-manager:8002"
    "sutazai-testing-qa-validator:8003"
    "sutazai-agent-orchestrator:8004"
    "sutazai- system-architect:8005"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --max-restarts)
            MAX_RESTARTS_PER_HOUR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--max-restarts N]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/agent_monitor_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Setup monitoring state directory
setup_state_directory() {
    log "INFO" "Setting up agent monitoring state directory..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$STATE_DIR" "$LOG_DIR"
        
        # Initialize state files for each agent if they don't exist
        for agent_config in "${EXPECTED_AGENTS[@]}"; do
            IFS=':' read -r agent_name port <<< "$agent_config"
            local state_file="$STATE_DIR/${agent_name}.json"
            
            if [[ ! -f "$state_file" ]]; then
                cat > "$state_file" << EOF
{
    "agent_name": "$agent_name",
    "port": $port,
    "consecutive_failures": 0,
    "last_successful_check": null,
    "last_restart": null,
    "restarts_in_hour": [],
    "total_restarts": 0,
    "status": "unknown"
}
EOF
                log "INFO" "Initialized state file for: $agent_name"
            fi
        done
    else
        log "INFO" "[DRY RUN] Would create state directory and initialize agent state files"
    fi
}

# Check if Docker daemon is running
check_docker_daemon() {
    if ! docker info >/dev/null 2>&1; then
        log "ERROR" "Docker daemon is not running or accessible"
        return 1
    fi
    return 0
}

# Get current restart count for agent in the last hour
get_restart_count_last_hour() {
    local agent_name="$1"
    local state_file="$STATE_DIR/${agent_name}.json"
    local current_time=$(date +%s)
    local one_hour_ago=$((current_time - 3600))
    
    if [[ ! -f "$state_file" ]]; then
        echo 0
        return
    fi
    
    # Count restarts in the last hour
    local restart_count=$(jq -r --arg threshold "$one_hour_ago" '
        .restarts_in_hour 
        | map(select(. > ($threshold | tonumber))) 
        | length' "$state_file" 2>/dev/null || echo 0)
    
    echo "$restart_count"
}

# Update agent state
update_agent_state() {
    local agent_name="$1"
    local status="$2"
    local state_file="$STATE_DIR/${agent_name}.json"
    local current_time=$(date +%s)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would update state for $agent_name: $status"
        return
    fi
    
    if [[ ! -f "$state_file" ]]; then
        log "ERROR" "State file not found: $state_file"
        return 1
    fi
    
    # Update the state based on status
    case $status in
        "healthy")
            jq --arg timestamp "$current_time" '
                .status = "healthy" |
                .consecutive_failures = 0 |
                .last_successful_check = $timestamp' \
                "$state_file" > "${state_file}.tmp" && mv "${state_file}.tmp" "$state_file"
            ;;
        "unhealthy")
            jq '.consecutive_failures += 1 | .status = "unhealthy"' \
                "$state_file" > "${state_file}.tmp" && mv "${state_file}.tmp" "$state_file"
            ;;
        "restarted")
            jq --arg timestamp "$current_time" '
                .last_restart = $timestamp |
                .total_restarts += 1 |
                .restarts_in_hour += [$timestamp] |
                .consecutive_failures = 0 |
                .status = "restarted"' \
                "$state_file" > "${state_file}.tmp" && mv "${state_file}.tmp" "$state_file"
            ;;
    esac
}

# Clean old restart timestamps (older than 1 hour)
clean_old_restart_timestamps() {
    local agent_name="$1"
    local state_file="$STATE_DIR/${agent_name}.json"
    local current_time=$(date +%s)
    local one_hour_ago=$((current_time - 3600))
    
    if [[ "$DRY_RUN" == "false" && -f "$state_file" ]]; then
        jq --arg threshold "$one_hour_ago" '
            .restarts_in_hour = (.restarts_in_hour | map(select(. > ($threshold | tonumber))))' \
            "$state_file" > "${state_file}.tmp" && mv "${state_file}.tmp" "$state_file"
    fi
}

# Check if container is running
is_container_running() {
    local container_name="$1"
    docker ps --format "{{.Names}}" | grep -q "^${container_name}$"
}

# Check agent health via HTTP endpoint
check_agent_health() {
    local agent_name="$1"
    local port="$2"
    local health_url="http://localhost:${port}/health"
    
    log "INFO" "Checking health for $agent_name on port $port..."
    
    # Check if container is running first
    if ! is_container_running "$agent_name"; then
        log "WARN" "$agent_name container is not running"
        return 1
    fi
    
    # Check health endpoint with timeout
    local response=$(curl -s -m "$MAX_RESPONSE_TIME" -w "%{http_code}:%{time_total}" "$health_url" 2>/dev/null || echo "000:0")
    IFS=':' read -r http_code response_time <<< "$response"
    
    if [[ "$http_code" == "200" ]]; then
        log "SUCCESS" "$agent_name is healthy (response time: ${response_time}s)"
        update_agent_state "$agent_name" "healthy"
        return 0
    else
        log "WARN" "$agent_name health check failed (HTTP $http_code, response time: ${response_time}s)"
        update_agent_state "$agent_name" "unhealthy"
        return 1
    fi
}

# Get consecutive failure count for agent
get_consecutive_failures() {
    local agent_name="$1"
    local state_file="$STATE_DIR/${agent_name}.json"
    
    if [[ -f "$state_file" ]]; then
        jq -r '.consecutive_failures' "$state_file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Restart agent container
restart_agent() {
    local agent_name="$1"
    
    log "INFO" "Attempting to restart agent: $agent_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would restart agent: $agent_name"
        return 0
    fi
    
    # Check current restart count
    local restart_count=$(get_restart_count_last_hour "$agent_name")
    
    if [[ $restart_count -ge $MAX_RESTARTS_PER_HOUR ]]; then
        log "ERROR" "Maximum restarts exceeded for $agent_name ($restart_count/$MAX_RESTARTS_PER_HOUR in last hour)"
        return 1
    fi
    
    # Attempt to restart the container
    if docker restart "$agent_name" >/dev/null 2>&1; then
        log "SUCCESS" "Restarted agent: $agent_name"
        update_agent_state "$agent_name" "restarted"
        
        # Wait for container to start
        sleep 10
        
        # Perform immediate health check
        IFS=':' read -r name port <<< "$(echo "${EXPECTED_AGENTS[@]}" | tr ' ' '\n' | grep "$agent_name")"
        if check_agent_health "$agent_name" "${port##*:}"; then
            log "SUCCESS" "Agent $agent_name is healthy after restart"
        else
            log "WARN" "Agent $agent_name is still unhealthy after restart"
        fi
        
        return 0
    else
        log "ERROR" "Failed to restart agent: $agent_name"
        return 1
    fi
}

# Check if agent needs restart based on consecutive failures
needs_restart() {
    local agent_name="$1"
    local consecutive_failures=$(get_consecutive_failures "$agent_name")
    
    if [[ $consecutive_failures -ge $CONSECUTIVE_FAILURES ]]; then
        return 0  # Needs restart
    else
        return 1  # Does not need restart
    fi
}

# Monitor all configured agents
monitor_agents() {
    log "INFO" "Starting agent monitoring cycle..."
    
    local total_agents=${#EXPECTED_AGENTS[@]}
    local healthy_agents=0
    local unhealthy_agents=0
    local restarted_agents=0
    
    for agent_config in "${EXPECTED_AGENTS[@]}"; do
        IFS=':' read -r agent_name port <<< "$agent_config"
        
        # Clean old restart timestamps
        clean_old_restart_timestamps "$agent_name"
        
        # Check agent health
        if check_agent_health "$agent_name" "$port"; then
            ((healthy_agents++))
        else
            ((unhealthy_agents++))
            
            # Check if restart is needed
            if needs_restart "$agent_name"; then
                log "WARN" "$agent_name has failed $CONSECUTIVE_FAILURES consecutive health checks, attempting restart..."
                
                if restart_agent "$agent_name"; then
                    ((restarted_agents++))
                else
                    log "ERROR" "Failed to restart $agent_name - manual intervention may be required"
                fi
            else
                local failures=$(get_consecutive_failures "$agent_name")
                log "INFO" "$agent_name has $failures consecutive failures (threshold: $CONSECUTIVE_FAILURES)"
            fi
        fi
    done
    
    log "INFO" "Monitoring cycle complete: $healthy_agents healthy, $unhealthy_agents unhealthy, $restarted_agents restarted"
    
    # Return status based on results
    if [[ $healthy_agents -eq $total_agents ]]; then
        return 0  # All healthy
    elif [[ $unhealthy_agents -gt $((total_agents / 2)) ]]; then
        return 2  # More than half unhealthy
    else
        return 1  # Some issues but manageable
    fi
}

# Generate monitoring report
generate_monitoring_report() {
    log "INFO" "Generating agent monitoring report..."
    
    local report_file="$LOG_DIR/agent_monitoring_report_$TIMESTAMP.json"
    local agents_status=()
    
    # Collect status for each agent
    for agent_config in "${EXPECTED_AGENTS[@]}"; do
        IFS=':' read -r agent_name port <<< "$agent_config"
        local state_file="$STATE_DIR/${agent_name}.json"
        
        if [[ -f "$state_file" ]]; then
            local agent_status=$(cat "$state_file")
            agents_status+=("$agent_status")
        fi
    done
    
    # Create agents array for JSON
    local agents_json="["
    local first=true
    for status in "${agents_status[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            agents_json+=","
        fi
        agents_json+="$status"
    done
    agents_json+="]"
    
    # Generate report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "monitoring_config": {
        "max_restarts_per_hour": $MAX_RESTARTS_PER_HOUR,
        "health_check_interval": $HEALTH_CHECK_INTERVAL,
        "max_response_time": $MAX_RESPONSE_TIME,
        "consecutive_failures_threshold": $CONSECUTIVE_FAILURES
    },
    "agents": $agents_json,
    "summary": {
        "total_agents": ${#EXPECTED_AGENTS[@]},
        "monitoring_mode": "$([ "$DRY_RUN" == "true" ] && echo "dry_run" || echo "active")"
    },
    "next_check": "$(date -d "+$((HEALTH_CHECK_INTERVAL / 60)) minutes" -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    log "SUCCESS" "Monitoring report saved to: $report_file"
    
    # Create symlink to latest report
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$report_file" "$LOG_DIR/latest_agent_monitoring_report.json"
    fi
}

# Main execution
main() {
    log "INFO" "Starting AI agent monitoring and restart system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no agents will be restarted"
    fi
    
    log "INFO" "Configuration: max_restarts=$MAX_RESTARTS_PER_HOUR/hour, consecutive_failures=$CONSECUTIVE_FAILURES, response_timeout=${MAX_RESPONSE_TIME}s"
    
    # Check prerequisites
    if ! check_docker_daemon; then
        log "ERROR" "Docker daemon check failed, aborting monitoring"
        exit 1
    fi
    
    # Setup state directory
    setup_state_directory
    
    # Monitor agents
    local monitoring_result=0
    monitor_agents || monitoring_result=$?
    
    # Generate report
    generate_monitoring_report
    
    log "SUCCESS" "Agent monitoring cycle completed"
    
    # Show summary
    echo
    echo "============================================"
    echo "       AGENT MONITORING SUMMARY"
    echo "============================================"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "ACTIVE MONITORING")"
    echo "Total agents: ${#EXPECTED_AGENTS[@]}"
    echo "Max restarts per hour: $MAX_RESTARTS_PER_HOUR"
    echo "Consecutive failure threshold: $CONSECUTIVE_FAILURES"
    echo "Timestamp: $(date)"
    echo "============================================"
    
    # Exit with appropriate code
    exit $monitoring_result
}

# Handle signals for graceful shutdown
trap 'log "INFO" "Received signal, shutting down gracefully..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"