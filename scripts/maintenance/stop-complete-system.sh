#!/bin/bash

# Sutazai Hygiene Enforcement System - Graceful Shutdown Script
# Purpose: Safely stop all system components with zero data loss
# Usage: ./stop-complete-system.sh [--force] [--timeout SECONDS]

set -euo pipefail

# Configuration

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

PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${LOG_DIR}/pids"
SYSTEM_LOG="${LOG_DIR}/system-shutdown.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
FORCE_STOP=false
SHUTDOWN_TIMEOUT=30

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_STOP=true
            shift
            ;;
        --timeout)
            SHUTDOWN_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force              Force immediate shutdown without graceful period"
            echo "  --timeout SECONDS    Graceful shutdown timeout (default: 30)"
            echo "  --help              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Color based on level
    local color=""
    case $level in
        ERROR) color=$RED ;;
        WARNING) color=$YELLOW ;;
        SUCCESS) color=$GREEN ;;
        INFO) color=$BLUE ;;
    esac
    
    echo -e "${color}[$timestamp] [$level] $message${NC}"
    echo "[$timestamp] [$level] $message" >> "$SYSTEM_LOG" 2>/dev/null || true
}

# Stop individual service gracefully
stop_service() {
    local service_name=$1
    local pid_file="${PID_DIR}/${service_name}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        log INFO "Service $service_name is not running (no PID file)"
        return 0
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        log INFO "Service $service_name is not running (PID $pid not found)"
        rm -f "$pid_file"
        return 0
    fi
    
    log INFO "Stopping service: $service_name (PID: $pid)"
    
    if [[ "$FORCE_STOP" == "true" ]]; then
        # Force stop immediately
        kill -KILL "$pid" 2>/dev/null || true
        log WARNING "Force killed $service_name"
    else
        # Graceful shutdown
        kill -TERM "$pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local wait_count=0
        while [[ $wait_count -lt $SHUTDOWN_TIMEOUT ]]; do
            if ! kill -0 "$pid" 2>/dev/null; then
                log SUCCESS "Service $service_name stopped gracefully"
                rm -f "$pid_file"
                return 0
            fi
            
            sleep 1
            wait_count=$((wait_count + 1))
            
            # Show progress every 5 seconds
            if [[ $((wait_count % 5)) -eq 0 ]]; then
                log INFO "Waiting for $service_name to shutdown... (${wait_count}s)"
            fi
        done
        
        # Force kill if graceful shutdown failed
        if kill -0 "$pid" 2>/dev/null; then
            log WARNING "Service $service_name did not stop gracefully, force killing..."
            kill -KILL "$pid" 2>/dev/null || true
            
            # Wait a moment for the kill to take effect
            sleep 2
            
            if kill -0 "$pid" 2>/dev/null; then
                log ERROR "Failed to stop service $service_name (PID: $pid)"
                return 1
            else
                log WARNING "Service $service_name force killed"
            fi
        fi
    fi
    
    rm -f "$pid_file"
    return 0
}

# Get list of running services in proper shutdown order
get_services_shutdown_order() {
    local services=()
    
    # Add services in reverse dependency order
    # (most dependent services first, infrastructure services last)
    
    [[ -f "${PID_DIR}/health-monitor.pid" ]] && services+=("health-monitor")
    [[ -f "${PID_DIR}/testing-qa-validator.pid" ]] && services+=("testing-qa-validator")
    [[ -f "${PID_DIR}/dashboard-server.pid" ]] && services+=("dashboard-server")
    [[ -f "${PID_DIR}/system-orchestrator.pid" ]] && services+=("system-orchestrator")
    [[ -f "${PID_DIR}/rule-control-api.pid" ]] && services+=("rule-control-api")
    
    echo "${services[@]}"
}

# Stop all system services
stop_all_services() {
    log INFO "🛑 Initiating system shutdown..."
    
    local services=($(get_services_shutdown_order))
    
    if [[ ${#services[@]} -eq 0 ]]; then
        log INFO "No running services found"
        return 0
    fi
    
    log INFO "Services to stop: ${services[*]}"
    
    local failed_services=()
    
    for service_name in "${services[@]}"; do
        if ! stop_service "$service_name"; then
            failed_services+=("$service_name")
        fi
        
        # Small delay between service stops to avoid overwhelming the system
        sleep 1
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log ERROR "Failed to stop services: ${failed_services[*]}"
        return 1
    fi
    
    log SUCCESS "All services stopped successfully"
    return 0
}

# Clean up system resources
cleanup_resources() {
    log INFO "🧹 Cleaning up system resources..."
    
    # Clean up temporary files
    local temp_patterns=(
        "${LOG_DIR}/*.tmp"
        "${LOG_DIR}/pids/*.tmp"
        "/tmp/sutazai-*"
        "/tmp/hygiene-*"
    )
    
    for pattern in "${temp_patterns[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            rm -f $pattern
            log INFO "Cleaned up temporary files: $pattern"
        fi
    done
    
    # Clean up empty PID files
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$pid_file"
                log INFO "Removed stale PID file: $(basename "$pid_file")"
            fi
        fi
    done
    
    # Compress old log files
    find "${LOG_DIR}" -name "*.log" -size +10M -mtime +1 -exec gzip {} \; 2>/dev/null || true
    
    log SUCCESS "Resource cleanup completed"
}

# Generate shutdown report
generate_shutdown_report() {
    log INFO "📊 Generating shutdown report..."
    
    local report_file="${LOG_DIR}/shutdown-report-$(date +%Y%m%d_%H%M%S).json"
    local remaining_processes=()
    
    # Check for any remaining processes
    while IFS= read -r line; do
        remaining_processes+=("$line")
    done < <(pgrep -f "sutazai|hygiene|rule-control" 2>/dev/null || true)
    
    # System resource usage at shutdown
    local memory_usage=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
    local cpu_load=$(uptime | grep -oP 'load average: \K[0-9.]+')
    
    # Create shutdown report
    cat > "$report_file" << EOF
{
    "shutdown_timestamp": "$(date -Iseconds)",
    "shutdown_type": "$([[ "$FORCE_STOP" == "true" ]] && echo "forced" || echo "graceful")",
    "shutdown_timeout": $SHUTDOWN_TIMEOUT,
    "system_info": {
        "hostname": "$(hostname)",
        "final_memory_usage_percent": $memory_usage,
        "final_cpu_load": "$cpu_load",
        "remaining_processes": [$(printf '"%s",' "${remaining_processes[@]}" | sed 's/,$//')]
    },
    "cleanup_performed": true,
    "status": "$([ ${#remaining_processes[@]} -eq 0 ] && echo "clean_shutdown" || echo "partial_shutdown")"
}
EOF
    
    log SUCCESS "Shutdown report generated: $report_file"
    
    # Display summary
    echo ""
    echo "🏁 SHUTDOWN SUMMARY"
    echo "==================="
    echo "Shutdown Type: $([[ "$FORCE_STOP" == "true" ]] && echo "Forced" || echo "Graceful")"
    echo "Shutdown Time: $(date)"
    echo "Timeout Used: ${SHUTDOWN_TIMEOUT}s"
    
    if [[ ${#remaining_processes[@]} -eq 0 ]]; then
        echo "Status: ✅ Clean shutdown - all processes stopped"
    else
        echo "Status: ⚠️ Partial shutdown - ${#remaining_processes[@]} processes may still be running"
        echo "Remaining PIDs: ${remaining_processes[*]}"
    fi
    
    echo ""
    echo "📋 Final System State:"
    echo "  Memory Usage: ${memory_usage}%"
    echo "  CPU Load: ${cpu_load}"
    echo "  Log Directory: $LOG_DIR"
    echo ""
}

# Verify complete shutdown
verify_shutdown() {
    log INFO "🔍 Verifying complete shutdown..."
    
    local remaining_services=()
    
    # Check for any remaining PID files with running processes
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local service_name=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file" 2>/dev/null || echo "")
            
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                remaining_services+=("$service_name:$pid")
            fi
        fi
    done
    
    # Check for any sutazai-related processes
    local orphaned_processes=()
    while IFS= read -r line; do
        orphaned_processes+=("$line")
    done < <(pgrep -f "sutazai|hygiene|rule-control" 2>/dev/null || true)
    
    # Report findings
    if [[ ${#remaining_services[@]} -eq 0 ]] && [[ ${#orphaned_processes[@]} -eq 0 ]]; then
        log SUCCESS "✅ Complete shutdown verified - no remaining processes"
        return 0
    else
        if [[ ${#remaining_services[@]} -gt 0 ]]; then
            log WARNING "⚠️ Services still running: ${remaining_services[*]}"
        fi
        
        if [[ ${#orphaned_processes[@]} -gt 0 ]]; then
            log WARNING "⚠️ Orphaned processes found: ${orphaned_processes[*]}"
            
            if [[ "$FORCE_STOP" == "true" ]]; then
                log INFO "Force stopping orphaned processes..."
                for pid in "${orphaned_processes[@]}"; do
                    kill -KILL "$pid" 2>/dev/null || true
                done
                sleep 2
                log INFO "Orphaned processes terminated"
            fi
        fi
        
        return 1
    fi
}

# Main execution flow
main() {
    log INFO "🛑 Starting Sutazai Hygiene Enforcement System shutdown..."
    
    # Check if system is running
    if [[ ! -d "$PID_DIR" ]] || [[ -z "$(ls -A "$PID_DIR" 2>/dev/null)" ]]; then
        log INFO "No system components appear to be running"
        exit 0
    fi
    
    local start_time=$(date +%s)
    
    # Stop all services
    if stop_all_services; then
        log SUCCESS "Service shutdown completed successfully"
    else
        log ERROR "Some services failed to stop properly"
        if [[ "$FORCE_STOP" == "false" ]]; then
            log INFO "Consider using --force flag for immediate termination"
        fi
    fi
    
    # Clean up resources
    cleanup_resources
    
    # Verify shutdown
    verify_shutdown
    
    # Generate report
    generate_shutdown_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log SUCCESS "🎉 System shutdown completed in ${duration} seconds"
    
    echo ""
    echo "💡 Helpful Commands:"
    echo "  Check for any remaining processes: pgrep -f 'sutazai|hygiene'"
    echo "  View shutdown logs: tail -f $SYSTEM_LOG"
    echo "  Start system again: ./start-complete-system.sh"
    echo ""
}

# Ensure log directory exists
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Execute main function
main "$@"