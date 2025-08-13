#!/bin/bash
#
# Script Name: health_check.sh
# Purpose: Comprehensive health monitoring for SutazAI system components
# Author: Shell Automation Specialist (CLEAN-001)
# Date: August 8, 2025
# Usage: ./health_check.sh [--cron] [--auto-cleanup] [-- ] [--component=<name>]
# Dependencies: curl, docker, jq (optional)
# Environment: Production, staging, development
#

set -euo pipefail  # Strict error handling

# Script metadata

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

SCRIPT_VERSION="2.0.0"
SCRIPT_NAME="SutazAI Health Check"

# Project root detection
if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="${PROJECT_ROOT}"
elif [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
    PROJECT_ROOT="/opt/sutazaiapp"
else
    PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

# Colors and formatting
if [ -t 1 ]; then  # Check if stdout is a terminal
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Configuration
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/health_check.log"
CRON_MODE=0
AUTO_CLEANUP=0
 _MODE=0
COMPONENT_CHECK=""

# Thresholds
DISK_THRESHOLD=90
MEMORY_THRESHOLD=90
CPU_THRESHOLD=95

# Service endpoints configuration
declare -A SERVICE_ENDPOINTS=(
    ["PostgreSQL"]="docker:sutazai-postgres:pg_isready -U sutazai"
    ["Redis"]="docker:sutazai-redis:redis-cli ping"
    ["Ollama"]="http://localhost:10104/api/tags:200"
    ["Backend API"]="http://localhost:10010/health:200"
    ["Frontend"]="http://localhost:10011:200"
    ["Neo4j"]="http://localhost:10002:200"
    ["Qdrant"]="http://localhost:10101/health:200"
    ["FAISS"]="http://localhost:10103/health:200"
    ["ChromaDB"]="http://localhost:10100/api/v1/heartbeat:200"
    ["Prometheus"]="http://localhost:10200/-/healthy:200"
    ["Grafana"]="http://localhost:10201/api/health:200"
)

# Create log directory
mkdir -p "$LOG_DIR"

# Parse command line arguments
parse_args() {
    for arg in "$@"; do
        case $arg in
            --cron)
                CRON_MODE=1
                AUTO_CLEANUP=1
                ;;
            --auto-cleanup)
                AUTO_CLEANUP=1
                ;;
            -- )
                 _MODE=1
                ;;
            --component=*)
                COMPONENT_CHECK="${arg#*=}"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
$SCRIPT_NAME v$SCRIPT_VERSION

Usage: $0 [OPTIONS]

OPTIONS:
    --cron              Run in cron mode (log only, auto-cleanup enabled)
    --auto-cleanup      Enable automatic cleanup of issues
    --            Run   checks only (faster)
    --component=NAME    Check specific component only
    --help, -h          Show this help message

EXAMPLES:
    $0                      # Full interactive health check
    $0 --             # Quick health check
    $0 --component=Ollama   # Check only Ollama service
    $0 --cron               # Cron-friendly mode with auto-cleanup

COMPONENTS:
    PostgreSQL, Redis, Ollama, Backend API, Frontend, Neo4j,
    Qdrant, FAISS, ChromaDB, Prometheus, Grafana

EOF
}

# Logging functions
log_to_file() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

log_message() {
    local level="$1"
    local message="$2"
    
    log_to_file "$level" "$message"
    
    if [ $CRON_MODE -eq 0 ]; then
        case $level in
            INFO)
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            WARNING)
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
            *)
                echo "$message"
                ;;
        esac
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Print section headers
print_header() {
    local title="$1"
    if [ $CRON_MODE -eq 0 ]; then
        echo
        echo -e "${BOLD}=== $title ===${NC}"
    fi
    log_to_file "INFO" "Starting: $title"
}

# Service check functions
check_http_endpoint() {
    local service_name="$1"
    local url="$2"
    local expected_code="${3:-200}"
    local timeout="${4:-10}"
    
    if ! command_exists curl; then
        log_message "WARNING" "$service_name: curl not available, skipping HTTP check"
        return 2
    fi
    
    local response_code
    if response_code=$(timeout "$timeout" curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response_code" -eq "$expected_code" ]; then
            log_message "SUCCESS" "$service_name: HTTP $response_code ✓"
            return 0
        else
            log_message "ERROR" "$service_name: HTTP $response_code (expected $expected_code)"
            return 1
        fi
    else
        log_message "ERROR" "$service_name: HTTP endpoint unreachable"
        return 1
    fi
}

check_docker_service() {
    local service_name="$1"
    local container_name="$2"
    local docker_command="$3"
    
    if ! command_exists docker; then
        log_message "WARNING" "$service_name: docker not available"
        return 2
    fi
    
    # Check if container is running
    if ! docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^${container_name}$"; then
        log_message "ERROR" "$service_name: container $container_name not running"
        return 1
    fi
    
    # Execute health check command in container
    if docker exec "$container_name" sh -c "$docker_command" >/dev/null 2>&1; then
        log_message "SUCCESS" "$service_name: container health check ✓"
        return 0
    else
        log_message "ERROR" "$service_name: container health check failed"
        return 1
    fi
}

# Universal service checker
check_service() {
    local service_name="$1"
    local check_spec="$2"
    
    if [ "$COMPONENT_CHECK" != "" ] && [ "$service_name" != "$COMPONENT_CHECK" ]; then
        return 0  # Skip non-matching components
    fi
    
    case "$check_spec" in
        http://*)
            local url="${check_spec%:*}"
            local expected_code="${check_spec##*:}"
            check_http_endpoint "$service_name" "$url" "$expected_code"
            ;;
        docker:*)
            local container_name=$(echo "$check_spec" | cut -d: -f2)
            local docker_command=$(echo "$check_spec" | cut -d: -f3-)
            check_docker_service "$service_name" "$container_name" "$docker_command"
            ;;
        *)
            log_message "ERROR" "$service_name: unknown check specification: $check_spec"
            return 1
            ;;
    esac
}

# System resource monitoring
check_system_resources() {
    if [ $ _MODE -eq 1 ]; then
        return 0
    fi
    
    print_header "System Resources"
    
    # Disk usage
    if command_exists df; then
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
        if [ "$disk_usage" -gt "$DISK_THRESHOLD" ]; then
            log_message "ERROR" "Disk usage critical: ${disk_usage}%"
        elif [ "$disk_usage" -gt $((DISK_THRESHOLD - 10)) ]; then
            log_message "WARNING" "Disk usage high: ${disk_usage}%"
        else
            log_message "SUCCESS" "Disk usage normal: ${disk_usage}%"
        fi
    fi
    
    # Memory usage
    if command_exists free; then
        local mem_usage=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
        if [ "$mem_usage" -gt "$MEMORY_THRESHOLD" ]; then
            log_message "ERROR" "Memory usage critical: ${mem_usage}%"
        elif [ "$mem_usage" -gt $((MEMORY_THRESHOLD - 10)) ]; then
            log_message "WARNING" "Memory usage high: ${mem_usage}%"
        else
            log_message "SUCCESS" "Memory usage normal: ${mem_usage}%"
        fi
    fi
    
    # CPU load
    if [ -f /proc/loadavg ]; then
        local cpu_cores=$(grep -c ^processor /proc/cpuinfo)
        local cpu_load=$(cat /proc/loadavg | awk '{print $1}')
        local cpu_load_percent=$(echo "$cpu_load $cpu_cores" | awk '{printf "%d", ($1/$2)*100}')
        
        if [ "$cpu_load_percent" -gt "$CPU_THRESHOLD" ]; then
            log_message "ERROR" "CPU load critical: ${cpu_load_percent}%"
        elif [ "$cpu_load_percent" -gt $((CPU_THRESHOLD - 10)) ]; then
            log_message "WARNING" "CPU load high: ${cpu_load_percent}%"
        else
            log_message "SUCCESS" "CPU load normal: ${cpu_load_percent}%"
        fi
    fi
}

# Container resource usage
check_container_resources() {
    if [ $ _MODE -eq 1 ] || ! command_exists docker; then
        return 0
    fi
    
    print_header "Container Resources"
    
    # Get container stats
    local container_stats
    if container_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null); then
        log_message "INFO" "Container resource usage:"
        if [ $CRON_MODE -eq 0 ]; then
            echo "$container_stats" | grep sutazai || echo "No sutazai containers found"
        fi
    else
        log_message "WARNING" "Could not retrieve container stats"
    fi
}

# Test model functionality
test_model_functionality() {
    if [ $ _MODE -eq 1 ]; then
        return 0
    fi
    
    print_header "Model Testing"
    
    if command_exists curl && command_exists jq; then
        local response
        if response=$(curl -s -X POST "http://localhost:10104/api/generate" \
            -d '{"model": "tinyllama", "prompt": "Hello", "stream": false}' \
            --max-time 30 2>/dev/null); then
            
            local model_response
            if model_response=$(echo "$response" | jq -r .response 2>/dev/null); then
                if [ -n "$model_response" ] && [ "$model_response" != "null" ]; then
                    log_message "SUCCESS" "TinyLlama model responding ✓"
                    log_to_file "INFO" "Model response preview: ${model_response:0:50}..."
                else
                    log_message "ERROR" "TinyLlama model not responding properly"
                fi
            else
                log_message "WARNING" "Could not parse model response"
            fi
        else
            log_message "ERROR" "TinyLlama model unreachable"
        fi
    else
        log_message "WARNING" "Model testing requires curl and jq"
    fi
}

# Cleanup functions
cleanup_stale_processes() {
    if [ $AUTO_CLEANUP -eq 0 ]; then
        return 0
    fi
    
    print_header "Process Cleanup"
    
    # Clean up zombie processes
    local zombie_count=$(ps aux | awk '$8 ~ /^Z/ {print $2}' | wc -l)
    if [ "$zombie_count" -gt 0 ]; then
        log_message "WARNING" "Found $zombie_count zombie processes"
    fi
    
    # Clean up temporary files
    if [ -d "${PROJECT_ROOT}/tmp" ]; then
        find "${PROJECT_ROOT}/tmp" -type f -mtime +1 -delete 2>/dev/null || true
        log_message "INFO" "Cleaned temporary files"
    fi
    
    # Clean old log files
    if [ -d "$LOG_DIR" ]; then
        find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
        find "$LOG_DIR" -name "*.log.gz" -mtime +30 -delete 2>/dev/null || true
        log_message "INFO" "Rotated old log files"
    fi
}

# Generate summary
generate_summary() {
    local error_count=$(grep -c "\[ERROR\]" "$LOG_FILE" 2>/dev/null || echo 0)
    local warning_count=$(grep -c "\[WARNING\]" "$LOG_FILE" 2>/dev/null || echo 0)
    local success_count=$(grep -c "\[SUCCESS\]" "$LOG_FILE" 2>/dev/null || echo 0)
    
    print_header "Health Check Summary"
    
    local health_status
    if [ "$error_count" -gt 0 ]; then
        health_status="${RED}CRITICAL${NC}"
        local exit_code=2
    elif [ "$warning_count" -gt 0 ]; then
        health_status="${YELLOW}WARNING${NC}"
        local exit_code=1
    else
        health_status="${GREEN}HEALTHY${NC}"
        local exit_code=0
    fi
    
    if [ $CRON_MODE -eq 0 ]; then
        echo -e "Overall Status: $health_status"
        echo -e "Errors: ${RED}$error_count${NC}"
        echo -e "Warnings: ${YELLOW}$warning_count${NC}"
        echo -e "Healthy: ${GREEN}$success_count${NC}"
        echo -e "Log file: $LOG_FILE"
        echo
    fi
    
    log_to_file "INFO" "Health check completed - Status: $health_status, Errors: $error_count, Warnings: $warning_count, Healthy: $success_count"
    
    return $exit_code
}

# Main execution
main() {
    # Parse arguments
    parse_args "$@"
    
    # Print startup banner
    if [ $CRON_MODE -eq 0 ]; then
        echo -e "${BOLD}$SCRIPT_NAME v$SCRIPT_VERSION${NC}"
        echo -e "${BOLD}Timestamp: $TIMESTAMP${NC}"
        echo -e "${BOLD}================================${NC}"
    fi
    
    log_to_file "INFO" "Health check started - Mode: $([ $CRON_MODE -eq 1 ] && echo "CRON" || echo "INTERACTIVE"), Component: ${COMPONENT_CHECK:-ALL}"
    
    # Core service checks
    print_header "Core Services"
    local service_errors=0
    
    for service_name in "${!SERVICE_ENDPOINTS[@]}"; do
        if ! check_service "$service_name" "${SERVICE_ENDPOINTS[$service_name]}"; then
            ((service_errors++))
        fi
    done
    
    # System resource monitoring
    check_system_resources
    
    # Container monitoring
    check_container_resources
    
    # Model testing
    test_model_functionality
    
    # Cleanup if enabled
    cleanup_stale_processes
    
    # Generate final summary and exit with appropriate code
    generate_summary
    return $?
}

# Execute main function
main "$@"