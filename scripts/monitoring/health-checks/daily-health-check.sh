#!/bin/bash
# Purpose: Daily comprehensive health check and reporting for SutazAI system
# Usage: ./daily-health-check.sh [--verbose] [--email recipient@domain.com]
# Requires: Docker, curl, jq

set -euo pipefail


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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"
REPORT_DIR="$BASE_DIR/reports"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$REPORT_DIR/daily_health_report_$TIMESTAMP.json"
LOG_FILE="$LOG_DIR/daily_health_check_$TIMESTAMP.log"

# Create directories if they don't exist
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
EMAIL_RECIPIENT=""
CRITICAL_THRESHOLD=5  # Number of failed checks before marking system as critical

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --email)
            EMAIL_RECIPIENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--verbose] [--email recipient@domain.com]"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $level: $message" | tee -a "$LOG_FILE"
    
    if [[ "$VERBOSE" == "true" || "$level" == "ERROR" || "$level" == "WARN" ]]; then
        case $level in
            ERROR) echo -e "${RED}[$timestamp] $level: $message${NC}" ;;
            WARN) echo -e "${YELLOW}[$timestamp] $level: $message${NC}" ;;
            INFO) echo -e "${BLUE}[$timestamp] $level: $message${NC}" ;;
            SUCCESS) echo -e "${GREEN}[$timestamp] $level: $message${NC}" ;;
        esac
    fi
}

# Initialize report structure
init_report() {
    cat > "$REPORT_FILE" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "system_info": {
        "uptime": "$(uptime -p)",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}' | xargs)",
        "memory_usage": "$(free -h | grep Mem | awk '{print $3"/"$2" ("$4" free)"}')",
        "disk_usage": "$(df -h / | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')"
    },
    "checks": [],
    "summary": {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "status": "unknown"
    }
}
EOF
}

# Add check result to report
add_check_result() {
    local name="$1"
    local status="$2"
    local message="$3"
    local details="${4:-{}}"
    
    # Update the JSON report
    jq --arg name "$name" \
       --arg status "$status" \
       --arg message "$message" \
       --argjson details "$details" \
       '.checks += [{
           "name": $name,
           "status": $status,
           "message": $message,
           "timestamp": (now | strftime("%Y-%m-%dT%H:%M:%SZ")),
           "details": $details
       }] | .summary.total_checks += 1' \
       "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    
    case $status in
        "PASS")
            jq '.summary.passed += 1' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
            log "SUCCESS" "$name: $message"
            ;;
        "FAIL")
            jq '.summary.failed += 1' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
            log "ERROR" "$name: $message"
            ;;
        "WARN")
            jq '.summary.warnings += 1' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
            log "WARN" "$name: $message"
            ;;
    esac
}

# Check Docker daemon
check_docker() {
    log "INFO" "Checking Docker daemon..."
    if docker info >/dev/null 2>&1; then
        local version=$(docker --version | cut -d' ' -f3 | tr -d ',')
        add_check_result "Docker Daemon" "PASS" "Docker is running" "{\"version\": \"$version\"}"
    else
        add_check_result "Docker Daemon" "FAIL" "Docker daemon is not running or accessible" "{}"
        return 1
    fi
}

# Check core services
check_core_services() {
    log "INFO" "Checking core services..."
    
    local services=("sutazai-postgres- " "sutazai-redis- " "sutazai-ollama")
    
    for service in "${services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${service}$"; then
            local status=$(docker inspect --format='{{.State.Status}}' "$service")
            local uptime=$(docker inspect --format='{{.State.StartedAt}}' "$service")
            add_check_result "Service: $service" "PASS" "Container is running" "{\"status\": \"$status\", \"started_at\": \"$uptime\"}"
        else
            add_check_result "Service: $service" "FAIL" "Container is not running" "{}"
        fi
    done
}

# Check API endpoints
check_api_endpoints() {
    log "INFO" "Checking API endpoints..."
    
    local endpoints=(
        "Ollama API:http://localhost:10104/api/tags:200"
        "Backend Health:http://localhost:8000/health:200"
        "Backend API v1:http://localhost:8000/api/v1/health:200"
        "Frontend:http://localhost:8501:200"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r name url expected_code <<< "$endpoint_info"
        
        local response=$(curl -s -o /dev/null -w "%{http_code}:%{time_total}" "$url" 2>/dev/null || echo "000:0")
        IFS=':' read -r actual_code response_time <<< "$response"
        
        if [[ "$actual_code" == "$expected_code" ]]; then
            add_check_result "$name" "PASS" "Endpoint is responding correctly" "{\"response_code\": $actual_code, \"response_time\": $response_time}"
        else
            add_check_result "$name" "FAIL" "Endpoint returned code $actual_code (expected $expected_code)" "{\"response_code\": $actual_code, \"expected_code\": $expected_code}"
        fi
    done
}

# Check database connectivity
check_database_connectivity() {
    log "INFO" "Checking database connectivity..."
    
    # PostgreSQL
    if docker exec sutazai-postgres-  pg_isready -U sutazai >/dev/null 2>&1; then
        local db_size=$(docker exec sutazai-postgres-  psql -U sutazai -d sutazai -t -c "SELECT pg_size_pretty(pg_database_size('sutazai'));" 2>/dev/null | xargs)
        add_check_result "PostgreSQL Connection" "PASS" "Database is accessible" "{\"database_size\": \"$db_size\"}"
    else
        add_check_result "PostgreSQL Connection" "FAIL" "Cannot connect to PostgreSQL" "{}"
    fi
    
    # Redis
    if docker exec sutazai-redis-  redis-cli ping >/dev/null 2>&1; then
        local redis_info=$(docker exec sutazai-redis-  redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
        add_check_result "Redis Connection" "PASS" "Redis is responding" "{\"memory_usage\": \"$redis_info\"}"
    else
        add_check_result "Redis Connection" "FAIL" "Cannot connect to Redis" "{}"
    fi
}

# Check AI agents
check_ai_agents() {
    log "INFO" "Checking AI agents..."
    
    local expected_agents=("sutazai-senior-ai-engineer" "sutazai-infrastructure-devops-manager" "sutazai-testing-qa-validator")
    local running_agents=0
    
    for agent in "${expected_agents[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "$agent"; then
            local cpu_mem=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" "$agent" 2>/dev/null || echo "0.00%,0B / 0B")
            IFS=',' read -r cpu mem <<< "$cpu_mem"
            add_check_result "Agent: $agent" "PASS" "Agent is running" "{\"cpu_usage\": \"$cpu\", \"memory_usage\": \"$mem\"}"
            ((running_agents++))
        else
            add_check_result "Agent: $agent" "FAIL" "Agent is not running" "{}"
        fi
    done
    
    local agent_percentage=$((running_agents * 100 / ${#expected_agents[@]}))
    if [[ $agent_percentage -ge 80 ]]; then
        add_check_result "AI Agents Summary" "PASS" "$running_agents/${#expected_agents[@]} agents running ($agent_percentage%)" "{\"running\": $running_agents, \"total\": ${#expected_agents[@]}}"
    elif [[ $agent_percentage -ge 50 ]]; then
        add_check_result "AI Agents Summary" "WARN" "$running_agents/${#expected_agents[@]} agents running ($agent_percentage%)" "{\"running\": $running_agents, \"total\": ${#expected_agents[@]}}"
    else
        add_check_result "AI Agents Summary" "FAIL" "Only $running_agents/${#expected_agents[@]} agents running ($agent_percentage%)" "{\"running\": $running_agents, \"total\": ${#expected_agents[@]}}"
    fi
}

# Check model availability
check_model_availability() {
    log "INFO" "Checking model availability..."
    
    # Test tinyllama model
    local test_response=$(curl -s -X POST http://localhost:10104/api/generate \
        -d '{"model": "tinyllama", "prompt": "Hello", "stream": false}' \
        --max-time 30 2>/dev/null | jq -r '.response // empty' 2>/dev/null)
    
    if [[ -n "$test_response" && "$test_response" != "null" ]]; then
        local model_size=$(curl -s http://localhost:10104/api/show -d '{"name": "tinyllama"}' | jq -r '.details.parameter_size // "unknown"' 2>/dev/null)
        add_check_result "tinyllama Model" "PASS" "Model is responding correctly" "{\"parameter_size\": \"$model_size\", \"test_response_length\": ${#test_response}}"
    else
        add_check_result "tinyllama Model" "FAIL" "Model is not responding or returned empty response" "{}"
    fi
}

# Check system resources
check_system_resources() {
    log "INFO" "Checking system resources..."
    
    # Memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local mem_usage_int=${mem_usage%.*}
    
    if [[ $mem_usage_int -lt 80 ]]; then
        add_check_result "Memory Usage" "PASS" "Memory usage is ${mem_usage}%" "{\"usage_percent\": $mem_usage}"
    elif [[ $mem_usage_int -lt 90 ]]; then
        add_check_result "Memory Usage" "WARN" "Memory usage is ${mem_usage}% (approaching limit)" "{\"usage_percent\": $mem_usage}"
    else
        add_check_result "Memory Usage" "FAIL" "Memory usage is ${mem_usage}% (critical)" "{\"usage_percent\": $mem_usage}"
    fi
    
    # Disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [[ $disk_usage -lt 80 ]]; then
        add_check_result "Disk Usage" "PASS" "Disk usage is ${disk_usage}%" "{\"usage_percent\": $disk_usage}"
    elif [[ $disk_usage -lt 90 ]]; then
        add_check_result "Disk Usage" "WARN" "Disk usage is ${disk_usage}% (approaching limit)" "{\"usage_percent\": $disk_usage}"
    else
        add_check_result "Disk Usage" "FAIL" "Disk usage is ${disk_usage}% (critical)" "{\"usage_percent\": $disk_usage}"
    fi
    
    # Load average
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)
    local cpu_cores=$(nproc)
    local load_threshold=$(echo "$cpu_cores * 1.5" | bc -l 2>/dev/null || echo "$cpu_cores")
    
    if (( $(echo "$load_1min < $load_threshold" | bc -l 2>/dev/null || echo "1") )); then
        add_check_result "System Load" "PASS" "Load average is ${load_1min} (${cpu_cores} cores)" "{\"load_1min\": $load_1min, \"cpu_cores\": $cpu_cores}"
    else
        add_check_result "System Load" "WARN" "Load average is ${load_1min} (high for ${cpu_cores} cores)" "{\"load_1min\": $load_1min, \"cpu_cores\": $cpu_cores}"
    fi
}

# Check log file sizes
check_log_files() {
    log "INFO" "Checking log file sizes..."
    
    local large_logs=()
    local total_log_size=0
    
    while IFS= read -r -d '' logfile; do
        local size_bytes=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))
        total_log_size=$((total_log_size + size_mb))
        
        if [[ $size_mb -gt 100 ]]; then
            large_logs+=("$(basename "$logfile"):${size_mb}MB")
        fi
    done < <(find "$LOG_DIR" -name "*.log" -print0 2>/dev/null)
    
    if [[ ${#large_logs[@]} -eq 0 ]]; then
        add_check_result "Log File Sizes" "PASS" "No large log files found (total: ${total_log_size}MB)" "{\"total_size_mb\": $total_log_size}"
    else
        local large_logs_str=$(IFS=','; echo "${large_logs[*]}")
        add_check_result "Log File Sizes" "WARN" "Found ${#large_logs[@]} large log files: $large_logs_str" "{\"large_files\": \"$large_logs_str\", \"total_size_mb\": $total_log_size}"
    fi
}

# Generate final report summary
finalize_report() {
    log "INFO" "Finalizing health check report..."
    
    local failed_count=$(jq -r '.summary.failed' "$REPORT_FILE")
    local warnings_count=$(jq -r '.summary.warnings' "$REPORT_FILE")
    local total_checks=$(jq -r '.summary.total_checks' "$REPORT_FILE")
    
    local overall_status
    if [[ $failed_count -ge $CRITICAL_THRESHOLD ]]; then
        overall_status="CRITICAL"
    elif [[ $failed_count -gt 0 ]]; then
        overall_status="DEGRADED"
    elif [[ $warnings_count -gt 0 ]]; then
        overall_status="WARNING"
    else
        overall_status="HEALTHY"
    fi
    
    # Update final status
    jq --arg status "$overall_status" '.summary.status = $status' "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    
    # Create symlink to latest report
    ln -sf "$REPORT_FILE" "$REPORT_DIR/latest_health_report.json"
    
    # Display summary
    echo
    echo "============================================"
    echo "          DAILY HEALTH CHECK SUMMARY"
    echo "============================================"
    echo "Timestamp: $(date)"
    echo "Overall Status: $overall_status"
    echo "Total Checks: $total_checks"
    echo "Passed: $(jq -r '.summary.passed' "$REPORT_FILE")"
    echo "Failed: $failed_count"
    echo "Warnings: $warnings_count"
    echo
    echo "Report saved to: $REPORT_FILE"
    echo "Log saved to: $LOG_FILE"
    echo "============================================"
    
    # Send email notification if configured and status is not healthy
    if [[ -n "$EMAIL_RECIPIENT" && "$overall_status" != "HEALTHY" ]]; then
        send_email_notification "$overall_status" "$failed_count" "$warnings_count"
    fi
}

# Send email notification
send_email_notification() {
    local status="$1"
    local failed="$2"
    local warnings="$3"
    
    local subject="SutazAI System Health Alert - Status: $status"
    local body="Daily health check completed with $status status.

Failed checks: $failed
Warning checks: $warnings

Full report: $REPORT_FILE
Log file: $LOG_FILE

System: $(hostname)
Timestamp: $(date)

Please investigate any failed or warning checks immediately."
    
    if command -v mail >/dev/null 2>&1; then
        echo "$body" | mail -s "$subject" "$EMAIL_RECIPIENT"
        log "INFO" "Email notification sent to $EMAIL_RECIPIENT"
    else
        log "WARN" "Mail command not available, cannot send email notification"
    fi
}

# Main execution
main() {
    log "INFO" "Starting daily health check for SutazAI system"
    
    # Initialize report
    init_report
    
    # Run all checks
    check_docker || true
    check_core_services || true
    check_api_endpoints || true
    check_database_connectivity || true
    check_ai_agents || true
    check_model_availability || true
    check_system_resources || true
    check_log_files || true
    
    # Finalize report
    finalize_report
    
    log "INFO" "Daily health check completed"
    
    # Return appropriate exit code based on status
    local status=$(jq -r '.summary.status' "$REPORT_FILE")
    case $status in
        "HEALTHY") exit 0 ;;
        "WARNING") exit 1 ;;
        "DEGRADED") exit 2 ;;
        "CRITICAL") exit 3 ;;
        *) exit 4 ;;
    esac
}

# Run main function
main "$@"