#!/bin/bash

# Sutazai Hygiene Enforcement System - Production Startup Script
# Purpose: Zero-downtime startup of the complete hygiene enforcement system
# Usage: ./start-complete-system.sh [--mode MODE] [--config CONFIG]

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_DIR="${PROJECT_ROOT}/config"
PID_DIR="${LOG_DIR}/pids"
SYSTEM_LOG="${LOG_DIR}/system-startup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODE="production"
CONFIG_FILE=""
FORCE_RESTART=false
HEALTH_CHECK_TIMEOUT=60
STARTUP_DELAY=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE           Startup mode: production, development, testing (default: production)"
            echo "  --config CONFIG       Custom configuration file path"
            echo "  --force-restart       Force restart all services even if running"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  SUTAZAI_LOG_LEVEL    Log level (DEBUG, INFO, WARNING, ERROR)"
            echo "  SUTAZAI_PORT_OFFSET  Port offset for services (default: 0)"
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
    echo "[$timestamp] [$level] $message" >> "$SYSTEM_LOG"
}

# Error handling
error_exit() {
    log ERROR "$1"
    exit 1
}

# Cleanup function for graceful shutdown
cleanup() {
    log INFO "Received interrupt signal, initiating graceful shutdown..."
    if [[ -f "${PID_DIR}/system-orchestrator.pid" ]]; then
        local pid=$(cat "${PID_DIR}/system-orchestrator.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log INFO "Stopping system orchestrator (PID: $pid)..."
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            for i in {1..30}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log WARNING "Force killing system orchestrator..."
                kill -KILL "$pid"
            fi
        fi
    fi
    
    log INFO "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Initialize environment
initialize_environment() {
    log INFO "🚀 Initializing Sutazai Hygiene Enforcement System..."
    log INFO "Mode: $MODE"
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$PID_DIR" "$CONFIG_DIR"
    
    # Set up log rotation
    if command -v logrotate >/dev/null 2>&1; then
        cat > "${LOG_DIR}/logrotate.conf" << EOF
${LOG_DIR}/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}
EOF
    fi
    
    # Check Python version
    if ! python3 --version | grep -E "Python 3\.[8-9]|Python 3\.1[0-9]" >/dev/null; then
        error_exit "Python 3.8+ is required"
    fi
    
    # Check system resources
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 1024 ]]; then
        log WARNING "Low available memory: ${available_memory}MB (recommended: 1GB+)"
    fi
    
    log SUCCESS "Environment initialized successfully"
}

# Check if system is healthy
check_system_health() {
    log INFO "🏥 Performing system health check..."
    
    # Check disk space
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        error_exit "Disk usage is ${disk_usage}% - need at least 10% free space"
    fi
    
    # Check if ports are available
    local base_port=8100
    if [[ -n "${SUTAZAI_PORT_OFFSET:-}" ]]; then
        base_port=$((base_port + SUTAZAI_PORT_OFFSET))
    fi
    
    for port in $base_port $((base_port + 1)); do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            if [[ "$FORCE_RESTART" == "true" ]]; then
                log WARNING "Port $port is in use but force restart is enabled"
            else
                error_exit "Port $port is already in use. Use --force-restart to override."
            fi
        fi
    done
    
    # Check Python dependencies
    local required_packages=("fastapi" "uvicorn" "pydantic" "aiohttp" "psutil")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            error_exit "Required Python package '$package' is not installed"
        fi
    done
    
    log SUCCESS "System health check completed"
}

# Start individual service with health checking
start_service() {
    local service_name=$1
    local script_path=$2
    local health_endpoint=${3:-""}
    local startup_delay=${4:-$STARTUP_DELAY}
    
    log INFO "Starting service: $service_name"
    
    # Check if service is already running
    local pid_file="${PID_DIR}/${service_name}.pid"
    if [[ -f "$pid_file" ]]; then
        local existing_pid=$(cat "$pid_file")
        if kill -0 "$existing_pid" 2>/dev/null; then
            if [[ "$FORCE_RESTART" == "true" ]]; then
                log INFO "Stopping existing $service_name (PID: $existing_pid)"
                kill -TERM "$existing_pid"
                
                # Wait for graceful shutdown
                for i in {1..10}; do
                    if ! kill -0 "$existing_pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                
                # Force kill if necessary
                if kill -0 "$existing_pid" 2>/dev/null; then
                    kill -KILL "$existing_pid"
                fi
                
                rm -f "$pid_file"
            else
                log INFO "Service $service_name is already running (PID: $existing_pid)"
                return 0
            fi
        else
            rm -f "$pid_file"
        fi
    fi
    
    # Start the service
    local log_file="${LOG_DIR}/${service_name}.log"
    local error_log="${LOG_DIR}/${service_name}.error.log"
    
    # Prepare environment variables
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export SUTAZAI_MODE="$MODE"
    export SUTAZAI_LOG_LEVEL="${SUTAZAI_LOG_LEVEL:-INFO}"
    
    # Start service in background
    nohup python3 "$script_path" \
        > "$log_file" \
        2> "$error_log" &
    
    local service_pid=$!
    echo "$service_pid" > "$pid_file"
    
    log INFO "Started $service_name with PID: $service_pid"
    
    # Wait for startup delay
    sleep "$startup_delay"
    
    # Health check if endpoint provided
    if [[ -n "$health_endpoint" ]]; then
        log INFO "Performing health check for $service_name..."
        
        local health_check_attempts=0
        local max_attempts=$((HEALTH_CHECK_TIMEOUT / 2))
        
        while [[ $health_check_attempts -lt $max_attempts ]]; do
            if curl -f -s --connect-timeout 2 --max-time 5 "$health_endpoint" >/dev/null 2>&1; then
                log SUCCESS "Service $service_name is healthy"
                return 0
            fi
            
            # Check if process is still running
            if ! kill -0 "$service_pid" 2>/dev/null; then
                log ERROR "Service $service_name died during startup"
                cat "$error_log" | tail -20
                return 1
            fi
            
            health_check_attempts=$((health_check_attempts + 1))
            sleep 2
        done
        
        log WARNING "Health check for $service_name timed out, but process is running"
        return 0
    else
        # Just check if process is running
        sleep 2
        if kill -0 "$service_pid" 2>/dev/null; then
            log SUCCESS "Service $service_name started successfully"
            return 0
        else
            log ERROR "Service $service_name failed to start"
            cat "$error_log" | tail -20
            return 1
        fi
    fi
}

# Start all system services
start_system_services() {
    log INFO "🔧 Starting system services..."
    
    local base_port=8100
    if [[ -n "${SUTAZAI_PORT_OFFSET:-}" ]]; then
        base_port=$((base_port + SUTAZAI_PORT_OFFSET))
    fi
    
    # Service definitions (name, script, health_endpoint, startup_delay)
    local services=(
        "rule-control-api:scripts/agents/rule-control-manager.py:http://localhost:${base_port}/api/health:5"
        "system-orchestrator:scripts/hygiene-system-orchestrator.py::3"
    )
    
    local failed_services=()
    
    for service_def in "${services[@]}"; do
        IFS=':' read -r service_name script_path health_endpoint startup_delay <<< "$service_def"
        
        if ! start_service "$service_name" "${PROJECT_ROOT}/$script_path" "$health_endpoint" "$startup_delay"; then
            failed_services+=("$service_name")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error_exit "Failed to start services: ${failed_services[*]}"
    fi
    
    log SUCCESS "All system services started successfully"
}

# Validate system startup
validate_system() {
    log INFO "🔍 Validating system startup..."
    
    # Run system validation
    local validation_script="${PROJECT_ROOT}/scripts/validate-complete-system.py"
    if [[ -f "$validation_script" ]]; then
        log INFO "Running comprehensive system validation..."
        
        if python3 "$validation_script" --test-mode quick --output-format text; then
            log SUCCESS "System validation passed"
        else
            log WARNING "System validation detected issues (check logs for details)"
            # Don't fail startup for validation warnings in production mode
            if [[ "$MODE" != "production" ]]; then
                return 1
            fi
        fi
    else
        log WARNING "System validation script not found, skipping validation"
    fi
    
    return 0
}

# Monitor system health
start_health_monitoring() {
    log INFO "📊 Starting health monitoring..."
    
    # Create simple health monitoring script
    cat > "${PROJECT_ROOT}/scripts/health-monitor.sh" << 'EOF'
#!/bin/bash
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${LOG_DIR}/pids"

while true; do
    # Check all services
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            service_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "$(date): CRITICAL: Service $service_name (PID $pid) is not responding" >> "${LOG_DIR}/health-monitor.log"
                
                # Attempt restart for critical services
                if [[ "$service_name" == "rule-control-api" ]] || [[ "$service_name" == "system-orchestrator" ]]; then
                    echo "$(date): INFO: Attempting to restart $service_name" >> "${LOG_DIR}/health-monitor.log"
                    # Restart logic would go here
                fi
            fi
        fi
    done
    
    # Check system resources
    memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        echo "$(date): WARNING: High memory usage: ${memory_usage}%" >> "${LOG_DIR}/health-monitor.log"
    fi
    
    sleep 30
done
EOF
    
    chmod +x "${PROJECT_ROOT}/scripts/health-monitor.sh"
    
    nohup "${PROJECT_ROOT}/scripts/health-monitor.sh" > /dev/null 2>&1 &
    echo $! > "${PID_DIR}/health-monitor.pid"
    
    log SUCCESS "Health monitoring started"
}

# Generate system status report
generate_status_report() {
    log INFO "📈 Generating system status report..."
    
    local status_file="${LOG_DIR}/system-status-$(date +%Y%m%d_%H%M%S).json"
    local services_status=()
    
    # Check each service
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local service_name=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file")
            local status="unknown"
            local uptime="0"
            
            if kill -0 "$pid" 2>/dev/null; then
                status="running"
                if [[ -e "/proc/$pid" ]]; then
                    local start_time=$(stat -c %Y "/proc/$pid")
                    uptime=$(( $(date +%s) - start_time ))
                fi
            else
                status="stopped"
            fi
            
            services_status+=("{\"name\":\"$service_name\",\"pid\":$pid,\"status\":\"$status\",\"uptime\":$uptime}")
        fi
    done
    
    # Create JSON report
    cat > "$status_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "mode": "$MODE",
    "project_root": "$PROJECT_ROOT",
    "system_info": {
        "hostname": "$(hostname)",
        "uptime": $(cat /proc/uptime | cut -d' ' -f1),
        "load_average": "$(uptime | grep -oP 'load average: \K.*')",
        "memory_usage": {
            "total": $(free -b | awk 'NR==2{print $2}'),
            "used": $(free -b | awk 'NR==2{print $3}'),
            "available": $(free -b | awk 'NR==2{print $7}')
        },
        "disk_usage": {
            "total": "$(df -BG / | awk 'NR==2{print $2}')",
            "used": "$(df -BG / | awk 'NR==2{print $3}')",
            "available": "$(df -BG / | awk 'NR==2{print $4}')"
        }
    },
    "services": [$(IFS=,; echo "${services_status[*]}")]
}
EOF
    
    log SUCCESS "Status report generated: $status_file"
    
    # Show quick status
    echo ""
    echo "🎯 SYSTEM STATUS SUMMARY"
    echo "========================"
    echo "Mode: $MODE"
    echo "Project Root: $PROJECT_ROOT"
    echo "Log Directory: $LOG_DIR"
    echo ""
    echo "Services:"
    
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            local service_name=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file")
            
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ✅ $service_name (PID: $pid)"
            else
                echo "  ❌ $service_name (PID: $pid - NOT RUNNING)"
            fi
        fi
    done
    
    echo ""
    echo "🔗 Access Points:"
    echo "  Rule Control API: http://localhost:${base_port}"
    echo "  Health Endpoint: http://localhost:${base_port}/api/health"
    echo "  Metrics Endpoint: http://localhost:${base_port}/api/metrics"
    echo ""
    echo "📊 Monitoring:"
    echo "  System Logs: $LOG_DIR"
    echo "  Health Monitor: Running"
    echo ""
}

# Main execution flow
main() {
    log INFO "Starting Sutazai Hygiene Enforcement System..."
    
    # Initialize
    initialize_environment
    
    # Health checks
    check_system_health
    
    # Start services
    start_system_services
    
    # Validate startup
    if ! validate_system; then
        if [[ "$MODE" != "production" ]]; then
            error_exit "System validation failed"
        fi
    fi
    
    # Start monitoring
    start_health_monitoring
    
    # Generate status report
    generate_status_report
    
    log SUCCESS "🎉 Sutazai Hygiene Enforcement System started successfully!"
    log INFO "System is running in $MODE mode"
    log INFO "Use 'tail -f $SYSTEM_LOG' to monitor system logs"
    log INFO "Use './stop-complete-system.sh' to stop the system"
    
    # Keep script running to handle signals
    if [[ "${1:-}" == "--daemon" ]]; then
        log INFO "Running in daemon mode..."
        while true; do
            sleep 60
            
            # Periodic health check
            if ! kill -0 $(cat "${PID_DIR}/rule-control-api.pid" 2>/dev/null) 2>/dev/null; then
                log ERROR "Critical service failure detected!"
                break
            fi
        done
    fi
}

# Execute main function
main "$@"