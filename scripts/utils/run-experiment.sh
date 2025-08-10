#!/bin/bash
# SutazAI Chaos Engineering - Experiment Runner
# Main script for executing chaos experiments

set -euo pipefail

# Configuration
CHAOS_DIR="/opt/sutazaiapp/chaos"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/chaos_experiment_${TIMESTAMP}.log"

# Default values
EXPERIMENT_NAME=""
SAFE_MODE=false
DRY_RUN=false
EMERGENCY_STOP=false
CONFIG_FILE="$CHAOS_DIR/config/chaos-config.yaml"

# Ensure directories exist
mkdir -p "$LOG_DIR"

# Logging functions
log_info() {
    echo "[$(date +'%H:%M:%S')] INFO: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date +'%H:%M:%S')] SUCCESS: $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo "[$(date +'%H:%M:%S')] WARNING: $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
}

# Usage information
show_usage() {
    cat << EOF
SutazAI Chaos Engineering - Experiment Runner

USAGE:
    $0 --experiment <name> [OPTIONS]

REQUIRED:
    --experiment <name>     Name of the chaos experiment to run

OPTIONS:
    --safe-mode            Run experiment in safe mode (limited impact)
    --dry-run              Show what would be done without executing
    --config <path>        Path to chaos configuration file
    --help                 Show this help message

AVAILABLE EXPERIMENTS:
    basic-container-chaos  Basic container failure and recovery test
    network-chaos         Network latency and partition testing
    resource-stress       CPU, memory, and disk stress testing

EXAMPLES:
    # Run basic container chaos in safe mode
    $0 --experiment basic-container-chaos --safe-mode

    # Dry run network chaos experiment
    $0 --experiment network-chaos --dry-run

    # Run resource stress with custom config
    $0 --experiment resource-stress --config /path/to/config.yaml

SAFETY FEATURES:
    - All experiments respect maintenance windows
    - Safe mode limits impact to non-critical services
    - Emergency stop capability (kill -TERM PID)
    - Automatic recovery monitoring
    - Health prerequisites validation

For more information, see: $CHAOS_DIR/README.md
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --experiment)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --safe-mode)
                SAFE_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$EXPERIMENT_NAME" ]]; then
        log_error "Experiment name is required"
        show_usage
        exit 1
    fi
}

# Validate prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check if chaos framework is initialized
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Chaos framework not initialized. Run init-chaos.sh first"
        exit 1
    fi
    
    # Check if experiment exists
    local experiment_file="$CHAOS_DIR/experiments/${EXPERIMENT_NAME}.yaml"
    if [[ ! -f "$experiment_file" ]]; then
        log_error "Experiment file not found: $experiment_file"
        exit 1
    fi
    
    # Check required commands
    local required_commands=("docker" "python3" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_success "$cmd is available"
        else
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Python dependencies
    if python3 -c "import yaml, docker, requests" &> /dev/null; then
        log_success "Python dependencies are available"
    else
        log_error "Required Python modules not found (yaml, docker, requests)"
        exit 1
    fi
}

# Check system health before experiment
check_system_health() {
    log_header "Checking System Health"
    
    # Get container health status
    local unhealthy_containers
    unhealthy_containers=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -v "healthy\|Up" | wc -l)
    
    if [[ $unhealthy_containers -gt 3 ]]; then
        log_error "Too many unhealthy containers ($unhealthy_containers). System not ready for chaos"
        return 1
    fi
    
    # Check if critical services are running
    local critical_services=("sutazai-postgres" "sutazai-redis" "sutazai-backend")
    for service in "${critical_services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "Critical service running: $service"
        else
            log_error "Critical service not running: $service"
            return 1
        fi
    done
    
    # Check system resources
    local cpu_usage memory_usage disk_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    log_info "System resources - CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Disk: ${disk_usage}%"
    
    if [[ ${cpu_usage%.*} -gt 80 ]] || [[ $memory_usage -gt 85 ]] || [[ $disk_usage -gt 90 ]]; then
        log_warning "High resource usage detected. Consider postponing experiment"
        if [[ "$SAFE_MODE" != "true" ]]; then
            log_error "Aborting due to high resource usage (use --safe-mode to override)"
            return 1
        fi
    fi
    
    log_success "System health check passed"
    return 0
}

# Check maintenance window
check_maintenance_window() {
    log_header "Checking Maintenance Window"
    
    local current_hour current_day
    current_hour=$(date +%H)
    current_day=$(date +%A | tr '[:upper:]' '[:lower:]')
    
    # Check if we're in maintenance window (2-4 AM)
    if [[ $current_hour -ge 2 ]] && [[ $current_hour -lt 4 ]] && \
       [[ "$current_day" =~ ^(monday|wednesday|friday)$ ]]; then
        log_success "Running during maintenance window"
        return 0
    fi
    
    # Allow manual runs in safe mode
    if [[ "$SAFE_MODE" == "true" ]]; then
        log_warning "Running outside maintenance window in safe mode"
        return 0
    fi
    
    log_error "Not in maintenance window and safe mode not enabled"
    log_info "Maintenance windows: Mon/Wed/Fri 2-4 AM UTC"
    log_info "Use --safe-mode to run outside maintenance windows"
    return 1
}

# Setup emergency stop handler
setup_emergency_stop() {
    # Create emergency stop function
    emergency_stop() {
        log_warning "Emergency stop signal received"
        EMERGENCY_STOP=true
        
        # Kill the chaos engine if running
        if [[ -n "${CHAOS_PID:-}" ]]; then
            kill -TERM "$CHAOS_PID" 2>/dev/null || true
        fi
        
        # Run emergency cleanup
        python3 "$CHAOS_DIR/scripts/emergency-cleanup.py" || true
        
        log_error "Emergency stop completed"
        exit 130
    }
    
    # Setup signal handlers
    trap emergency_stop SIGTERM SIGINT
}

# Execute experiment
run_experiment() {
    log_header "Executing Chaos Experiment: $EXPERIMENT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN - Would execute experiment: $EXPERIMENT_NAME"
        log_info "Safe mode: $SAFE_MODE"
        log_info "Config file: $CONFIG_FILE"
        
        # Show experiment details
        if command -v yq &> /dev/null; then
            echo "Experiment configuration:"
            yq # SECURITY FIX: eval replaced
# Original: eval "$CHAOS_DIR/experiments/${EXPERIMENT_NAME}.yaml"
$CHAOS_DIR/experiments/${EXPERIMENT_NAME}.yaml
        else
            log_info "Install 'yq' to see experiment configuration preview"
        fi
        
        log_success "Dry run completed"
        return 0
    fi
    
    # Build command arguments
    local cmd_args=("--experiment" "$EXPERIMENT_NAME" "--config" "$CONFIG_FILE")
    
    if [[ "$SAFE_MODE" == "true" ]]; then
        cmd_args+=("--safe-mode")
    fi
    
    # Execute chaos engine
    log_info "Starting chaos engine..."
    python3 "$CHAOS_DIR/scripts/chaos-engine.py" "${cmd_args[@]}" &
    CHAOS_PID=$!
    
    # Monitor experiment
    local start_time
    start_time=$(date +%s)
    
    while kill -0 "$CHAOS_PID" 2>/dev/null; do
        if [[ "$EMERGENCY_STOP" == "true" ]]; then
            break
        fi
        
        # Check for timeout (max 30 minutes)
        local current_time duration
        current_time=$(date +%s)
        duration=$((current_time - start_time))
        
        if [[ $duration -gt 1800 ]]; then
            log_error "Experiment timeout reached (30 minutes)"
            kill -TERM "$CHAOS_PID" 2>/dev/null || true
            break
        fi
        
        sleep 10
    done
    
    # Wait for process to complete
    wait "$CHAOS_PID" 2>/dev/null || true
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Chaos experiment completed successfully"
    else
        log_error "Chaos experiment failed with exit code: $exit_code"
    fi
    
    return $exit_code
}

# Post-experiment validation
validate_experiment_results() {
    log_header "Validating Experiment Results"
    
    # Check if all critical services recovered
    local critical_services=("sutazai-postgres" "sutazai-redis" "sutazai-backend")
    local failed_services=()
    
    for service in "${critical_services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "Service recovered: $service"
        else
            log_error "Service failed to recover: $service"
            failed_services+=("$service")
        fi
    done
    
    # Check system health
    sleep 30  # Wait for services to stabilize
    
    local health_score
    health_score=$(python3 -c "
import sys
sys.path.append('$CHAOS_DIR/scripts')
from chaos_engine import ChaosEngine, HealthMonitor, ChaosLogger
import asyncio

async def check_health():
    logger = ChaosLogger()
    health_monitor = HealthMonitor(logger)
    engine = ChaosEngine()
    targets = await engine.get_targets()
    health = await health_monitor.get_system_health(targets)
    print(health['health_score'])

asyncio.run(check_health())
" 2>/dev/null || echo "0")
    
    log_info "Post-experiment health score: ${health_score}%"
    
    if [[ ${health_score%.*} -lt 80 ]]; then
        log_warning "System health below acceptable threshold after experiment"
        
        # Attempt automatic recovery
        log_info "Attempting automatic recovery..."
        python3 "$CHAOS_DIR/scripts/auto-recovery.py" || true
        
        return 1
    fi
    
    # Check for recent experiment report
    local report_dir="$CHAOS_DIR/reports"
    local latest_report
    latest_report=$(find "$report_dir" -name "experiment_${EXPERIMENT_NAME}_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$latest_report" ]] && [[ -f "$latest_report" ]]; then
        log_success "Experiment report generated: $latest_report"
        
        # Extract key metrics
        local status recovery_time errors
        status=$(jq -r '.status' "$latest_report" 2>/dev/null || echo "unknown")
        recovery_time=$(jq -r '.recovery_time' "$latest_report" 2>/dev/null || echo "unknown")
        errors=$(jq -r '.errors | length' "$latest_report" 2>/dev/null || echo "unknown")
        
        log_info "Experiment status: $status"
        log_info "Recovery time: ${recovery_time} seconds"
        log_info "Error count: $errors"
        
        if [[ "$status" == "completed" ]] && [[ ${#failed_services[@]} -eq 0 ]]; then
            log_success "Experiment validation passed"
            return 0
        fi
    fi
    
    log_error "Experiment validation failed"
    return 1
}

# Generate experiment summary
generate_summary() {
    log_header "Experiment Summary"
    
    log_info "Experiment: $EXPERIMENT_NAME"
    log_info "Safe mode: $SAFE_MODE"
    log_info "Start time: $(date -d @$start_time 2>/dev/null || date)"
    log_info "End time: $(date)"
    log_info "Log file: $LOG_FILE"
    
    # Find latest report
    local report_dir="$CHAOS_DIR/reports"
    local latest_report
    latest_report=$(find "$report_dir" -name "experiment_${EXPERIMENT_NAME}_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$latest_report" ]] && [[ -f "$latest_report" ]]; then
        log_info "Detailed report: $latest_report"
        
        # Show report summary if jq is available
        if command -v jq &> /dev/null; then
            echo ""
            echo "=== EXPERIMENT METRICS ==="
            jq -r '
                "Status: " + .status,
                "Targets affected: " + (.targets_affected | join(", ")),
                "Recovery time: " + (.recovery_time | tostring) + " seconds",
                "Error count: " + (.errors | length | tostring),
                "Scenarios executed: " + (.metrics.scenarios_executed | tostring),
                "Successful scenarios: " + (.metrics.successful_scenarios | tostring)
            ' "$latest_report" 2>/dev/null
            echo "=========================="
        fi
    fi
    
    log_success "Chaos experiment session completed"
}

# Main execution function
main() {
    local start_time
    start_time=$(date +%s)
    
    log_header "SutazAI Chaos Engineering - Experiment Runner"
    
    parse_arguments "$@"
    setup_emergency_stop
    check_prerequisites
    
    if ! check_system_health; then
        exit 1
    fi
    
    if ! check_maintenance_window; then
        exit 1
    fi
    
    if ! run_experiment; then
        log_error "Experiment execution failed"
        exit 1
    fi
    
    if ! validate_experiment_results; then
        log_warning "Experiment validation failed - manual review required"
        exit 1
    fi
    
    generate_summary
    
    log_success "Chaos experiment completed successfully"
}

# Execute main function
main "$@"