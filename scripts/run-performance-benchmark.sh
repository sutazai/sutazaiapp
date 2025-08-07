#!/bin/bash
# SutazAI System Performance Benchmark Execution Script
# =====================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
REPORTS_DIR="$PROJECT_ROOT/reports/performance"
DATA_DIR="$PROJECT_ROOT/data"

# Create necessary directories
mkdir -p "$LOG_DIR" "$REPORTS_DIR" "$DATA_DIR"

# Logging setup
LOG_FILE="$LOG_DIR/performance_benchmark_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "=================================================="
echo "SutazAI System Performance Benchmark"
echo "Started: $(date)"
echo "=================================================="

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python and required packages
    if ! command -v python3 &> /dev/null; then
        log "ERROR: Python3 is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR: Docker is required but not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log "ERROR: Docker daemon is not running"
        exit 1
    fi
    
    # Install Python dependencies if needed
    log "Installing Python dependencies..."
    pip3 install -q \
        psutil \
        docker \
        requests \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        pyyaml \
        prometheus-client \
        asyncio \
        aiohttp
    
    log "Prerequisites check completed"
}

# Function to validate system state
validate_system_state() {
    log "Validating system state..."
    
    # Check if core services are running
    local required_services=("consul" "kong" "rabbitmq" "prometheus" "grafana")
    local missing_services=()
    
    for service in "${required_services[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "$service"; then
            missing_services+=("$service")
        fi
    done
    
    if [ ${#missing_services[@]} -gt 0 ]; then
        log "WARNING: Missing services: ${missing_services[*]}"
        log "Some benchmarks may fail. Consider starting missing services."
    fi
    
    # Check agent containers
    local agent_count=$(docker ps --filter "name=sutazaiapp-" --format "table {{.Names}}" | wc -l)
    log "Found $agent_count agent containers running"
    
    if [ "$agent_count" -lt 10 ]; then
        log "WARNING: Only $agent_count agents running. Expected 90+."
        log "Consider starting more agents for comprehensive benchmarking."
    fi
    
    # Check system resources
    local available_memory=$(free -g | awk '/^Mem:/{print $7}')
    local cpu_count=$(nproc)
    
    log "System resources: $cpu_count CPU cores, ${available_memory}GB available memory"
    
    if [ "$available_memory" -lt 5 ]; then
        log "WARNING: Low available memory ($available_memory GB). Consider freeing up memory."
    fi
    
    log "System state validation completed"
}

# Function to run pre-benchmark cleanup
pre_benchmark_cleanup() {
    log "Performing pre-benchmark cleanup..."
    
    # Clear old benchmark data (keep last 7 days)
    find "$REPORTS_DIR" -name "benchmark_report_*.json" -mtime +7 -delete 2>/dev/null || true
    find "$REPORTS_DIR" -name "*.png" -mtime +7 -delete 2>/dev/null || true
    
    # Clear old logs (keep last 30 days)
    find "$LOG_DIR" -name "performance_benchmark_*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Restart any unhealthy containers
    log "Checking for unhealthy containers..."
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}")
    
    if [ -n "$unhealthy_containers" ]; then
        log "Restarting unhealthy containers: $unhealthy_containers"
        echo "$unhealthy_containers" | xargs -r docker restart
        sleep 30  # Wait for containers to stabilize
    fi
    
    log "Pre-benchmark cleanup completed"
}

# Function to run the benchmark
run_benchmark() {
    log "Starting performance benchmark..."
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export BENCHMARK_CONFIG="$PROJECT_ROOT/config/benchmark_config.yaml"
    
    # Run the benchmark with timeout
    local benchmark_timeout=3600  # 1 hour timeout
    
    if timeout "$benchmark_timeout" python3 "$PROJECT_ROOT/monitoring/system_performance_benchmark_suite.py"; then
        log "Benchmark completed successfully"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log "ERROR: Benchmark timed out after $benchmark_timeout seconds"
        else
            log "ERROR: Benchmark failed with exit code $exit_code"
        fi
        return $exit_code
    fi
}

# Function to post-benchmark analysis
post_benchmark_analysis() {
    log "Performing post-benchmark analysis..."
    
    # Find the latest benchmark report
    local latest_report=$(find "$REPORTS_DIR" -name "benchmark_report_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_report" ]; then
        log "ERROR: No benchmark report found"
        return 1
    fi
    
    log "Latest benchmark report: $latest_report"
    
    # Extract key metrics using jq if available
    if command -v jq &> /dev/null; then
        log "Extracting key metrics..."
        
        local total_agents=$(jq -r '.agent_performance | length' "$latest_report" 2>/dev/null || echo "unknown")
        local compliance_score=$(jq -r '.sla_compliance.compliance_score' "$latest_report" 2>/dev/null || echo "unknown")
        local total_duration=$(jq -r '.total_duration' "$latest_report" 2>/dev/null || echo "unknown")
        
        log "Key Metrics:"
        log "  - Total Agents Tested: $total_agents"
        log "  - SLA Compliance Score: $compliance_score%"
        log "  - Total Duration: $total_duration seconds"
        
        # Check for critical issues
        local violations=$(jq -r '.sla_compliance.violations | length' "$latest_report" 2>/dev/null || echo "0")
        if [ "$violations" -gt "0" ]; then
            log "WARNING: $violations SLA violations detected"
            log "Review the detailed report for optimization recommendations"
        fi
    fi
    
    # Generate summary email/notification if configured
    if [ -n "${NOTIFICATION_EMAIL:-}" ]; then
        send_notification "$latest_report"
    fi
    
    log "Post-benchmark analysis completed"
}

# Function to send notifications
send_notification() {
    local report_file="$1"
    
    log "Sending notification to $NOTIFICATION_EMAIL..."
    
    # Create simple email body
    local email_body="SutazAI Performance Benchmark Completed
    
Report Location: $report_file
Generated: $(date)

Key Results:
$(jq -r '.sla_compliance.compliance_score' "$report_file" 2>/dev/null || echo "unknown")% SLA Compliance

Please review the detailed report for complete analysis.
"
    
    # Send email using mail command if available
    if command -v mail &> /dev/null; then
        echo "$email_body" | mail -s "SutazAI Performance Benchmark Report" "$NOTIFICATION_EMAIL"
        log "Notification sent successfully"
    else
        log "WARNING: mail command not available, skipping email notification"
    fi
}

# Function to handle errors
error_handler() {
    local exit_code=$?
    log "ERROR: Benchmark script failed with exit code $exit_code"
    
    # Collect diagnostic information
    log "Collecting diagnostic information..."
    
    echo "=== System Info ===" >> "$LOG_FILE"
    uname -a >> "$LOG_FILE" 2>&1
    
    echo "=== Memory Usage ===" >> "$LOG_FILE"
    free -h >> "$LOG_FILE" 2>&1
    
    echo "=== Disk Usage ===" >> "$LOG_FILE"
    df -h >> "$LOG_FILE" 2>&1
    
    echo "=== Docker Containers ===" >> "$LOG_FILE"
    docker ps -a >> "$LOG_FILE" 2>&1
    
    echo "=== Recent Docker Logs ===" >> "$LOG_FILE"
    docker logs --since=1h sutazaiapp-agent-orchestrator >> "$LOG_FILE" 2>&1 || true
    
    log "Diagnostic information collected in $LOG_FILE"
    exit $exit_code
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -c, --config FILE       Use custom config file
    -o, --output DIR        Custom output directory
    -e, --email EMAIL       Send notification to email
    -q, --quick             Run quick benchmark (reduced duration)
    --skip-cleanup          Skip pre-benchmark cleanup
    --skip-validation       Skip system state validation
    --dry-run              Show what would be done without executing

Examples:
    $0                                 # Run full benchmark
    $0 --quick                         # Run quick benchmark
    $0 --email admin@company.com       # Send results via email
    $0 --config custom_config.yaml     # Use custom configuration

Environment Variables:
    BENCHMARK_CONFIG        Path to benchmark configuration file
    NOTIFICATION_EMAIL      Email for notifications
    BENCHMARK_TIMEOUT       Timeout in seconds (default: 3600)
EOF
}

# Parse command line arguments
QUICK_MODE=false
SKIP_CLEANUP=false
SKIP_VALIDATION=false
DRY_RUN=false
CUSTOM_CONFIG=""
CUSTOM_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        -e|--email)
            export NOTIFICATION_EMAIL="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log "ERROR: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Set custom configuration if provided
if [ -n "$CUSTOM_CONFIG" ]; then
    export BENCHMARK_CONFIG="$CUSTOM_CONFIG"
fi

# Set custom output directory if provided
if [ -n "$CUSTOM_OUTPUT" ]; then
    REPORTS_DIR="$CUSTOM_OUTPUT"
    mkdir -p "$REPORTS_DIR"
fi

# Set up error handling
trap error_handler ERR

# Main execution
main() {
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would execute the following steps:"
        log "1. Check prerequisites"
        log "2. Validate system state"
        log "3. Pre-benchmark cleanup"
        log "4. Run performance benchmark"
        log "5. Post-benchmark analysis"
        log "Configuration: ${BENCHMARK_CONFIG:-default}"
        log "Output directory: $REPORTS_DIR"
        exit 0
    fi
    
    log "Starting SutazAI performance benchmark execution"
    
    # Execute benchmark steps
    check_prerequisites
    
    if [ "$SKIP_VALIDATION" = false ]; then
        validate_system_state
    fi
    
    if [ "$SKIP_CLEANUP" = false ]; then
        pre_benchmark_cleanup
    fi
    
    # Modify config for quick mode
    if [ "$QUICK_MODE" = true ]; then
        log "Quick mode enabled - reducing benchmark duration"
        export BENCHMARK_QUICK_MODE=true
    fi
    
    # Run the actual benchmark
    if run_benchmark; then
        post_benchmark_analysis
        log "Benchmark execution completed successfully"
        
        # Print summary
        echo ""
        echo "=================================================="
        echo "BENCHMARK COMPLETED SUCCESSFULLY"
        echo "Reports available in: $REPORTS_DIR"
        echo "Logs available in: $LOG_FILE"
        echo "=================================================="
        
    else
        log "Benchmark execution failed"
        exit 1
    fi
}

# Execute main function
main "$@"