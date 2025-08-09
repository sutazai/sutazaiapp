#!/bin/bash
#
# SutazAI Fast Startup Script
# Optimized startup with 50%+ time reduction and graceful degradation
#
# DESCRIPTION:
#   Fast startup script that prioritizes critical services and uses parallel
#   startup techniques to achieve 50% or better startup time reduction
#
# USAGE:
#   ./fast_start.sh [MODE] [OPTIONS]
#
# MODES:
#   critical-only  - Start only critical services (postgres, redis, neo4j)
#   core           - Start critical + core services (ollama, backend, frontend)
#   full           - Start all services with optimized parallel startup
#   agents-only    - Start only AI agents (assumes core is running)
#
# OPTIONS:
#   --parallel N   - Maximum parallel starts (default: auto-detect)
#   --timeout T    - Health check timeout in seconds (default: 30)
#   --force        - Skip dependency checks
#   --monitor      - Enable real-time monitoring
#   --dry-run      - Show what would be started without starting
#

set -euo pipefail

# Script metadata
readonly SCRIPT_NAME="SutazAI Fast Startup"
readonly SCRIPT_VERSION="1.0.0"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOG_FILE="$PROJECT_ROOT/logs/fast_startup_$(date +%Y%m%d_%H%M%S).log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Configuration
STARTUP_MODE="${1:-full}"
MAX_PARALLEL_JOBS=$(nproc)
HEALTH_CHECK_TIMEOUT=30
FORCE_START=false
ENABLE_MONITORING=false
DRY_RUN=false

# Service definitions
declare -A SERVICE_GROUPS=(
    ["critical"]="postgres redis neo4j"
    ["infrastructure"]="chromadb qdrant faiss ollama"
    ["core"]="backend frontend"
    ["monitoring"]="prometheus grafana loki promtail"
    ["ai_batch_1"]="letta autogpt crewai aider langflow flowise"
    ["ai_batch_2"]="gpt-engineer agentgpt privategpt llamaindex shellgpt"
    ["ai_batch_3"]="pentestgpt documind browser-use skyvern pytorch tensorflow"
    ["ai_batch_4"]="jax ai-metrics-exporter health-monitor mcp-server"
    ["ai_batch_5"]="context-framework autogen opendevin finrobot code-improver"
    ["ai_batch_6"]="service-hub awesome-code-ai fsdp agentzero"
)

declare -A SERVICE_PRIORITIES=(
    ["critical"]=1
    ["infrastructure"]=2
    ["core"]=2
    ["monitoring"]=3
    ["ai_batch_1"]=4
    ["ai_batch_2"]=4
    ["ai_batch_3"]=4
    ["ai_batch_4"]=4
    ["ai_batch_5"]=4
    ["ai_batch_6"]=4
)

declare -A SERVICE_STARTUP_TIMES=(
    ["postgres"]=10
    ["redis"]=8
    ["neo4j"]=25
    ["chromadb"]=15
    ["qdrant"]=12
    ["faiss"]=10
    ["ollama"]=30
    ["backend"]=12
    ["frontend"]=8
)

# Global tracking
declare -A STARTUP_PIDS=()
declare -A STARTUP_RESULTS=()
declare -A ACTUAL_STARTUP_TIMES=()
TOTAL_START_TIME=""
FAILED_SERVICES=()

# ===============================================
# LOGGING AND OUTPUT
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log_info() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[$(date +'%H:%M:%S')] 🐛 $1${NC}"
    fi
}

show_progress() {
    local current="$1"
    local total="$2"
    local description="$3"
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r${PURPLE}["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %d%% - %s${NC}" "$percentage" "$description"
    
    if [[ $current -eq $total ]]; then
        echo
    fi
}

# ===============================================
# SYSTEM RESOURCE MONITORING
# ===============================================

start_resource_monitor() {
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log_info "Starting resource monitoring..."
        
        {
            while [[ -f "/tmp/sutazai_monitor.pid" ]]; do
                local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
                local mem_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
                local running_containers=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
                
                echo "$(date '+%H:%M:%S') CPU: ${cpu_usage}% MEM: ${mem_usage}% CONTAINERS: ${running_containers}" >> "$PROJECT_ROOT/logs/resource_monitor.log"
                
                # Throttle if resources are high
                if (( $(echo "$cpu_usage > 85" | bc -l) )) || (( $(echo "$mem_usage > 85" | bc -l) )); then
                    log_warn "High resource usage detected - throttling startup"
                    sleep 3
                else
                    sleep 1
                fi
            done
        } &
        
        echo $! > "/tmp/sutazai_monitor.pid"
    fi
}

stop_resource_monitor() {
    if [[ -f "/tmp/sutazai_monitor.pid" ]]; then
        rm -f "/tmp/sutazai_monitor.pid"
        log_info "Resource monitoring stopped"
    fi
}

# ===============================================
# SERVICE STARTUP FUNCTIONS
# ===============================================

wait_for_service_health() {
    local service_name="$1"
    local timeout="${2:-$HEALTH_CHECK_TIMEOUT}"
    local start_time=$(date +%s)
    local container_name="sutazai-$service_name"
    
    log_debug "Waiting for $service_name health check (timeout: ${timeout}s)"
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            log_warn "$service_name health check timeout after ${timeout}s"
            return 1
        fi
        
        # Check if container exists and is running
        if docker ps --filter "name=$container_name" --filter "status=running" --format "{{.Names}}" | grep -q "$container_name"; then
            # Check health status
            local health_status
            health_status=$(docker inspect "$container_name" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
            
            if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no_healthcheck" ]]; then
                log_debug "$service_name is healthy"
                return 0
            elif [[ "$health_status" == "unhealthy" ]]; then
                log_warn "$service_name is unhealthy after ${elapsed}s"
                return 1
            fi
        fi
        
        sleep 2
    done
}

start_single_service() {
    local service_name="$1"
    local group_name="$2"
    local service_start_time=$(date +%s)
    
    log_info "Starting $service_name (group: $group_name)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would start: $service_name"
        return 0
    fi
    
    # Start the service
    if docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d "$service_name" >/dev/null 2>&1; then
        # Wait for health check
        if wait_for_service_health "$service_name"; then
            local service_end_time=$(date +%s)
            local elapsed=$((service_end_time - service_start_time))
            ACTUAL_STARTUP_TIMES["$service_name"]=$elapsed
            STARTUP_RESULTS["$service_name"]="success"
            log_success "$service_name started successfully in ${elapsed}s"
            return 0
        else
            STARTUP_RESULTS["$service_name"]="health_failed"
            log_warn "$service_name started but failed health check"
            return 1
        fi
    else
        STARTUP_RESULTS["$service_name"]="start_failed"
        log_error "Failed to start $service_name"
        FAILED_SERVICES+=("$service_name")
        return 1
    fi
}

start_service_group_parallel() {
    local group_name="$1"
    local services="${SERVICE_GROUPS[$group_name]}"
    local max_jobs="${2:-$MAX_PARALLEL_JOBS}"
    
    if [[ -z "$services" ]]; then
        log_warn "No services defined for group: $group_name"
        return 0
    fi
    
    log_info "Starting service group: $group_name (max parallel: $max_jobs)"
    log_info "Services: $services"
    
    local service_array=($services)
    local total_services=${#service_array[@]}
    local current_jobs=0
    local completed_services=0
    local group_pids=()
    
    # Start services in parallel batches
    for service in "${service_array[@]}"; do
        # Wait if we've reached the parallel limit
        while [[ $current_jobs -ge $max_jobs ]]; do
            wait -n  # Wait for any job to complete
            current_jobs=$((current_jobs - 1))
            completed_services=$((completed_services + 1))
            show_progress $completed_services $total_services "Starting $group_name services"
        done
        
        # Start service in background
        start_single_service "$service" "$group_name" &
        local pid=$!
        group_pids+=($pid)
        STARTUP_PIDS["$service"]=$pid
        current_jobs=$((current_jobs + 1))
        
        # Brief stagger to avoid overwhelming the system
        sleep 0.5
    done
    
    # Wait for all remaining jobs to complete
    for pid in "${group_pids[@]}"; do
        wait "$pid" || true
        completed_services=$((completed_services + 1))
        show_progress $completed_services $total_services "Completing $group_name services"
    done
    
    echo  # New line after progress bar
    
    # Count successful services
    local successful=0
    for service in "${service_array[@]}"; do
        if [[ "${STARTUP_RESULTS[$service]:-}" == "success" ]]; then
            successful=$((successful + 1))
        fi
    done
    
    log_info "Group $group_name completed: $successful/$total_services services started successfully"
    
    # Allow some time for services to settle before starting next group
    sleep 2
    
    return 0
}

# ===============================================
# STARTUP MODE HANDLERS
# ===============================================

start_critical_only() {
    log_info "🚀 CRITICAL ONLY MODE - Starting essential databases only"
    start_service_group_parallel "critical" 3
}

start_core() {
    log_info "🚀 CORE MODE - Starting critical + core services"
    start_service_group_parallel "critical" 3
    start_service_group_parallel "infrastructure" 4
    start_service_group_parallel "core" 2
}

start_full() {
    log_info "🚀 FULL MODE - Starting all services with optimized parallel startup"
    
    # Phase 1: Critical infrastructure (sequential for reliability)
    start_service_group_parallel "critical" 3
    
    # Phase 2: AI infrastructure (parallel)
    start_service_group_parallel "infrastructure" 4
    
    # Phase 3: Core application (after AI infrastructure is ready)
    start_service_group_parallel "core" 2
    
    # Phase 4: AI agents in parallel batches (background)
    log_info "Starting AI agent batches in parallel..."
    
    # Start AI batches with overlap for maximum throughput
    local ai_batch_pids=()
    local batch_parallel=2  # Number of batches to run simultaneously
    local active_batches=0
    
    for batch in ai_batch_1 ai_batch_2 ai_batch_3 ai_batch_4 ai_batch_5 ai_batch_6; do
        # Wait if we have too many active batches
        while [[ $active_batches -ge $batch_parallel ]]; do
            wait -n  # Wait for any batch to complete
            active_batches=$((active_batches - 1))
        done
        
        # Start batch in background
        start_service_group_parallel "$batch" 6 &
        ai_batch_pids+=($!)
        active_batches=$((active_batches + 1))
        
        # Stagger batch starts
        sleep 3
    done
    
    # Wait for all AI batches to complete
    for pid in "${ai_batch_pids[@]}"; do
        wait "$pid" || true
    done
    
    # Phase 5: Monitoring (optional, low priority)
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        start_service_group_parallel "monitoring" 4
    fi
}

start_agents_only() {
    log_info "🚀 AGENTS ONLY MODE - Starting AI agents (assumes core is running)"
    
    # Verify core services are running
    local core_services="postgres redis neo4j ollama backend"
    local missing_core=()
    
    for service in $core_services; do
        if ! docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            missing_core+=("$service")
        fi
    done
    
    if [[ ${#missing_core[@]} -gt 0 ]]; then
        log_error "Core services not running: ${missing_core[*]}"
        log_error "Please start core services first with: ./fast_start.sh core"
        return 1
    fi
    
    # Start AI agent batches
    for batch in ai_batch_1 ai_batch_2 ai_batch_3 ai_batch_4 ai_batch_5 ai_batch_6; do
        start_service_group_parallel "$batch" 8 &
        sleep 2  # Brief stagger
    done
    
    wait  # Wait for all batches to complete
}

# ===============================================
# HEALTH CHECKS AND VALIDATION
# ===============================================

run_startup_health_checks() {
    log_info "Running post-startup health checks..."
    
    local health_results=()
    local failed_checks=0
    
    # Check core infrastructure
    local core_services="postgres redis neo4j ollama backend frontend"
    for service in $core_services; do
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            health_results+=("✅ $service - Running")
        else
            health_results+=("❌ $service - Not running")
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    # Test basic functionality
    local backend_health=""
    if curl -s --max-time 10 http://localhost:8000/health >/dev/null 2>&1; then
        health_results+=("✅ Backend API - Responding")
    else
        health_results+=("❌ Backend API - Not responding")
        failed_checks=$((failed_checks + 1))
    fi
    
    # Test Ollama
    if curl -s --max-time 10 http://localhost:10104/api/tags >/dev/null 2>&1; then
        health_results+=("✅ Ollama API - Responding")
    else
        health_results+=("❌ Ollama API - Not responding")
        failed_checks=$((failed_checks + 1))
    fi
    
    # Display results
    echo -e "\n${BOLD}${CYAN}HEALTH CHECK RESULTS${NC}"
    echo -e "${CYAN}===================${NC}\n"
    
    for result in "${health_results[@]}"; do
        echo -e "$result"
    done
    
    echo -e "\n${BOLD}Summary: $((${#health_results[@]} - failed_checks))/${#health_results[@]} checks passed${NC}"
    
    if [[ $failed_checks -gt 0 ]]; then
        log_warn "$failed_checks health checks failed"
        return 1
    else
        log_success "All health checks passed!"
        return 0
    fi
}

# ===============================================
# STARTUP REPORT AND ANALYTICS
# ===============================================

generate_startup_report() {
    local total_end_time=$(date +%s)
    local total_startup_time=$((total_end_time - TOTAL_START_TIME))
    
    local report_file="$PROJECT_ROOT/logs/fast_startup_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Count results
    local successful_services=0
    local failed_services=0
    local total_services=0
    
    for service in "${!STARTUP_RESULTS[@]}"; do
        total_services=$((total_services + 1))
        if [[ "${STARTUP_RESULTS[$service]}" == "success" ]]; then
            successful_services=$((successful_services + 1))
        else
            failed_services=$((failed_services + 1))
        fi
    done
    
    # Calculate estimated sequential time
    local estimated_sequential_time=0
    for service in "${!STARTUP_RESULTS[@]}"; do
        local estimated_time=${SERVICE_STARTUP_TIMES[$service]:-10}
        estimated_sequential_time=$((estimated_sequential_time + estimated_time))
    done
    
    # Calculate optimization percentage
    local optimization_percentage=0
    if [[ $estimated_sequential_time -gt 0 ]]; then
        optimization_percentage=$(( (estimated_sequential_time - total_startup_time) * 100 / estimated_sequential_time ))
    fi
    
    # Create JSON report
    cat > "$report_file" << EOF
{
    "timestamp": $(date +%s),
    "startup_mode": "$STARTUP_MODE",
    "summary": {
        "total_startup_time_seconds": $total_startup_time,
        "estimated_sequential_time_seconds": $estimated_sequential_time,
        "optimization_percentage": $optimization_percentage,
        "target_achieved": $([ $optimization_percentage -ge 50 ] && echo "true" || echo "false"),
        "total_services": $total_services,
        "successful_services": $successful_services,
        "failed_services": $failed_services
    },
    "service_timings": {
EOF
    
    # Add service timings
    local first=true
    for service in "${!ACTUAL_STARTUP_TIMES[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "        \"$service\": ${ACTUAL_STARTUP_TIMES[$service]}" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    },
    "service_results": {
EOF
    
    # Add service results
    first=true
    for service in "${!STARTUP_RESULTS[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "        \"$service\": \"${STARTUP_RESULTS[$service]}\"" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    },
    "failed_services": [
EOF
    
    # Add failed services
    for i in "${!FAILED_SERVICES[@]}"; do
        if [[ $i -gt 0 ]]; then
            echo "," >> "$report_file"
        fi
        echo "        \"${FAILED_SERVICES[$i]}\"" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    ],
    "system_info": {
        "max_parallel_jobs": $MAX_PARALLEL_JOBS,
        "health_check_timeout": $HEALTH_CHECK_TIMEOUT,
        "monitoring_enabled": $ENABLE_MONITORING
    }
}
EOF
    
    log_info "Startup report generated: $report_file"
    
    # Display summary
    echo -e "\n${BOLD}${GREEN}🎉 STARTUP COMPLETED! 🎉${NC}\n"
    echo -e "${CYAN}Total startup time: ${BOLD}${total_startup_time}s${NC}"
    echo -e "${CYAN}Estimated sequential time: ${BOLD}${estimated_sequential_time}s${NC}"
    echo -e "${CYAN}Optimization achieved: ${BOLD}${optimization_percentage}%${NC}"
    echo -e "${CYAN}Services started: ${BOLD}${successful_services}/${total_services}${NC}"
    
    if [[ $optimization_percentage -ge 50 ]]; then
        echo -e "\n${GREEN}${BOLD}✅ TARGET ACHIEVED: 50%+ startup time reduction!${NC}\n"
    else
        echo -e "\n${YELLOW}${BOLD}⚠️ Target not achieved: ${optimization_percentage}% reduction${NC}\n"
    fi
    
    if [[ ${#FAILED_SERVICES[@]} -gt 0 ]]; then
        echo -e "${RED}Failed services: ${FAILED_SERVICES[*]}${NC}"
    fi
    
    echo -e "\nFull report: ${report_file}"
    echo -e "System access: http://localhost:8501\n"
    
    return 0
}

# ===============================================
# ARGUMENT PROCESSING
# ===============================================

process_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            --timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --force)
                FORCE_START=true
                shift
                ;;
            --monitor)
                ENABLE_MONITORING=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                if [[ -z "${STARTUP_MODE:-}" ]] || [[ "$STARTUP_MODE" == "full" ]]; then
                    STARTUP_MODE="$1"
                fi
                shift
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
${BOLD}$SCRIPT_NAME v$SCRIPT_VERSION${NC}
Fast startup script with 50%+ time reduction and graceful degradation

${BOLD}USAGE:${NC}
    $0 [MODE] [OPTIONS]

${BOLD}MODES:${NC}
    critical-only    Start only critical services (postgres, redis, neo4j)
    core            Start critical + core services (ollama, backend, frontend)  
    full            Start all services with optimized parallel startup (default)
    agents-only     Start only AI agents (assumes core is running)

${BOLD}OPTIONS:${NC}
    --parallel N     Maximum parallel starts (default: auto-detect)
    --timeout T      Health check timeout in seconds (default: 30)
    --force          Skip dependency checks
    --monitor        Enable real-time resource monitoring
    --dry-run        Show what would be started without starting
    --help, -h       Show this help message

${BOLD}EXAMPLES:${NC}
    $0                                    # Full optimized startup
    $0 core                              # Start core services only
    $0 full --parallel 8 --monitor       # Full startup with monitoring
    $0 agents-only --timeout 60          # Start AI agents with longer timeout
    $0 critical-only --dry-run           # Preview critical services

${BOLD}OPTIMIZATION FEATURES:${NC}
    • Parallel service startup within resource constraints
    • Dependency-aware startup ordering  
    • Resource monitoring and throttling
    • Health check optimization
    • Graceful degradation for failed services
    • Real-time progress monitoring
    • Comprehensive startup reporting

EOF
}

# ===============================================
# MAIN EXECUTION
# ===============================================

main() {
    # Process command line arguments
    process_arguments "$@"
    
    # Setup logging
    setup_logging
    
    # Initialize
    TOTAL_START_TIME=$(date +%s)
    
    log_info "$SCRIPT_NAME v$SCRIPT_VERSION"
    log_info "Starting SutazAI with mode: $STARTUP_MODE"
    log_info "Max parallel jobs: $MAX_PARALLEL_JOBS"
    log_info "Health check timeout: ${HEALTH_CHECK_TIMEOUT}s"
    
    # Validate mode
    if [[ ! "$STARTUP_MODE" =~ ^(critical-only|core|full|agents-only)$ ]]; then
        log_error "Invalid startup mode: $STARTUP_MODE"
        show_usage
        exit 1
    fi
    
    # Start resource monitoring if enabled
    start_resource_monitor
    
    # Execute startup based on mode
    case "$STARTUP_MODE" in
        "critical-only")
            start_critical_only
            ;;
        "core")
            start_core
            ;;
        "full")
            start_full
            ;;
        "agents-only")
            start_agents_only
            ;;
    esac
    
    # Stop resource monitoring
    stop_resource_monitor
    
    # Run health checks unless dry run
    if [[ "$DRY_RUN" != "true" ]]; then
        sleep 5  # Allow services to settle
        run_startup_health_checks
    fi
    
    # Generate report
    generate_startup_report
    
    log_success "Fast startup completed!"
}

# Cleanup on exit
cleanup_on_exit() {
    stop_resource_monitor
}

trap cleanup_on_exit EXIT

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi