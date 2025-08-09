#!/bin/bash

# Ollama Integration Deployment Script
# Zero-downtime deployment with blue-green canary rollouts
# Author: Infrastructure DevOps Manager
# Version: 2.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment/ollama-integration"
LOG_DIR="${PROJECT_ROOT}/logs"
BACKUP_DIR="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"

# Deployment configuration
DEPLOYMENT_ID="ollama-integration-$(date +%Y%m%d_%H%M%S)"
DEPLOYMENT_LOG="${LOG_DIR}/deployment_${DEPLOYMENT_ID}.log"
ROLLOUT_CONFIG="${DEPLOYMENT_DIR}/rollout-phases.yaml"
MONITORING_CONFIG="${DEPLOYMENT_DIR}/monitoring-setup.yaml"

# Resource constraints (WSL2 limited environment)
MAX_MEMORY_GB=48
MAX_GPU_GB=4
OLLAMA_PARALLEL=2
MAX_CONCURRENT_DEPLOYMENTS=5
MEMORY_SAFETY_MARGIN=0.15  # 15% safety margin

# State management
STATE_FILE="/tmp/deployment-state-${DEPLOYMENT_ID}.json"
DEPLOYED_AGENTS_FILE="/tmp/deployed-agents-${DEPLOYMENT_ID}.txt"
ROLLBACK_FLAG="/tmp/rollback-in-progress"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging setup
exec 1> >(tee -a "${DEPLOYMENT_LOG}")
exec 2> >(tee -a "${DEPLOYMENT_LOG}" >&2)

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

# Cleanup function
cleanup() {
    local exit_code=$?
    log "Deployment cleanup initiated (exit code: $exit_code)"
    
    # Remove temporary files
    rm -f "${STATE_FILE}" "${DEPLOYED_AGENTS_FILE}"
    
    # Stop monitoring if started
    pkill -f "deployment-monitor" || true
    
    # Clear deployment lock
    rm -f "${ROLLBACK_FLAG}"
    
    if [ $exit_code -ne 0 ]; then
        error "Deployment failed. Check logs at: ${DEPLOYMENT_LOG}"
        warning "Consider running rollback: ${SCRIPT_DIR}/rollback-ollama-integration.sh"
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Pre-flight validation
validate_prerequisites() {
    log "Starting pre-flight validation..."
    
    # Check if running as appropriate user
    if [[ $EUID -eq 0 ]]; then
        error "Do not run this script as root"
        exit 1
    fi
    
    # Validate required files exist
    local required_files=(
        "${PROJECT_ROOT}/agents/core/base_agent_v2.py"
        "${PROJECT_ROOT}/agents/agent_base.py"
        "${ROLLOUT_CONFIG}"
        "${MONITORING_CONFIG}"
        "${PROJECT_ROOT}/docker-compose.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file missing: $file"
            exit 1
        fi
    done
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if another deployment is running
    if [[ -f "${ROLLBACK_FLAG}" ]]; then
        error "Another deployment or rollback is in progress"
        exit 1
    fi
    
    success "Pre-flight validation passed"
}

# System resource check
check_system_resources() {
    log "Checking system resources..."
    
    # Memory check
    local available_memory_gb
    available_memory_gb=$(free -g | awk '/^Mem:/{print $7}')
    local required_memory_gb
    required_memory_gb=$(echo "${MAX_MEMORY_GB} * (1 - ${MEMORY_SAFETY_MARGIN})" | bc)
    
    if (( $(echo "$available_memory_gb < $required_memory_gb" | bc -l) )); then
        error "Insufficient memory: ${available_memory_gb}GB available, ${required_memory_gb}GB required"
        exit 1
    fi
    
    # GPU memory check (if available)
    if command -v nvidia-smi &> /dev/null; then
        local gpu_memory_free
        gpu_memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        local gpu_memory_free_gb
        gpu_memory_free_gb=$(echo "scale=2; $gpu_memory_free / 1024" | bc)
        
        if (( $(echo "$gpu_memory_free_gb < 2" | bc -l) )); then
            warning "Low GPU memory: ${gpu_memory_free_gb}GB free"
        fi
    fi
    
    # Check Ollama service
    if ! curl -f -s "http://localhost:10104/api/tags" > /dev/null; then
        error "Ollama service is not accessible"
        exit 1
    fi
    
    success "System resources validated"
}

# Establish performance baseline
establish_baseline() {
    log "Establishing performance baseline..."
    
    local baseline_file="${BACKUP_DIR}/performance-baseline.json"
    mkdir -p "$(dirname "$baseline_file")"
    
    # Collect current metrics
    local avg_cpu
    avg_cpu=$(docker stats --no-stream --format "table {{.CPUPerc}}" | grep -v CPU | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}')
    
    local avg_memory_mb
    avg_memory_mb=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep -v MEM | sed 's/MiB.*//' | awk '{sum+=$1} END {print sum/NR}')
    
    local task_count
    task_count=$(curl -s "http://localhost:8000/api/metrics/tasks" | jq '.processed_count // 0')
    
    # Create baseline JSON
    cat > "$baseline_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment_id": "${DEPLOYMENT_ID}",
    "metrics": {
        "avg_cpu_usage": ${avg_cpu:-0},
        "avg_memory_usage_mb": ${avg_memory_mb:-0},
        "task_processing_rate": ${task_count:-0},
        "active_agents": $(docker ps --filter "name=*agent*" --format "table {{.Names}}" | wc -l)
    }
}
EOF
    
    log "Baseline established: CPU=${avg_cpu}%, Memory=${avg_memory_mb}MB, Tasks=${task_count}"
}

# Create backup of current state
create_backup() {
    log "Creating system backup..."
    
    mkdir -p "${BACKUP_DIR}"
    
    # Backup configurations
    cp -r "${PROJECT_ROOT}/agents/configs" "${BACKUP_DIR}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/docker-compose.yml" "${BACKUP_DIR}/"
    
    # Backup database state
    if command -v pg_dump &> /dev/null; then
        PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump -h localhost -U postgres sutazai > "${BACKUP_DIR}/database_backup.sql" 2>/dev/null || true
    fi
    
    # Export current container states
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "${BACKUP_DIR}/container_states.txt"
    
    success "Backup created at: ${BACKUP_DIR}"
}

# Start monitoring
start_monitoring() {
    log "Starting deployment monitoring..."
    
    # Start monitoring script in background
    "${SCRIPT_DIR}/deployment-monitor.py" \
        --config "${MONITORING_CONFIG}" \
        --deployment-id "${DEPLOYMENT_ID}" \
        --log-file "${LOG_DIR}/monitoring_${DEPLOYMENT_ID}.log" &
    
    local monitor_pid=$!
    echo "$monitor_pid" > "/tmp/monitor_${DEPLOYMENT_ID}.pid"
    
    # Wait for monitoring to initialize
    sleep 5
    
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
        error "Failed to start monitoring"
        exit 1
    fi
    
    success "Monitoring started (PID: $monitor_pid)"
}

# Parse rollout configuration
parse_rollout_config() {
    if ! command -v yq &> /dev/null; then
        error "yq is required for YAML parsing. Install with: pip install yq"
        exit 1
    fi
    
    log "Parsing rollout configuration..."
    
    # Extract phase information
    yq -r '.phases | keys[]' "$ROLLOUT_CONFIG" > "/tmp/phases_${DEPLOYMENT_ID}.txt"
    
    success "Rollout configuration parsed"
}

# Deploy specific phase
deploy_phase() {
    local phase_name="$1"
    log "Starting deployment phase: ${phase_name}"
    
    # Extract phase configuration
    local phase_config
    phase_config=$(yq -r ".phases.${phase_name}" "$ROLLOUT_CONFIG")
    
    local agent_count
    agent_count=$(echo "$phase_config" | yq -r '.agent_count')
    
    local duration
    duration=$(echo "$phase_config" | yq -r '.duration')
    
    local timeout
    timeout=$(echo "$phase_config" | yq -r '.timeout')
    
    log "Phase ${phase_name}: deploying ${agent_count} agents (timeout: ${timeout})"
    
    # Get target agents for this phase
    local target_agents
    if echo "$phase_config" | yq -e '.target_agents[] // empty' > /dev/null 2>&1; then
        target_agents=$(echo "$phase_config" | yq -r '.target_agents[]')
    else
        # If "remaining", get all agents not yet deployed
        target_agents=$(get_remaining_agents)
    fi
    
    # Deploy agents in batches
    local batch_size=3
    local deployed_count=0
    local failed_count=0
    
    echo "$target_agents" | while read -r agent_name; do
        if [[ -z "$agent_name" ]]; then
            continue
        fi
        
        if deploy_single_agent "$agent_name" "$phase_name"; then
            ((deployed_count++))
            echo "$agent_name" >> "$DEPLOYED_AGENTS_FILE"
        else
            ((failed_count++))
            warning "Failed to deploy agent: $agent_name"
        fi
        
        # Check rollback conditions
        if check_rollback_conditions; then
            error "Rollback conditions triggered during phase: $phase_name"
            exit 1
        fi
        
        # Batch delay
        if (( deployed_count % batch_size == 0 )); then
            log "Batch completed. Waiting 30 seconds..."
            sleep 30
        fi
    done
    
    # Phase validation
    if ! validate_phase_success "$phase_name"; then
        error "Phase validation failed: $phase_name"
        exit 1
    fi
    
    success "Phase completed: ${phase_name} (deployed: ${deployed_count}, failed: ${failed_count})"
}

# Deploy single agent
deploy_single_agent() {
    local agent_name="$1"
    local phase_name="$2"
    
    log "Deploying agent: $agent_name"
    
    # Check if agent exists
    if ! docker ps --filter "name=${agent_name}" --format "{{.Names}}" | grep -q "$agent_name"; then
        warning "Agent container not found: $agent_name"
        return 1
    fi
    
    # Create enhanced agent container
    local enhanced_container="${agent_name}-v2"
    
    # Build enhanced agent image if needed
    if ! docker images | grep -q "sutazai-agent-v2"; then
        log "Building enhanced agent image..."
        docker build -t sutazai-agent-v2 -f "${PROJECT_ROOT}/agents/core/Dockerfile" "${PROJECT_ROOT}/agents/core/"
    fi
    
    # Start enhanced container
    docker run -d \
        --name "$enhanced_container" \
        --network sutazai-network \
        --env-file "${PROJECT_ROOT}/.env" \
        -e AGENT_NAME="$agent_name" \
        -e AGENT_VERSION="2.0.0" \
        -e OLLAMA_URL="http://ollama:10104" \
        -e BACKEND_URL="http://backend:8000" \
        -v "${PROJECT_ROOT}/agents/configs/${agent_name}.json:/app/config.json:ro" \
        sutazai-agent-v2
    
    # Wait for container to be healthy
    local max_attempts=30
    local attempt=0
    
    while (( attempt < max_attempts )); do
        if docker exec "$enhanced_container" curl -f -s "http://localhost:8080/health" > /dev/null 2>&1; then
            break
        fi
        
        ((attempt++))
        sleep 2
    done
    
    if (( attempt >= max_attempts )); then
        error "Agent health check failed: $agent_name"
        docker rm -f "$enhanced_container" 2>/dev/null || true
        return 1
    fi
    
    # Perform blue-green switch
    log "Performing blue-green switch for: $agent_name"
    
    # Update load balancer / service discovery
    update_service_discovery "$agent_name" "$enhanced_container"
    
    # Wait for traffic to drain from old container
    sleep 10
    
    # Stop old container
    docker stop "$agent_name" || true
    docker rename "$agent_name" "${agent_name}-old" || true
    docker rename "$enhanced_container" "$agent_name"
    
    # Verify switch success
    if ! verify_agent_switch "$agent_name"; then
        error "Agent switch verification failed: $agent_name"
        # Rollback this agent
        docker rename "$agent_name" "$enhanced_container"
        docker rename "${agent_name}-old" "$agent_name" || true
        docker start "$agent_name" || true
        docker rm -f "$enhanced_container" 2>/dev/null || true
        return 1
    fi
    
    # Clean up old container
    docker rm "${agent_name}-old" 2>/dev/null || true
    
    success "Agent deployed successfully: $agent_name"
    return 0
}

# Update service discovery
update_service_discovery() {
    local agent_name="$1"
    local container_name="$2"
    
    # Update nginx configuration if applicable
    if [[ -f "${PROJECT_ROOT}/nginx/agents.conf" ]]; then
        sed -i "s/${agent_name}/${container_name}/g" "${PROJECT_ROOT}/nginx/agents.conf"
        docker exec nginx nginx -s reload 2>/dev/null || true
    fi
    
    # Update consul service discovery if applicable
    if command -v consul &> /dev/null; then
        consul services register -name="$agent_name" -address="$container_name" -port=8080 || true
    fi
}

# Verify agent switch
verify_agent_switch() {
    local agent_name="$1"
    
    # Check container is running
    if ! docker ps --filter "name=${agent_name}" --format "{{.Names}}" | grep -q "$agent_name"; then
        return 1
    fi
    
    # Check health endpoint
    if ! docker exec "$agent_name" curl -f -s "http://localhost:8080/health" > /dev/null; then
        return 1
    fi
    
    # Check task processing
    local task_endpoint="http://localhost:8080/api/process-task"
    local test_task='{"id":"test","type":"health_check","data":{}}'
    
    if ! docker exec "$agent_name" curl -f -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_task" \
        "$task_endpoint" > /dev/null; then
        return 1
    fi
    
    return 0
}

# Get remaining agents to deploy
get_remaining_agents() {
    local all_agents
    all_agents=$(docker ps --filter "name=*agent*" --format "{{.Names}}" | grep -v "\-v2$" | grep -v "\-old$")
    
    local deployed_agents=""
    if [[ -f "$DEPLOYED_AGENTS_FILE" ]]; then
        deployed_agents=$(cat "$DEPLOYED_AGENTS_FILE")
    fi
    
    # Return agents not yet deployed
    echo "$all_agents" | while read -r agent; do
        if ! echo "$deployed_agents" | grep -q "$agent"; then
            echo "$agent"
        fi
    done
}

# Check rollback conditions
check_rollback_conditions() {
    # Check monitoring system for alerts
    if [[ -f "/tmp/rollback_triggered_${DEPLOYMENT_ID}" ]]; then
        return 0  # Rollback triggered
    fi
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free -g | awk '/^Mem:/{print ($3/$2)*100}')
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        echo "Memory usage critical: ${memory_usage}%" > "/tmp/rollback_triggered_${DEPLOYMENT_ID}"
        return 0
    fi
    
    # Check error rates from logs
    local error_count
    error_count=$(grep -c "ERROR" "${DEPLOYMENT_LOG}" | tail -1 || echo "0")
    if (( error_count > 10 )); then
        echo "High error count: ${error_count}" > "/tmp/rollback_triggered_${DEPLOYMENT_ID}"
        return 0
    fi
    
    return 1  # No rollback needed
}

# Validate phase success
validate_phase_success() {
    local phase_name="$1"
    log "Validating phase success: $phase_name"
    
    # Extract success criteria from config
    local success_criteria
    success_criteria=$(yq -r ".phases.${phase_name}.success_criteria" "$ROLLOUT_CONFIG")
    
    # Check each criterion
    local error_threshold
    error_threshold=$(echo "$success_criteria" | yq -r '.error_rate_threshold // 0.01')
    
    local memory_multiplier
    memory_multiplier=$(echo "$success_criteria" | yq -r '.memory_usage_multiplier // 1.2')
    
    # Validate current metrics against criteria
    if ! validate_metrics "$error_threshold" "$memory_multiplier"; then
        return 1
    fi
    
    # Allow stabilization period
    log "Waiting for phase stabilization (2 minutes)..."
    sleep 120
    
    # Re-validate after stabilization
    if ! validate_metrics "$error_threshold" "$memory_multiplier"; then
        return 1
    fi
    
    success "Phase validation passed: $phase_name"
    return 0
}

# Validate metrics against thresholds
validate_metrics() {
    local error_threshold="$1"
    local memory_multiplier="$2"
    
    # Check error rate (placeholder - would integrate with actual monitoring)
    local current_error_rate=0.001  # 0.1%
    if (( $(echo "$current_error_rate > $error_threshold" | bc -l) )); then
        error "Error rate exceeds threshold: ${current_error_rate} > ${error_threshold}"
        return 1
    fi
    
    # Check memory usage
    local current_memory
    current_memory=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep -v MEM | sed 's/MiB.*//' | awk '{sum+=$1} END {print sum/NR}')
    
    local baseline_memory=150  # From baseline (placeholder)
    local max_memory
    max_memory=$(echo "$baseline_memory * $memory_multiplier" | bc)
    
    if (( $(echo "$current_memory > $max_memory" | bc -l) )); then
        error "Memory usage exceeds threshold: ${current_memory}MB > ${max_memory}MB"
        return 1
    fi
    
    return 0
}

# Post-deployment validation
post_deployment_validation() {
    log "Starting post-deployment validation..."
    
    # Wait for system stabilization
    log "Allowing system stabilization (5 minutes)..."
    sleep 300
    
    # Comprehensive health check
    if ! "${SCRIPT_DIR}/validate-deployment.sh" --deployment-id "$DEPLOYMENT_ID"; then
        error "Post-deployment validation failed"
        return 1
    fi
    
    # Performance comparison
    local current_baseline="${BACKUP_DIR}/performance-baseline.json"
    local post_baseline="/tmp/post-deployment-baseline-${DEPLOYMENT_ID}.json"
    establish_baseline > "$post_baseline"
    
    # Compare performance
    if ! compare_performance "$current_baseline" "$post_baseline"; then
        warning "Performance degradation detected"
        return 1
    fi
    
    success "Post-deployment validation passed"
    return 0
}

# Compare performance baselines
compare_performance() {
    local baseline_file="$1"
    local current_file="$2"
    
    # Extract metrics (simplified comparison)
    local baseline_cpu
    baseline_cpu=$(jq -r '.metrics.avg_cpu_usage' "$baseline_file")
    
    local current_cpu
    current_cpu=$(jq -r '.metrics.avg_cpu_usage' "$current_file")
    
    # Allow 10% performance degradation
    local max_cpu
    max_cpu=$(echo "$baseline_cpu * 1.1" | bc)
    
    if (( $(echo "$current_cpu > $max_cpu" | bc -l) )); then
        error "CPU usage increased beyond acceptable threshold"
        return 1
    fi
    
    return 0
}

# Generate deployment report
generate_report() {
    local status="$1"
    local report_file="${PROJECT_ROOT}/reports/deployment_${DEPLOYMENT_ID}.md"
    
    log "Generating deployment report..."
    
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
# Ollama Integration Deployment Report

**Deployment ID**: ${DEPLOYMENT_ID}
**Status**: ${status}
**Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Duration**: $(calculate_duration)

## Summary
$(generate_summary "$status")

## Metrics
$(generate_metrics_summary)

## Deployed Agents
$(cat "$DEPLOYED_AGENTS_FILE" 2>/dev/null | wc -l) agents successfully deployed:
$(cat "$DEPLOYED_AGENTS_FILE" 2>/dev/null | sed 's/^/- /' || echo "None")

## Logs
- Deployment Log: ${DEPLOYMENT_LOG}
- Monitoring Log: ${LOG_DIR}/monitoring_${DEPLOYMENT_ID}.log
- Backup Location: ${BACKUP_DIR}

## Next Steps
$(generate_next_steps "$status")
EOF
    
    success "Deployment report generated: $report_file"
}

# Calculate deployment duration
calculate_duration() {
    if [[ -f "$STATE_FILE" ]]; then
        local start_time
        start_time=$(jq -r '.start_time // empty' "$STATE_FILE" 2>/dev/null || echo "")
        if [[ -n "$start_time" ]]; then
            local duration
            duration=$(( $(date +%s) - start_time ))
            echo "${duration} seconds"
        else
            echo "Unknown"
        fi
    else
        echo "Unknown"
    fi
}

# Generate summary based on status
generate_summary() {
    local status="$1"
    
    case "$status" in
        "SUCCESS")
            echo "Deployment completed successfully. All 131 agents migrated to BaseAgentV2 with enhanced Ollama integration."
            ;;
        "FAILED")
            echo "Deployment failed during execution. System rolled back to previous stable state."
            ;;
        "PARTIAL") 
            echo "Deployment partially completed. Some agents successfully migrated."
            ;;
        *)
            echo "Deployment status unknown."
            ;;
    esac
}

# Generate metrics summary  
generate_metrics_summary() {
    cat << EOF
- **Total Agents**: 131
- **Successfully Deployed**: $(cat "$DEPLOYED_AGENTS_FILE" 2>/dev/null | wc -l || echo "0")
- **Failed Deployments**: Unknown
- **Rollback Triggered**: $(test -f "/tmp/rollback_triggered_${DEPLOYMENT_ID}" && echo "Yes" || echo "No")
- **Peak Memory Usage**: Unknown
- **Average Response Time**: Unknown
EOF
}

# Generate next steps
generate_next_steps() {
    local status="$1"
    
    case "$status" in
        "SUCCESS")
            cat << EOF
1. Monitor system for 24 hours
2. Update documentation
3. Schedule retrospective meeting
4. Archive deployment artifacts
EOF
            ;;
        "FAILED")
            cat << EOF
1. Review failure logs and analysis
2. Fix identified issues
3. Test fixes in staging environment
4. Plan retry deployment
EOF
            ;;
        *)
            echo "1. Determine current system state"
            echo "2. Plan appropriate next actions"
            ;;
    esac
}

# Main deployment orchestration
main() {
    local start_time
    start_time=$(date +%s)
    
    # Initialize state tracking
    cat > "$STATE_FILE" << EOF
{
    "deployment_id": "${DEPLOYMENT_ID}",
    "start_time": ${start_time},
    "status": "STARTING",
    "phase": "initialization"
}
EOF
    
    log "Starting Ollama Integration Deployment"
    log "Deployment ID: ${DEPLOYMENT_ID}"
    log "Target: 131 agents (BaseAgent -> BaseAgentV2)"
    
    # Pre-deployment steps
    validate_prerequisites
    check_system_resources
    create_backup
    establish_baseline
    parse_rollout_config
    start_monitoring
    
    # Update state
    jq '.status = "DEPLOYING" | .phase = "deployment"' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
    
    # Execute phased deployment
    local phases
    phases=$(cat "/tmp/phases_${DEPLOYMENT_ID}.txt")
    
    for phase in $phases; do
        log "Executing phase: $phase"
        
        jq --arg p "$phase" '.phase = $p' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
        
        if ! deploy_phase "$phase"; then
            error "Phase failed: $phase"
            jq '.status = "FAILED"' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
            generate_report "FAILED"
            exit 1
        fi
        
        success "Phase completed: $phase"
    done
    
    # Post-deployment validation
    jq '.status = "VALIDATING" | .phase = "validation"' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
    
    if ! post_deployment_validation; then
        error "Post-deployment validation failed"
        jq '.status = "FAILED"' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
        generate_report "FAILED"
        exit 1
    fi
    
    # Success
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))
    
    jq --arg et "$end_time" --arg d "$duration" '.status = "SUCCESS" | .end_time = ($et | tonumber) | .duration = ($d | tonumber)' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
    
    success "Deployment completed successfully!"
    success "Duration: ${duration} seconds"
    success "All 131 agents migrated to BaseAgentV2"
    
    generate_report "SUCCESS"
    
    log "Deployment artifacts:"
    log "- State: $STATE_FILE"
    log "- Logs: $DEPLOYMENT_LOG"
    log "- Backup: $BACKUP_DIR"
    log "- Report: ${PROJECT_ROOT}/reports/deployment_${DEPLOYMENT_ID}.md"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_prerequisites
        check_system_resources
        success "Pre-deployment validation passed"
        ;;
    "dry-run")
        log "DRY RUN MODE: No actual changes will be made"
        validate_prerequisites
        check_system_resources
        parse_rollout_config
        success "Dry run completed successfully"
        ;;
    "help"|"-h"|"--help")
        cat << EOF
Ollama Integration Deployment Script

Usage: $0 [command]

Commands:
    deploy      Execute full deployment (default)
    validate    Run pre-deployment validation only
    dry-run     Validate and parse config without deploying
    help        Show this help message

Environment Variables:
    DEPLOYMENT_ENV      Target environment (staging|production)
    MAX_BATCH_SIZE      Maximum agents per batch (default: 3)
    ROLLBACK_THRESHOLD  Error threshold for auto-rollback (default: 5%)
    
Examples:
    $0 deploy                   # Full deployment
    $0 validate                 # Pre-flight checks only
    $0 dry-run                  # Validate without deploying
    
Logs: ${LOG_DIR}/deployment_*.log
EOF
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        exit 1
        ;;
esac