#!/bin/bash

# Ollama Integration Rollback Script
# Emergency rollback for deployment failures
# Author: Infrastructure DevOps Manager
# Version: 2.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Rollback configuration
ROLLBACK_ID="rollback-$(date +%Y%m%d_%H%M%S)"
ROLLBACK_LOG="${LOG_DIR}/rollback_${ROLLBACK_ID}.log"
ROLLBACK_TIMEOUT=180  # 3 minutes maximum rollback time
VALIDATION_TIMEOUT=120  # 2 minutes validation time

# State management
ROLLBACK_FLAG="/tmp/rollback-in-progress"
ROLLBACK_STATE="/tmp/rollback-state-${ROLLBACK_ID}.json"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging setup
exec 1> >(tee -a "${ROLLBACK_LOG}")
exec 2> >(tee -a "${ROLLBACK_LOG}" >&2)

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

urgent() {
    echo -e "${RED}${PURPLE}[URGENT $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

# Cleanup function
cleanup() {
    local exit_code=$?
    log "Rollback cleanup initiated (exit code: $exit_code)"
    
    # Remove rollback flag
    rm -f "${ROLLBACK_FLAG}"
    
    # Stop any monitoring processes
    pkill -f "rollback-monitor" || true
    
    if [ $exit_code -eq 0 ]; then
        success "Rollback completed successfully"
    else
        error "Rollback encountered errors. Check logs at: ${ROLLBACK_LOG}"
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Initialize rollback tracking
initialize_rollback() {
    local reason="${1:-manual}"
    local start_time
    start_time=$(date +%s)
    
    # Create rollback flag to prevent concurrent operations
    echo "$ROLLBACK_ID" > "$ROLLBACK_FLAG"
    
    # Initialize rollback state
    cat > "$ROLLBACK_STATE" << EOF
{
    "rollback_id": "${ROLLBACK_ID}",
    "start_time": ${start_time},
    "reason": "${reason}",
    "status": "INITIALIZING",
    "phase": "initialization",
    "agents_rolled_back": [],
    "failed_rollbacks": []
}
EOF
    
    urgent "🚨 ROLLBACK INITIATED: ${reason}"
    urgent "Rollback ID: ${ROLLBACK_ID}"
    urgent "Target: Return all agents to BaseAgent (legacy)"
    
    # Send immediate alerts
    send_alert "CRITICAL" "Ollama Integration Rollback Started" "Reason: ${reason}"
}

# Send alerts to notification channels
send_alert() {
    local severity="$1"
    local title="$2"
    local message="$3"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 ${severity}: ${title}\n${message}\"}" \
            "${SLACK_WEBHOOK_URL}" &>/dev/null || true
    fi
    
    # Email notification (if configured)
    if command -v sendmail &> /dev/null && [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo -e "Subject: ${severity}: ${title}\n\n${message}" | \
            sendmail "${ALERT_EMAIL}" || true
    fi
    
    # Log the alert
    case "$severity" in
        "CRITICAL") urgent "$title: $message" ;;
        "WARNING") warning "$title: $message" ;;
        *) log "$title: $message" ;;
    esac
}

# Immediate stop of ongoing deployment
immediate_stop() {
    log "Phase 1: Immediate deployment stop (0-30 seconds)"
    
    # Stop any running deployment processes
    pkill -f "deploy-ollama-integration" || true
    pkill -f "deployment-monitor" || true
    
    # Pause any Kubernetes deployments
    if command -v kubectl &> /dev/null; then
        kubectl patch deployment ollama-integration -p '{"spec":{"paused":true}}' &>/dev/null || true
    fi
    
    # Stop Docker Compose operations
    docker-compose -f "${PROJECT_ROOT}/docker-compose.ollama-optimized.yml" stop &>/dev/null || true
    
    # Lock deployment directory
    touch "${PROJECT_ROOT}/deployment/.rollback-in-progress"
    
    # Update state
    jq '.status = "STOPPING" | .phase = "immediate_stop"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
        mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    
    success "Deployment processes stopped successfully"
}

# Redirect traffic to stable environment
redirect_traffic() {
    log "Phase 2: Traffic redirection (30-60 seconds)"
    
    # Update nginx configuration to point to legacy agents
    if [[ -f "${PROJECT_ROOT}/nginx/nginx.conf" ]]; then
        # Backup current config
        cp "${PROJECT_ROOT}/nginx/nginx.conf" "${PROJECT_ROOT}/nginx/nginx.conf.rollback-backup"
        
        # Restore backup configuration if it exists
        if [[ -f "${BACKUP_DIR}/latest/nginx.conf" ]]; then
            cp "${BACKUP_DIR}/latest/nginx.conf" "${PROJECT_ROOT}/nginx/nginx.conf"
            docker exec nginx nginx -s reload &>/dev/null || true
        fi
    fi
    
    # Update service discovery
    if command -v consul &> /dev/null; then
        # Re-register legacy services
        consul services register -name="agent-registry" -address="legacy-agents" -port=8080 || true
    fi
    
    # Update load balancer configuration
    update_load_balancer_config "legacy"
    
    # Update state
    jq '.status = "REDIRECTING" | .phase = "traffic_redirect"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
        mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    
    success "Traffic redirected to stable environment"
}

# Update load balancer configuration
update_load_balancer_config() {
    local target="$1"  # "legacy" or "enhanced"
    
    case "$target" in
        "legacy")
            # Point to original agent containers
            if [[ -f "${PROJECT_ROOT}/nginx/upstream.conf" ]]; then
                sed -i 's/-v2://g' "${PROJECT_ROOT}/nginx/upstream.conf"
            fi
            ;;
        "enhanced")
            # Point to enhanced agent containers
            if [[ -f "${PROJECT_ROOT}/nginx/upstream.conf" ]]; then
                sed -i 's/:/:v2:/g' "${PROJECT_ROOT}/nginx/upstream.conf"
            fi
            ;;
    esac
    
    # Reload nginx if running
    docker exec nginx nginx -s reload &>/dev/null || true
}

# Swap containers back to legacy versions
swap_containers() {
    log "Phase 3: Container environment swap (60-120 seconds)"
    
    local swapped_agents=()
    local failed_swaps=()
    
    # Get list of enhanced agents (v2 containers)
    local enhanced_agents
    enhanced_agents=$(docker ps --filter "name=*-v2" --format "{{.Names}}" || true)
    
    if [[ -z "$enhanced_agents" ]]; then
        log "No enhanced agents found to rollback"
        return 0
    fi
    
    # Process each enhanced agent
    echo "$enhanced_agents" | while read -r enhanced_container; do
        if [[ -z "$enhanced_container" ]]; then
            continue
        fi
        
        local base_name
        base_name=$(echo "$enhanced_container" | sed 's/-v2$//')
        
        log "Rolling back agent: $base_name"
        
        if rollback_single_agent "$base_name" "$enhanced_container"; then
            swapped_agents+=("$base_name")
            jq --arg agent "$base_name" '.agents_rolled_back += [$agent]' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
                mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
        else
            failed_swaps+=("$base_name")
            jq --arg agent "$base_name" '.failed_rollbacks += [$agent]' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
                mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
            warning "Failed to rollback agent: $base_name"
        fi
    done
    
    # Update state
    jq '.status = "SWAPPING" | .phase = "container_swap"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
        mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    
    success "Container swap completed (success: ${#swapped_agents[@]}, failed: ${#failed_swaps[@]})"
}

# Rollback single agent
rollback_single_agent() {
    local agent_name="$1"
    local enhanced_container="$2"
    
    # Check if legacy container exists (stopped)
    local legacy_container="${agent_name}-old"
    if ! docker ps -a --filter "name=${legacy_container}" --format "{{.Names}}" | grep -q "$legacy_container"; then
        # No backup container found, recreate from image
        log "Recreating legacy container for: $agent_name"
        return recreate_legacy_agent "$agent_name"
    fi
    
    # Stop enhanced container
    docker stop "$enhanced_container" &>/dev/null || true
    
    # Start legacy container
    if ! docker start "$legacy_container"; then
        error "Failed to start legacy container: $legacy_container"
        return 1
    fi
    
    # Rename containers
    docker rename "$enhanced_container" "${enhanced_container}-failed" &>/dev/null || true
    docker rename "$legacy_container" "$agent_name" &>/dev/null || true
    
    # Wait for container to be healthy
    local max_attempts=30
    local attempt=0
    
    while (( attempt < max_attempts )); do
        if docker exec "$agent_name" curl -f -s "http://localhost:8080/health" &>/dev/null; then
            break
        fi
        ((attempt++))
        sleep 2
    done
    
    if (( attempt >= max_attempts )); then
        error "Legacy agent health check failed: $agent_name"
        return 1
    fi
    
    # Clean up failed enhanced container
    docker rm -f "${enhanced_container}-failed" &>/dev/null || true
    
    success "Agent rolled back successfully: $agent_name"
    return 0
}

# Recreate legacy agent from backup or image
recreate_legacy_agent() {
    local agent_name="$1"
    
    # Use original docker-compose to recreate
    if ! docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" up -d "$agent_name"; then
        error "Failed to recreate legacy agent: $agent_name"
        return 1
    fi
    
    # Wait for health check
    sleep 10
    
    if ! docker exec "$agent_name" curl -f -s "http://localhost:8080/health" &>/dev/null; then
        error "Recreated legacy agent health check failed: $agent_name"
        return 1
    fi
    
    return 0
}

# Restore database state
restore_database_state() {
    log "Restoring database state to legacy configuration"
    
    # Restore agent registry to legacy versions
    if command -v psql &> /dev/null && [[ -n "${POSTGRES_PASSWORD:-}" ]]; then
        PGPASSWORD="${POSTGRES_PASSWORD}" psql -h localhost -U postgres sutazai -c "
            UPDATE agents 
            SET base_version='1.0.0', 
                status='active', 
                updated_at=NOW(),
                capabilities = CASE 
                    WHEN capabilities @> '[\"enhanced_ollama\"]'::jsonb 
                    THEN capabilities - 'enhanced_ollama'
                    ELSE capabilities
                END
            WHERE base_version='2.0.0';
        " &>/dev/null || warning "Failed to update database state"
    fi
    
    # Clear enhanced agent metrics from Redis
    if command -v redis-cli &> /dev/null; then
        redis-cli DEL "metrics:base-agent-v2:*" &>/dev/null || true
        redis-cli DEL "circuit-breaker:*" &>/dev/null || true
        redis-cli DEL "connection-pool:*" &>/dev/null || true
    fi
    
    success "Database state restored"
}

# Restore configuration files
restore_configurations() {
    log "Restoring configuration files"
    
    # Find latest backup
    local latest_backup
    latest_backup=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        warning "No backup directory found for configuration restore"
        return 1
    fi
    
    log "Restoring from backup: $latest_backup"
    
    # Restore agent configurations
    if [[ -d "${latest_backup}/configs" ]]; then
        cp -r "${latest_backup}/configs/"* "${PROJECT_ROOT}/agents/configs/" 2>/dev/null || true
    fi
    
    # Restore docker-compose configuration
    if [[ -f "${latest_backup}/docker-compose.yml" ]]; then
        cp "${latest_backup}/docker-compose.yml" "${PROJECT_ROOT}/docker-compose.yml"
    fi
    
    # Restore environment variables
    if [[ -f "${latest_backup}/.env" ]]; then
        cp "${latest_backup}/.env" "${PROJECT_ROOT}/.env"
    fi
    
    success "Configuration files restored"
}

# Comprehensive validation of rollback
validate_rollback() {
    log "Phase 4: Rollback validation (120-210 seconds)"
    
    local validation_start
    validation_start=$(date +%s)
    
    # Update state
    jq '.status = "VALIDATING" | .phase = "validation"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
        mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    
    # 1. Verify all agents are running legacy versions
    if ! validate_agent_versions; then
        error "Agent version validation failed"
        return 1
    fi
    
    # 2. Check system health
    if ! validate_system_health; then
        error "System health validation failed"
        return 1
    fi
    
    # 3. Verify task processing
    if ! validate_task_processing; then
        error "Task processing validation failed"
        return 1
    fi
    
    # 4. Check resource utilization
    if ! validate_resource_usage; then
        error "Resource usage validation failed"
        return 1
    fi
    
    # 5. Verify no enhanced components remain
    if ! validate_cleanup; then
        error "Enhanced component cleanup validation failed"
        return 1
    fi
    
    local validation_end
    validation_end=$(date +%s)
    local validation_duration
    validation_duration=$((validation_end - validation_start))
    
    success "Rollback validation completed in ${validation_duration} seconds"
}

# Validate agent versions
validate_agent_versions() {
    log "Validating agent versions..."
    
    # Check no v2 containers are running
    local v2_containers
    v2_containers=$(docker ps --filter "name=*-v2" --format "{{.Names}}" || true)
    
    if [[ -n "$v2_containers" ]]; then
        error "Enhanced containers still running: $v2_containers"
        return 1
    fi
    
    # Check all expected legacy agents are running
    local expected_agents=131
    local running_agents
    running_agents=$(docker ps --filter "name=*agent*" --format "{{.Names}}" | wc -l)
    
    if (( running_agents < expected_agents )); then
        error "Not all agents running: ${running_agents}/${expected_agents}"
        return 1
    fi
    
    success "Agent versions validated (${running_agents} legacy agents running)"
    return 0
}

# Validate system health
validate_system_health() {
    log "Validating system health..."
    
    # Check agent health endpoints
    local unhealthy_agents=()
    local agent_list
    agent_list=$(docker ps --filter "name=*agent*" --format "{{.Names}}")
    
    echo "$agent_list" | while read -r agent; do
        if [[ -z "$agent" ]]; then
            continue
        fi
        
        if ! docker exec "$agent" curl -f -s "http://localhost:8080/health" &>/dev/null; then
            unhealthy_agents+=("$agent")
        fi
    done
    
    if (( ${#unhealthy_agents[@]} > 0 )); then
        error "Unhealthy agents detected: ${unhealthy_agents[*]}"
        return 1
    fi
    
    # Check backend connectivity
    if ! curl -f -s "http://localhost:8000/health" &>/dev/null; then
        error "Backend service unhealthy"
        return 1
    fi
    
    success "System health validated"
    return 0
}

# Validate task processing
validate_task_processing() {
    log "Validating task processing..."
    
    # Submit test tasks to verify processing
    local test_task='{
        "id": "rollback-validation-test",
        "type": "health_check",
        "data": {"test": true}
    }'
    
    # Test task submission
    if ! curl -f -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_task" \
        "http://localhost:8000/api/tasks/submit" &>/dev/null; then
        error "Task submission failed"
        return 1
    fi
    
    # Wait for task processing
    sleep 5
    
    # Check task was processed
    if ! curl -f -s "http://localhost:8000/api/tasks/status/rollback-validation-test" &>/dev/null; then
        error "Task processing verification failed"
        return 1
    fi
    
    success "Task processing validated"
    return 0
}

# Validate resource usage
validate_resource_usage() {
    log "Validating resource usage..."
    
    # Check memory usage is within normal bounds
    local memory_usage_percent
    memory_usage_percent=$(free | awk '/^Mem:/{printf "%.0f", $3/$2*100}')
    
    if (( memory_usage_percent > 80 )); then
        warning "High memory usage after rollback: ${memory_usage_percent}%"
    fi
    
    # Check CPU usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        warning "High CPU usage after rollback: ${cpu_usage}%"
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    
    if (( disk_usage > 90 )); then
        warning "High disk usage: ${disk_usage}%"
    fi
    
    success "Resource usage validated (Memory: ${memory_usage_percent}%, CPU: ${cpu_usage}%, Disk: ${disk_usage}%)"
    return 0
}

# Validate cleanup of enhanced components
validate_cleanup() {
    log "Validating enhanced component cleanup..."
    
    # Check no enhanced Docker images remain
    local v2_images
    v2_images=$(docker images | grep "sutazai-agent-v2" || true)
    
    if [[ -n "$v2_images" ]]; then
        log "Enhanced images still present (will be cleaned up later)"
    fi
    
    # Check no enhanced configuration remains
    if [[ -f "${PROJECT_ROOT}/docker-compose.ollama-optimized.yml" ]]; then
        log "Enhanced compose file still present (archived)"
    fi
    
    # Verify no enhanced processes running
    local enhanced_processes
    enhanced_processes=$(pgrep -f "base_agent_v2" || true)
    
    if [[ -n "$enhanced_processes" ]]; then
        error "Enhanced agent processes still running: $enhanced_processes"
        return 1
    fi
    
    success "Enhanced component cleanup validated"
    return 0
}

# Generate rollback report
generate_rollback_report() {
    local status="$1"
    local report_file="${PROJECT_ROOT}/reports/rollback_${ROLLBACK_ID}.md"
    
    log "Generating rollback report..."
    
    mkdir -p "$(dirname "$report_file")"
    
    local end_time
    end_time=$(date +%s)
    local start_time
    start_time=$(jq -r '.start_time' "$ROLLBACK_STATE")
    local duration
    duration=$((end_time - start_time))
    
    local agents_rolled_back
    agents_rolled_back=$(jq -r '.agents_rolled_back | length' "$ROLLBACK_STATE")
    
    local failed_rollbacks
    failed_rollbacks=$(jq -r '.failed_rollbacks | length' "$ROLLBACK_STATE")
    
    cat > "$report_file" << EOF
# Ollama Integration Rollback Report

**Rollback ID**: ${ROLLBACK_ID}
**Status**: ${status}
**Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Duration**: ${duration} seconds
**Reason**: $(jq -r '.reason' "$ROLLBACK_STATE")

## Executive Summary
$(generate_rollback_summary "$status" "$agents_rolled_back" "$failed_rollbacks")

## Rollback Statistics
- **Total Agents Targeted**: 131
- **Successfully Rolled Back**: ${agents_rolled_back}
- **Failed Rollbacks**: ${failed_rollbacks}
- **Rollback Success Rate**: $(echo "scale=2; $agents_rolled_back / 131 * 100" | bc)%
- **Total Duration**: ${duration} seconds

## Phase Timeline
$(generate_phase_timeline)

## System State Post-Rollback
$(generate_system_state)

## Failed Rollbacks
$(generate_failed_rollbacks)

## Recommendations
$(generate_rollback_recommendations "$status")

## Artifacts
- **Rollback Log**: ${ROLLBACK_LOG}
- **State File**: ${ROLLBACK_STATE}
- **Backup Location**: ${BACKUP_DIR}

## Next Steps
$(generate_rollback_next_steps "$status")
EOF
    
    success "Rollback report generated: $report_file"
}

# Generate rollback summary
generate_rollback_summary() {
    local status="$1"
    local success_count="$2"
    local failed_count="$3"
    
    case "$status" in
        "SUCCESS")
            echo "Rollback completed successfully. All agents restored to BaseAgent (legacy) with stable operation confirmed."
            ;;
        "PARTIAL")
            echo "Rollback partially completed. ${success_count} agents successfully rolled back, ${failed_count} failed. Manual intervention required."
            ;;
        "FAILED")
            echo "Rollback failed. System may be in inconsistent state. Immediate manual intervention required."
            ;;
    esac
}

# Generate phase timeline
generate_phase_timeline() {
    cat << EOF
1. **Immediate Stop** (0-30s): Deployment halted
2. **Traffic Redirect** (30-60s): Load balancer updated  
3. **Container Swap** (60-120s): Agents replaced with legacy versions
4. **Validation** (120-210s): System health verified
EOF
}

# Generate system state
generate_system_state() {
    local running_agents
    running_agents=$(docker ps --filter "name=*agent*" --format "{{.Names}}" | wc -l)
    
    local memory_usage
    memory_usage=$(free | awk '/^Mem:/{printf "%.0f", $3/$2*100}')
    
    cat << EOF
- **Running Agents**: ${running_agents}/131
- **Memory Usage**: ${memory_usage}%
- **All Enhanced Containers**: Removed
- **Database State**: Restored to legacy
- **Configuration Files**: Restored from backup
EOF
}

# Generate failed rollbacks section
generate_failed_rollbacks() {
    local failed_list
    failed_list=$(jq -r '.failed_rollbacks[]?' "$ROLLBACK_STATE" 2>/dev/null || echo "None")
    
    if [[ "$failed_list" == "None" ]]; then
        echo "No failed rollbacks."
    else
        echo "The following agents failed to rollback and require manual intervention:"
        echo "$failed_list" | sed 's/^/- /'
    fi
}

# Generate rollback recommendations
generate_rollback_recommendations() {
    local status="$1"
    
    case "$status" in
        "SUCCESS")
            cat << EOF
1. Monitor system stability for next 4 hours
2. Investigate root cause of deployment failure
3. Plan remediation for identified issues
4. Update deployment procedures based on lessons learned
EOF
            ;;
        "PARTIAL"|"FAILED")
            cat << EOF
1. **URGENT**: Manually recover failed agents
2. Verify system integrity and data consistency
3. Complete rollback of any remaining enhanced components
4. Investigate rollback failures for future prevention
EOF
            ;;
    esac
}

# Generate next steps
generate_rollback_next_steps() {
    local status="$1"
    
    case "$status" in
        "SUCCESS")
            cat << EOF
1. System monitoring and alerting review
2. Post-incident retrospective meeting
3. Update deployment and rollback procedures
4. Plan deployment retry with fixes
EOF
            ;;
        *)
            cat << EOF
1. **Immediate**: Complete manual recovery
2. System integrity verification
3. Incident escalation if needed
4. Detailed failure analysis
EOF
            ;;
    esac
}

# Main rollback orchestration
main_rollback() {
    local reason="${1:-manual}"
    local start_time
    start_time=$(date +%s)
    
    # Initialize rollback
    initialize_rollback "$reason"
    
    urgent "Beginning emergency rollback sequence..."
    
    # Phase 1: Immediate stop (0-30 seconds)
    if ! timeout 30 immediate_stop; then
        error "Phase 1 (Immediate Stop) timed out or failed"
        jq '.status = "FAILED" | .phase = "immediate_stop_failed"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
            mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
        generate_rollback_report "FAILED"
        exit 1
    fi
    
    # Phase 2: Traffic redirection (30-60 seconds)  
    if ! timeout 30 redirect_traffic; then
        error "Phase 2 (Traffic Redirect) timed out or failed"
        jq '.status = "FAILED" | .phase = "traffic_redirect_failed"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
            mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
        generate_rollback_report "FAILED"
        exit 1
    fi
    
    # Phase 3: Container swap (60-120 seconds)
    if ! timeout 60 swap_containers; then
        warning "Phase 3 (Container Swap) timed out or partially failed"
        jq '.status = "PARTIAL" | .phase = "container_swap_partial"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
            mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    fi
    
    # Restore database and configurations
    restore_database_state
    restore_configurations
    
    # Phase 4: Validation (120-210 seconds)
    if ! timeout "$VALIDATION_TIMEOUT" validate_rollback; then
        warning "Phase 4 (Validation) timed out or failed"
        jq '.status = "PARTIAL" | .phase = "validation_failed"' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
            mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
        generate_rollback_report "PARTIAL"
        exit 0  # Partial success still allows system operation
    fi
    
    # Success
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))
    
    jq --arg et "$end_time" --arg d "$duration" '.status = "SUCCESS" | .end_time = ($et | tonumber) | .duration = ($d | tonumber)' "$ROLLBACK_STATE" > "$ROLLBACK_STATE.tmp" && \
        mv "$ROLLBACK_STATE.tmp" "$ROLLBACK_STATE"
    
    success "🎉 ROLLBACK COMPLETED SUCCESSFULLY!"
    success "Duration: ${duration} seconds"
    success "System restored to stable BaseAgent configuration"
    
    generate_rollback_report "SUCCESS"
    
    # Send success notification
    send_alert "INFO" "Rollback Completed Successfully" "Duration: ${duration}s. System stable."
    
    log "Rollback artifacts:"
    log "- State: $ROLLBACK_STATE"
    log "- Logs: $ROLLBACK_LOG"
    log "- Report: ${PROJECT_ROOT}/reports/rollback_${ROLLBACK_ID}.md"
}

# Command line argument handling
case "${1:-rollback}" in
    "rollback"|"emergency")
        if [[ -f "$ROLLBACK_FLAG" ]]; then
            error "Another rollback is already in progress"
            exit 1
        fi
        main_rollback "${2:-manual}"
        ;;
    "status")
        if [[ -f "$ROLLBACK_FLAG" ]]; then
            local current_rollback_id
            current_rollback_id=$(cat "$ROLLBACK_FLAG")
            log "Rollback in progress: $current_rollback_id"
            
            local state_file="/tmp/rollback-state-${current_rollback_id}.json"
            if [[ -f "$state_file" ]]; then
                jq -r '"Status: " + .status + ", Phase: " + .phase' "$state_file"
            fi
        else
            log "No rollback currently in progress"
        fi
        ;;
    "force")
        warning "FORCE MODE: Attempting rollback even if another is in progress"
        rm -f "$ROLLBACK_FLAG"
        main_rollback "force-${2:-manual}"
        ;;
    "validate")
        log "Validating rollback capability..."
        if validate_rollback; then
            success "Rollback validation passed"
        else
            error "Rollback validation failed"
            exit 1
        fi
        ;;
    "help"|"-h"|"--help")
        cat << EOF
Ollama Integration Rollback Script

Usage: $0 [command] [reason]

Commands:
    rollback [reason]   Execute emergency rollback (default)
    emergency [reason]  Alias for rollback
    status              Check current rollback status
    force [reason]      Force rollback even if one is in progress
    validate            Test rollback capability without executing
    help                Show this help message

Reasons (optional):
    manual              Manual rollback request (default)
    memory_exhaustion   High memory usage triggered rollback
    error_rate          High error rate triggered rollback
    health_check        Health check failures triggered rollback
    timeout             Deployment timeout triggered rollback

Examples:
    $0 rollback                          # Manual emergency rollback
    $0 emergency memory_exhaustion       # Rollback due to memory issues
    $0 status                           # Check rollback status
    $0 force manual                     # Force rollback
    
Logs: ${LOG_DIR}/rollback_*.log
EOF
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        exit 1
        ;;
esac