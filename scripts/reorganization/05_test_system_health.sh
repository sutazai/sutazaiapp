#!/bin/bash
# SutazAI System Health Testing Script
# Comprehensive health testing after reorganization

set -euo pipefail

# Configuration
LOG_FILE="/opt/sutazaiapp/logs/reorganization.log"
HEALTH_REPORT="/opt/sutazaiapp/logs/health_report_$(date +"%Y%m%d_%H%M%S").md"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1" | tee -a "$LOG_FILE"
}

warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1" | tee -a "$LOG_FILE"
}

# Initialize health report
init_health_report() {
    cat > "$HEALTH_REPORT" << 'EOF'
# SutazAI System Health Report

**Generated:** $(date)
**Test Suite:** Post-Reorganization Health Check

## Test Results Summary

EOF
}

# Test Docker infrastructure
test_docker_infrastructure() {
    log "Testing Docker infrastructure..."
    
    local docker_status="✅ PASSED"
    local details=""
    
    # Test Docker daemon
    if ! docker info >/dev/null 2>&1; then
        docker_status="❌ FAILED"
        details+="- Docker daemon not running\n"
        error "Docker daemon not accessible"
        return 1
    fi
    
    # Test essential containers
    local essential_containers=(
        "sutazai-backend-minimal"
        "sutazai-ollama-minimal"
        "sutazai-postgres-minimal"
        "sutazai-redis-minimal" 
        "sutazai-frontend-minimal"
    )
    
    local running_containers=0
    
    for container in "${essential_containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            ((running_containers++))
            details+="- ✅ $container: Running\n"
        else
            docker_status="⚠️ DEGRADED"
            details+="- ❌ $container: Not running\n"
            warning "$container is not running"
        fi
    done
    
    # Test container health
    for container in "${essential_containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-healthcheck")
            if [[ "$health" == "healthy" ]]; then
                details+="- ✅ $container health: Healthy\n"
            elif [[ "$health" == "no-healthcheck" ]]; then
                details+="- ℹ️ $container health: No healthcheck configured\n"
            else
                docker_status="⚠️ DEGRADED"
                details+="- ⚠️ $container health: $health\n"
                warning "$container health status: $health"
            fi
        fi
    done
    
    cat >> "$HEALTH_REPORT" << EOF
### Docker Infrastructure: $docker_status

**Running Containers:** $running_containers/${#essential_containers[@]}

**Details:**
$details

EOF
    
    success "Docker infrastructure test: $docker_status"
    [[ "$docker_status" == "✅ PASSED" ]]
}

# Test backend API
test_backend_api() {
    log "Testing backend API..."
    
    local api_status="✅ PASSED"
    local details=""
    
    # Test health endpoint
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
        details+="- ✅ Health endpoint: Accessible\n"
        
        # Get health details
        local health_data=$(curl -s "http://localhost:8000/health" | jq -r '.status' 2>/dev/null || echo "unknown")
        details+="- ℹ️ Health status: $health_data\n"
    else
        api_status="❌ FAILED"
        details+="- ❌ Health endpoint: Not accessible\n"
        error "Backend health endpoint not accessible"
    fi
    
    # Test core endpoints
    local endpoints=(
        "/agents"
        "/models" 
        "/public/metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -sf "http://localhost:8000$endpoint" >/dev/null 2>&1; then
            details+="- ✅ $endpoint: Accessible\n"
        else
            api_status="⚠️ DEGRADED"
            details+="- ❌ $endpoint: Not accessible\n"
            warning "Endpoint not accessible: $endpoint"
        fi
    done
    
    # Test chat functionality
    if curl -sf -X POST "http://localhost:8000/simple-chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "test"}' >/dev/null 2>&1; then
        details+="- ✅ Chat endpoint: Functional\n"
    else
        api_status="⚠️ DEGRADED"
        details+="- ❌ Chat endpoint: Not functional\n"
        warning "Chat endpoint not functional"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### Backend API: $api_status

**Details:**
$details

EOF
    
    success "Backend API test: $api_status"
    [[ "$api_status" != "❌ FAILED" ]]
}

# Test frontend accessibility
test_frontend() {
    log "Testing frontend accessibility..."
    
    local frontend_status="✅ PASSED"
    local details=""
    
    # Test frontend endpoint
    if curl -sf "http://localhost:8501" >/dev/null 2>&1; then
        details+="- ✅ Frontend endpoint: Accessible\n"
    else
        frontend_status="❌ FAILED"
        details+="- ❌ Frontend endpoint: Not accessible\n"
        error "Frontend not accessible"
    fi
    
    # Test if Streamlit is running
    if docker logs sutazai-frontend-minimal 2>&1 | grep -q "Streamlit"; then
        details+="- ✅ Streamlit: Running\n"
    else
        frontend_status="⚠️ DEGRADED"
        details+="- ⚠️ Streamlit: Status unknown\n"
        warning "Streamlit status unclear"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### Frontend: $frontend_status

**Details:**
$details

EOF
    
    success "Frontend test: $frontend_status"
    [[ "$frontend_status" != "❌ FAILED" ]]
}

# Test AI model availability
test_ai_models() {
    log "Testing AI model availability..."
    
    local models_status="✅ PASSED"
    local details=""
    
    # Test Ollama service
    if curl -sf "http://localhost:10104/api/tags" >/dev/null 2>&1; then
        details+="- ✅ Ollama service: Accessible\n"
        
        # Get model count
        local model_count=$(curl -s "http://localhost:10104/api/tags" | jq '.models | length' 2>/dev/null || echo 0)
        details+="- ℹ️ Available models: $model_count\n"
        
        if [ "$model_count" -gt 0 ]; then
            details+="- ✅ Models: Available\n"
        else
            models_status="⚠️ DEGRADED"
            details+="- ⚠️ Models: None loaded\n"
            warning "No AI models loaded"
        fi
    else
        models_status="❌ FAILED"
        details+="- ❌ Ollama service: Not accessible\n"
        error "Ollama service not accessible"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### AI Models: $models_status

**Details:**
$details

EOF
    
    success "AI models test: $models_status"
    [[ "$models_status" != "❌ FAILED" ]]
}

# Test database connectivity
test_databases() {
    log "Testing database connectivity..."
    
    local db_status="✅ PASSED"
    local details=""
    
    # Test PostgreSQL
    if docker exec sutazai-postgres-minimal pg_isready -U sutazai >/dev/null 2>&1; then
        details+="- ✅ PostgreSQL: Connected\n"
    else
        db_status="❌ FAILED"
        details+="- ❌ PostgreSQL: Connection failed\n"
        error "PostgreSQL connection failed"
    fi
    
    # Test Redis
    if docker exec sutazai-redis-minimal redis-cli ping | grep -q "PONG"; then
        details+="- ✅ Redis: Connected\n"
    else
        db_status="❌ FAILED"
        details+="- ❌ Redis: Connection failed\n"
        error "Redis connection failed"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### Databases: $db_status

**Details:**
$details

EOF
    
    success "Database test: $db_status"
    [[ "$db_status" != "❌ FAILED" ]]
}

# Test system resources
test_system_resources() {
    log "Testing system resources..."
    
    local resources_status="✅ PASSED"
    local details=""
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1 || echo "unknown")
    details+="- ℹ️ CPU usage: ${cpu_usage}%\n"
    
    # Memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    details+="- ℹ️ Memory usage: ${memory_usage}%\n"
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        resources_status="⚠️ DEGRADED"
        details+="- ⚠️ High memory usage detected\n"
        warning "High memory usage: ${memory_usage}%"
    fi
    
    # Disk usage
    local disk_usage=$(df /opt/sutazaiapp | awk 'NR==2 {print $5}' | cut -d'%' -f1)
    details+="- ℹ️ Disk usage: ${disk_usage}%\n"
    
    if [ "$disk_usage" -gt 85 ]; then
        resources_status="⚠️ DEGRADED"
        details+="- ⚠️ High disk usage detected\n"
        warning "High disk usage: ${disk_usage}%"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### System Resources: $resources_status

**Details:**
$details

EOF
    
    success "System resources test: $resources_status"
    [[ "$resources_status" != "❌ FAILED" ]]
}

# Test essential scripts
test_essential_scripts() {
    log "Testing essential scripts..."
    
    local scripts_status="✅ PASSED"
    local details=""
    
    # Essential scripts that must exist
    local essential_scripts=(
        "/opt/sutazaiapp/scripts/live_logs.sh"
        "/opt/sutazaiapp/health_check.sh"
        "/opt/sutazaiapp/backend/app/main.py"
        "/opt/sutazaiapp/frontend/app.py"
    )
    
    for script in "${essential_scripts[@]}"; do
        if [ -f "$script" ]; then
            details+="- ✅ $(basename "$script"): Present\n"
        else
            scripts_status="❌ FAILED"
            details+="- ❌ $(basename "$script"): Missing\n"
            error "Essential script missing: $script"
        fi
    done
    
    # Test script executability
    local executable_scripts=(
        "/opt/sutazaiapp/scripts/live_logs.sh"
        "/opt/sutazaiapp/health_check.sh"
    )
    
    for script in "${executable_scripts[@]}"; do
        if [ -x "$script" ]; then
            details+="- ✅ $(basename "$script"): Executable\n"
        else
            scripts_status="⚠️ DEGRADED"
            details+="- ⚠️ $(basename "$script"): Not executable\n"
            warning "Script not executable: $script"
        fi
    done
    
    cat >> "$HEALTH_REPORT" << EOF
### Essential Scripts: $scripts_status

**Details:**
$details

EOF
    
    success "Essential scripts test: $scripts_status"
    [[ "$scripts_status" != "❌ FAILED" ]]
}

# Perform end-to-end functionality test
test_end_to_end() {
    log "Performing end-to-end functionality test..."
    
    local e2e_status="✅ PASSED"
    local details=""
    
    # Test complete chat workflow
    log "Testing complete chat workflow..."
    
    local chat_response=$(curl -s -X POST "http://localhost:8000/simple-chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, test system health"}' 2>/dev/null)
    
    if echo "$chat_response" | jq -e '.response' >/dev/null 2>&1; then
        details+="- ✅ Complete chat workflow: Functional\n"
        
        # Check if we got a reasonable response
        local response_text=$(echo "$chat_response" | jq -r '.response' 2>/dev/null || echo "")
        if [ ${#response_text} -gt 10 ]; then
            details+="- ✅ AI response generation: Working\n"
        else
            e2e_status="⚠️ DEGRADED"
            details+="- ⚠️ AI response generation: Limited\n"
            warning "AI response seems limited"
        fi
    else
        e2e_status="❌ FAILED"
        details+="- ❌ Complete chat workflow: Failed\n"
        error "End-to-end chat workflow failed"
    fi
    
    # Test health monitoring
    if bash /opt/sutazaiapp/health_check.sh >/dev/null 2>&1; then
        details+="- ✅ Health monitoring: Functional\n"
    else
        e2e_status="⚠️ DEGRADED"
        details+="- ⚠️ Health monitoring: Issues detected\n"
        warning "Health monitoring script has issues"
    fi
    
    cat >> "$HEALTH_REPORT" << EOF
### End-to-End Testing: $e2e_status

**Details:**
$details

EOF
    
    success "End-to-end test: $e2e_status"
    [[ "$e2e_status" != "❌ FAILED" ]]
}

# Generate final health summary
generate_health_summary() {
    log "Generating health summary..."
    
    local overall_status="✅ HEALTHY"
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local degraded_tests=0
    
    # Count test results
    while IFS= read -r line; do
        if [[ "$line" =~ ^###.*: ]]; then
            ((total_tests++))
            if [[ "$line" =~ "✅ PASSED" ]]; then
                ((passed_tests++))
            elif [[ "$line" =~ "❌ FAILED" ]]; then
                ((failed_tests++))
                overall_status="❌ CRITICAL"
            elif [[ "$line" =~ "⚠️ DEGRADED" ]]; then
                ((degraded_tests++))
                if [[ "$overall_status" == "✅ HEALTHY" ]]; then
                    overall_status="⚠️ DEGRADED"
                fi
            fi
        fi
    done < "$HEALTH_REPORT"
    
    cat >> "$HEALTH_REPORT" << EOF

## Overall System Health: $overall_status

**Test Summary:**
- Total tests: $total_tests
- Passed: $passed_tests
- Degraded: $degraded_tests  
- Failed: $failed_tests

**Recommendations:**
EOF

    if [ "$failed_tests" -gt 0 ]; then
        cat >> "$HEALTH_REPORT" << EOF
- ❌ **CRITICAL**: $failed_tests critical failures detected. Immediate action required.
- Review failed components and restore from backup if necessary.
EOF
    elif [ "$degraded_tests" -gt 0 ]; then
        cat >> "$HEALTH_REPORT" << EOF
- ⚠️ **WARNING**: $degraded_tests components showing degraded performance.
- Monitor closely and consider optimization.
EOF
    else
        cat >> "$HEALTH_REPORT" << EOF
- ✅ **SUCCESS**: All systems operating normally.
- Reorganization completed successfully with no negative impact.
EOF
    fi
    
    cat >> "$HEALTH_REPORT" << EOF

**Next Steps:**
- Continue normal operations
- Monitor system performance over next 24 hours
- Archive can be cleaned up after 7 days if no issues arise

---
*Report generated by SutazAI health testing system*
EOF

    log "Health summary generated: $overall_status"
    
    # Return appropriate exit code
    case "$overall_status" in
        "✅ HEALTHY") return 0 ;;
        "⚠️ DEGRADED") return 1 ;;
        "❌ CRITICAL") return 2 ;;
    esac
}

# Main testing function
main() {
    log "Starting comprehensive system health testing..."
    
    # Initialize report
    init_health_report
    
    # Run all tests
    local test_results=0
    
    test_docker_infrastructure || ((test_results++))
    test_backend_api || ((test_results++))
    test_frontend || ((test_results++))
    test_ai_models || ((test_results++))
    test_databases || ((test_results++))
    test_system_resources || ((test_results++))
    test_essential_scripts || ((test_results++))
    test_end_to_end || ((test_results++))
    
    # Generate summary
    generate_health_summary
    local summary_result=$?
    
    log "System health testing completed"
    
    echo ""
    echo "🏥 System Health Testing Complete"
    echo "=================================="
    echo "📋 Full report: $HEALTH_REPORT"
    
    case $summary_result in
        0)
            echo "✅ Overall Status: HEALTHY"
            echo "🎉 Reorganization completed successfully with no negative impact"
            ;;
        1)
            echo "⚠️ Overall Status: DEGRADED"
            echo "⚠️ Some components showing degraded performance - monitor closely"
            ;;
        2)
            echo "❌ Overall Status: CRITICAL"
            echo "🚨 Critical failures detected - immediate action required"
            ;;
    esac
    
    echo ""
    echo "Test Summary:"
    echo "  - Docker Infrastructure: $(grep "### Docker Infrastructure:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - Backend API: $(grep "### Backend API:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - Frontend: $(grep "### Frontend:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - AI Models: $(grep "### AI Models:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - Databases: $(grep "### Databases:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"  
    echo "  - System Resources: $(grep "### System Resources:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - Essential Scripts: $(grep "### Essential Scripts:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    echo "  - End-to-End: $(grep "### End-to-End Testing:" "$HEALTH_REPORT" | cut -d':' -f2 | tr -d ' ')"
    
    return $summary_result
}

# Run main function
main "$@"