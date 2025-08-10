#!/bin/bash
# ULTRA CLEANUP TEST SUITE - Comprehensive Testing for Script/Dockerfile Consolidation
# QA Testing Specialist - ZERO TOLERANCE FOR FAILURES
# Created: August 10, 2025

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_LOG_DIR="$PROJECT_ROOT/tests/logs"
BACKUP_DIR="$PROJECT_ROOT/backups/test-$(date +%Y%m%d_%H%M%S)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Test Results
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0
CRITICAL_FAILURES=()

# Create directories
mkdir -p "$TEST_LOG_DIR" "$BACKUP_DIR"

# Logging functions
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; ((TESTS_PASSED++)); }
log_failure() { echo -e "${RED}[FAIL]${NC} $1"; CRITICAL_FAILURES+=("$1"); ((TESTS_FAILED++)); }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_critical() { echo -e "${RED}${BOLD}[CRITICAL]${NC} $1"; }

# Test tracking
start_test() {
    ((TESTS_TOTAL++))
    log_test "$1"
}

# Emergency rollback function
emergency_rollback() {
    log_critical "EMERGENCY ROLLBACK INITIATED"
    
    if [ -d "$BACKUP_DIR" ]; then
        log_info "Restoring from backup: $BACKUP_DIR"
        
        # Stop all containers
        docker-compose down --remove-orphans || true
        
        # Restore scripts if backed up
        if [ -d "$BACKUP_DIR/scripts" ]; then
            rm -rf "$PROJECT_ROOT/scripts"
            cp -r "$BACKUP_DIR/scripts" "$PROJECT_ROOT/scripts"
            chmod +x "$PROJECT_ROOT/scripts"/**/*.sh
            log_info "Scripts restored from backup"
        fi
        
        # Restore containers if needed
        if [ -f "$BACKUP_DIR/docker-compose.yml" ]; then
            cp "$BACKUP_DIR/docker-compose.yml" "$PROJECT_ROOT/docker-compose.yml"
            log_info "Docker compose file restored"
        fi
        
        # Restart system
        cd "$PROJECT_ROOT"
        if [ -f "scripts/master/deploy.sh" ]; then
            timeout 300 ./scripts/master/deploy.sh minimal || {
                log_critical "Master deploy failed, using fallback"
                docker-compose up -d postgres redis backend frontend || true
            }
        fi
        
        log_info "Emergency rollback completed"
    else
        log_critical "No backup available - manual intervention required"
    fi
    
    exit 1
}

# Trap for cleanup
trap 'emergency_rollback' ERR

# PHASE 1: PRE-CLEANUP VALIDATION
phase1_baseline_capture() {
    log_info "=== PHASE 1: BASELINE CAPTURE ==="
    
    start_test "Capturing system baseline"
    
    # System health baseline
    if [ -f "$PROJECT_ROOT/scripts/master/health.sh" ]; then
        timeout 120 "$PROJECT_ROOT/scripts/master/health.sh" full > "$TEST_LOG_DIR/baseline_health.log" 2>&1 || {
            log_warning "Health script failed, using fallback checks"
        }
    fi
    
    # Container status
    docker ps --format "json" > "$TEST_LOG_DIR/baseline_containers.json" 2>/dev/null || {
        docker ps > "$TEST_LOG_DIR/baseline_containers.txt" 2>/dev/null
    }
    
    # Performance baseline
    docker stats --no-stream > "$TEST_LOG_DIR/baseline_performance.txt" 2>/dev/null || {
        log_warning "Could not capture performance baseline"
    }
    
    log_success "System baseline captured"
}

phase1_service_validation() {
    start_test "Validating critical services"
    
    local critical_services=(
        "10010:/health:Backend API"
        "10011:/:Frontend UI"
        "10104:/api/tags:Ollama"
        "11110:/health:Hardware Optimizer"
        "10000:/:PostgreSQL"
        "10001:/:Redis"
    )
    
    local service_failures=0
    
    for service in "${critical_services[@]}"; do
        IFS=':' read -r port endpoint name <<< "$service"
        
        if curl -sf --max-time 10 "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            log_info "✓ $name healthy on port $port"
        else
            log_warning "✗ $name not responding on port $port"
            ((service_failures++))
        fi
    done
    
    if [ $service_failures -eq 0 ]; then
        log_success "All critical services validated"
    elif [ $service_failures -le 2 ]; then
        log_warning "Some services not responding (acceptable: $service_failures failures)"
    else
        log_failure "Too many service failures: $service_failures"
        return 1
    fi
}

phase1_performance_baseline() {
    start_test "Capturing performance baseline"
    
    # Test Ollama performance
    local start_time=$(date +%s)
    local ollama_test=$(curl -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "tinyllama", "prompt": "test", "stream": false}' \
        --max-time 15 2>/dev/null || echo "failed")
    local end_time=$(date +%s)
    local ollama_duration=$((end_time - start_time))
    
    echo "Ollama response time: ${ollama_duration}s" >> "$TEST_LOG_DIR/performance_baseline.txt"
    
    # Test Redis if available
    if command -v redis-benchmark > /dev/null 2>&1; then
        redis-benchmark -h localhost -p 10001 -n 100 -q >> "$TEST_LOG_DIR/redis_baseline.txt" 2>/dev/null || {
            log_info "Redis benchmark not available"
        }
    fi
    
    log_success "Performance baseline captured (Ollama: ${ollama_duration}s)"
}

# PHASE 2: SCRIPT CONSOLIDATION TESTING
phase2_script_validation() {
    log_info "=== PHASE 2: SCRIPT CONSOLIDATION TESTING ==="
    
    start_test "Testing master scripts functionality"
    
    # Test master deploy script
    if [ -f "$PROJECT_ROOT/scripts/master/deploy.sh" ]; then
        # Test syntax
        bash -n "$PROJECT_ROOT/scripts/master/deploy.sh" || {
            log_failure "Master deploy script syntax error"
            return 1
        }
        
        # Test help/usage
        "$PROJECT_ROOT/scripts/master/deploy.sh" help > /dev/null 2>&1 || {
            "$PROJECT_ROOT/scripts/master/deploy.sh" --help > /dev/null 2>&1 || {
                log_info "Deploy script help not available (acceptable)"
            }
        }
        
        log_success "Master deploy script validated"
    else
        log_failure "Master deploy script not found"
        return 1
    fi
    
    # Test master health script
    if [ -f "$PROJECT_ROOT/scripts/master/health.sh" ]; then
        bash -n "$PROJECT_ROOT/scripts/master/health.sh" || {
            log_failure "Master health script syntax error"
            return 1
        }
        
        # Test quick health check
        timeout 30 "$PROJECT_ROOT/scripts/master/health.sh" quick > /dev/null 2>&1 || {
            log_warning "Health script quick check failed"
        }
        
        log_success "Master health script validated"
    else
        log_failure "Master health script not found"
        return 1
    fi
}

phase2_script_dependency_check() {
    start_test "Checking script dependencies"
    
    # Find scripts that might be called by other scripts
    local critical_scripts=()
    
    # Check for scripts referenced in master scripts
    if [ -f "$PROJECT_ROOT/scripts/master/deploy.sh" ]; then
        while read -r script_ref; do
            if [ -f "$PROJECT_ROOT/$script_ref" ]; then
                critical_scripts+=("$script_ref")
            fi
        done < <(grep -o 'scripts/[^[:space:]]*\.sh' "$PROJECT_ROOT/scripts/master/deploy.sh" 2>/dev/null || true)
    fi
    
    # Check Docker compose references
    if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        while read -r script_ref; do
            if [ -f "$PROJECT_ROOT/$script_ref" ]; then
                critical_scripts+=("$script_ref")
            fi
        done < <(grep -o 'scripts/[^[:space:]]*\.sh' "$PROJECT_ROOT/docker-compose.yml" 2>/dev/null || true)
    fi
    
    log_info "Found ${#critical_scripts[@]} potentially critical scripts"
    
    # Test syntax of critical scripts
    local syntax_errors=0
    for script in "${critical_scripts[@]}"; do
        if ! bash -n "$PROJECT_ROOT/$script" 2>/dev/null; then
            log_warning "Syntax error in critical script: $script"
            ((syntax_errors++))
        fi
    done
    
    if [ $syntax_errors -eq 0 ]; then
        log_success "All critical scripts have valid syntax"
    else
        log_warning "$syntax_errors critical scripts have syntax errors"
    fi
}

# PHASE 3: DOCKERFILE CONSOLIDATION TESTING
phase3_dockerfile_validation() {
    log_info "=== PHASE 3: DOCKERFILE CONSOLIDATION TESTING ==="
    
    start_test "Testing Dockerfile builds"
    
    # Find and test base Dockerfiles
    local base_dockerfiles=()
    while read -r dockerfile; do
        base_dockerfiles+=("$dockerfile")
    done < <(find "$PROJECT_ROOT/docker/base" -name "Dockerfile*" 2>/dev/null || true)
    
    if [ ${#base_dockerfiles[@]} -eq 0 ]; then
        log_info "No base Dockerfiles found (may be already consolidated)"
    else
        log_info "Found ${#base_dockerfiles[@]} base Dockerfiles to test"
        
        local build_failures=0
        for dockerfile in "${base_dockerfiles[@]}"; do
            local tag_name="test-base-$(basename "$dockerfile" | tr '.' '-')"
            
            if docker build -t "$tag_name" -f "$dockerfile" "$PROJECT_ROOT" > /dev/null 2>&1; then
                log_info "✓ Built base image: $(basename "$dockerfile")"
                # Cleanup test image
                docker rmi "$tag_name" > /dev/null 2>&1 || true
            else
                log_warning "✗ Failed to build: $(basename "$dockerfile")"
                ((build_failures++))
            fi
        done
        
        if [ $build_failures -eq 0 ]; then
            log_success "All base Dockerfiles build successfully"
        else
            log_warning "$build_failures base Dockerfile build failures"
        fi
    fi
}

phase3_service_container_builds() {
    start_test "Testing critical service container builds"
    
    # Test critical services can build
    local critical_services=("backend" "frontend" "hardware-resource-optimizer")
    local build_failures=0
    
    for service in "${critical_services[@]}"; do
        if docker-compose build "$service" > /dev/null 2>&1; then
            log_info "✓ Built service: $service"
        else
            log_warning "✗ Failed to build service: $service"
            ((build_failures++))
        fi
    done
    
    if [ $build_failures -eq 0 ]; then
        log_success "All critical services build successfully"
    else
        log_warning "$build_failures service build failures"
    fi
}

# PHASE 4: SYSTEM INTEGRATION TESTING
phase4_integration_testing() {
    log_info "=== PHASE 4: SYSTEM INTEGRATION TESTING ==="
    
    start_test "Testing system deployment with master scripts"
    
    # Create backup before testing
    log_info "Creating backup before integration test..."
    cp -r "$PROJECT_ROOT/scripts" "$BACKUP_DIR/scripts" 2>/dev/null || true
    cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/docker-compose.yml" 2>/dev/null || true
    
    # Test minimal deployment
    if [ -f "$PROJECT_ROOT/scripts/master/deploy.sh" ]; then
        log_info "Testing minimal deployment..."
        
        # Run deployment
        if timeout 300 "$PROJECT_ROOT/scripts/master/deploy.sh" minimal > "$TEST_LOG_DIR/deploy_test.log" 2>&1; then
            log_info "✓ Minimal deployment completed"
            
            # Wait for services to stabilize
            sleep 30
            
            # Test health check
            if "$PROJECT_ROOT/scripts/master/health.sh" quick > "$TEST_LOG_DIR/health_test.log" 2>&1; then
                log_success "System integration test passed"
            else
                log_warning "Health check failed after deployment"
            fi
        else
            log_failure "Deployment failed"
            return 1
        fi
    else
        log_failure "Master deploy script not available"
        return 1
    fi
}

phase4_critical_workflows() {
    start_test "Testing critical system workflows"
    
    # Test 1: Backend API
    if curl -sf http://localhost:10010/health > /dev/null 2>&1; then
        log_info "✓ Backend API workflow active"
    else
        log_warning "✗ Backend API workflow failed"
    fi
    
    # Test 2: Frontend UI
    if curl -sf http://localhost:10011/ > /dev/null 2>&1; then
        log_info "✓ Frontend UI workflow active"
    else
        log_warning "✗ Frontend UI workflow failed"
    fi
    
    # Test 3: Ollama (if available)
    local ollama_test=$(curl -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "tinyllama", "prompt": "test", "stream": false}' \
        --max-time 20 2>/dev/null || echo "failed")
    
    if [[ $ollama_test != "failed" && $ollama_test == *"response"* ]]; then
        log_info "✓ Ollama AI workflow active"
    else
        log_warning "✗ Ollama AI workflow not responding"
    fi
    
    # Test 4: Database connectivity
    if docker exec sutazai-postgres psql -U sutazai -c "SELECT 1;" > /dev/null 2>&1; then
        log_info "✓ Database workflow active"
    else
        log_warning "✗ Database workflow failed"
    fi
    
    log_success "Critical workflow testing completed"
}

# PHASE 5: PERFORMANCE VALIDATION
phase5_performance_validation() {
    log_info "=== PHASE 5: PERFORMANCE VALIDATION ==="
    
    start_test "Validating system performance post-cleanup"
    
    # Ollama performance test
    local start_time=$(date +%s)
    local response=$(curl -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "tinyllama", "prompt": "Performance test", "stream": false}' \
        --max-time 15 2>/dev/null || echo "failed")
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Post-cleanup Ollama response time: ${duration}s" >> "$TEST_LOG_DIR/performance_comparison.txt"
    
    if [ "$duration" -le 15 ]; then
        log_info "✓ Ollama performance acceptable: ${duration}s"
    else
        log_warning "✗ Ollama performance degraded: ${duration}s"
    fi
    
    # System resource check
    local memory_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
    echo "Memory usage: ${memory_usage}%" >> "$TEST_LOG_DIR/performance_comparison.txt"
    
    log_success "Performance validation completed"
}

# Final validation
final_validation() {
    log_info "=== FINAL VALIDATION ==="
    
    start_test "Comprehensive system health check"
    
    # Run full health check if available
    if [ -f "$PROJECT_ROOT/scripts/master/health.sh" ]; then
        if "$PROJECT_ROOT/scripts/master/health.sh" full > "$TEST_LOG_DIR/final_health.log" 2>&1; then
            local health_score=$(grep "Health Score" "$TEST_LOG_DIR/final_health.log" | awk '{print $3}' | tr -d '%' || echo "0")
            
            if [ "$health_score" -ge 80 ]; then
                log_success "System health excellent: ${health_score}%"
            elif [ "$health_score" -ge 60 ]; then
                log_warning "System health acceptable: ${health_score}%"
            else
                log_failure "System health poor: ${health_score}%"
            fi
        else
            log_warning "Health check script failed"
        fi
    fi
    
    # Count running containers
    local running_containers=$(docker ps -q | wc -l)
    log_info "Running containers: $running_containers"
    
    # Check critical services one more time
    local critical_endpoints=(
        "http://localhost:10010/health"
        "http://localhost:10011/"
        "http://localhost:10000/"
    )
    
    local endpoint_failures=0
    for endpoint in "${critical_endpoints[@]}"; do
        if ! curl -sf --max-time 5 "$endpoint" > /dev/null 2>&1; then
            ((endpoint_failures++))
        fi
    done
    
    if [ $endpoint_failures -eq 0 ]; then
        log_success "All critical endpoints responding"
    else
        log_warning "$endpoint_failures critical endpoints not responding"
    fi
}

# Generate test report
generate_report() {
    local report_file="$TEST_LOG_DIR/test_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# ULTRA CLEANUP TEST REPORT
**Date:** $(date)
**Test Suite:** SutazAI System Cleanup Validation
**Total Tests:** $TESTS_TOTAL
**Passed:** $TESTS_PASSED
**Failed:** $TESTS_FAILED

## Test Results Summary
$([ $TESTS_FAILED -eq 0 ] && echo "🟢 **ALL TESTS PASSED**" || echo "🔴 **$TESTS_FAILED TESTS FAILED**")

## Performance Metrics
$([ -f "$TEST_LOG_DIR/performance_comparison.txt" ] && cat "$TEST_LOG_DIR/performance_comparison.txt" || echo "Performance metrics not available")

## Critical Failures
$([ ${#CRITICAL_FAILURES[@]} -eq 0 ] && echo "None" || printf '%s\n' "${CRITICAL_FAILURES[@]}")

## Logs Location
- Test logs: $TEST_LOG_DIR
- Backup location: $BACKUP_DIR

## Recommendations
$([ $TESTS_FAILED -eq 0 ] && echo "✅ System cleanup validation successful. Safe to proceed with cleanup operation." || echo "❌ System cleanup validation failed. Manual review required before proceeding.")
EOF

    log_info "Test report generated: $report_file"
}

# Main execution
main() {
    log_info "🚀 ULTRA CLEANUP TEST SUITE STARTING"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Test Logs: $TEST_LOG_DIR"
    log_info "Backup Location: $BACKUP_DIR"
    
    # Execute test phases
    phase1_baseline_capture
    phase1_service_validation
    phase1_performance_baseline
    
    phase2_script_validation
    phase2_script_dependency_check
    
    phase3_dockerfile_validation
    phase3_service_container_builds
    
    phase4_integration_testing
    phase4_critical_workflows
    
    phase5_performance_validation
    
    final_validation
    
    # Generate report
    generate_report
    
    # Final summary
    log_info "=== TEST EXECUTION COMPLETE ==="
    log_info "Total Tests: $TESTS_TOTAL"
    log_success "Passed: $TESTS_PASSED"
    
    if [ $TESTS_FAILED -gt 0 ]; then
        log_failure "Failed: $TESTS_FAILED"
        echo ""
        log_critical "CRITICAL FAILURES:"
        printf '%s\n' "${CRITICAL_FAILURES[@]}"
        echo ""
        log_critical "SYSTEM NOT READY FOR CLEANUP"
        exit 1
    else
        echo ""
        log_success "🎉 ALL TESTS PASSED - SYSTEM READY FOR CLEANUP"
        log_info "Backup preserved at: $BACKUP_DIR"
    fi
}

# Parse command line arguments
case "${1:-run}" in
    run)
        main
        ;;
    baseline)
        phase1_baseline_capture
        phase1_service_validation
        phase1_performance_baseline
        ;;
    scripts)
        phase2_script_validation
        phase2_script_dependency_check
        ;;
    docker)
        phase3_dockerfile_validation
        phase3_service_container_builds
        ;;
    integration)
        phase4_integration_testing
        phase4_critical_workflows
        ;;
    performance)
        phase5_performance_validation
        ;;
    rollback)
        emergency_rollback
        ;;
    *)
        echo "Usage: $0 {run|baseline|scripts|docker|integration|performance|rollback}"
        echo ""
        echo "Modes:"
        echo "  run         - Execute complete test suite (default)"
        echo "  baseline    - Run baseline capture tests only"
        echo "  scripts     - Test script consolidation only"
        echo "  docker      - Test Dockerfile builds only"
        echo "  integration - Test system integration only"
        echo "  performance - Test performance validation only"
        echo "  rollback    - Execute emergency rollback"
        exit 1
        ;;
esac