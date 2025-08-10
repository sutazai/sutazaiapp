#!/bin/bash

# SutazAI Dockerfile Consolidation Validation Suite
# Ultra QA Validator - Comprehensive Testing & Validation
# 
# Purpose: Validates Dockerfile consolidation to ensure:
#   1. All services build successfully with new templates
#   2. No functionality is lost
#   3. Container security is maintained  
#   4. Resource usage is optimized
#   5. Health checks work properly
#
# Author: ULTRA QA VALIDATOR
# Date: August 10, 2025
# Version: 1.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
VALIDATION_REPORT="$PROJECT_ROOT/dockerfile_validation_report_$(date +%Y%m%d_%H%M%S).json"
TEMP_DIR="/tmp/dockerfile_validation_$$"
BUILD_TIMEOUT=600  # 10 minutes per build
TEST_TIMEOUT=300   # 5 minutes per test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Initialize validation environment
initialize_validation() {
    log_info "Initializing Dockerfile validation environment"
    
    # Create required directories
    mkdir -p "$LOG_DIR" "$TEMP_DIR"
    
    # Create initial validation report structure
    cat > "$VALIDATION_REPORT" << EOF
{
  "validation_started": "$(date -Iseconds)",
  "validation_id": "dockerfile_validation_$(date +%s)",
  "project_root": "$PROJECT_ROOT",
  "summary": {
    "total_services": 0,
    "successful_builds": 0,
    "failed_builds": 0,
    "security_compliant": 0,
    "security_violations": 0,
    "functionality_preserved": 0,
    "functionality_lost": 0,
    "resource_optimized": 0,
    "resource_degraded": 0,
    "health_checks_working": 0,
    "health_checks_failing": 0
  },
  "services": {},
  "base_images": {},
  "consolidation_analysis": {},
  "recommendations": []
}
EOF
    
    log_success "Validation environment initialized"
}

# Discover all services with Dockerfiles
discover_services() {
    log_info "Discovering services with Dockerfiles"
    
    # Find all Dockerfiles and their service contexts
    SERVICES=()
    
    # Core application services
    if [[ -f "$PROJECT_ROOT/backend/Dockerfile" ]]; then
        SERVICES+=("backend:$PROJECT_ROOT/backend")
    fi
    
    if [[ -f "$PROJECT_ROOT/frontend/Dockerfile" ]]; then
        SERVICES+=("frontend:$PROJECT_ROOT/frontend")
    fi
    
    # Agent services
    while IFS= read -r -d '' dockerfile; do
        service_dir=$(dirname "$dockerfile")
        service_name=$(basename "$service_dir")
        
        # Skip test and backup directories
        if [[ "$dockerfile" =~ (test-|backup|\.backup) ]]; then
            continue
        fi
        
        # Add to services list
        SERVICES+=("$service_name:$service_dir")
    done < <(find "$PROJECT_ROOT/agents" "$PROJECT_ROOT/docker" -name "Dockerfile" -print0 2>/dev/null || true)
    
    log_success "Discovered ${#SERVICES[@]} services with Dockerfiles"
    
    # Update validation report
    jq --argjson count "${#SERVICES[@]}" \
       '.summary.total_services = $count' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
}

# Validate base image templates
validate_base_images() {
    log_info "Validating base image templates"
    
    local base_images=(
        "docker/base/Dockerfile.python-agent-master"
        "docker/base/Dockerfile.nodejs-agent-master"
    )
    
    local base_validation=()
    
    for base_image in "${base_images[@]}"; do
        local base_path="$PROJECT_ROOT/$base_image"
        local base_name=$(basename "$base_image")
        
        if [[ ! -f "$base_path" ]]; then
            log_warning "Base image template not found: $base_image"
            base_validation+=("{\"image\": \"$base_name\", \"exists\": false, \"buildable\": false}")
            continue
        fi
        
        log_info "Testing base image build: $base_name"
        
        # Test build the base image
        local build_result=false
        local build_output
        
        if timeout "$BUILD_TIMEOUT" docker build \
            -f "$base_path" \
            -t "sutazai-base-test:$base_name" \
            "$PROJECT_ROOT" \
            > "$LOG_DIR/base_build_${base_name}.log" 2>&1; then
            build_result=true
            log_success "Base image builds successfully: $base_name"
        else
            log_error "Base image build failed: $base_name"
        fi
        
        base_validation+=("{\"image\": \"$base_name\", \"exists\": true, \"buildable\": $build_result}")
    done
    
    # Update validation report
    local base_json=$(IFS=,; echo "[${base_validation[*]}]")
    jq --argjson bases "$base_json" \
       '.base_images = $bases' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
}

# Test individual service builds
test_service_builds() {
    log_info "Testing individual service builds"
    
    local build_success=0
    local build_failures=0
    
    for service_info in "${SERVICES[@]}"; do
        IFS=':' read -r service_name service_dir <<< "$service_info"
        
        log_info "Testing build for service: $service_name"
        
        local dockerfile="$service_dir/Dockerfile"
        local build_result=false
        local build_time_start=$(date +%s)
        local build_logs="$LOG_DIR/build_${service_name}_$(date +%s).log"
        
        # Test the build
        if timeout "$BUILD_TIMEOUT" docker build \
            -f "$dockerfile" \
            -t "sutazai-test-$service_name:latest" \
            "$service_dir" \
            > "$build_logs" 2>&1; then
            
            build_result=true
            ((build_success++))
            log_success "Build successful: $service_name"
        else
            ((build_failures++))
            log_error "Build failed: $service_name (see $build_logs)"
        fi
        
        local build_time_end=$(date +%s)
        local build_duration=$((build_time_end - build_time_start))
        
        # Analyze build output for optimization opportunities
        local image_size=""
        if [[ "$build_result" == true ]]; then
            image_size=$(docker images --format "table {{.Size}}" "sutazai-test-$service_name:latest" | tail -n1)
        fi
        
        # Update service validation record
        jq --arg service "$service_name" \
           --argjson success "$build_result" \
           --arg duration "$build_duration" \
           --arg size "$image_size" \
           --arg logs "$build_logs" \
           '.services[$service] = {
               "build_successful": $success,
               "build_duration_seconds": ($duration | tonumber),
               "image_size": $size,
               "build_logs": $logs,
               "dockerfile_path": "'$dockerfile'"
           }' \
           "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
           mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    done
    
    # Update summary
    jq --argjson success "$build_success" \
       --argjson failures "$build_failures" \
       '.summary.successful_builds = $success |
        .summary.failed_builds = $failures' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Service build testing completed: $build_success successful, $build_failures failed"
}

# Test container security compliance
test_security_compliance() {
    log_info "Testing container security compliance"
    
    local security_compliant=0
    local security_violations=0
    
    for service_info in "${SERVICES[@]}"; do
        IFS=':' read -r service_name service_dir <<< "$service_info"
        
        local dockerfile="$service_dir/Dockerfile"
        local violations=()
        
        # Check for root user (should use non-root)
        if grep -q "^USER root" "$dockerfile" 2>/dev/null || ! grep -q "^USER " "$dockerfile" 2>/dev/null; then
            violations+=("running_as_root")
        fi
        
        # Check for hardcoded secrets
        if grep -iE "(password|secret|key|token).*=" "$dockerfile" 2>/dev/null; then
            violations+=("hardcoded_secrets")
        fi
        
        # Check for insecure package installations
        if grep -q "pip install.*--trusted-host" "$dockerfile" 2>/dev/null; then
            violations+=("insecure_pip_install")
        fi
        
        # Check for missing security updates
        if grep -q "apt-get install" "$dockerfile" && ! grep -q "apt-get update.*upgrade" "$dockerfile" 2>/dev/null; then
            violations+=("missing_security_updates")
        fi
        
        # Check for proper health checks
        if ! grep -q "HEALTHCHECK" "$dockerfile" 2>/dev/null; then
            violations+=("missing_healthcheck")
        fi
        
        if [[ ${#violations[@]} -eq 0 ]]; then
            ((security_compliant++))
            log_success "Security compliant: $service_name"
        else
            ((security_violations++))
            log_warning "Security violations in $service_name: ${violations[*]}"
        fi
        
        # Update service security record
        local violations_json=$(printf '%s\n' "${violations[@]}" | jq -R . | jq -s .)
        jq --arg service "$service_name" \
           --argjson violations "$violations_json" \
           '.services[$service].security_violations = $violations |
            .services[$service].security_compliant = (($violations | length) == 0)' \
           "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
           mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    done
    
    # Update summary
    jq --argjson compliant "$security_compliant" \
       --argjson violations "$security_violations" \
       '.summary.security_compliant = $compliant |
        .summary.security_violations = $violations' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Security compliance testing completed: $security_compliant compliant, $security_violations with violations"
}

# Test functionality preservation
test_functionality_preservation() {
    log_info "Testing functionality preservation"
    
    local functionality_preserved=0
    local functionality_lost=0
    
    # Test core services that should be running
    local core_services=(
        "backend:10010:/health"
        "frontend:10011:/"
        "ai-agent-orchestrator:8589:/health"
        "ollama-integration:8090:/health"
        "hardware-resource-optimizer:11110:/health"
    )
    
    for service_info in "${core_services[@]}"; do
        IFS=':' read -r service_name port endpoint <<< "$service_info"
        
        log_info "Testing functionality: $service_name on port $port"
        
        local functional=false
        
        # Try to connect to the service
        if curl -f --max-time 10 --silent "http://localhost:$port$endpoint" >/dev/null 2>&1; then
            functional=true
            ((functionality_preserved++))
            log_success "Functionality preserved: $service_name"
        else
            ((functionality_lost++))
            log_warning "Functionality lost or unavailable: $service_name"
        fi
        
        # Update service functionality record
        jq --arg service "$service_name" \
           --argjson functional "$functional" \
           --arg port "$port" \
           --arg endpoint "$endpoint" \
           '.services[$service].functionality = {
               "preserved": $functional,
               "port": ($port | tonumber),
               "endpoint": $endpoint
           }' \
           "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
           mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    done
    
    # Update summary
    jq --argjson preserved "$functionality_preserved" \
       --argjson lost "$functionality_lost" \
       '.summary.functionality_preserved = $preserved |
        .summary.functionality_lost = $lost' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Functionality preservation testing completed: $functionality_preserved preserved, $functionality_lost lost"
}

# Test resource optimization
test_resource_optimization() {
    log_info "Testing resource optimization"
    
    local optimized=0
    local degraded=0
    
    # Get current container resource usage
    local containers=$(docker ps --format "{{.Names}}" | grep -E "^sutazai-" || true)
    
    for container in $containers; do
        log_info "Analyzing resource usage: $container"
        
        # Get container stats
        local stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" "$container" 2>/dev/null || echo "0.00%,0B / 0B")
        local cpu_usage=$(echo "$stats" | cut -d',' -f1 | sed 's/%//')
        local memory_usage=$(echo "$stats" | cut -d',' -f2)
        
        # Check if resource usage is reasonable
        local is_optimized=true
        
        # CPU usage should be reasonable (less than 80% for non-ML services)
        if [[ $(echo "$cpu_usage > 80" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
            is_optimized=false
            log_warning "High CPU usage in $container: $cpu_usage%"
        fi
        
        if [[ "$is_optimized" == true ]]; then
            ((optimized++))
            log_success "Resource usage optimized: $container"
        else
            ((degraded++))
            log_warning "Resource usage concerns: $container"
        fi
        
        # Update container resource record
        jq --arg container "$container" \
           --arg cpu "$cpu_usage" \
           --arg memory "$memory_usage" \
           --argjson optimized "$is_optimized" \
           '.services[$container].resources = {
               "cpu_usage_percent": $cpu,
               "memory_usage": $memory,
               "optimized": $optimized
           }' \
           "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
           mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    done
    
    # Update summary
    jq --argjson opt "$optimized" \
       --argjson deg "$degraded" \
       '.summary.resource_optimized = $opt |
        .summary.resource_degraded = $deg' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Resource optimization testing completed: $optimized optimized, $degraded with concerns"
}

# Test health check functionality
test_health_checks() {
    log_info "Testing health check functionality"
    
    local health_working=0
    local health_failing=0
    
    # Get running containers
    local containers=$(docker ps --format "{{.Names}}" | grep -E "^sutazai-" || true)
    
    for container in $containers; do
        log_info "Testing health check: $container"
        
        # Get container health status
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
        local is_healthy=false
        
        case "$health_status" in
            "healthy")
                is_healthy=true
                ((health_working++))
                log_success "Health check working: $container"
                ;;
            "unhealthy")
                ((health_failing++))
                log_error "Health check failing: $container"
                ;;
            "starting")
                log_info "Health check starting: $container"
                # Wait a bit and check again
                sleep 10
                health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
                if [[ "$health_status" == "healthy" ]]; then
                    is_healthy=true
                    ((health_working++))
                    log_success "Health check now working: $container"
                else
                    ((health_failing++))
                    log_warning "Health check still not ready: $container"
                fi
                ;;
            "none")
                log_warning "No health check defined: $container"
                ;;
        esac
        
        # Update container health record
        jq --arg container "$container" \
           --arg status "$health_status" \
           --argjson healthy "$is_healthy" \
           '.services[$container].health = {
               "status": $status,
               "working": $healthy
           }' \
           "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
           mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    done
    
    # Update summary
    jq --argjson working "$health_working" \
       --argjson failing "$health_failing" \
       '.summary.health_checks_working = $working |
        .summary.health_checks_failing = $failing' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Health check testing completed: $health_working working, $health_failing failing"
}

# Generate consolidation analysis
generate_consolidation_analysis() {
    log_info "Generating consolidation analysis"
    
    # Analyze Dockerfile patterns for consolidation opportunities
    local total_dockerfiles=0
    local consolidated_dockerfiles=0
    local base_image_usage={}
    
    # Count total Dockerfiles
    total_dockerfiles=$(find "$PROJECT_ROOT" -name "Dockerfile" -o -name "*.Dockerfile" | wc -l)
    
    # Check how many use base images
    local base_usage=0
    while IFS= read -r -d '' dockerfile; do
        if grep -q "FROM.*python.*agent.*master\|FROM.*nodejs.*agent.*master" "$dockerfile" 2>/dev/null; then
            ((base_usage++))
        fi
    done < <(find "$PROJECT_ROOT" -name "Dockerfile" -print0 2>/dev/null || true)
    
    # Calculate consolidation metrics
    local consolidation_ratio=0
    if [[ $total_dockerfiles -gt 0 ]]; then
        consolidation_ratio=$(echo "scale=2; $base_usage * 100 / $total_dockerfiles" | bc -l 2>/dev/null || echo "0")
    fi
    
    # Generate recommendations
    local recommendations=()
    
    if [[ $(echo "$consolidation_ratio < 50" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
        recommendations+=("\"More services should be migrated to use base image templates\"")
    fi
    
    if [[ $(jq '.summary.security_violations' "$VALIDATION_REPORT") -gt 0 ]]; then
        recommendations+=("\"Address security violations in non-compliant containers\"")
    fi
    
    if [[ $(jq '.summary.failed_builds' "$VALIDATION_REPORT") -gt 0 ]]; then
        recommendations+=("\"Fix build failures before deploying consolidation changes\"")
    fi
    
    if [[ $(jq '.summary.health_checks_failing' "$VALIDATION_REPORT") -gt 0 ]]; then
        recommendations+=("\"Repair failing health checks to ensure service reliability\"")
    fi
    
    # Update consolidation analysis
    local rec_json=$(IFS=,; echo "[${recommendations[*]}]")
    jq --argjson total "$total_dockerfiles" \
       --argjson consolidated "$base_usage" \
       --arg ratio "$consolidation_ratio" \
       --argjson recs "$rec_json" \
       '.consolidation_analysis = {
           "total_dockerfiles": $total,
           "using_base_images": $consolidated,
           "consolidation_percentage": ($ratio | tonumber),
           "consolidation_status": (if ($ratio | tonumber) >= 80 then "excellent" elif ($ratio | tonumber) >= 50 then "good" elif ($ratio | tonumber) >= 25 then "fair" else "poor" end)
       } |
        .recommendations = $recs' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Consolidation analysis completed: $consolidation_ratio% consolidation achieved"
}

# Generate final validation report
generate_final_report() {
    log_info "Generating final validation report"
    
    # Add completion timestamp
    jq --arg timestamp "$(date -Iseconds)" \
       '.validation_completed = $timestamp |
        .validation_duration_minutes = ((now - (.validation_started | fromdateiso8601)) / 60 | floor)' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    # Calculate overall success score
    local total_services=$(jq '.summary.total_services' "$VALIDATION_REPORT")
    local successful_builds=$(jq '.summary.successful_builds' "$VALIDATION_REPORT")
    local security_compliant=$(jq '.summary.security_compliant' "$VALIDATION_REPORT")
    local functionality_preserved=$(jq '.summary.functionality_preserved' "$VALIDATION_REPORT")
    local resource_optimized=$(jq '.summary.resource_optimized' "$VALIDATION_REPORT")
    local health_working=$(jq '.summary.health_checks_working' "$VALIDATION_REPORT")
    
    # Calculate weighted success score
    local build_score=0
    local security_score=0
    local function_score=0
    local resource_score=0
    local health_score=0
    
    if [[ $total_services -gt 0 ]]; then
        build_score=$(echo "scale=2; $successful_builds * 100 / $total_services" | bc -l 2>/dev/null || echo "0")
        security_score=$(echo "scale=2; $security_compliant * 100 / $total_services" | bc -l 2>/dev/null || echo "0")
        function_score=$(echo "scale=2; $functionality_preserved * 100 / 5" | bc -l 2>/dev/null || echo "0")  # Based on 5 core services
        resource_score=$(echo "scale=2; $resource_optimized * 100 / ($resource_optimized + $(jq '.summary.resource_degraded' "$VALIDATION_REPORT"))" | bc -l 2>/dev/null || echo "100")
        health_score=$(echo "scale=2; $health_working * 100 / ($health_working + $(jq '.summary.health_checks_failing' "$VALIDATION_REPORT"))" | bc -l 2>/dev/null || echo "100")
    fi
    
    # Overall score (weighted average)
    local overall_score=$(echo "scale=1; ($build_score * 0.3 + $security_score * 0.25 + $function_score * 0.25 + $resource_score * 0.1 + $health_score * 0.1)" | bc -l 2>/dev/null || echo "0")
    
    # Add scores to report
    jq --arg build "$build_score" \
       --arg security "$security_score" \
       --arg function "$function_score" \
       --arg resource "$resource_score" \
       --arg health "$health_score" \
       --arg overall "$overall_score" \
       '.validation_scores = {
           "build_success_percentage": ($build | tonumber),
           "security_compliance_percentage": ($security | tonumber),
           "functionality_preservation_percentage": ($function | tonumber),
           "resource_optimization_percentage": ($resource | tonumber),
           "health_check_success_percentage": ($health | tonumber),
           "overall_score": ($overall | tonumber),
           "grade": (if ($overall | tonumber) >= 90 then "A" elif ($overall | tonumber) >= 80 then "B" elif ($overall | tonumber) >= 70 then "C" elif ($overall | tonumber) >= 60 then "D" else "F" end)
       }' \
       "$VALIDATION_REPORT" > "$TEMP_DIR/report.tmp" && \
       mv "$TEMP_DIR/report.tmp" "$VALIDATION_REPORT"
    
    log_success "Final validation report generated: $VALIDATION_REPORT"
}

# Print validation summary
print_summary() {
    echo
    echo "========================================"
    echo "  DOCKERFILE CONSOLIDATION VALIDATION  "
    echo "========================================"
    echo
    
    # Extract key metrics from report
    local total=$(jq '.summary.total_services' "$VALIDATION_REPORT")
    local builds=$(jq '.summary.successful_builds' "$VALIDATION_REPORT")
    local security=$(jq '.summary.security_compliant' "$VALIDATION_REPORT")
    local overall=$(jq '.validation_scores.overall_score' "$VALIDATION_REPORT")
    local grade=$(jq -r '.validation_scores.grade' "$VALIDATION_REPORT")
    local consolidation=$(jq '.consolidation_analysis.consolidation_percentage' "$VALIDATION_REPORT")
    
    echo "Services Analyzed:     $total"
    echo "Successful Builds:     $builds"
    echo "Security Compliant:    $security"
    echo "Consolidation Rate:    $consolidation%"
    echo "Overall Score:         $overall/100"
    echo "Grade:                 $grade"
    echo
    echo "Detailed Report:       $VALIDATION_REPORT"
    echo
    
    # Print recommendations
    local rec_count=$(jq '.recommendations | length' "$VALIDATION_REPORT")
    if [[ $rec_count -gt 0 ]]; then
        echo "RECOMMENDATIONS:"
        jq -r '.recommendations[]' "$VALIDATION_REPORT" | sed 's/^/  • /'
        echo
    fi
    
    # Final status
    if [[ $(echo "$overall >= 80" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
        log_success "Dockerfile consolidation validation PASSED (Score: $overall/100)"
        echo "✅ System is ready for production deployment"
    else
        log_warning "Dockerfile consolidation validation NEEDS IMPROVEMENT (Score: $overall/100)"
        echo "⚠️  Address recommendations before production deployment"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up validation environment"
    
    # Remove test containers and images
    docker images -q --filter "reference=sutazai-test-*" | xargs -r docker rmi -f 2>/dev/null || true
    docker images -q --filter "reference=sutazai-base-test:*" | xargs -r docker rmi -f 2>/dev/null || true
    
    # Remove temporary directory
    rm -rf "$TEMP_DIR"
    
    log_success "Cleanup completed"
}

# Main execution function
main() {
    echo "Starting Dockerfile Consolidation Validation Suite"
    echo "=================================================="
    
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Execute validation steps
    initialize_validation
    discover_services
    validate_base_images
    test_service_builds
    test_security_compliance
    test_functionality_preservation
    test_resource_optimization
    test_health_checks
    generate_consolidation_analysis
    generate_final_report
    print_summary
    
    echo
    echo "Validation completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi