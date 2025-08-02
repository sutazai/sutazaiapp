#!/bin/bash
# SutazAI Chaos Engineering Framework - Deployment Validation
# Comprehensive validation of chaos engineering framework deployment

set -euo pipefail

# Configuration
CHAOS_DIR="/opt/sutazaiapp/chaos"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/chaos_validation_${TIMESTAMP}.log"

# Validation results
VALIDATION_RESULTS=()
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Ensure directories exist
mkdir -p "$LOG_DIR"

# Logging functions
log_info() {
    echo "[$(date +'%H:%M:%S')] INFO: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date +'%H:%M:%S')] ✅ SUCCESS: $1" | tee -a "$LOG_FILE"
    ((PASSED_TESTS++))
}

log_error() {
    echo "[$(date +'%H:%M:%S')] ❌ ERROR: $1" | tee -a "$LOG_FILE"
    ((FAILED_TESTS++))
}

log_warning() {
    echo "[$(date +'%H:%M:%S')] ⚠️  WARNING: $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TOTAL_TESTS++))
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        log_success "$test_name"
        VALIDATION_RESULTS+=("✅ $test_name")
        return 0
    else
        log_error "$test_name"
        VALIDATION_RESULTS+=("❌ $test_name")
        return 1
    fi
}

# File existence tests
test_file_exists() {
    local file_path="$1"
    local description="$2"
    
    if [[ -f "$file_path" ]]; then
        log_success "File exists: $description ($file_path)"
        return 0
    else
        log_error "Missing file: $description ($file_path)"
        return 1
    fi
}

# Directory structure validation
validate_directory_structure() {
    log_header "Validating Directory Structure"
    
    local required_dirs=(
        "$CHAOS_DIR"
        "$CHAOS_DIR/config"
        "$CHAOS_DIR/scripts"
        "$CHAOS_DIR/experiments"
        "$CHAOS_DIR/monitoring"
        "$CHAOS_DIR/reports"
        "$CHAOS_DIR/tools"
    )
    
    for dir in "${required_dirs[@]}"; do
        run_test "Directory exists: $(basename $dir)" "[[ -d '$dir' ]]"
    done
}

# Configuration file validation
validate_configuration_files() {
    log_header "Validating Configuration Files"
    
    # Main configuration
    run_test "Main config file exists" "test_file_exists '$CHAOS_DIR/config/chaos-config.yaml' 'Main configuration'"
    
    # Validate YAML syntax
    run_test "Main config YAML syntax valid" "python3 -c 'import yaml; yaml.safe_load(open(\"$CHAOS_DIR/config/chaos-config.yaml\"))'"
    
    # Experiment definitions
    local experiments=(
        "basic-container-chaos.yaml"
        "network-chaos.yaml"
        "resource-stress.yaml"
    )
    
    for experiment in "${experiments[@]}"; do
        run_test "Experiment exists: $experiment" "test_file_exists '$CHAOS_DIR/experiments/$experiment' 'Experiment definition'"
        run_test "Experiment YAML valid: $experiment" "python3 -c 'import yaml; yaml.safe_load(open(\"$CHAOS_DIR/experiments/$experiment\"))'"
    done
}

# Script validation
validate_scripts() {
    log_header "Validating Scripts"
    
    local required_scripts=(
        "init-chaos.sh"
        "run-experiment.sh"
        "chaos-engine.py"
        "chaos-monkey.py"
        "resilience-tester.py"
        "docker-integration.py"
        "monitoring-integration.py"
    )
    
    for script in "${required_scripts[@]}"; do
        local script_path="$CHAOS_DIR/scripts/$script"
        run_test "Script exists: $script" "test_file_exists '$script_path' 'Script'"
        run_test "Script executable: $script" "[[ -x '$script_path' ]]"
        
        # Syntax validation for Python scripts
        if [[ "$script" == *.py ]]; then
            run_test "Python syntax valid: $script" "python3 -m py_compile '$script_path'"
        fi
        
        # Basic shell script validation
        if [[ "$script" == *.sh ]]; then
            run_test "Shell script syntax valid: $script" "bash -n '$script_path'"
        fi
    done
}

# Dependency validation
validate_dependencies() {
    log_header "Validating Dependencies"
    
    # System dependencies
    local system_deps=("docker" "python3" "curl" "jq")
    for dep in "${system_deps[@]}"; do
        run_test "System dependency: $dep" "command -v $dep &> /dev/null"
    done
    
    # Python dependencies
    local python_deps=("yaml" "docker" "requests" "schedule" "networkx")
    for dep in "${python_deps[@]}"; do
        run_test "Python module: $dep" "python3 -c 'import $dep' 2>/dev/null"
    done
    
    # Docker daemon
    run_test "Docker daemon running" "docker info &> /dev/null"
    
    # Docker Compose
    run_test "Docker Compose available" "docker-compose --version &> /dev/null"
}

# Framework functionality tests
validate_framework_functionality() {
    log_header "Validating Framework Functionality"
    
    # Test chaos engine help
    run_test "Chaos engine help command" "python3 '$CHAOS_DIR/scripts/chaos-engine.py' --help &> /dev/null"
    
    # Test experiment runner help
    run_test "Experiment runner help command" "'$CHAOS_DIR/scripts/run-experiment.sh' --help &> /dev/null"
    
    # Test chaos monkey help
    run_test "Chaos monkey help command" "python3 '$CHAOS_DIR/scripts/chaos-monkey.py' --help &> /dev/null"
    
    # Test resilience tester help
    run_test "Resilience tester help command" "python3 '$CHAOS_DIR/scripts/resilience-tester.py' --help &> /dev/null"
    
    # Test configuration loading
    run_test "Configuration loading" "python3 -c 'import sys; sys.path.append(\"$CHAOS_DIR/scripts\"); from chaos_engine import ChaosEngine; ChaosEngine()'"
    
    # Test dry run capability
    run_test "Dry run experiment" "'$CHAOS_DIR/scripts/run-experiment.sh' --experiment basic-container-chaos --dry-run &> /dev/null"
}

# Integration validation
validate_integrations() {
    log_header "Validating Integrations"
    
    # Docker integration
    run_test "Docker integration validation" "python3 '$CHAOS_DIR/scripts/docker-integration.py' --validate &> /dev/null"
    
    # Monitoring integration
    run_test "Monitoring integration validation" "python3 '$CHAOS_DIR/scripts/monitoring-integration.py' --validate &> /dev/null"
    
    # Check if SutazAI services are available
    local sutazai_services=("sutazai-backend" "sutazai-frontend" "sutazai-postgres" "sutazai-redis")
    local available_services=0
    
    for service in "${sutazai_services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            ((available_services++))
        fi
    done
    
    if [[ $available_services -gt 0 ]]; then
        log_success "SutazAI services available for testing ($available_services/${#sutazai_services[@]})"
    else
        log_warning "No SutazAI services running - chaos experiments will have limited targets"
    fi
}

# Monitoring setup validation
validate_monitoring_setup() {
    log_header "Validating Monitoring Setup"
    
    # Check monitoring files
    local monitoring_files=(
        "chaos-prometheus.yml"
        "chaos-dashboard.json"
    )
    
    for file in "${monitoring_files[@]}"; do
        run_test "Monitoring file exists: $file" "test_file_exists '$CHAOS_DIR/monitoring/$file' 'Monitoring configuration'"
    done
    
    # Test Prometheus connectivity (if available)
    if curl -s http://localhost:9090/api/v1/status/config &> /dev/null; then
        log_success "Prometheus accessible for integration"
    else
        log_warning "Prometheus not accessible - monitoring integration will be limited"
    fi
    
    # Test Grafana connectivity (if available)
    if curl -s http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana accessible for dashboards"
    else
        log_warning "Grafana not accessible - dashboard integration will be limited"
    fi
}

# Safety mechanism validation
validate_safety_mechanisms() {
    log_header "Validating Safety Mechanisms"
    
    # Test safe mode configuration
    run_test "Safe mode in configuration" "grep -q 'safe_mode.*true' '$CHAOS_DIR/config/chaos-config.yaml'"
    
    # Test protected services configuration
    run_test "Protected services configured" "grep -q 'protected_services' '$CHAOS_DIR/config/chaos-config.yaml'"
    
    # Test maintenance window configuration
    run_test "Maintenance windows configured" "grep -q 'maintenance_window' '$CHAOS_DIR/config/chaos-config.yaml'"
    
    # Test emergency stop capability
    run_test "Emergency stop script availability" "grep -q 'emergency_stop' '$CHAOS_DIR/scripts/chaos-monkey.py'"
    
    # Test health check integration
    run_test "Health check integration" "grep -q 'health_check' '$CHAOS_DIR/scripts/chaos-engine.py'"
}

# Performance and resource validation
validate_performance_requirements() {
    log_header "Validating Performance Requirements"
    
    # Check available disk space
    local available_space
    available_space=$(df "$CHAOS_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_space -gt 1000000 ]]; then  # 1GB in KB
        log_success "Sufficient disk space available ($(($available_space / 1024 / 1024))GB)"
    else
        log_warning "Limited disk space available for chaos reports"
    fi
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [[ $memory_usage -lt 85 ]]; then
        log_success "Memory usage acceptable (${memory_usage}%)"
    else
        log_warning "High memory usage detected (${memory_usage}%)"
    fi
    
    # Check CPU load
    local cpu_load
    cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    if (( $(echo "$cpu_load < 2.0" | bc -l 2>/dev/null || echo "1") )); then
        log_success "CPU load acceptable ($cpu_load)"
    else
        log_warning "High CPU load detected ($cpu_load)"
    fi
}

# Generate validation report
generate_validation_report() {
    log_header "Validation Report"
    
    local success_rate
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    else
        success_rate=0
    fi
    
    log_info "Total tests: $TOTAL_TESTS"
    log_info "Passed: $PASSED_TESTS"
    log_info "Failed: $FAILED_TESTS"
    log_info "Success rate: ${success_rate}%"
    
    echo ""
    echo "=== DETAILED RESULTS ==="
    for result in "${VALIDATION_RESULTS[@]}"; do
        echo "$result"
    done
    echo "========================="
    
    # Overall status
    if [[ $success_rate -ge 90 ]]; then
        log_success "Chaos Engineering Framework validation PASSED (${success_rate}%)"
        echo ""
        echo "🎉 The SutazAI Chaos Engineering Framework is ready for use!"
        echo ""
        echo "Next steps:"
        echo "1. Run: ./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode"
        echo "2. Monitor: tail -f /opt/sutazaiapp/logs/chaos.log"
        echo "3. Dashboard: http://localhost:3000 (SutazAI Chaos Engineering)"
        echo ""
        return 0
    elif [[ $success_rate -ge 75 ]]; then
        log_warning "Chaos Engineering Framework validation PASSED with warnings (${success_rate}%)"
        echo ""
        echo "⚠️  The framework is functional but has some issues to address."
        echo "Review the failed tests above and fix before production use."
        echo ""
        return 0
    else
        log_error "Chaos Engineering Framework validation FAILED (${success_rate}%)"
        echo ""
        echo "❌ The framework has significant issues that must be resolved."
        echo "Review all failed tests and run validation again."
        echo ""
        return 1
    fi
}

# Main execution
main() {
    log_header "SutazAI Chaos Engineering Framework - Deployment Validation"
    log_info "Starting comprehensive validation at $(date)"
    log_info "Log file: $LOG_FILE"
    
    # Run all validation tests
    validate_directory_structure
    validate_configuration_files
    validate_scripts
    validate_dependencies
    validate_framework_functionality
    validate_integrations
    validate_monitoring_setup
    validate_safety_mechanisms
    validate_performance_requirements
    
    # Generate final report
    generate_validation_report
}

# Execute main function
main "$@"