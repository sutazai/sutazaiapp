#!/bin/bash
# Rule System Validation Script
# ============================
#
# Purpose: Comprehensive validation of the rule system infrastructure
# Usage: ./validate-rule-system.sh [--quick|--full|--stress] [--report-only] [--fix-issues]
# Requirements: Python 3.8+, Docker, sufficient system resources
#
# This script validates:
# - Rule system configuration integrity
# - Test infrastructure functionality  
# - Resource availability and limits
# - Dependency satisfaction
# - Conflict detection and resolution
# - Performance characteristics
# - Recovery and rollback mechanisms
#
# Author: AI Testing and QA Validation Specialist
# Version: 1.0.0
# Date: 2025-08-03

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"
REPORT_DIR="$BASE_DIR/reports"
CONFIG_DIR="$BASE_DIR/config"
TEST_DIR="$BASE_DIR/tests"

# Create necessary directories
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Logging setup
LOG_FILE="$LOG_DIR/rule-system-validation.log"
VALIDATION_REPORT="$REPORT_DIR/rule_system_validation_$(date +%Y%m%d_%H%M%S).md"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)  echo -e "[\033[32m$level\033[0m] $timestamp - $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "[\033[33m$level\033[0m] $timestamp - $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "[\033[31m$level\033[0m] $timestamp - $message" | tee -a "$LOG_FILE" ;;
        DEBUG) echo -e "[\033[36m$level\033[0m] $timestamp - $message" | tee -a "$LOG_FILE" ;;
        *)     echo -e "[$level] $timestamp - $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Initialize validation report
init_report() {
    cat > "$VALIDATION_REPORT" << 'EOF'
# Rule System Validation Report

## Executive Summary
This report provides a comprehensive validation of the rule system infrastructure, 
including configuration integrity, test framework functionality, and performance characteristics.

**Report Generated**: $(date '+%Y-%m-%d %H:%M:%S')
**Validation Mode**: %VALIDATION_MODE%
**System Information**:
- Platform: $(uname -a)
- Python Version: $(python3 --version 2>/dev/null || echo "Not available")
- Docker Version: $(docker --version 2>/dev/null || echo "Not available")
- Available Memory: $(free -h | grep Mem | awk '{print $2}' 2>/dev/null || echo "Unknown")
- Available Disk: $(df -h . | tail -1 | awk '{print $4}' 2>/dev/null || echo "Unknown")

---

EOF
}

# System prerequisites validation
validate_prerequisites() {
    log INFO "Validating system prerequisites..."
    local issues=0
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log ERROR "Python 3 is not installed or not in PATH"
        ((issues++))
    else
        local python_version=$(python3 --version | cut -d' ' -f2)
        local min_version="3.8.0"
        if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
            log ERROR "Python version $python_version is too old (minimum: $min_version)"
            ((issues++))
        else
            log INFO "Python version $python_version is compatible"
        fi
    fi
    
    # Check required Python packages
    local required_packages=("psutil" "sqlite3" "json" "pathlib" "concurrent.futures")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log WARN "Python package '$package' not available (may cause test failures)"
        fi
    done
    
    # Check Docker availability
    if ! command -v docker &> /dev/null; then
        log WARN "Docker is not installed (some validation tests will be skipped)"
    else
        if ! docker info &> /dev/null; then
            log WARN "Docker daemon is not running (Docker-related tests will fail)"
        else
            log INFO "Docker is available and running"
        fi
    fi
    
    # Check system resources
    local available_memory=$(free -m | grep Mem | awk '{print $7}')
    local min_memory=2048  # 2GB minimum
    if [[ $available_memory -lt $min_memory ]]; then
        log WARN "Available memory ($available_memory MB) is below recommended minimum (${min_memory} MB)"
    fi
    
    local available_disk=$(df . | tail -1 | awk '{print int($4/1024)}')
    local min_disk=5120  # 5GB minimum
    if [[ $available_disk -lt $min_disk ]]; then
        log WARN "Available disk space ($available_disk MB) is below recommended minimum (${min_disk} MB)"
    fi
    
    # Check file permissions
    local critical_files=(
        "$BASE_DIR/scripts/test-all-rule-combinations.py"
        "$BASE_DIR/tests/rule-combination-matrix.json"
        "$BASE_DIR/CLAUDE.md"
        "$BASE_DIR/CLAUDE.local.md"
    )
    
    for file in "${critical_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log ERROR "Critical file missing: $file"
            ((issues++))
        elif [[ ! -r "$file" ]]; then
            log ERROR "Critical file not readable: $file"
            ((issues++))
        fi
    done
    
    # Check write permissions for output directories
    local output_dirs=("$LOG_DIR" "$REPORT_DIR")
    for dir in "${output_dirs[@]}"; do
        if [[ ! -w "$dir" ]]; then
            log ERROR "Output directory not writable: $dir"
            ((issues++))
        fi
    done
    
    if [[ $issues -eq 0 ]]; then
        log INFO "All prerequisites validated successfully"
        return 0
    else
        log ERROR "Prerequisites validation failed with $issues issues"
        return 1
    fi
}

# Configuration validation
validate_configuration() {
    log INFO "Validating rule system configuration..."
    local issues=0
    
    # Validate rule configuration matrix
    local config_file="$TEST_DIR/rule-combination-matrix.json"
    if [[ ! -f "$config_file" ]]; then
        log ERROR "Rule combination matrix not found: $config_file"
        ((issues++))
        return $issues
    fi
    
    # Parse and validate JSON structure
    if ! python3 -c "
import json
import sys

try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    required_sections = ['meta', 'rules', 'combination_categories', 'test_execution_plan', 'validation_criteria']
    for section in required_sections:
        if section not in config:
            print(f'ERROR: Missing required section: {section}')
            sys.exit(1)
    
    # Validate rules section
    rules = config.get('rules', {})
    if len(rules) == 0:
        print('ERROR: No rules defined in configuration')
        sys.exit(1)
    
    rule_ids = set()
    for rule_key, rule_data in rules.items():
        if 'id' not in rule_data:
            print(f'ERROR: Rule {rule_key} missing id field')
            sys.exit(1)
        
        rule_id = rule_data['id']
        if rule_id in rule_ids:
            print(f'ERROR: Duplicate rule ID: {rule_id}')
            sys.exit(1)
        rule_ids.add(rule_id)
        
        # Validate required fields
        required_fields = ['name', 'category', 'priority', 'test_scenarios']
        for field in required_fields:
            if field not in rule_data:
                print(f'ERROR: Rule {rule_key} missing required field: {field}')
                sys.exit(1)
    
    # Validate dependencies and conflicts reference valid rule IDs
    all_rule_ids = list(rule_ids)
    for rule_key, rule_data in rules.items():
        for dep_id in rule_data.get('dependencies', []):
            if dep_id not in all_rule_ids:
                print(f'ERROR: Rule {rule_key} references invalid dependency: {dep_id}')
                sys.exit(1)
        
        for conflict_id in rule_data.get('conflicts', []):
            if conflict_id not in all_rule_ids:
                print(f'ERROR: Rule {rule_key} references invalid conflict: {conflict_id}')
                sys.exit(1)
    
    print(f'INFO: Configuration validated successfully - {len(rules)} rules defined')
    
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON in configuration file: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Configuration validation failed: {e}')
    sys.exit(1)
" 2>&1; then
        log INFO "Configuration validation passed"
    else
        log ERROR "Configuration validation failed"
        ((issues++))
    fi
    
    # Validate CLAUDE.md and CLAUDE.local.md consistency
    if [[ -f "$BASE_DIR/CLAUDE.md" && -f "$BASE_DIR/CLAUDE.local.md" ]]; then
        local claude_rules=$(grep -c "Rule [0-9]" "$BASE_DIR/CLAUDE.md" "$BASE_DIR/CLAUDE.local.md" | cut -d: -f2 | paste -sd+ | bc 2>/dev/null || echo "0")
        local config_rules=$(python3 -c "import json; print(len(json.load(open('$config_file'))['rules']))" 2>/dev/null || echo "0")
        
        if [[ $claude_rules -ne $config_rules ]]; then
            log WARN "Mismatch between CLAUDE.md rules ($claude_rules) and configuration rules ($config_rules)"
        fi
    fi
    
    return $issues
}

# Test infrastructure validation
validate_test_infrastructure() {
    log INFO "Validating test infrastructure..."
    local issues=0
    
    # Validate main test script
    local test_script="$BASE_DIR/scripts/test-all-rule-combinations.py"
    if ! python3 -m py_compile "$test_script" 2>/dev/null; then
        log ERROR "Test script has syntax errors: $test_script"
        ((issues++))
    else
        log INFO "Test script syntax validation passed"
    fi
    
    # Test script help functionality
    if ! python3 "$test_script" --help &>/dev/null; then
        log ERROR "Test script help functionality not working"
        ((issues++))
    fi
    
    # Validate test database creation
    local test_db="$LOG_DIR/test_validation.db"
    if python3 -c "
import sqlite3
import sys
import os

try:
    # Remove existing test db
    if os.path.exists('$test_db'):
        os.remove('$test_db')
    
    # Test database creation
    conn = sqlite3.connect('$test_db')
    conn.execute('CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)')
    conn.execute('INSERT INTO test_table (data) VALUES (?)', ('test_data',))
    conn.commit()
    
    # Test database query
    cursor = conn.execute('SELECT data FROM test_table WHERE id = 1')
    result = cursor.fetchone()
    if result[0] != 'test_data':
        raise Exception('Database query failed')
    
    conn.close()
    os.remove('$test_db')
    print('INFO: Database functionality validated')
    
except Exception as e:
    print(f'ERROR: Database validation failed: {e}')
    sys.exit(1)
"; then
        log INFO "Database functionality validation passed"
    else
        log ERROR "Database functionality validation failed"
        ((issues++))
    fi
    
    # Validate concurrent execution capability
    if ! python3 -c "
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def test_task(task_id):
    time.sleep(0.1)
    return task_id * 2

try:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(test_task, i) for i in range(10)]
        results = [future.result() for future in as_completed(futures)]
    
    if len(results) != 10:
        raise Exception(f'Expected 10 results, got {len(results)}')
    
    print('INFO: Concurrent execution capability validated')
    
except Exception as e:
    print(f'ERROR: Concurrent execution validation failed: {e}')
    import sys
    sys.exit(1)
"; then
        log INFO "Concurrent execution validation passed"
    else
        log ERROR "Concurrent execution validation failed"
        ((issues++))
    fi
    
    return $issues
}

# Performance baseline measurement
measure_performance_baseline() {
    log INFO "Measuring performance baseline..."
    
    local baseline_script=$(cat << 'EOF'
import time
import psutil
import json
import sys

def measure_baseline():
    start_time = time.time()
    
    # Collect initial metrics
    initial_cpu = psutil.cpu_percent(interval=0.1)
    initial_memory = psutil.virtual_memory().percent
    initial_processes = len(psutil.pids())
    
    # Simulate minimal workload
    for i in range(1000):
        _ = i * i
    
    # Collect final metrics
    final_cpu = psutil.cpu_percent(interval=0.1)
    final_memory = psutil.virtual_memory().percent
    final_processes = len(psutil.pids())
    
    end_time = time.time()
    
    baseline_metrics = {
        'duration': end_time - start_time,
        'cpu_start': initial_cpu,
        'cpu_end': final_cpu,
        'memory_start': initial_memory,
        'memory_end': final_memory,
        'processes_start': initial_processes,
        'processes_end': final_processes
    }
    
    return baseline_metrics

if __name__ == "__main__":
    try:
        metrics = measure_baseline()
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"ERROR: Baseline measurement failed: {e}", file=sys.stderr)
        sys.exit(1)
EOF
)
    
    local baseline_result=$(python3 -c "$baseline_script" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo "$baseline_result" > "$REPORT_DIR/performance_baseline.json"
        log INFO "Performance baseline measured and saved"
    else
        log WARN "Performance baseline measurement failed"
    fi
}

# Quick validation test
run_quick_validation() {
    log INFO "Running quick validation test..."
    
    # Test with minimal rule combinations
    local test_output=$(python3 "$BASE_DIR/scripts/test-all-rule-combinations.py" \
        --max-combinations 5 \
        --phases baseline individual \
        --timeout 30 \
        --workers 2 2>&1)
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Quick validation test passed"
        echo "Quick validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 0
    else
        log ERROR "Quick validation test failed (exit code: $exit_code)"
        echo "Quick validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 1
    fi
}

# Full validation test
run_full_validation() {
    log INFO "Running full validation test..."
    
    # Test with extended rule combinations
    local test_output=$(python3 "$BASE_DIR/scripts/test-all-rule-combinations.py" \
        --max-combinations 100 \
        --phases baseline individual categories \
        --timeout 60 \
        --workers 4 2>&1)
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Full validation test passed"
        echo "Full validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 0
    else
        log ERROR "Full validation test failed (exit code: $exit_code)"
        echo "Full validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 1
    fi
}

# Stress validation test
run_stress_validation() {
    log INFO "Running stress validation test..."
    
    # Test with maximum rule combinations and stress conditions
    local test_output=$(python3 "$BASE_DIR/scripts/test-all-rule-combinations.py" \
        --max-combinations 1000 \
        --phases baseline individual categories stress \
        --timeout 120 \
        --workers 8 2>&1)
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Stress validation test passed"
        echo "Stress validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 0
    else
        log ERROR "Stress validation test failed (exit code: $exit_code)"
        echo "Stress validation test output:" >> "$LOG_FILE"
        echo "$test_output" >> "$LOG_FILE"
        return 1
    fi
}

# Fix common issues
fix_issues() {
    log INFO "Attempting to fix common issues..."
    local fixes_applied=0
    
    # Fix file permissions
    if [[ -f "$BASE_DIR/scripts/test-all-rule-combinations.py" ]]; then
        chmod +x "$BASE_DIR/scripts/test-all-rule-combinations.py"
        log INFO "Fixed script permissions"
        ((fixes_applied++))
    fi
    
    # Create missing directories
    local required_dirs=("$LOG_DIR" "$REPORT_DIR" "$TEST_DIR")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log INFO "Created missing directory: $dir"
            ((fixes_applied++))
        fi
    done
    
    # Install missing Python packages
    local pip_packages=("psutil")
    for package in "${pip_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            if command -v pip3 &> /dev/null; then
                pip3 install "$package" &>/dev/null
                log INFO "Installed Python package: $package"
                ((fixes_applied++))
            fi
        fi
    done
    
    log INFO "Applied $fixes_applied fixes"
    return 0
}

# Generate final validation report
generate_final_report() {
    local validation_mode="$1"
    local overall_status="$2"
    local validation_summary="$3"
    
    # Replace placeholders in report template
    sed -i "s/%VALIDATION_MODE%/$validation_mode/g" "$VALIDATION_REPORT"
    
    # Add validation results
    cat >> "$VALIDATION_REPORT" << EOF

## Validation Results

### Overall Status: $overall_status

### Validation Summary
$validation_summary

### Detailed Findings

#### Prerequisites Validation
$(grep "Prerequisites" "$LOG_FILE" | tail -10)

#### Configuration Validation  
$(grep "Configuration" "$LOG_FILE" | tail -10)

#### Test Infrastructure Validation
$(grep "Test infrastructure" "$LOG_FILE" | tail -10)

#### Performance Validation
$(grep "Performance" "$LOG_FILE" | tail -10)

### Log Files
- **Main Log**: $LOG_FILE
- **Validation Report**: $VALIDATION_REPORT
- **Performance Baseline**: $REPORT_DIR/performance_baseline.json

### System Information at Completion
- **Timestamp**: $(date '+%Y-%m-%d %H:%M:%S')
- **Memory Usage**: $(free -h | grep Mem | awk '{print $3 "/" $2}')
- **Disk Usage**: $(df -h . | tail -1 | awk '{print $3 "/" $2}')
- **Load Average**: $(uptime | awk -F'load average:' '{print $2}')

### Recommendations

EOF

    # Add specific recommendations based on findings
    if grep -q "ERROR" "$LOG_FILE"; then
        echo "- **CRITICAL**: Address all ERROR-level issues before running comprehensive tests" >> "$VALIDATION_REPORT"
    fi
    
    if grep -q "WARN" "$LOG_FILE"; then
        echo "- **WARNING**: Review all WARNING-level issues for potential impact" >> "$VALIDATION_REPORT"
    fi
    
    echo "- Run \`validate-rule-system.sh --fix-issues\` to attempt automatic fixes" >> "$VALIDATION_REPORT"
    echo "- Monitor system resources during comprehensive testing" >> "$VALIDATION_REPORT"
    echo "- Review log files for detailed diagnostic information" >> "$VALIDATION_REPORT"
    
    log INFO "Final validation report generated: $VALIDATION_REPORT"
}

# Main execution function
main() {
    local validation_mode="quick"
    local report_only=false
    local fix_issues_flag=false
    local overall_status="UNKNOWN"
    local validation_summary=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                validation_mode="quick"
                shift
                ;;
            --full)
                validation_mode="full"
                shift
                ;;
            --stress)
                validation_mode="stress"
                shift
                ;;
            --report-only)
                report_only=true
                shift
                ;;
            --fix-issues)
                fix_issues_flag=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--quick|--full|--stress] [--report-only] [--fix-issues]"
                echo ""
                echo "Options:"
                echo "  --quick       Run quick validation (default)"
                echo "  --full        Run full validation suite"
                echo "  --stress      Run stress testing validation"
                echo "  --report-only Generate report without running tests"
                echo "  --fix-issues  Attempt to fix common issues"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Initialize report
    init_report
    
    log INFO "Starting rule system validation in $validation_mode mode"
    log INFO "Report will be generated at: $VALIDATION_REPORT"
    
    # Fix issues if requested
    if [[ "$fix_issues_flag" == true ]]; then
        fix_issues
    fi
    
    # Track validation status
    local total_checks=0
    local passed_checks=0
    local failed_checks=0
    
    # Run validation checks
    if [[ "$report_only" != true ]]; then
        # Prerequisites validation
        ((total_checks++))
        if validate_prerequisites; then
            ((passed_checks++))
        else
            ((failed_checks++))
        fi
        
        # Configuration validation
        ((total_checks++))
        if validate_configuration; then
            ((passed_checks++))
        else
            ((failed_checks++))
        fi
        
        # Test infrastructure validation
        ((total_checks++))
        if validate_test_infrastructure; then
            ((passed_checks++))
        else
            ((failed_checks++))
        fi
        
        # Performance baseline
        measure_performance_baseline
        
        # Run appropriate validation test
        ((total_checks++))
        case "$validation_mode" in
            quick)
                if run_quick_validation; then
                    ((passed_checks++))
                else
                    ((failed_checks++))
                fi
                ;;
            full)
                if run_full_validation; then
                    ((passed_checks++))
                else
                    ((failed_checks++))
                fi
                ;;
            stress)
                if run_stress_validation; then
                    ((passed_checks++))
                else
                    ((failed_checks++))
                fi
                ;;
        esac
    fi
    
    # Determine overall status
    if [[ $failed_checks -eq 0 ]]; then
        overall_status="PASSED"
    elif [[ $passed_checks -gt $failed_checks ]]; then
        overall_status="PASSED_WITH_WARNINGS"
    else
        overall_status="FAILED"
    fi
    
    validation_summary="Completed $total_checks validation checks: $passed_checks passed, $failed_checks failed"
    
    # Generate final report
    generate_final_report "$validation_mode" "$overall_status" "$validation_summary"
    
    # Print summary
    echo ""
    echo "======================================"
    echo "Rule System Validation Complete"
    echo "======================================"
    echo "Mode: $validation_mode"
    echo "Status: $overall_status"
    echo "Summary: $validation_summary"
    echo "Report: $VALIDATION_REPORT"
    echo "Log: $LOG_FILE"
    echo "======================================"
    
    # Exit with appropriate code
    case "$overall_status" in
        PASSED)
            exit 0
            ;;
        PASSED_WITH_WARNINGS)
            exit 0
            ;;
        FAILED)
            exit 1
            ;;
        *)
            exit 2
            ;;
    esac
}

# Signal handlers
trap 'log ERROR "Validation interrupted by signal"; exit 130' INT TERM

# Execute main function
main "$@"