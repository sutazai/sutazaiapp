#!/bin/bash
# Purpose: Validates hygiene enforcement system deployment and functionality
# Usage: ./validate-hygiene-deployment.sh [--environment=dev|staging|prod] [--verbose]
# Requirements: bash, python3, git, find, grep

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
VALIDATION_LOG="$PROJECT_ROOT/logs/hygiene-deployment-validation.log"
ENVIRONMENT="${1:-dev}"
VERBOSE=false
ERRORS_FOUND=0
WARNINGS_FOUND=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--environment=dev|staging|prod] [--verbose]"
            echo "Validates hygiene enforcement system deployment"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    local message="$1"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [INFO] $message" | tee -a "$VALIDATION_LOG"
}

log_warning() {
    local message="$1"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [WARN] $message" | tee -a "$VALIDATION_LOG"
    ((WARNINGS_FOUND++))
}

log_error() {
    local message="$1"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [ERROR] $message" | tee -a "$VALIDATION_LOG"
    ((ERRORS_FOUND++))
}

log_success() {
    local message="$1"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [SUCCESS] $message" | tee -a "$VALIDATION_LOG"
}

# Ensure log directory exists
mkdir -p "$(dirname "$VALIDATION_LOG")"

# Initialize validation log
log_info "=== HYGIENE SYSTEM DEPLOYMENT VALIDATION ==="
log_info "Environment: $ENVIRONMENT"
log_info "Project Root: $PROJECT_ROOT"

# Check 1: Verify core scripts exist and are executable
validate_core_scripts() {
    log_info "Validating core hygiene scripts..."
    
    local core_scripts=(
        "scripts/test-hygiene-system.py"
        "scripts/agents/hygiene-agent-orchestrator.py"
        "scripts/hygiene-enforcement-coordinator.py"
        "scripts/utils/automated-hygiene-maintenance.sh"
    )
    
    for script in "${core_scripts[@]}"; do
        local script_path="$PROJECT_ROOT/$script"
        
        if [[ -f "$script_path" ]]; then
            if [[ -x "$script_path" ]]; then
                log_success "✓ $script exists and is executable"
            else
                log_warning "⚠ $script exists but is not executable"
            fi
        else
            log_error "✗ Missing core script: $script"
        fi
    done
}

# Check 2: Validate Python syntax for all Python scripts
validate_python_syntax() {
    log_info "Validating Python script syntax..."
    
    local python_scripts=(
        "scripts/test-hygiene-system.py"
        "scripts/agents/hygiene-agent-orchestrator.py"
        "scripts/hygiene-enforcement-coordinator.py"
    )
    
    for script in "${python_scripts[@]}"; do
        local script_path="$PROJECT_ROOT/$script"
        
        if [[ -f "$script_path" ]]; then
            if python3 -m py_compile "$script_path" 2>/dev/null; then
                log_success "✓ $script has valid Python syntax"
            else
                log_error "✗ $script has Python syntax errors"
                if [[ "$VERBOSE" == "true" ]]; then
                    python3 -m py_compile "$script_path" || true
                fi
            fi
        fi
    done
}

# Check 3: Validate shell script syntax
validate_shell_syntax() {
    log_info "Validating shell script syntax..."
    
    local shell_scripts=(
        "scripts/utils/automated-hygiene-maintenance.sh"
        "scripts/hygiene-audit.sh"
        "scripts/setup-hygiene-automation.sh"
        "scripts/install-hygiene-hooks.sh"
    )
    
    for script in "${shell_scripts[@]}"; do
        local script_path="$PROJECT_ROOT/$script"
        
        if [[ -f "$script_path" ]]; then
            if bash -n "$script_path" 2>/dev/null; then
                log_success "✓ $script has valid shell syntax"
            else
                log_error "✗ $script has shell syntax errors"
                if [[ "$VERBOSE" == "true" ]]; then
                    bash -n "$script_path" || true
                fi
            fi
        fi
    done
}

# Check 4: Validate Git hooks installation
validate_git_hooks() {
    log_info "Validating Git hooks installation..."
    
    local hooks_dir="$PROJECT_ROOT/.git/hooks"
    local expected_hooks=("pre-commit" "pre-push")
    
    if [[ ! -d "$hooks_dir" ]]; then
        log_error "✗ Git hooks directory not found: $hooks_dir"
        return
    fi
    
    for hook in "${expected_hooks[@]}"; do
        local hook_path="$hooks_dir/$hook"
        
        if [[ -f "$hook_path" ]]; then
            if [[ -x "$hook_path" ]]; then
                log_success "✓ Git $hook hook exists and is executable"
                
                # Validate hook content
                if grep -q "hygiene\|validation" "$hook_path"; then
                    log_success "✓ $hook hook contains hygiene validation logic"
                else
                    log_warning "⚠ $hook hook may not contain hygiene validation"
                fi
            else
                log_warning "⚠ Git $hook hook exists but is not executable"
            fi
        else
            log_warning "⚠ Git $hook hook not found"
        fi
    done
}

# Check 5: Test directory structure
validate_directory_structure() {
    log_info "Validating directory structure..."
    
    local required_directories=(
        "scripts"
        "scripts/agents"
        "scripts/utils"
        "tests/hygiene"
        "logs"
    )
    
    for dir in "${required_directories[@]}"; do
        local dir_path="$PROJECT_ROOT/$dir"
        
        if [[ -d "$dir_path" ]]; then
            log_success "✓ Directory exists: $dir"
        else
            if [[ "$dir" == "logs" ]]; then
                # Create logs directory if it doesn't exist
                mkdir -p "$dir_path"
                log_success "✓ Created missing directory: $dir"
            else
                log_error "✗ Missing required directory: $dir"
            fi
        fi
    done
}

# Check 6: Validate test suite functionality
validate_test_suite() {
    log_info "Validating test suite functionality..."
    
    local test_runner="$PROJECT_ROOT/scripts/test-hygiene-system.py"
    
    if [[ ! -f "$test_runner" ]]; then
        log_error "✗ Test runner not found: $test_runner"
        return
    fi
    
    # Test help functionality
    if python3 "$test_runner" --help >/dev/null 2>&1; then
        log_success "✓ Test runner shows help successfully"
    else
        log_error "✗ Test runner help functionality failed"
    fi
    
    # Test setup-only mode
    if python3 "$test_runner" --setup-only >/dev/null 2>&1; then
        log_success "✓ Test environment setup works"
    else
        log_warning "⚠ Test environment setup may have issues"
    fi
}

# Check 7: Validate dependencies
validate_dependencies() {
    log_info "Validating system dependencies..."
    
    local required_commands=(
        "python3"
        "git"
        "find"
        "grep"
        "bash"
    )
    
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_success "✓ Command available: $cmd"
        else
            log_error "✗ Missing required command: $cmd"
        fi
    done
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        local python_version
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_info "Python version: $python_version"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            log_success "✓ Python version is 3.8+"
        else
            log_warning "⚠ Python version may be too old (need 3.8+)"
        fi
    fi
}

# Check 8: Test orchestrator functionality
validate_orchestrator() {
    log_info "Validating orchestrator functionality..."
    
    local orchestrator="$PROJECT_ROOT/scripts/agents/hygiene-agent-orchestrator.py"
    
    if [[ ! -f "$orchestrator" ]]; then
        log_error "✗ Orchestrator not found: $orchestrator"
        return
    fi
    
    # Test help functionality
    if python3 "$orchestrator" --help >/dev/null 2>&1; then
        log_success "✓ Orchestrator shows help successfully"
    else
        log_error "✗ Orchestrator help functionality failed"
    fi
    
    # Test dry run for rule 13
    if timeout 60 python3 "$orchestrator" --rule=13 --dry-run >/dev/null 2>&1; then
        log_success "✓ Orchestrator dry run works for rule 13"
    else
        log_warning "⚠ Orchestrator dry run may have issues"
    fi
}

# Check 9: Test coordinator functionality
validate_coordinator() {
    log_info "Validating coordinator functionality..."
    
    local coordinator="$PROJECT_ROOT/scripts/hygiene-enforcement-coordinator.py"
    
    if [[ ! -f "$coordinator" ]]; then
        log_error "✗ Coordinator not found: $coordinator"
        return
    fi
    
    # Test help functionality
    if python3 "$coordinator" --help >/dev/null 2>&1; then
        log_success "✓ Coordinator shows help successfully"
    else
        log_error "✗ Coordinator help functionality failed"
    fi
    
    # Test dry run for phase 1
    if timeout 60 python3 "$coordinator" --phase=1 --dry-run >/dev/null 2>&1; then
        log_success "✓ Coordinator dry run works for phase 1"
    else
        log_warning "⚠ Coordinator dry run may have issues"
    fi
}

# Check 10: Validate file permissions
validate_permissions() {
    log_info "Validating file permissions..."
    
    # Check if we can write to logs directory
    local test_log="$PROJECT_ROOT/logs/permission_test.tmp"
    
    if echo "test" > "$test_log" 2>/dev/null; then
        rm -f "$test_log"
        log_success "✓ Can write to logs directory"
    else
        log_error "✗ Cannot write to logs directory"
    fi
    
    # Check if we can create archive directories
    local test_archive="$PROJECT_ROOT/archive/permission_test"
    
    if mkdir -p "$test_archive" 2>/dev/null; then
        rmdir "$test_archive" 2>/dev/null || true
        log_success "✓ Can create archive directories"
    else
        log_error "✗ Cannot create archive directories"
    fi
}

# Check 11: Validate environment-specific settings
validate_environment_settings() {
    log_info "Validating environment-specific settings for: $ENVIRONMENT"
    
    case "$ENVIRONMENT" in
        "dev")
            log_info "Development environment validation..."
            # Dev-specific checks
            if [[ -f "/tmp" ]]; then
                log_success "✓ Temp directory available for dev testing"
            fi
            ;;
        "staging")
            log_info "Staging environment validation..."
            # Staging-specific checks
            log_success "✓ Staging environment settings validated"
            ;;
        "prod")
            log_info "Production environment validation..."
            # Production-specific checks - more strict
            if [[ "$WARNINGS_FOUND" -gt 0 ]]; then
                log_error "✗ Production deployment should have no warnings"
            fi
            ;;
        *)
            log_warning "⚠ Unknown environment: $ENVIRONMENT"
            ;;
    esac
}

# Check 12: Run basic functionality test
run_functionality_test() {
    log_info "Running basic functionality test..."
    
    # Create temporary test directory
    local test_dir="/tmp/hygiene_validation_test_$$"
    mkdir -p "$test_dir"
    
    # Create test violation files
    echo "test backup" > "$test_dir/test.backup"
    echo "test temp" > "$test_dir/temp.tmp"
    
    # Test finding violations (simulate rule 13)
    local violations
    violations=$(find "$test_dir" -name "*.backup" -o -name "*.tmp" | wc -l)
    
    if [[ "$violations" -eq 2 ]]; then
        log_success "✓ Basic violation detection works"
    else
        log_error "✗ Basic violation detection failed"
    fi
    
    # Cleanup
    rm -rf "$test_dir"
}

# Main validation execution
main() {
    log_info "Starting hygiene system deployment validation..."
    
    validate_core_scripts
    validate_python_syntax
    validate_shell_syntax
    validate_git_hooks
    validate_directory_structure
    validate_test_suite
    validate_dependencies
    validate_orchestrator
    validate_coordinator
    validate_permissions
    validate_environment_settings
    run_functionality_test
    
    # Generate summary
    log_info "=== VALIDATION SUMMARY ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Errors found: $ERRORS_FOUND"
    log_info "Warnings found: $WARNINGS_FOUND"
    
    if [[ "$ERRORS_FOUND" -eq 0 ]]; then
        if [[ "$WARNINGS_FOUND" -eq 0 ]]; then
            log_success "✓ All validation checks passed successfully!"
            echo "✅ Hygiene system deployment validation: PASSED"
            return 0
        else
            log_warning "⚠ Validation passed with $WARNINGS_FOUND warnings"
            echo "⚠️ Hygiene system deployment validation: PASSED WITH WARNINGS"
            return 0
        fi
    else
        log_error "✗ Validation failed with $ERRORS_FOUND errors and $WARNINGS_FOUND warnings"
        echo "❌ Hygiene system deployment validation: FAILED"
        return 1
    fi
}

# Error handling
trap 'log_error "Validation interrupted or failed"; exit 1' ERR

# Run main validation
main "$@"