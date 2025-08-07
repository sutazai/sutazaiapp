#!/bin/bash
"""
Comprehensive test runner for Ollama integration testing
Executes all test suites with proper configuration and reporting
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTS_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$PROJECT_ROOT/test-reports"
LOGS_DIR="$PROJECT_ROOT/logs"

# Test configuration
PYTHON_PATH="$PROJECT_ROOT/agents:$PROJECT_ROOT:$PYTHONPATH"
PYTEST_ARGS="-v --tb=short --strict-markers"
COVERAGE_THRESHOLD=80
PERFORMANCE_TIMEOUT=300  # 5 minutes for performance tests

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Ollama Integration Test Runner

Usage: $0 [OPTIONS] [TEST_SUITE]

TEST_SUITE options:
    unit            Run unit tests only
    integration     Run integration tests only
    performance     Run performance tests only
    failure         Run failure scenario tests only
    regression      Run regression tests only
    all             Run all test suites (default)

OPTIONS:
    -h, --help      Show this help message
    -c, --coverage  Generate coverage report
    -f, --fast      Skip slow tests (performance and long-running tests)
    -v, --verbose   Verbose output
    -q, --quiet     Quiet output (errors only)
    --no-cleanup    Don't cleanup test artifacts
    --parallel      Run tests in parallel where possible
    --junit         Generate JUnit XML reports
    --html          Generate HTML coverage report
    --ci            CI mode (includes all reports, stricter requirements)

Examples:
    $0                          # Run all tests
    $0 unit                     # Run only unit tests
    $0 --coverage --html        # Run all tests with HTML coverage report
    $0 --ci                     # Run in CI mode with all reports
    $0 performance --verbose    # Run performance tests with verbose output

EOF
}

# Parse command line arguments
COVERAGE=false
FAST_MODE=false
VERBOSE=false
QUIET=false
NO_CLEANUP=false
PARALLEL=false
JUNIT=false
HTML_REPORT=false
CI_MODE=false
TEST_SUITE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -f|--fast)
            FAST_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --junit)
            JUNIT=true
            shift
            ;;
        --html)
            HTML_REPORT=true
            shift
            ;;
        --ci)
            CI_MODE=true
            COVERAGE=true
            JUNIT=true
            HTML_REPORT=true
            shift
            ;;
        unit|integration|performance|failure|regression|all)
            TEST_SUITE="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set verbosity
if [[ "$VERBOSE" == "true" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -vv -s"
elif [[ "$QUIET" == "true" ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -q"
fi

# Set up directories
setup_directories() {
    log "Setting up test directories..."
    
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$LOGS_DIR"
    
    # Create test report subdirectories
    mkdir -p "$REPORTS_DIR/unit"
    mkdir -p "$REPORTS_DIR/integration"
    mkdir -p "$REPORTS_DIR/performance"
    mkdir -p "$REPORTS_DIR/failure"
    mkdir -p "$REPORTS_DIR/regression"
    mkdir -p "$REPORTS_DIR/coverage"
}

# Check test environment
check_environment() {
    log "Checking test environment..."
    
    # Check Python version
    if ! python3 --version &>/dev/null; then
        error "Python 3 is required but not found"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "${python_version}" < "3.8" ]]; then
        error "Python 3.8 or higher is required, found $python_version"
        exit 1
    fi
    
    # Check required packages
    local required_packages=("pytest" "pytest-asyncio" "pytest-mock" "pytest-cov" "httpx")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &>/dev/null; then
            error "Required package '$package' not found. Please install test requirements:"
            error "pip install -r $PROJECT_ROOT/tests/requirements-test.txt"
            exit 1
        fi
    done
    
    # Check test files exist
    local test_files=(
        "$TESTS_DIR/test_ollama_integration.py"
        "$TESTS_DIR/test_base_agent_v2.py"
        "$TESTS_DIR/test_connection_pool.py"
        "$TESTS_DIR/test_performance.py"
        "$TESTS_DIR/test_failure_scenarios.py"
        "$TESTS_DIR/test_integration.py"
        "$TESTS_DIR/test_regression.py"
    )
    
    for test_file in "${test_files[@]}"; do
        if [[ ! -f "$test_file" ]]; then
            error "Test file not found: $test_file"
            exit 1
        fi
    done
    
    success "Environment check passed"
}

# Install test requirements if needed
install_requirements() {
    local requirements_file="$TESTS_DIR/requirements-test.txt"
    
    if [[ -f "$requirements_file" ]]; then
        log "Installing test requirements..."
        python3 -m pip install -r "$requirements_file" --quiet
    else
        warning "Test requirements file not found: $requirements_file"
        log "Installing basic test dependencies..."
        python3 -m pip install pytest pytest-asyncio pytest-mock pytest-cov httpx psutil --quiet
    fi
}

# Build pytest command
build_pytest_command() {
    local test_files=()
    local pytest_cmd="python3 -m pytest"
    
    # Add coverage if requested
    if [[ "$COVERAGE" == "true" ]]; then
        pytest_cmd="$pytest_cmd --cov=$PROJECT_ROOT/agents/core --cov-report=term-missing"
        
        if [[ "$HTML_REPORT" == "true" ]]; then
            pytest_cmd="$pytest_cmd --cov-report=html:$REPORTS_DIR/coverage/html"
        fi
        
        pytest_cmd="$pytest_cmd --cov-report=xml:$REPORTS_DIR/coverage/coverage.xml"
        pytest_cmd="$pytest_cmd --cov-fail-under=$COVERAGE_THRESHOLD"
    fi
    
    # Add JUnit XML if requested
    if [[ "$JUNIT" == "true" ]]; then
        pytest_cmd="$pytest_cmd --junit-xml=$REPORTS_DIR/junit.xml"
    fi
    
    # Add parallel execution if requested
    if [[ "$PARALLEL" == "true" ]]; then
        pytest_cmd="$pytest_cmd -n auto"
    fi
    
    # Select test files based on suite
    case $TEST_SUITE in
        unit)
            test_files=(
                "$TESTS_DIR/test_ollama_integration.py"
                "$TESTS_DIR/test_base_agent_v2.py"
                "$TESTS_DIR/test_connection_pool.py"
            )
            ;;
        integration)
            test_files=("$TESTS_DIR/test_integration.py")
            ;;
        performance)
            test_files=("$TESTS_DIR/test_performance.py")
            if [[ "$FAST_MODE" == "true" ]]; then
                pytest_cmd="$pytest_cmd -m 'not slow'"
            fi
            ;;
        failure)
            test_files=("$TESTS_DIR/test_failure_scenarios.py")
            ;;
        regression)
            test_files=("$TESTS_DIR/test_regression.py")
            ;;
        all)
            test_files=(
                "$TESTS_DIR/test_ollama_integration.py"
                "$TESTS_DIR/test_base_agent_v2.py"
                "$TESTS_DIR/test_connection_pool.py"
                "$TESTS_DIR/test_integration.py"
                "$TESTS_DIR/test_failure_scenarios.py"
                "$TESTS_DIR/test_regression.py"
            )
            
            if [[ "$FAST_MODE" != "true" ]]; then
                test_files+=("$TESTS_DIR/test_performance.py")
            fi
            ;;
        *)
            error "Unknown test suite: $TEST_SUITE"
            exit 1
            ;;
    esac
    
    # Add test files to command
    pytest_cmd="$pytest_cmd ${test_files[*]}"
    
    # Add pytest args
    pytest_cmd="$pytest_cmd $PYTEST_ARGS"
    
    echo "$pytest_cmd"
}

# Run test suite
run_tests() {
    local pytest_cmd
    pytest_cmd=$(build_pytest_command)
    
    log "Running $TEST_SUITE tests..."
    log "Command: $pytest_cmd"
    
    # Set environment variables
    export PYTHONPATH="$PYTHON_PATH"
    export LOG_LEVEL="WARNING"  # Reduce log noise during tests
    
    # Add CI-specific environment variables
    if [[ "$CI_MODE" == "true" ]]; then
        export CI=true
        export PYTEST_CURRENT_TEST=true
    fi
    
    # Set timeout for performance tests
    if [[ "$TEST_SUITE" == "performance" || "$TEST_SUITE" == "all" ]]; then
        timeout "$PERFORMANCE_TIMEOUT" bash -c "$pytest_cmd" || {
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                error "Performance tests timed out after $PERFORMANCE_TIMEOUT seconds"
            fi
            return $exit_code
        }
    else
        eval "$pytest_cmd"
    fi
}

# Generate test report
generate_report() {
    local report_file="$REPORTS_DIR/test_summary_$(date +%Y%m%d_%H%M%S).md"
    
    log "Generating test report: $report_file"
    
    cat > "$report_file" << EOF
# Ollama Integration Test Report

**Generated:** $(date)
**Test Suite:** $TEST_SUITE
**Mode:** $(if [[ "$CI_MODE" == "true" ]]; then echo "CI"; else echo "Local"; fi)

## Configuration
- Coverage: $COVERAGE
- Fast Mode: $FAST_MODE
- Parallel: $PARALLEL
- JUnit: $JUNIT
- HTML Report: $HTML_REPORT

## Test Results

EOF
    
    # Add coverage summary if available
    if [[ "$COVERAGE" == "true" && -f "$REPORTS_DIR/coverage/coverage.xml" ]]; then
        echo "### Coverage Summary" >> "$report_file"
        echo "" >> "$report_file"
        
        # Parse coverage percentage from XML (basic parsing)
        if command -v xmllint &> /dev/null; then
            local coverage_pct=$(xmllint --xpath "string(//coverage/@line-rate)" "$REPORTS_DIR/coverage/coverage.xml" 2>/dev/null || echo "N/A")
            if [[ "$coverage_pct" != "N/A" ]]; then
                coverage_pct=$(python3 -c "print(f'{float('$coverage_pct') * 100:.1f}%')" 2>/dev/null || echo "N/A")
            fi
            echo "**Line Coverage:** $coverage_pct" >> "$report_file"
        fi
        echo "" >> "$report_file"
    fi
    
    # Add performance metrics if available
    if [[ "$TEST_SUITE" == "performance" || "$TEST_SUITE" == "all" ]]; then
        echo "### Performance Metrics" >> "$report_file"
        echo "" >> "$report_file"
        echo "See individual test outputs for detailed performance metrics." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add recommendations
    echo "### Recommendations" >> "$report_file"
    echo "" >> "$report_file"
    
    if [[ "$COVERAGE" == "true" ]]; then
        echo "- Review coverage report for areas needing additional tests" >> "$report_file"
    fi
    
    if [[ "$TEST_SUITE" == "performance" || "$TEST_SUITE" == "all" ]]; then
        echo "- Monitor performance benchmarks for regression detection" >> "$report_file"
    fi
    
    echo "- Run full test suite regularly to catch regressions early" >> "$report_file"
    echo "" >> "$report_file"
    
    success "Test report generated: $report_file"
}

# Cleanup function
cleanup_test_artifacts() {
    if [[ "$NO_CLEANUP" == "true" ]]; then
        log "Skipping cleanup (--no-cleanup specified)"
        return
    fi
    
    log "Cleaning up test artifacts..."
    
    # Remove temporary files
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove old test databases or temporary files
    find "$PROJECT_ROOT" -name "test_*.db" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.tmp" -delete 2>/dev/null || true
    
    success "Cleanup completed"
}

# Health check for Ollama service (if available)
check_ollama_service() {
    log "Checking Ollama service availability..."
    
    local ollama_url="${OLLAMA_URL:-http://localhost:11434}"
    
    if command -v curl &> /dev/null; then
        if curl -s --connect-timeout 5 "$ollama_url/api/version" &>/dev/null; then
            success "Ollama service is available at $ollama_url"
            return 0
        else
            warning "Ollama service not available at $ollama_url"
            log "Tests will run with mocked Ollama interactions"
            return 1
        fi
    else
        warning "curl not available, cannot check Ollama service"
        return 1
    fi
}

# Main execution function
main() {
    local start_time=$(date +%s)
    local exit_code=0
    
    log "Starting Ollama Integration Test Runner"
    log "Test Suite: $TEST_SUITE"
    log "Project Root: $PROJECT_ROOT"
    
    # Setup
    setup_directories
    check_environment
    install_requirements
    check_ollama_service
    
    # Run tests
    if run_tests; then
        success "All tests passed!"
    else
        exit_code=$?
        error "Some tests failed (exit code: $exit_code)"
    fi
    
    # Generate reports
    generate_report
    
    # Cleanup
    cleanup_test_artifacts
    
    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Test execution completed in ${duration} seconds"
    
    if [[ $exit_code -eq 0 ]]; then
        success "✅ All tests passed successfully!"
        
        # CI-specific success actions
        if [[ "$CI_MODE" == "true" ]]; then
            log "CI Mode: Test results are ready for artifact collection"
            log "  - JUnit XML: $REPORTS_DIR/junit.xml"
            log "  - Coverage XML: $REPORTS_DIR/coverage/coverage.xml"
            if [[ "$HTML_REPORT" == "true" ]]; then
                log "  - HTML Coverage: $REPORTS_DIR/coverage/html/index.html"
            fi
        fi
    else
        error "❌ Some tests failed. Check the output above for details."
        
        # Provide helpful debugging information
        echo ""
        error "Debugging information:"
        error "  - Test reports: $REPORTS_DIR"
        error "  - Logs directory: $LOGS_DIR"
        error "  - Re-run with --verbose for more details"
        error "  - Run specific test suite: $0 <suite_name>"
    fi
    
    exit $exit_code
}

# Signal handlers for graceful shutdown
trap 'error "Test execution interrupted"; cleanup_test_artifacts; exit 130' INT TERM

# Execute main function
main "$@"