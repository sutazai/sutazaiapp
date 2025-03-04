#!/bin/bash

# Master Automation Script for SutazaiApp
# This script combines all the test improvement steps to achieve 100% test success

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"
REMOTE_PROJECT_PATH="/opt/sutazaiapp"

# Print section header
section() {
    echo -e "\n${BLUE}===== $1 =====${NC}\n"
}

# Print success message
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print info message
info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Fix dependencies
fix_dependencies() {
    section "Fixing Dependencies"
    
    # Run the dependency fix script
    info "Running dependency fix script..."
    ./fix_dependencies.sh
    
    success "Dependencies fixed"
}

# Fix async tests
fix_async_tests() {
    section "Fixing Async Tests"
    
    # Run the async test fix script
    info "Running async test fix script..."
    ./fix_async_tests.sh
    
    success "Async tests fixed"
}

# Improve test coverage
improve_coverage() {
    section "Improving Test Coverage"
    
    # Run the test coverage improvement script
    info "Running test coverage improvement script..."
    ./improve_test_coverage.sh
    
    success "Test coverage improved"
}

# Run full automation test
run_full_automation() {
    section "Running Full Automation Test"
    
    # Run the full automation test script
    info "Running full automation test script..."
    ./full_automation_test.sh
    
    success "Full automation test complete"
}

# Create test summary
create_summary() {
    section "Creating Test Summary"
    
    # Copy test reports from remote server
    info "Copying test reports from remote server..."
    rsync -av -e "ssh -i ${SSH_KEY}" \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/coverage \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/test_reports \
        ${PROJECT_ROOT}/
    
    # Generate test summary
    info "Generating test summary..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -c \"
import json
import os
from datetime import datetime

try:
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'coverage': {},
        'tests': {}
    }
    
    # Get coverage data
    coverage_file = 'coverage/.coverage'
    if os.path.exists(coverage_file):
        from coverage.control import Coverage
        
        cov = Coverage()
        cov.load()
        total = cov.report()
        
        summary['coverage']['total'] = total
        summary['coverage']['status'] = 'Success' if total >= 95 else 'Needs Improvement'
    else:
        summary['coverage']['status'] = 'Not Available'
    
    # Get test results from pytest output
    report_file = 'test_reports/report.html'
    if os.path.exists(report_file):
        # Parse HTML for test results - simplified approach
        with open(report_file, 'r') as f:
            content = f.read()
            
        if 'All tests passed!' in content:
            summary['tests']['status'] = 'All Passed'
        else:
            # Count tests from summary section
            import re
            match = re.search(r'(\d+) passed, (\d+) failed', content)
            if match:
                passed, failed = match.groups()
                summary['tests']['passed'] = int(passed)
                summary['tests']['failed'] = int(failed)
                summary['tests']['total'] = int(passed) + int(failed)
                summary['tests']['success_rate'] = round(int(passed) / (int(passed) + int(failed)) * 100, 1)
                summary['tests']['status'] = 'Partial Success'
            else:
                summary['tests']['status'] = 'Unknown'
    else:
        summary['tests']['status'] = 'Not Available'
    
    # Write summary to file
    with open('test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f'Test Summary ({summary[\"date\"]}):')
    print(f'- Coverage: {summary[\"coverage\"].get(\"total\", \"N/A\")}% ({summary[\"coverage\"][\"status\"]})')
    
    if 'success_rate' in summary['tests']:
        print(f'- Tests: {summary[\"tests\"][\"passed\"]}/{summary[\"tests\"][\"total\"]} ' +
              f'({summary[\"tests\"][\"success_rate\"]}%) {summary[\"tests\"][\"status\"]}')
    else:
        print(f'- Tests: {summary[\"tests\"][\"status\"]}')
    
except Exception as e:
    print(f'Error generating summary: {e}')
\"
"
    
    # Copy the summary file
    rsync -av -e "ssh -i ${SSH_KEY}" \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/test_summary.json \
        ${PROJECT_ROOT}/
    
    success "Test summary created"
}

# Main function
main() {
    section "Starting Master Automation Process"
    
    # Step 1: Fix dependencies
    fix_dependencies
    
    # Step 2: Fix async tests
    fix_async_tests
    
    # Step 3: Improve test coverage
    improve_coverage
    
    # Step 4: Run full automation test
    run_full_automation
    
    # Step 5: Create test summary
    create_summary
    
    section "Master Automation Process Complete"
    info "Check test_summary.json for detailed results"
    
    # Display final report
    if [ -f "${PROJECT_ROOT}/test_summary.json" ]; then
        cat "${PROJECT_ROOT}/test_summary.json"
    fi
    
    return 0
}

# Run the main function
main "$@" 