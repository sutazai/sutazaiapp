#!/bin/bash

# Test Coverage Improvement Script for SutazaiApp
# This script analyzes test coverage and generates test stubs for uncovered code

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"
REMOTE_PROJECT_PATH="/opt/sutazaiapp"
COVERAGE_THRESHOLD=95 # Target coverage percentage

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

# Run tests and generate coverage report
generate_coverage_report() {
    section "Generating Coverage Report"
    
    # Run the tests on the remote server with detailed coverage
    info "Running tests with detailed coverage reporting..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/ \
        -v \
        --asyncio-mode=auto \
        --cov=core_system \
        --cov=ai_agents \
        --cov=scripts \
        --cov=backend \
        --cov-report=html:coverage \
        --cov-report=xml:coverage/coverage.xml \
        --cov-report=term \
        --html=test_reports/report.html \
        --self-contained-html"
    
    success "Coverage report generated"
}

# Identify modules with low coverage
identify_low_coverage() {
    section "Identifying Modules with Low Coverage"
    
    # Extract coverage data from XML report
    info "Analyzing coverage data..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -c \"
import xml.etree.ElementTree as ET
import os

try:
    tree = ET.parse('coverage/coverage.xml')
    root = tree.getroot()
    
    # Find classes with coverage < ${COVERAGE_THRESHOLD}%
    print('Modules with coverage below ${COVERAGE_THRESHOLD}%:')
    print('-' * 60)
    print(f'{'Module':<40} {'Coverage %':<10} {'Missing Lines':<10}')
    print('-' * 60)
    
    for cls in root.findall('.//class'):
        filename = cls.attrib['filename']
        
        # Skip test files and __init__.py
        if '/tests/' in filename or filename.endswith('__init__.py'):
            continue
            
        line_rate = float(cls.attrib['line-rate']) * 100
        
        if line_rate < ${COVERAGE_THRESHOLD}:
            # Get missing lines
            missing_lines = []
            for line in cls.findall('.//line'):
                if int(line.attrib['hits']) == 0:
                    missing_lines.append(int(line.attrib['number']))
            
            # Format missing lines as ranges
            ranges = []
            start = end = None
            for line in sorted(missing_lines):
                if start is None:
                    start = end = line
                elif line == end + 1:
                    end = line
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f'{start}-{end}')
                    start = end = line
            
            if start is not None:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f'{start}-{end}')
            
            missing = ','.join(ranges) if ranges else 'None'
            if len(missing) > 20:
                missing = missing[:17] + '...'
                
            print(f'{filename:<40} {line_rate:<10.2f} {missing:<10}')
except Exception as e:
    print(f'Error analyzing coverage: {e}')
\"
"
    success "Low coverage modules identified"
}

# Generate test stubs for uncovered code
generate_test_stubs() {
    section "Generating Test Stubs for Uncovered Code"
    
    # Extract modules needing tests and generate stubs
    info "Creating test stubs for uncovered modules..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -c \"
import xml.etree.ElementTree as ET
import os
import re

try:
    tree = ET.parse('coverage/coverage.xml')
    root = tree.getroot()
    
    test_dir = 'tests'
    os.makedirs(test_dir, exist_ok=True)
    
    # Track what we've generated
    generated_stubs = []
    
    for cls in root.findall('.//class'):
        filename = cls.attrib['filename']
        
        # Skip test files and __init__.py
        if '/tests/' in filename or filename.endswith('__init__.py'):
            continue
            
        line_rate = float(cls.attrib['line-rate']) * 100
        
        if line_rate < ${COVERAGE_THRESHOLD}:
            # Determine module name from filename
            module_path = filename.replace('/', '.')
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            
            # Extract base module name for test file
            base_name = os.path.basename(filename)
            if base_name.endswith('.py'):
                base_name = base_name[:-3]
            
            test_file = os.path.join(test_dir, f'test_{base_name}.py')
            
            # Check if test file already exists
            if os.path.exists(test_file):
                print(f'Test file already exists: {test_file}')
                continue
                
            # Get missing lines to determine functions needing tests
            missing_lines = []
            for line in cls.findall('.//line'):
                if int(line.attrib['hits']) == 0:
                    missing_lines.append(int(line.attrib['number']))
            
            if not missing_lines:
                continue
                
            # Read the source file to find function names
            try:
                with open(filename, 'r') as f:
                    source_lines = f.readlines()
                
                # Find function definitions in missing lines
                functions_to_test = []
                class_name = None
                
                for i, line in enumerate(source_lines, 1):
                    if any(i <= ml <= i + 3 for ml in missing_lines):  # Check line and a few lines after
                        # Check for class definition
                        class_match = re.match(r'\\s*class\\s+(\\w+)', line)
                        if class_match:
                            class_name = class_match.group(1)
                            
                        # Check for function definition
                        func_match = re.match(r'\\s*def\\s+(\\w+)', line)
                        if func_match:
                            func_name = func_match.group(1)
                            # Skip private methods and special methods
                            if not func_name.startswith('_'):
                                functions_to_test.append((func_name, class_name))
                
                if functions_to_test:
                    # Generate test stub
                    with open(test_file, 'w') as f:
                        f.write(f'''\"\"\"Tests for {base_name} module.\"\"\"
import pytest
from {module_path} import *

''')
                        # Add test functions
                        for func_name, class_name in functions_to_test:
                            if class_name:
                                f.write(f'''
def test_{func_name}_in_{class_name.lower()}():
    \"\"\"Test the {func_name} method in {class_name} class.\"\"\"
    # TODO: Implement test for {class_name}.{func_name}
    # 1. Create instance of {class_name}
    # 2. Call the {func_name} method
    # 3. Assert expected behavior
    pass
''')
                            else:
                                f.write(f'''
def test_{func_name}():
    \"\"\"Test the {func_name} function.\"\"\"
    # TODO: Implement test for {func_name}
    # 1. Setup test data
    # 2. Call the {func_name} function
    # 3. Assert expected behavior
    pass
''')
                    
                    print(f'Generated test stub: {test_file}')
                    generated_stubs.append(test_file)
                    
            except Exception as e:
                print(f'Error generating test stub for {filename}: {e}')
    
    if generated_stubs:
        print(f'\\nGenerated {len(generated_stubs)} test stubs.')
    else:
        print('No new test stubs needed.')
                
except Exception as e:
    print(f'Error generating test stubs: {e}')
\"
"
    success "Test stubs generated"
}

# Fix missing test dependencies
fix_test_dependencies() {
    section "Fixing Test Dependencies"
    
    # Install dependencies that might be missing
    info "Installing potential missing dependencies..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-html \
        psutil fastapi httpx sqlalchemy aiohttp requests mock \
        pandas numpy"
    
    success "Dependencies installed"
}

# Create or update conftest.py with common fixtures
update_conftest() {
    section "Updating Test Configuration"
    
    # Create or update conftest.py
    info "Updating conftest.py with common fixtures..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && cat > tests/conftest.py << 'EOF'
\"\"\"Common test fixtures for all tests.\"\"\"
import os
import pytest
from pathlib import Path
import tempfile
import shutil
import json

@pytest.fixture
def temp_dir():
    \"\"\"Create a temporary directory for tests.\"\"\"
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    \"\"\"Create a test configuration file.\"\"\"
    # Create directories
    log_dir = temp_dir / 'logs'
    data_dir = temp_dir / 'data'
    backup_dir = temp_dir / 'backups'
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    (log_dir / 'code_audit.log').touch()
    
    # Create config file
    config = {
        'log_dir': str(log_dir),
        'data_dir': str(data_dir),
        'backup_dir': str(backup_dir),
        'max_log_age_days': 7,
        'max_backup_age_days': 30,
        'backup_retention': 5
    }
    config_path = temp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return config_path

@pytest.fixture
def mock_environment(monkeypatch):
    \"\"\"Set up mock environment variables for tests.\"\"\"
    monkeypatch.setenv('SUTAZAI_ENV', 'test')
    monkeypatch.setenv('SUTAZAI_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('SUTAZAI_CONFIG_PATH', '/tmp/test_config.json')
    
@pytest.fixture
def sample_data():
    \"\"\"Return sample data for tests.\"\"\"
    return {
        'id': 1,
        'name': 'Test Item',
        'value': 42,
        'active': True,
        'tags': ['test', 'sample', 'fixture']
    }
EOF"
    
    success "Test configuration updated"
}

# Run the improved test suite
run_improved_tests() {
    section "Running Improved Test Suite"
    
    # Run tests again after improvements
    info "Executing tests with all improvements..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/ \
        -v \
        --asyncio-mode=auto \
        --cov=core_system \
        --cov=ai_agents \
        --cov=scripts \
        --cov=backend \
        --cov-report=html:coverage \
        --cov-report=term \
        --html=test_reports/report.html \
        --self-contained-html"
    
    # Check if we reached the target coverage
    info "Checking if we reached target coverage..."
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && \
        python -c \"
import json
import os

try:
    # Try to parse coverage data from .coverage
    coverage_file = '.coverage'
    if not os.path.exists(coverage_file):
        coverage_file = 'coverage/.coverage'
    
    if os.path.exists(coverage_file):
        import json
        from coverage.control import Coverage
        
        cov = Coverage()
        cov.load()
        total = cov.report()
        
        if total >= ${COVERAGE_THRESHOLD}:
            print(f'SUCCESS: Reached {total:.2f}% coverage, which meets or exceeds target of ${COVERAGE_THRESHOLD}%')
        else:
            print(f'NEEDS IMPROVEMENT: Current coverage is {total:.2f}%, below target of ${COVERAGE_THRESHOLD}%')
    else:
        print('Could not find coverage data file')
except Exception as e:
    print(f'Error checking coverage: {e}')
\"
"
    
    success "Improved tests executed"
}

# Copy results back to local machine
copy_results() {
    section "Copying Results"
    
    # Copy coverage and test reports
    info "Copying reports back to local machine..."
    rsync -av -e "ssh -i ${SSH_KEY}" \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/coverage \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/test_reports \
        ${PROJECT_ROOT}/
    
    success "Reports copied to local machine"
}

# Main function
main() {
    section "Starting Test Coverage Improvement"
    
    # Generate initial coverage report
    generate_coverage_report
    
    # Identify modules with low coverage
    identify_low_coverage
    
    # Fix dependencies
    fix_test_dependencies
    
    # Update conftest.py
    update_conftest
    
    # Generate test stubs
    generate_test_stubs
    
    # Run improved tests
    run_improved_tests
    
    # Copy results
    copy_results
    
    section "Test Coverage Improvement Complete"
    info "Check the coverage reports for detailed information"
    info "Test stubs have been generated for uncovered code"
    info "Edit the generated test stubs to implement proper tests"
    
    return 0
}

# Run the main function
main "$@" 