# Hygiene Enforcement System Testing Guide

## Overview

This guide provides comprehensive documentation for testing the hygiene enforcement infrastructure, ensuring 100% functionality and catching potential issues before they affect development workflows.

## Table of Contents

- [Test Suite Architecture](#test-suite-architecture)
- [Quick Start](#quick-start)
- [Test Components](#test-components)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Failure Scenarios](#failure-scenarios)
- [Performance Testing](#performance-testing)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Test Suite Architecture

The hygiene testing system consists of multiple layers:

```
/scripts/test-hygiene-system.py          # Master test runner
/tests/hygiene/                          # Unit tests directory
├── __init__.py                          # Test package initialization
├── test_orchestrator.py                # Agent orchestrator tests
├── test_coordinator.py                 # Enforcement coordinator tests
├── test_git_hooks.py                   # Git hooks functionality tests
├── test_monitoring.py                  # Monitoring system tests
└── test_fixtures.py                    # Test fixtures and mock violations
/scripts/validate-hygiene-deployment.sh # Deployment validation
```

## Quick Start

### Prerequisites

- Python 3.8+
- Git repository initialized
- All hygiene enforcement components installed
- Bash shell environment

### Run All Tests

```bash
# Run complete test suite
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py

# Run with verbose output
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py --verbose

# Generate test report
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py --report logs/test-report.json
```

### Run Specific Component Tests

```bash
# Test only the orchestrator
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py --component orchestrator

# Test only Git hooks
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py --component hooks

# Test monitoring system
python3 /opt/sutazaiapp/scripts/test-hygiene-system.py --component monitoring
```

### Validate Deployment

```bash
# Basic deployment validation
/opt/sutazaiapp/scripts/validate-hygiene-deployment.sh

# Validate for specific environment
/opt/sutazaiapp/scripts/validate-hygiene-deployment.sh --environment=prod

# Verbose validation output
/opt/sutazaiapp/scripts/validate-hygiene-deployment.sh --verbose
```

## Test Components

### 1. Master Test Runner (`test-hygiene-system.py`)

The master test runner coordinates all testing activities:

**Features:**
- Component-specific testing
- Test environment setup/teardown
- Comprehensive reporting
- Performance monitoring
- Failure scenario simulation

**Usage Examples:**

```bash
# Setup test environment only
python3 scripts/test-hygiene-system.py --setup-only

# Run performance tests
python3 scripts/test-hygiene-system.py --component performance

# Run failure scenario tests
python3 scripts/test-hygiene-system.py --component failures
```

### 2. Unit Tests (`/tests/hygiene/`)

#### Orchestrator Tests (`test_orchestrator.py`)

Tests for the hygiene agent orchestrator:

- Agent registry validation
- Task creation and execution
- Rule-based agent coordination
- Error handling and recovery
- Integration with specialized agents

**Key Test Cases:**
- `test_orchestrator_exists()` - Verify script presence
- `test_orchestrator_dry_run()` - Test safe execution mode
- `test_get_agents_for_rule()` - Rule-to-agent mapping
- `test_execute_agent_task_success()` - Successful task execution
- `test_execute_agent_task_failure()` - Failure handling

#### Coordinator Tests (`test_coordinator.py`)

Tests for the hygiene enforcement coordinator:

- Rule violation detection
- File safety verification
- Archive management
- Phase-based enforcement
- Reporting functionality

**Key Test Cases:**
- `test_find_violations_rule_13()` - Junk file detection
- `test_verify_file_safety()` - Safe removal checks
- `test_enforce_rule_13_dry_run()` - Rule enforcement simulation
- `test_generate_report()` - Report generation

#### Git Hooks Tests (`test_git_hooks.py`)

Tests for Git hook functionality:

- Hook installation verification
- Pre-commit validation logic
- Pre-push enforcement
- Error handling in hooks
- Hook configuration management

**Key Test Cases:**
- `test_pre_commit_hook_exists()` - Hook presence verification
- `test_pre_commit_hook_blocks_violations()` - Violation blocking
- `test_hook_handles_missing_files()` - Error resilience

#### Monitoring Tests (`test_monitoring.py`)

Tests for the monitoring system:

- Real-time file monitoring
- Automated maintenance scheduling
- Log management
- Performance monitoring
- Resource usage tracking

**Key Test Cases:**
- `test_monitor_script_syntax()` - Script validation
- `test_maintenance_daily_mode()` - Daily maintenance execution
- `test_log_directory_creation()` - Log management

#### Fixtures Tests (`test_fixtures.py`)

Tests for test fixture creation and management:

- Mock violation generation
- Test data cleanup
- Fixture metadata management
- Violation pattern testing

**Key Test Cases:**
- `test_create_rule_13_fixtures()` - Junk file fixtures
- `test_detect_rule_13_violations()` - Violation detection
- `test_fixture_cleanup_simulation()` - Cleanup testing

### 3. Deployment Validation (`validate-hygiene-deployment.sh`)

Comprehensive deployment validation script:

**Validation Checks:**
1. Core script existence and permissions
2. Python syntax validation
3. Shell script syntax validation
4. Git hooks installation
5. Directory structure verification
6. Test suite functionality
7. System dependencies
8. Component functionality testing
9. File permissions
10. Environment-specific settings
11. Basic functionality tests

**Usage:**

```bash
# Development environment validation
./validate-hygiene-deployment.sh --environment=dev

# Production environment validation (strict)
./validate-hygiene-deployment.sh --environment=prod --verbose

# Check validation logs
tail -f logs/hygiene-deployment-validation.log
```

## Running Tests

### Local Development Testing

```bash
# Quick validation during development
python3 scripts/test-hygiene-system.py --component orchestrator --verbose

# Test specific functionality
cd /opt/sutazaiapp
python3 -m pytest tests/hygiene/test_orchestrator.py::TestAgentOrchestrator::test_get_agents_for_rule -v
```

### Integration Testing

```bash
# Full integration test
python3 scripts/test-hygiene-system.py

# Test with real files (careful in production)
python3 scripts/test-hygiene-system.py --component performance

# Validate complete deployment
scripts/validate-hygiene-deployment.sh --environment=staging
```

### Automated Testing

```bash
# Set up automated daily testing
crontab -e
# Add: 0 6 * * * cd /opt/sutazaiapp && python3 scripts/test-hygiene-system.py --report logs/daily-test-report.json

# Weekly comprehensive validation
# Add: 0 7 * * 0 cd /opt/sutazaiapp && scripts/validate-hygiene-deployment.sh --environment=prod
```

## Test Coverage

### Component Coverage Matrix

| Component | Unit Tests | Integration Tests | Performance Tests | Failure Tests |
|-----------|------------|-------------------|-------------------|---------------|
| Orchestrator | ✅ | ✅ | ✅ | ✅ |
| Coordinator | ✅ | ✅ | ✅ | ✅ |
| Git Hooks | ✅ | ✅ | ⚠️ | ✅ |
| Monitoring | ✅ | ⚠️ | ✅ | ✅ |
| Automation | ✅ | ✅ | ✅ | ⚠️ |

**Legend:**
- ✅ Full coverage
- ⚠️ Partial coverage
- ❌ No coverage

### Rule Coverage Testing

| Rule | Description | Test Coverage | Mock Violations |
|------|-------------|---------------|-----------------|
| 1 | No Fantasy Elements | ✅ | ✅ |
| 2 | No Breaking Changes | ✅ | ✅ |
| 3 | Analyze Everything | ✅ | ✅ |
| 8 | Python Documentation | ✅ | ✅ |
| 9 | Directory Consolidation | ✅ | ✅ |
| 11 | Docker Structure | ✅ | ✅ |
| 12 | Single Deployment Script | ✅ | ✅ |
| 13 | No Garbage Files | ✅ | ✅ |
| Others | Various Rules | ⚠️ | ⚠️ |

## Failure Scenarios

### Testing Failure Recovery

The test suite includes comprehensive failure scenario testing:

#### 1. Missing Dependencies

```bash
# Test behavior when Python modules are missing
python3 scripts/test-hygiene-system.py --component failures
```

**Tested Scenarios:**
- Missing Python packages
- Unavailable system commands
- Permission denied errors
- Disk space issues
- Network connectivity problems

#### 2. Corrupted Files

**Mock Scenarios:**
- Corrupted Python scripts
- Invalid shell scripts
- Missing Git hooks
- Broken configuration files

#### 3. Resource Exhaustion

**Test Coverage:**
- Memory usage limits
- CPU timeout handling
- Large file processing
- Concurrent execution limits

### Recovery Validation

```bash
# Test recovery mechanisms
python3 -c "
import sys
sys.path.insert(0, '/opt/sutazaiapp')
from tests.hygiene.test_orchestrator import TestOrchestratorErrorHandling
import unittest

suite = unittest.TestLoader().loadTestsFromTestCase(TestOrchestratorErrorHandling)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
"
```

## Performance Testing

### Performance Benchmarks

| Operation | Target Time | Memory Limit | Test Method |
|-----------|-------------|--------------|-------------|
| Orchestrator Dry Run | < 60s | < 100MB | `test_execution_time()` |
| Rule 13 Detection | < 30s | < 50MB | `test_large_directory_scanning_performance()` |
| Git Hook Execution | < 10s | < 25MB | Hook execution timing |
| Report Generation | < 5s | < 20MB | Report creation timing |

### Running Performance Tests

```bash
# Specific performance testing
python3 scripts/test-hygiene-system.py --component performance --verbose

# Monitor resource usage during tests
top -p $(pgrep -f test-hygiene-system) &
python3 scripts/test-hygiene-system.py
```

### Performance Monitoring

```bash
# Generate performance report
python3 scripts/test-hygiene-system.py --report logs/performance-report.json

# Analyze performance trends
grep "execution_time" logs/performance-report.json
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/hygiene-tests.yml
name: Hygiene System Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  hygiene-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        # Install any required packages
    
    - name: Run hygiene tests
      run: |
        python3 scripts/test-hygiene-system.py --report test-results.json
    
    - name: Validate deployment
      run: |
        scripts/validate-hygiene-deployment.sh --environment=staging
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test-results.json
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Hygiene Tests') {
            steps {
                script {
                    sh 'python3 scripts/test-hygiene-system.py --report test-results.json'
                    
                    def testResults = readJSON file: 'test-results.json'
                    
                    if (!testResults.summary.overall_success) {
                        error('Hygiene tests failed')
                    }
                }
            }
        }
        
        stage('Deployment Validation') {
            steps {
                sh 'scripts/validate-hygiene-deployment.sh --environment=staging'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test-results.json,logs/*.log', fingerprint: true
        }
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Test Environment Setup Failures

**Problem:** Test environment creation fails
```bash
# Check permissions
ls -la /tmp/
# Verify disk space
df -h /tmp/
```

**Solution:**
```bash
# Clean up old test directories
find /tmp -name "hygiene_test_*" -type d -mtime +1 -exec rm -rf {} \;
```

#### 2. Import Errors in Tests

**Problem:** `ImportError: No module named 'scripts.agents.hygiene_agent_orchestrator'`

**Solution:**
```bash
# Verify Python path
export PYTHONPATH="/opt/sutazaiapp:$PYTHONPATH"

# Check script syntax
python3 -m py_compile scripts/agents/hygiene-agent-orchestrator.py
```

#### 3. Permission Denied Errors

**Problem:** Cannot execute scripts or write to logs

**Solution:**
```bash
# Fix script permissions
find scripts/ -name "*.py" -exec chmod +x {} \;
find scripts/ -name "*.sh" -exec chmod +x {} \;

# Fix directory permissions
chmod 755 logs/
chmod 755 tests/hygiene/
```

#### 4. Git Hook Test Failures

**Problem:** Git hooks not found or not executable

**Solution:**
```bash
# Check Git repository status
git status

# Reinstall hooks
scripts/install-hygiene-hooks.sh

# Verify hook permissions
ls -la .git/hooks/pre-commit
```

### Debug Mode Testing

```bash
# Enable debug output
export HYGIENE_TEST_DEBUG=1
python3 scripts/test-hygiene-system.py --verbose

# Run single test with debugging
python3 -m pytest tests/hygiene/test_orchestrator.py::TestAgentOrchestrator::test_get_agents_for_rule -v -s
```

### Log Analysis

```bash
# View recent test logs
tail -f logs/hygiene-deployment-validation.log

# Search for specific errors
grep -i error logs/*.log

# Analyze test performance
grep "execution_time" logs/test-report.json | jq '.'
```

## Best Practices

### 1. Test Development

- Always create tests for new hygiene components
- Use descriptive test names and docstrings
- Include both positive and negative test cases
- Test error conditions and edge cases
- Mock external dependencies appropriately

### 2. Test Execution

- Run tests in isolated environments
- Clean up test artifacts after execution
- Use appropriate timeouts for long-running tests
- Monitor resource usage during tests
- Generate and review test reports regularly

### 3. Test Maintenance

- Update tests when components change
- Remove obsolete tests
- Keep test fixtures current
- Review test coverage regularly
- Document test requirements clearly

## Reporting and Metrics

### Test Report Structure

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "summary": {
    "total_components_tested": 8,
    "successful_components": 7,
    "total_tests_run": 45,
    "total_failures": 1,
    "total_errors": 0,
    "overall_success": false
  },
  "component_results": {
    "orchestrator": {
      "tests_run": 12,
      "failures": 0,
      "errors": 0,
      "success": true
    }
  },
  "recommendations": [
    "Address test failures before deployment",
    "Consider adding more comprehensive tests"
  ]
}
```

### Metrics Tracking

Track these metrics over time:
- Test execution time trends
- Test failure rates by component
- Coverage percentage
- Performance benchmark trends
- Resource usage patterns

---

## Conclusion

This testing guide provides comprehensive coverage for validating the hygiene enforcement system. Regular execution of these tests ensures system reliability and catches issues before they impact development workflows.

For additional support or questions about the testing system, refer to the component-specific documentation or create an issue in the project repository.