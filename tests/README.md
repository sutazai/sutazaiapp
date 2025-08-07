# Ollama Integration Testing Framework

## Overview

This comprehensive testing framework ensures 100% reliability of the Ollama integration with all 131 agents in the SutazAI system. The framework includes unit tests, integration tests, performance tests, failure scenario tests, and regression tests.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Global pytest configuration and fixtures
├── pytest.ini                  # Pytest configuration
├── requirements-test.txt        # Test dependencies
├── test_ollama_integration.py   # Unit tests for OllamaIntegration class
├── test_base_agent_v2.py       # Unit tests for BaseAgentV2 class
├── test_connection_pool.py     # Unit tests for OllamaConnectionPool
├── test_integration.py         # Integration tests for system interactions
├── test_performance.py         # Performance and load testing
├── test_failure_scenarios.py   # Failure scenario and resilience tests
└── test_regression.py          # Regression tests for backward compatibility
```

## Test Categories

### 1. Unit Tests
- **File**: `test_ollama_integration.py`, `test_base_agent_v2.py`, `test_connection_pool.py`
- **Purpose**: Test individual components in isolation
- **Coverage**: 
  - OllamaIntegration class methods
  - BaseAgentV2 lifecycle and methods
  - OllamaConnectionPool functionality
  - Circuit breaker behavior
  - Request queue management

### 2. Integration Tests
- **File**: `test_integration.py`
- **Purpose**: Test component interactions and end-to-end workflows
- **Coverage**:
  - Agent-to-Ollama communication
  - Multi-agent coordination
  - System-wide integration
  - Configuration integration
  - Error propagation

### 3. Performance Tests
- **File**: `test_performance.py`
- **Purpose**: Validate performance under load and resource constraints
- **Coverage**:
  - Connection pool performance
  - Concurrent request handling
  - Memory efficiency
  - Response time benchmarks
  - Resource optimization

### 4. Failure Scenario Tests
- **File**: `test_failure_scenarios.py`
- **Purpose**: Test resilience and error handling
- **Coverage**:
  - Ollama service failures
  - Network failures
  - Resource exhaustion
  - Data corruption scenarios
  - Recovery mechanisms

### 5. Regression Tests
- **File**: `test_regression.py`
- **Purpose**: Ensure backward compatibility
- **Coverage**:
  - Existing agent compatibility
  - API stability
  - Configuration compatibility
  - Performance regression detection

## Running Tests

### Prerequisites

1. **Python 3.8+** required
2. **Install test dependencies**:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

### Quick Start

```bash
# Run all tests
./scripts/run_ollama_tests.sh

# Run specific test suite
./scripts/run_ollama_tests.sh unit
./scripts/run_ollama_tests.sh integration
./scripts/run_ollama_tests.sh performance
./scripts/run_ollama_tests.sh failure
./scripts/run_ollama_tests.sh regression

# Run with coverage
./scripts/run_ollama_tests.sh --coverage --html

# Run in CI mode
./scripts/run_ollama_tests.sh --ci
```

### Manual Test Execution

```bash
# Unit tests only
pytest tests/test_ollama_integration.py tests/test_base_agent_v2.py tests/test_connection_pool.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests (may take longer)
pytest tests/test_performance.py -v -m "not slow"

# All tests with coverage
pytest --cov=agents/core --cov-report=html tests/
```

### Test Markers

Use pytest markers to run specific categories:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"

# Run tests that don't require network
pytest -m "not network"
```

## Test Runner Options

The `run_ollama_tests.sh` script provides comprehensive test execution with the following options:

```bash
Usage: ./scripts/run_ollama_tests.sh [OPTIONS] [TEST_SUITE]

TEST_SUITE options:
    unit            Run unit tests only
    integration     Run integration tests only
    performance     Run performance tests only
    failure         Run failure scenario tests only
    regression      Run regression tests only
    all             Run all test suites (default)

OPTIONS:
    -h, --help      Show help message
    -c, --coverage  Generate coverage report
    -f, --fast      Skip slow tests
    -v, --verbose   Verbose output
    -q, --quiet     Quiet output (errors only)
    --no-cleanup    Don't cleanup test artifacts
    --parallel      Run tests in parallel
    --junit         Generate JUnit XML reports
    --html          Generate HTML coverage report
    --ci            CI mode (all reports, stricter requirements)
```

## Test Configuration

### Environment Variables

The following environment variables can be set to customize test behavior:

```bash
# Service URLs (defaults provided)
export BACKEND_URL="http://test-backend:8000"
export OLLAMA_URL="http://test-ollama:10104"

# Test configuration
export LOG_LEVEL="WARNING"
export TESTING="true"
export PYTEST_CURRENT_TEST="true"

# CI/CD specific
export CI="true"  # Enables CI-specific behavior
```

### Pytest Configuration

Configuration is managed through:
- `pytest.ini`: Main pytest configuration
- `conftest.py`: Global fixtures and test utilities

## Test Data and Fixtures

### Global Fixtures

Available in all tests via `conftest.py`:

- `temp_config_file`: Temporary agent configuration file
- `mock_environment`: Mock environment variables
- `mock_ollama_service`: Mock Ollama service responses
- `mock_backend_service`: Mock backend coordinator
- `base_agent`: Configured BaseAgentV2 instance
- `sample_task`: Sample task for testing
- `sample_task_result`: Sample task result
- `mock_circuit_breaker`: Mock circuit breaker
- `mock_connection_pool`: Mock connection pool
- `assertions`: Custom test assertions

### Custom Assertions

Use the `assertions` fixture for specialized validations:

```python
def test_agent_metrics(assertions, base_agent):
    # Validate agent metrics structure
    assertions.assert_valid_agent_metrics(base_agent.metrics)
    
    # Check performance bounds
    assertions.assert_performance_within_bounds(response_time, 2.0, "API call")
    
    # Verify memory usage
    assertions.assert_memory_usage_reasonable(memory_mb, max_memory_mb=100)
```

## Test Coverage

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Critical Components**: >95% coverage required
- **New Features**: 100% coverage required

### Coverage Reports

Generated coverage reports include:

1. **Terminal Report**: Summary displayed after test run
2. **HTML Report**: Detailed coverage analysis at `test-reports/coverage/html/index.html`
3. **XML Report**: Machine-readable format at `test-reports/coverage/coverage.xml`

### Viewing Coverage

```bash
# Generate and view HTML coverage report
./scripts/run_ollama_tests.sh --coverage --html
open test-reports/coverage/html/index.html
```

## Performance Benchmarks

### Performance Thresholds

The following performance thresholds are enforced:

```python
PERFORMANCE_BENCHMARKS = {
    "connection_pool": {
        "max_response_time_avg": 2.0,
        "min_success_rate": 0.95,
        "max_memory_growth_mb": 200
    },
    "agent_processing": {
        "max_response_time_avg": 1.0,
        "min_success_rate": 0.98,
        "max_response_time_p95": 2.0
    },
    "system_wide": {
        "max_response_time_avg": 3.0,
        "min_success_rate": 0.95,
        "max_memory_usage_mb": 500
    }
}
```

### Performance Test Categories

1. **Connection Pool Performance**: Concurrent connection handling
2. **Agent Processing Performance**: Task processing efficiency
3. **Multi-Agent Performance**: System-wide coordination
4. **Resource Optimization**: Memory and CPU usage
5. **Model Switching Performance**: Overhead of model changes

## Failure Scenarios

### Tested Failure Modes

1. **Service Failures**:
   - Ollama service down
   - Backend coordinator unavailable
   - Network partitions
   - DNS resolution failures

2. **Resource Exhaustion**:
   - Memory pressure
   - Connection limits
   - Queue overflow
   - Timeout scenarios

3. **Data Corruption**:
   - Malformed JSON responses
   - Invalid configuration
   - Incomplete task data
   - Network corruption

4. **Recovery Testing**:
   - Circuit breaker activation
   - Connection pool recovery
   - Graceful degradation
   - Service restoration

## CI/CD Integration

### Running in CI/CD

Use CI mode for automated testing:

```bash
./scripts/run_ollama_tests.sh --ci
```

CI mode includes:
- Full test suite execution
- Coverage report generation
- JUnit XML output
- HTML coverage reports
- Performance benchmarking
- Failure scenario validation

### CI/CD Outputs

Generated artifacts for CI/CD:

- `test-reports/junit.xml`: JUnit test results
- `test-reports/coverage/coverage.xml`: Coverage data
- `test-reports/coverage/html/`: HTML coverage report
- `test-reports/test_summary_*.md`: Test execution summary

### GitHub Actions Example

```yaml
- name: Run Ollama Integration Tests
  run: |
    ./scripts/run_ollama_tests.sh --ci
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-results
    path: test-reports/
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure PYTHONPATH includes agents directory
   export PYTHONPATH="agents:.:$PYTHONPATH"
   pytest tests/
   ```

2. **Async Test Issues**:
   ```bash
   # Install pytest-asyncio
   pip install pytest-asyncio
   ```

3. **Missing Dependencies**:
   ```bash
   # Install all test requirements
   pip install -r tests/requirements-test.txt
   ```

4. **Permission Errors**:
   ```bash
   # Make test runner executable
   chmod +x scripts/run_ollama_tests.sh
   ```

### Debug Mode

For detailed debugging:

```bash
# Verbose output with debug logging
./scripts/run_ollama_tests.sh --verbose

# No cleanup for artifact inspection
./scripts/run_ollama_tests.sh --no-cleanup

# Single test file with maximum verbosity
pytest tests/test_ollama_integration.py -vvv -s --tb=long
```

### Performance Issues

If tests are running slowly:

```bash
# Use fast mode (skips slow tests)
./scripts/run_ollama_tests.sh --fast

# Run tests in parallel
./scripts/run_ollama_tests.sh --parallel

# Run specific test categories
./scripts/run_ollama_tests.sh unit  # Fastest
```

## Contributing

### Adding New Tests

1. **Choose the appropriate test file** based on test category
2. **Follow naming conventions**: `test_*` for functions, `Test*` for classes
3. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. **Add fixtures as needed** in `conftest.py`
5. **Update this README** if adding new test categories

### Test Writing Guidelines

1. **Test Isolation**: Each test should be independent
2. **Clear Naming**: Test names should describe what is being tested
3. **Appropriate Mocking**: Mock external dependencies
4. **Error Testing**: Include both success and failure scenarios
5. **Performance Awareness**: Consider test execution time
6. **Documentation**: Add docstrings for complex test logic

### Example Test Structure

```python
import pytest
from unittest.mock import patch, Mock

class TestNewFeature:
    """Test suite for new feature"""
    
    @pytest.mark.unit
    def test_basic_functionality(self, mock_environment):
        """Test basic functionality works as expected"""
        # Arrange
        # Act
        # Assert
        pass
    
    @pytest.mark.integration
    async def test_integration_scenario(self, base_agent):
        """Test integration with other components"""
        # Arrange
        # Act
        # Assert
        pass
    
    @pytest.mark.performance
    def test_performance_requirements(self, assertions):
        """Test performance meets requirements"""
        # Measure performance
        # Assert within bounds
        assertions.assert_performance_within_bounds(time, limit)
```

## Monitoring and Metrics

### Test Metrics

Track the following metrics:

- **Test Coverage**: Percentage of code covered by tests
- **Test Execution Time**: Duration of test suite execution
- **Test Success Rate**: Percentage of passing tests
- **Performance Benchmarks**: Response times and resource usage
- **Failure Recovery**: Time to recover from failures

### Continuous Monitoring

Set up monitoring for:

- Daily test execution
- Performance regression detection
- Coverage trend analysis
- Failure pattern identification
- Resource usage tracking

## Support

For questions or issues with the testing framework:

1. Check this README for common solutions
2. Review test output and logs in `test-reports/`
3. Run tests with `--verbose` for detailed information
4. Check the test artifacts in `test-reports/` directory

## Changelog

### Version 1.0.0
- Initial comprehensive testing framework
- Full coverage of Ollama integration
- Performance benchmarking
- Failure scenario testing
- CI/CD integration
- Documentation and examples