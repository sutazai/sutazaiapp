# Service Configuration Guide

This guide explains how to configure and use the system services for task automation and analysis.

## Service Structure

Each service is a specialized component designed for specific tasks. Here's how to configure them:

### Basic Service Configuration

```json
{
  "name": "code-analysis-service",
  "type": "analyzer",
  "capabilities": [
    "static_analysis",
    "style_check",
    "metrics_collection"
  ],
  "config": {
    "max_file_size": "1MB",
    "timeout": 300,
    "concurrent_jobs": 4
  }
```

## Service API Reference

### 1. List Available Services

```bash
curl http://localhost:8000/api/v1/services/
```

Response:
```json
{
  "services": [
    {
      "name": "code-analyzer",
      "description": "Static code analysis service",
      "status": "running",
      "version": "1.0.0"
    },
    {
      "name": "test-runner",
      "description": "Automated test execution service",
      "status": "running",
      "version": "1.0.0"
    }
    // ... more services
  ]
}
```

### 2. Execute Analysis Task

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "service": "code-analyzer",
    "operation": "analyze",
    "data": {
      "file_path": "./src/utils.py",
      "rules": ["style", "complexity", "bugs"]
    }
  }'
```

## Service Integration Examples

### Code Analysis Service

**Purpose**: Static code analysis and linting

```python
# Using in Python
import httpx
from typing import List, Dict

async def analyze_code(file_path: str, rules: List[str]) -> Dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/analyze",
            json={
                "service": "code-analyzer",
                "operation": "analyze",
                "data": {
                    "file_path": file_path,
                    "rules": rules,
                    "severity": "error"
                }
            }
        )
        return response.json()
```

### Security Scanner Service

**Purpose**: Security vulnerability scanning

```python
async def scan_security(directory_path: str) -> Dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/scan",
            json={
                "service": "security-scanner",
                "operation": "scan",
                "data": {
                    "target": directory_path,
                    "checks": ["cve", "secrets", "config"]
                }
            }
        )
        return response.json()
```

### Test Runner Service

**Purpose**: Execute test suites

```python
async def run_tests(test_path: str) -> Dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/test",
            json={
                "service": "test-runner",
                "operation": "run",
                "data": {
                    "path": test_path,
                    "framework": "pytest",
                    "coverage": true
                }
            }
        )
        return response.json()
```

## Service Types

### Development Services

| Service | Operations | Purpose |
|---------|------------|----------|
| `code-analyzer` | lint, analyze, measure | Static code analysis |
| `dependency-checker` | scan, update, audit | Dependency management |
| `api-validator` | validate, document, test | API verification |

### Testing Services

| Service | Operations | Use Case |
|---------|------------|-----------|
| `test-runner` | run, coverage, report | Test execution |
| `security-scanner` | scan, audit, report | Security testing |

### Operations Services

| Service | Operations | Use Case |
|---------|------------|-----------|
| `deployment-service` | deploy, rollback, verify | Deployment management |
| `resource-monitor` | collect, analyze, alert | Resource monitoring |

## Configuration Guidelines

### 1. Service Configuration

```json
{
  "service": "code-analyzer",
  "operation": "analyze",
  "config": {
    "rules": ["security", "performance"],
    "threshold": "warning",
    "output_format": "json"
  }
}
```

### 2. Resource Management

```json
{
  "service": "test-runner",
  "limits": {
    "timeout": 300,
    "max_memory": "1GB",
    "max_concurrent": 4
  }
}
```

### 3. Output Settings

```json
{
  "service": "security-scanner",
  "output": {
    "format": "json",
    "include_details": true,
    "sort_by": "severity"
  }
}
```

## Service Workflows

### 1. Code Analysis Pipeline

```python
async def analyze_codebase(project_path: str) -> Dict:
    # Step 1: Static analysis
    static_analysis = await run_static_analysis(project_path)
    
    # Step 2: Security scan
    security_scan = await run_security_scan(project_path)
    
    # Step 3: Generate report
    report = await generate_analysis_report(static_analysis, security_scan)
    
    return report
```

### 2. Test Execution Pipeline

```python
async def run_test_suite(test_path: str) -> Dict:
    # Step 1: Run unit tests
    unit_results = await run_unit_tests(test_path)
    
    # Step 2: Generate coverage
    coverage = await generate_coverage(test_path)
    
    # Step 3: Create test report
    report = await create_test_report(unit_results, coverage)
    
    return report
```

## Error Management

```python
from typing import Dict, Any

try:
    result = await execute_service_operation(
        service_name="test-runner",
        operation="run",
        data=operation_data
    )
except ServiceUnavailableError:
    # Handle service downtime
    logger.error("Service unavailable")
except OperationTimeoutError:
    # Handle operation timeout
    logger.error("Operation timed out")
except Exception as e:
    # General error handling
    logger.error(f"Operation failed: {str(e)}")
```

## Performance Optimization

1. **Request Batching**: Combine related operations
2. **Concurrent Execution**: Use async operations
3. **Result Caching**: Cache frequent lookups
4. **Connection Pooling**: Reuse service connections

## Service Monitoring

```python
from monitoring import get_service_metrics

# Get service metrics
metrics = await get_service_metrics("code-analyzer")
print(f"Operations completed: {metrics['operations_count']}")
print(f"Average response time: {metrics['avg_response_time']}ms")
print(f"Error rate: {metrics['error_rate']}%")
```

## Service Extensions

The system supports custom service extensions through the standard interface:

```python
from typing import Dict, Any

class CustomAnalyzer:
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Pre-process input
        processed_data = self.preprocess(data)
        
        # Run analysis
        results = await self.run_analysis(processed_data)
        
        # Format output
        return self.format_results(results)
```

## Troubleshooting Guide

### Service Unavailable

1. Check service status:
   ```bash
   docker ps | grep service-name
   ```

2. View service logs:
   ```bash
   docker logs service-name
   ```

3. Verify service health:
   ```bash
   curl http://localhost:8000/health/service-name
   ```

### Operation Timeouts

- Review timeout configuration
- Split large operations
- Use pagination for large datasets

### Resource Management

- Monitor resource usage
- Implement rate limiting
- Configure proper scaling

## Overview

The service system provides:
- **Specialized services** for specific operations
- **REST API** for integration
- **Containerized deployment** for isolation
- **Configurable resources** for scaling
- **Standard workflows** for common tasks

Best practice is to use appropriate services for each operation type and combine them for efficient processing pipelines.