# Infrastructure Health Verification System

Comprehensive infrastructure health verification and monitoring system for the SutazAI platform. This system provides production-ready health checks, CI/CD pipeline integration, and operational monitoring capabilities based on the CLAUDE.md truth document.

## Overview

The health verification system consists of specialized health check scripts for each service category, plus a comprehensive orchestrator script that coordinates all checks and provides detailed reporting.

**Created:** December 19, 2024  
**Author:** infrastructure-devops-manager agent  
**Version:** 1.0.0

## Service Categories

Based on the CLAUDE.md truth document, the following service categories are monitored:

### Critical Services (System Failure if Down)
- **Ollama + TinyLlama** (port 10104): Local LLM service with TinyLlama model
- **Core Data Services** (ports 10000, 10001, 10007, 10008): PostgreSQL, Redis, RabbitMQ

### Non-Critical Services (Degraded Operation if Down)
- **API Gateway Services** (ports 10005, 10006): Kong API Gateway, Consul Service Discovery
- **Vector Database Services** (ports 10100-10103): Qdrant, FAISS, ChromaDB
- **Monitoring Services** (ports 10200-10203): Prometheus, Grafana, Loki, AlertManager

## Health Check Scripts

### Individual Service Scripts

#### 1. Ollama + TinyLlama Health Check
**File:** `scripts/devops/health_check_ollama.py`

Verifies Ollama service and TinyLlama model availability according to CLAUDE.md Rule 16.

```bash
# Basic usage
python scripts/devops/health_check_ollama.py

# Custom configuration
python scripts/devops/health_check_ollama.py --host localhost --port 10104 --timeout 10

# Skip text generation test
python scripts/devops/health_check_ollama.py --skip-generation --verbose
```

**Checks performed:**
- TCP connectivity to Ollama server
- Ollama API health endpoint verification
- TinyLlama model presence verification
- Text generation functionality test (optional)

#### 2. API Gateway Health Check  
**File:** `scripts/devops/health_check_gateway.py`

Verifies Kong API Gateway and Consul service discovery health.

```bash
# Basic usage
python scripts/devops/health_check_gateway.py

# Custom ports
python scripts/devops/health_check_gateway.py --kong-port 10005 --consul-port 10006

# Skip specific services
python scripts/devops/health_check_gateway.py --skip-consul --verbose
```

**Checks performed:**
- Kong Gateway admin API status
- Kong services and routes configuration
- Consul cluster leadership and peers
- Consul service catalog and health checks

#### 3. Vector Database Health Check
**File:** `scripts/devops/health_check_vectordb.py`  

Verifies vector database services with awareness of known issues.

```bash
# Basic usage
python scripts/devops/health_check_vectordb.py

# Specific service checks
python scripts/devops/health_check_vectordb.py --specific-services --verbose

# Custom port range
python scripts/devops/health_check_vectordb.py --port-range 10100-10103
```

**Checks performed:**
- Qdrant HTTP and gRPC endpoints
- FAISS service availability
- ChromaDB service status (expects connection issues per CLAUDE.md)
- Collections and configuration verification

#### 4. Core Data Services Health Check
**File:** `scripts/devops/health_check_dataservices.py`

Verifies essential data layer services.

```bash
# Basic usage  
python scripts/devops/health_check_dataservices.py

# Custom ports
python scripts/devops/health_check_dataservices.py --redis-port 10001 --postgres-port 10000

# Skip specific services
python scripts/devops/health_check_dataservices.py --skip-rabbitmq --verbose
```

**Checks performed:**
- PostgreSQL database connectivity and protocol verification
- Redis cache connectivity and command execution
- RabbitMQ AMQP protocol and management interface

#### 5. Monitoring Services Health Check
**File:** `scripts/devops/health_check_monitoring.py`

Verifies comprehensive monitoring stack.

```bash
# Basic usage
python scripts/devops/health_check_monitoring.py

# Custom configuration
python scripts/devops/health_check_monitoring.py --prometheus-port 10200 --grafana-port 10201

# Skip specific services  
python scripts/devops/health_check_monitoring.py --skip-loki --verbose
```

**Checks performed:**
- Prometheus metrics collection and targets
- Grafana dashboard accessibility and datasources  
- Loki log aggregation and query API
- AlertManager alert routing and configuration

### Comprehensive Orchestrator Script

**File:** `scripts/devops/infrastructure_health_check.py`

The main orchestrator script that coordinates all health checks and provides comprehensive reporting.

```bash
# Run all health checks
python scripts/devops/infrastructure_health_check.py

# Parallel execution for faster results
python scripts/devops/infrastructure_health_check.py --parallel

# Check specific service groups
python scripts/devops/infrastructure_health_check.py --services ollama,dataservices

# Generate JSON report for CI/CD
python scripts/devops/infrastructure_health_check.py --json-output /tmp/health_report.json

# Custom timeouts and ports
python scripts/devops/infrastructure_health_check.py --timeout 30 --host 127.0.0.1 --verbose
```

**Features:**
- Parallel and sequential execution modes
- Comprehensive JSON reporting for CI/CD integration
- Critical vs non-critical service classification
- Performance timing and success rate calculation
- Actionable recommendations based on failures
- Configurable exit codes for pipeline integration

## CI/CD Integration

### Pipeline Integration Examples

#### GitHub Actions
```yaml
- name: Infrastructure Health Check
  run: |
    python scripts/devops/infrastructure_health_check.py \
      --parallel \
      --json-output health_report.json \
      --fail-on-critical
    
- name: Upload Health Report
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: infrastructure-health-report
    path: health_report.json
```

#### GitLab CI
```yaml
infrastructure_health:
  stage: test
  script:
    - python scripts/devops/infrastructure_health_check.py --parallel --json-output health_report.json
  artifacts:
    reports:
      junit: health_report.json
    when: always
    expire_in: 1 week
```

#### Jenkins Pipeline
```groovy
stage('Infrastructure Health Check') {
    steps {
        script {
            sh 'python scripts/devops/infrastructure_health_check.py --parallel --json-output health_report.json'
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'health_report.json', allowEmptyArchive: true
        }
    }
}
```

### Exit Codes

The health check scripts use standardized exit codes for CI/CD integration:

- **0**: All checks passed successfully
- **1**: One or more critical services failed (default behavior)
- **2**: Invalid configuration or missing dependencies
- **3**: Report generation or file I/O failure

### Configuration Options

#### Timeout Configuration
```bash
# Short timeout for fast feedback
--timeout 10

# Long timeout for thorough checks
--timeout 60
```

#### Service Selection
```bash
# Check only critical services
--services ollama,dataservices

# Check only monitoring stack
--services monitoring

# Check everything except vector databases
--services ollama,gateway,dataservices,monitoring
```

#### Failure Behavior
```bash
# Fail only on critical service failures (default)
--fail-on-critical

# Fail on any service failure
--fail-on-any
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama/TinyLlama Issues
```bash
# Check if model is loaded
curl http://localhost:10104/api/tags

# Load TinyLlama model if missing
docker exec sutazai-ollama ollama pull tinyllama
```

#### 2. PostgreSQL Connection Issues
```bash
# Check container status
docker-compose logs postgres

# Verify port mapping
docker-compose ps postgres
```

#### 3. ChromaDB Connection Issues (Expected)
According to CLAUDE.md, ChromaDB has known connection issues. This is expected behavior and should not fail critical checks.

#### 4. Monitoring Services Not Responding
```bash
# Check Prometheus targets
curl http://localhost:10200/api/v1/targets

# Verify Grafana login
curl http://localhost:10201/login
```

#### 5. Timeout Issues
```bash
# Increase timeout for slow systems
python scripts/devops/infrastructure_health_check.py --timeout 60

# Use sequential mode if parallel causes issues
python scripts/devops/infrastructure_health_check.py --timeout 30
```

### Debugging Commands

#### View detailed logs
```bash
python scripts/devops/infrastructure_health_check.py --verbose --services ollama
```

#### Test individual scripts
```bash
python scripts/devops/health_check_ollama.py --verbose --timeout 10
python scripts/devops/health_check_dataservices.py --verbose
```

#### Generate detailed report
```bash
python scripts/devops/infrastructure_health_check.py \
  --json-output detailed_report.json \
  --verbose
```

## Best Practices

### Development Environment
- Run health checks before committing major changes
- Use `--verbose` flag for debugging issues
- Focus on critical services for rapid feedback

### Staging Environment  
- Run comprehensive checks including all service groups
- Use parallel execution for faster results
- Generate JSON reports for trend analysis

### Production Environment
- Run health checks after deployments
- Monitor critical services more frequently
- Set up alerts based on health check failures
- Use `--fail-on-critical` to avoid false positives from non-critical services

### Performance Optimization
- Use `--parallel` flag for faster execution
- Adjust `--timeout` based on system performance
- Consider running critical and non-critical checks separately

## Integration with Existing Scripts

The new health verification system is designed to complement and eventually replace the existing basic scripts:

### Legacy Scripts (Still Available)
- `scripts/devops/check_services_health.sh`: Basic bash script for simple TCP checks
- `scripts/devops/check_services_health.py`: Basic Python script with limited functionality

### Migration Path
1. **Phase 1**: Use both legacy and new scripts in parallel
2. **Phase 2**: Gradually replace legacy script usage with new orchestrator
3. **Phase 3**: Deprecate legacy scripts once new system is fully validated

### Related Utilities
- `scripts/register_with_consul.py`: Service registration (idempotent)
- `scripts/configure_kong.sh`: Kong service/route configuration (idempotent) 
- `scripts/monitoring/`: Enhanced monitoring and alerting scripts

## System Requirements

### Dependencies
- Python 3.7+
- Standard library modules only (no external dependencies)
- Docker and docker-compose for service management

### Permissions
- Network access to all monitored service ports
- Read access to script directory
- Write access for JSON report generation (if used)

### Resource Usage
-   CPU and memory footprint
- Network bandwidth depends on number of services checked
- Parallel mode uses one thread per service group

## Future Enhancements

### Planned Features
- Integration with Prometheus metrics export
- Webhook notifications for health check failures
- Historical health data storage and trending
- Service dependency graph validation
- Auto-remediation for common failure scenarios

### Monitoring Integration
The health check system is designed to integrate with the existing monitoring stack:
- Health check results can be exported as Prometheus metrics
- Grafana dashboards can display health check trends
- AlertManager can route health check failure notifications

## Security Considerations

### Network Security
- Health checks only perform read-only operations
- No authentication credentials are stored in scripts
- All network connections use standard protocols (HTTP, TCP)

### Data Privacy
- Health check logs contain no sensitive information
- JSON reports include only operational metadata
- No user data or business logic is exposed

### Access Control
- Scripts require only network access to service endpoints
- No elevated privileges or root access required
- Integration with existing security frameworks is maintained

## Support and Maintenance

### Documentation
- This README provides comprehensive usage information
- Individual scripts include built-in help (`--help` flag)
- CLAUDE.md contains authoritative system truth information

### Updates and Changes
- All changes must follow the 19 comprehensive codebase rules
- Updates are logged in CHANGELOG.md
- Backward compatibility is maintained for CI/CD integration

### Contact and Issues
- Report issues through standard project channels
- Include verbose logs and JSON reports when reporting problems
- Follow existing project contribution guidelines

---

**Note**: This health verification system is based on the CLAUDE.md truth document and reflects the actual operational status of services as of the last system verification (August 6, 2025). Service expectations and known issues are documented according to real system behavior, not aspirational documentation.