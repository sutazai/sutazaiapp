# SutazAI Deployment Verification Guide

**Testing QA Validator Agent** | Version 1.0.0

## Overview

This guide provides comprehensive documentation for the SutazAI deployment verification system, which includes multiple tools to ensure your deployment is healthy and functioning correctly.

## Verification Tools

### 1. Quick Deployment Check (Shell-based)
**File:** `scripts/quick_deployment_check.sh`

A fast, lightweight verification script that uses shell commands to check basic system health.

**Features:**
- Docker container status
- Port connectivity checks
- Basic HTTP endpoint validation
- System resource monitoring
- Quick model inference test
- Minimal dependencies (only requires `nc`, `curl`, `jq`)

**Usage:**
```bash
./scripts/quick_deployment_check.sh
```

### 2. Comprehensive Deployment Verification (Python-based)
**File:** `scripts/comprehensive_deployment_verification.py`

An advanced verification system with detailed testing capabilities.

**Features:**
- Detailed service health checks
- Complete API endpoint validation
- Agent communication testing
- Database connectivity verification
- Ollama model validation and inference testing
- Resource usage monitoring with thresholds
- Comprehensive JSON reporting
- Async operations for better performance

**Requirements:**
- Python 3.8+
- See `scripts/requirements-verification.txt`

**Usage:**
```bash
# Install dependencies first
pip install -r scripts/requirements-verification.txt

# Run verification
python3 scripts/comprehensive_deployment_verification.py
```

### 3. Verification Runner (Unified Interface)
**File:** `scripts/run_deployment_verification.sh`

A unified interface that can run either or both verification tools.

**Usage:**
```bash
# Run both quick and comprehensive (default)
./scripts/run_deployment_verification.sh

# Run only quick verification
./scripts/run_deployment_verification.sh --quick

# Run only comprehensive verification
./scripts/run_deployment_verification.sh --full

# Install dependencies only
./scripts/run_deployment_verification.sh --install

# Show help
./scripts/run_deployment_verification.sh --help
```

## Configuration

### Verification Configuration
**File:** `config/deployment_verification.yaml`

Customize verification behavior including:
- Service definitions and timeouts
- Resource usage thresholds
- API endpoint tests
- Database connection parameters
- Model inference settings
- Report generation options

### Key Configuration Sections

#### Resource Thresholds
```yaml
thresholds:
  cpu_warning: 80
  cpu_critical: 95
  memory_warning: 85
  memory_critical: 95
  success_rate_good: 90
  success_rate_acceptable: 80
```

#### Service Categories
- **Core Infrastructure:** PostgreSQL, Redis, Neo4j
- **Vector Databases:** ChromaDB, Qdrant, FAISS
- **Application:** Backend, Frontend, Ollama
- **AI Services:** LiteLLM, LangFlow, Flowise, Dify
- **Monitoring:** Prometheus, Grafana, Loki
- **Workflow:** n8n

#### Agent Categories
- **Autonomous:** AutoGPT, AgentGPT, AgentZero
- **Collaborative:** CrewAI, Letta
- **Coding:** Aider, GPT-Engineer
- **Specialized:** PrivateGPT, PentestGPT

## Verification Process

### 1. Docker Container Checks
- Verifies all SutazAI containers are running
- Reports container status and health
- Identifies failed or missing containers

### 2. Service Health Validation
- Tests port connectivity
- Validates HTTP health endpoints
- Checks service responsiveness
- Categorizes services by importance

### 3. API Endpoint Testing
- Core backend endpoints
- Interactive endpoints (chat, reasoning)
- External service APIs
- Model management endpoints

### 4. Agent Communication Testing
- Verifies agent containers are running
- Tests agent health endpoints
- Validates agent responsiveness

### 5. Database Connectivity
- **PostgreSQL:** Connection and version check
- **Redis:** Connection and info retrieval
- **Neo4j:** Connection and component status

### 6. Ollama Model Verification
- Lists available models
- Tests model inference capability
- Validates model loading status
- Performance testing with sample prompts

### 7. Resource Usage Monitoring
- CPU usage and load average
- Memory consumption and availability
- Disk space utilization
- Docker container resource usage

## Exit Codes

The verification scripts return standardized exit codes:

- **0:** Success (≥80% checks passed)
- **1:** Warning (60-79% checks passed)
- **2:** Critical (<60% checks passed)
- **3:** Error (script execution failed)
- **130:** Interrupted by user

## Report Formats

### Console Output
Real-time colored output with:
- ✅ Success indicators
- ❌ Failure indicators
- ⚠️ Warning indicators
- 📊 Summary statistics

### JSON Reports
Detailed JSON reports saved to `logs/` directory:
```json
{
  "overall": {
    "status": "EXCELLENT",
    "score": 95.2,
    "timestamp": "2024-01-20T10:30:00Z"
  },
  "services": { /* service details */ },
  "agents": { /* agent details */ },
  "databases": { /* database details */ },
  "models": { /* model details */ },
  "resources": { /* resource usage */ }
}
```

## Troubleshooting

### Common Issues

#### 1. Docker Services Not Running
```bash
# Check Docker daemon
sudo systemctl status docker

# Restart SutazAI services
docker-compose down && docker-compose up -d
```

#### 2. Port Conflicts
```bash
# Check port usage
netstat -tlnp | grep :8000

# Kill processes using required ports
sudo fuser -k 8000/tcp
```

#### 3. Ollama Models Not Loaded
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull required models
docker exec sutazai-ollama ollama pull llama3.2:1b
```

#### 4. Database Connection Issues
```bash
# Check database logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs neo4j
```

#### 5. High Resource Usage
```bash
# Check resource usage
docker stats

# Restart resource-intensive services
docker-compose restart ollama
```

### Performance Optimization

#### For Low-Resource Systems
1. Reduce concurrent model loading in Ollama
2. Use smaller models (1B-3B parameters)
3. Increase verification timeouts
4. Disable non-essential services

#### Configuration Adjustments
```yaml
# In deployment_verification.yaml
global:
  timeout: 60  # Increase timeouts
  
thresholds:
  cpu_warning: 90  # Adjust thresholds
  memory_warning: 90
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Verify Deployment
  run: |
    ./scripts/run_deployment_verification.sh --full
  timeout-minutes: 10
```

### Jenkins Pipeline
```groovy
stage('Deployment Verification') {
    steps {
        sh './scripts/run_deployment_verification.sh --both'
    }
    post {
        always {
            archiveArtifacts 'logs/deployment_*.json'
        }
    }
}
```

## Monitoring Integration

### Prometheus Metrics
The verification scripts can expose metrics for Prometheus monitoring:
- Deployment health score
- Service availability percentages
- Resource usage trends
- Model inference performance

### Grafana Dashboards
Create dashboards using verification metrics:
- Overall system health
- Service status matrix
- Resource usage trends
- Agent communication status

## Customization

### Adding New Services
1. Edit `config/deployment_verification.yaml`
2. Add service definition with container, port, and health URL
3. Specify timeout and requirement level

### Custom Health Checks
Extend the verification scripts to add:
- Application-specific validations
- Business logic tests
- Performance benchmarks
- Security scans

### Report Customization
- Modify report templates
- Add custom metrics
- Integrate with external systems
- Create custom visualizations

## Best Practices

### Regular Verification
- Run verification after each deployment
- Schedule periodic health checks
- Monitor verification trends
- Set up alerting for failures

### Documentation
- Document custom configurations
- Maintain service inventories
- Update thresholds based on usage
- Share verification results with teams

### Security
- Secure verification credentials
- Limit verification script permissions
- Protect sensitive configuration data
- Audit verification access

## Support

For issues with the deployment verification system:

1. Check the logs in `/opt/sutazaiapp/logs/`
2. Review the configuration in `config/deployment_verification.yaml`
3. Run with increased verbosity for debugging
4. Consult the SutazAI documentation

## Version History

- **1.0.0:** Initial release with comprehensive verification capabilities
  - Shell-based quick verification
  - Python-based comprehensive verification
  - Unified runner interface
  - YAML-based configuration
  - JSON reporting system