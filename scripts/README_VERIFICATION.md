# SutazAI Deployment Verification System

**Created by:** Testing QA Validator Agent  
**Version:** 1.0.0  
**Purpose:** Comprehensive deployment verification and health monitoring

## Quick Start

### Option 1: Run Everything (Recommended)
```bash
# Run both quick and comprehensive verification
./scripts/run_deployment_verification.sh
```

### Option 2: Quick Check Only
```bash
# Fast shell-based verification (minimal dependencies)
./scripts/quick_deployment_check.sh
```

### Option 3: Comprehensive Check Only
```bash
# Advanced Python-based verification (requires dependencies)
./scripts/run_deployment_verification.sh --full
```

## Files Overview

### Main Scripts
- **`run_deployment_verification.sh`** - Unified runner for all verification tools
- **`quick_deployment_check.sh`** - Fast shell-based verification
- **`comprehensive_deployment_verification.py`** - Advanced Python verification

### Configuration & Dependencies
- **`requirements-verification.txt`** - Python dependencies for comprehensive verification
- **`../config/deployment_verification.yaml`** - Configuration file for customization

### Documentation
- **`../docs/DEPLOYMENT_VERIFICATION_GUIDE.md`** - Complete user guide
- **`README_VERIFICATION.md`** - This file

## What Gets Verified

### 🐳 Docker Infrastructure
- Container status and health
- Service availability
- Container resource usage

### 🏥 Core Services
- **Databases:** PostgreSQL, Redis, Neo4j
- **Vector DBs:** ChromaDB, Qdrant, FAISS
- **Core App:** Backend API, Frontend UI
- **AI Services:** Ollama, LiteLLM

### 🤖 AI Agents
- **Autonomous:** AutoGPT, AgentGPT, AgentZero
- **Collaborative:** CrewAI, Letta
- **Coding:** Aider, GPT-Engineer
- **Specialized:** PrivateGPT, PentestGPT

### 🔗 API Endpoints
- Health endpoints
- Core functionality APIs
- Interactive endpoints (chat, reasoning)
- External service APIs

### 🗄️ Database Connectivity
- Connection testing
- Query execution
- Version verification

### 🧠 Model Validation
- Ollama model availability
- Model inference testing
- Performance validation

### 📊 Resource Monitoring
- CPU, Memory, Disk usage
- Load averages
- Container resource consumption
- Threshold-based alerting

## Exit Codes

- **0:** Success (≥80% checks passed)
- **1:** Warning (60-79% checks passed)  
- **2:** Critical (<60% checks passed)
- **3:** Error (script execution failed)
- **130:** Interrupted by user

## Dependencies

### Required (for comprehensive verification)
```bash
pip install aiohttp asyncpg redis psutil docker neo4j pyyaml
```

### Minimal (for quick verification)
- `bash`, `nc` (netcat), `curl`, `jq`, `docker`

### Automatic Installation
```bash
# Install dependencies automatically
./scripts/run_deployment_verification.sh --install
```

## Configuration

Edit `config/deployment_verification.yaml` to customize:
- Service definitions and timeouts
- Resource usage thresholds
- API endpoint tests
- Database connection parameters
- Report generation options

Example customization:
```yaml
thresholds:
  cpu_warning: 80
  memory_warning: 85
  success_rate_good: 90

services:
  backend:
    timeout: 20  # Increase timeout for slow systems
    required: true
```

## Output Examples

### Console Output
```
🔍 SutazAI Deployment Verification
===============================================================================

🐳 Docker Container Status
--------------------------------------------------
  ✅ sutazai-backend        running
  ✅ sutazai-frontend       running
  ❌ sutazai-neo4j         exited

📊 Container Summary: 8/10 running

🏥 Service Health Checks  
--------------------------------------------------
  ✅ PostgreSQL Database   healthy
  ✅ Backend API          healthy
  ❌ Neo4j Graph DB       unavailable

📊 DEPLOYMENT VERIFICATION SUMMARY
===============================================================================
Overall Status: GOOD (85%)
Verification Time: 45.2s
Checks Passed: 34/40
```

### JSON Report
```json
{
  "overall": {
    "status": "GOOD",
    "score": 85.0,
    "passed_checks": 34,
    "total_checks": 40,
    "timestamp": "2024-01-20T10:30:00Z",
    "verification_time": 45.2
  },
  "services": {
    "backend": {
      "status": "healthy",
      "port_open": true,
      "health_check": true
    }
  },
  "recommendations": [
    "System is functional with minor issues",
    "Review failed checks in log file"
  ]
}
```

## Troubleshooting

### Common Issues

#### No Docker containers running
```bash
docker-compose up -d
```

#### Python dependencies missing
```bash
./scripts/run_deployment_verification.sh --install
```

#### Permission denied
```bash
chmod +x scripts/*.sh
```

#### High resource usage warnings
- Check `docker stats` for resource-hungry containers
- Consider using smaller AI models
- Adjust thresholds in configuration

#### Model inference failures
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull a lightweight model
docker exec sutazai-ollama ollama pull gpt-oss.2:1b
```

### Log Files
Verification logs are saved to:
- Quick verification: `/opt/sutazaiapp/logs/quick_deployment_check_YYYYMMDD_HHMMSS.log`
- Comprehensive: `/opt/sutazaiapp/logs/deployment_verification_YYYYMMDD_HHMMSS.log`
- JSON reports: `/opt/sutazaiapp/logs/deployment_report_YYYYMMDD_HHMMSS.json`

## Integration Examples

### GitHub Actions
```yaml
- name: Verify Deployment
  run: ./scripts/run_deployment_verification.sh --full
  timeout-minutes: 10
```

### Cron Job (Daily Health Check)
```bash
# Add to crontab
0 6 * * * /opt/sutazaiapp/scripts/quick_deployment_check.sh
```

### Docker Healthcheck
```yaml
healthcheck:
  test: ["/opt/sutazaiapp/scripts/quick_deployment_check.sh"]
  interval: 5m
  timeout: 2m
  retries: 3
```

## Performance Tips

### For Low-Resource Systems
1. Use quick verification only: `--quick`
2. Increase timeouts in configuration
3. Disable non-essential services
4. Use smaller AI models

### For Production
1. Run comprehensive verification: `--full`
2. Set up monitoring integration
3. Schedule regular health checks
4. Configure alerting on failures

## Support

1. **Check logs:** Review verification logs in `/opt/sutazaiapp/logs/`
2. **Run with debug:** Add `-x` to shell scripts for verbose output
3. **Check configuration:** Verify `config/deployment_verification.yaml`
4. **Review documentation:** See `docs/DEPLOYMENT_VERIFICATION_GUIDE.md`

## Version History

- **1.0.0 (Current):** Initial comprehensive verification system
  - Shell-based quick verification
  - Python-based comprehensive verification  
  - Unified runner interface
  - YAML configuration system
  - JSON reporting with detailed metrics
  - Graceful dependency handling
  - Docker and service health monitoring
  - AI model validation and testing
  - Resource usage monitoring with thresholds
  - Complete API endpoint validation