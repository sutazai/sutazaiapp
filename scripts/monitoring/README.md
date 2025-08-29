# Monitoring Scripts

Monitoring and health check scripts for SutazaiApp services.

## Scripts in this directory:

### comprehensive_system_audit.sh
Performs a complete system audit including resource usage, service health, and configuration validation.
```bash
./comprehensive_system_audit.sh
```

### ollama-health-fix.sh
Fixes Ollama service health issues and memory allocation problems.
```bash
./ollama-health-fix.sh
```

### fix-unhealthy-services.sh
Automated script to detect and fix unhealthy Docker services.
```bash
./fix-unhealthy-services.sh
```

### health-monitor-daemon.sh
Background daemon for continuous health monitoring.
```bash
./health-monitor-daemon.sh
```

### fix-ollama-semgrep.sh
Specific fixes for Ollama and Semgrep service health issues.
```bash
./fix-ollama-semgrep.sh
```

### live_logs.sh
Real-time aggregated log viewer for all services.
```bash
./live_logs.sh live  # Live streaming
./live_logs.sh tail  # Last 100 lines
```

## Health Check Endpoints

| Service | Health Check URL |
|---------|-----------------|
| Backend | http://localhost:10200/health |
| Frontend | http://localhost:11000/_stcore/health |
| MCP Bridge | http://localhost:11100/health |
| Ollama | http://localhost:11435/api/tags |

## Monitoring Commands

```bash
# Check all service health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Monitor resource usage
docker stats --no-stream

# Check specific service logs
docker logs [service-name] --tail 100 -f
```