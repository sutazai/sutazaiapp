# SutazAI Chaos Engineering Framework - Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Framework Components](#framework-components)
3. [Running Experiments](#running-experiments)
4. [Chaos Monkey](#chaos-monkey)
5. [Monitoring and Dashboards](#monitoring-and-dashboards)
6. [Safety Features](#safety-features)
7. [Integration with SutazAI](#integration-with-sutazai)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

## Quick Start

### 1. Initialize the Framework

```bash
# Navigate to chaos directory
cd /opt/sutazaiapp/chaos

# Initialize chaos engineering framework
sudo ./scripts/init-chaos.sh

# Verify installation
./scripts/run-experiment.sh --help
```

### 2. Run Your First Experiment

```bash
# Run a basic container chaos experiment in safe mode
./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Check experiment results
ls -la reports/

# View system health
python3 scripts/resilience-tester.py --test-type recovery --services sutazai-autogpt
```

### 3. Enable Chaos Monkey

```bash
# Start Chaos Monkey in safe mode
python3 scripts/chaos-monkey.py --mode safe --daemon

# Check Chaos Monkey status
python3 scripts/chaos-monkey.py --status
```

## Framework Components

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `init-chaos.sh` | Initialize framework | `./scripts/init-chaos.sh` |
| `run-experiment.sh` | Execute experiments | `./scripts/run-experiment.sh --experiment NAME` |
| `chaos-engine.py` | Core experiment engine | `python3 scripts/chaos-engine.py --experiment NAME` |
| `chaos-monkey.py` | Automated chaos monkey | `python3 scripts/chaos-monkey.py --daemon` |
| `resilience-tester.py` | Test resilience | `python3 scripts/resilience-tester.py` |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/chaos-config.yaml` | Main configuration |
| `experiments/*.yaml` | Experiment definitions |
| `monitoring/*.yml` | Monitoring configs |

## Running Experiments

### Available Experiments

#### 1. Basic Container Chaos
Tests container failure and recovery patterns.

```bash
# Safe mode (recommended for first time)
./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Production mode (maintenance window only)
./scripts/run-experiment.sh --experiment basic-container-chaos
```

**What it does:**
- Randomly kills containers
- Measures recovery time
- Validates service restoration
- Tests health check effectiveness

#### 2. Network Chaos
Tests network resilience and dependency handling.

```bash
# Run network latency and partition tests
./scripts/run-experiment.sh --experiment network-chaos --safe-mode
```

**What it does:**
- Injects network latency (100ms)
- Simulates packet loss (5%)
- Tests bandwidth limitations
- Creates temporary network partitions

#### 3. Resource Stress
Tests system behavior under resource pressure.

```bash
# Stress CPU, memory, and disk
./scripts/run-experiment.sh --experiment resource-stress --safe-mode
```

**What it does:**
- CPU stress testing (80% utilization)
- Memory pressure testing (75% usage)
- Disk I/O stress testing
- Combined resource exhaustion

### Experiment Parameters

Each experiment can be customized with:

- **Duration**: How long the experiment runs
- **Target Services**: Which services to affect
- **Intensity**: Severity of the chaos injection
- **Recovery Timeout**: Maximum time to wait for recovery

### Dry Run Mode

Test experiments without actually executing them:

```bash
# See what would happen without doing it
./scripts/run-experiment.sh --experiment basic-container-chaos --dry-run
```

## Chaos Monkey

### Starting Chaos Monkey

```bash
# Start in safe mode (recommended)
python3 scripts/chaos-monkey.py --mode safe --daemon

# Start in production mode (maintenance windows only)
python3 scripts/chaos-monkey.py --mode production --daemon

# Start with custom config
python3 scripts/chaos-monkey.py --config /path/to/config.yaml --daemon
```

### Chaos Monkey Modes

| Mode | Description | Safety Level | Scheduling |
|------|-------------|--------------|------------|
| `safe` | Limited impact, non-critical services only | High | Anytime |
| `production` | Full experiments during maintenance windows | Medium | 2-4 AM Mon/Wed/Fri |
| `aggressive` | Frequent experiments, higher impact | Low | Continuous |
| `disabled` | No experiments | Maximum | None |

### Monitoring Chaos Monkey

```bash
# Check current status
python3 scripts/chaos-monkey.py --status

# View recent activity
tail -f /opt/sutazaiapp/logs/chaos_monkey.log

# Check state file
cat /opt/sutazaiapp/chaos/chaos_monkey_state.json
```

### Emergency Controls

```bash
# Emergency stop all chaos activities
python3 scripts/chaos-monkey.py --emergency-stop

# Graceful stop
python3 scripts/chaos-monkey.py --stop
```

## Monitoring and Dashboards

### Grafana Dashboard

Access the chaos engineering dashboard at:
- **URL**: `http://localhost:3000`
- **Dashboard**: "SutazAI Chaos Engineering"
- **Username**: `admin`
- **Password**: Check `/opt/sutazaiapp/secrets/grafana_password.txt`

### Key Metrics

#### System Health Metrics
- **Service Availability**: Percentage of services healthy
- **Response Times**: API response time during experiments
- **Error Rates**: Increased errors during chaos
- **Recovery Times**: How fast services recover

#### Experiment Metrics
- **Success Rate**: Percentage of successful experiments
- **Impact Score**: Severity of chaos impact
- **Cascade Depth**: How many services affected
- **Experiment Duration**: Time to complete experiments

#### Chaos Monkey Metrics
- **Experiments Executed**: Total count
- **Target Distribution**: Which services targeted
- **Schedule Adherence**: Timing accuracy
- **Safety Violations**: Any safety breaches

### Prometheus Queries

```promql
# Experiment success rate over time
rate(chaos_experiments_success_total[5m]) / rate(chaos_experiments_total[5m]) * 100

# Mean recovery time
avg_over_time(chaos_recovery_time_seconds[1h])

# Service health during experiments
avg by (container_name) (up{container_name=~"sutazai-.*"}) * 100

# Chaos impact correlation
(100 - avg(up{job=~".*sutazai.*"}) * 100) * on() group_left() (chaos_experiments_active > 0)
```

### Log Analysis

```bash
# View chaos experiment logs
tail -f /opt/sutazaiapp/logs/chaos.log

# Search for specific experiment
grep "basic-container-chaos" /opt/sutazaiapp/logs/chaos.log

# Monitor resilience test results
grep "recovery_time" /opt/sutazaiapp/logs/resilience_test.log
```

## Safety Features

### Built-in Safety Mechanisms

#### 1. Protected Services
These services are never targeted by chaos experiments:
- `sutazai-postgres` (Critical database)
- `sutazai-prometheus` (Required for monitoring)
- `sutazai-grafana` (Required for dashboards)
- `sutazai-health-monitor` (System health tracking)

#### 2. System Health Checks
Before any experiment:
- ✅ Minimum 80% of services must be healthy
- ✅ CPU usage below 85%
- ✅ Memory usage below 90%
- ✅ No ongoing incidents

#### 3. Maintenance Windows
Production experiments only run during:
- **Days**: Monday, Wednesday, Friday
- **Time**: 2:00 AM - 4:00 AM UTC
- **Override**: Use `--safe-mode` to run anytime

#### 4. Emergency Stop
Multiple ways to stop chaos:
```bash
# Method 1: Emergency stop command
python3 scripts/chaos-monkey.py --emergency-stop

# Method 2: Kill process signal
kill -TERM $(pgrep -f chaos-monkey)

# Method 3: Disable in config
# Set mode: disabled in chaos-config.yaml
```

#### 5. Automatic Recovery
- Failed experiments trigger automatic recovery
- Services are restarted if needed
- Health checks validate recovery
- Cascading failures are contained

### Safety Overrides

```bash
# Run outside maintenance window (safe mode only)
./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Check system safety before experiment
python3 scripts/chaos-engine.py --experiment basic-container-chaos --dry-run
```

## Integration with SutazAI

### Docker Compose Integration

The chaos framework integrates seamlessly with SutazAI's Docker infrastructure:

```bash
# Deploy chaos services alongside SutazAI
cd /opt/sutazaiapp
docker-compose -f docker-compose.yml -f chaos/docker-compose.chaos.yml up -d

# Add chaos labels to existing services
python3 chaos/scripts/docker-integration.py --add-labels

# Validate integration
python3 chaos/scripts/docker-integration.py --validate
```

### Service Dependencies

The framework understands SutazAI service dependencies:

```
Backend → [Postgres, Redis, Ollama]
Frontend → [Backend]
Agents → [Backend, Ollama]
Monitoring → [Prometheus]
```

This enables:
- **Smart targeting**: Avoid critical path disruption
- **Cascade detection**: Monitor downstream effects
- **Recovery validation**: Ensure full system restoration

### Health Check Integration

Leverages existing SutazAI health checks:
- Container health status
- Service endpoint availability
- Application-specific metrics
- Custom health indicators

## Troubleshooting

### Common Issues

#### 1. Experiment Fails to Start

**Symptoms**: Experiment exits immediately
**Causes**: 
- System health too low
- Protected service targeted
- Outside maintenance window

**Solutions**:
```bash
# Check system health
python3 scripts/resilience-tester.py --test-type recovery --services sutazai-backend

# Use safe mode
./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Check logs
tail -f /opt/sutazaiapp/logs/chaos.log
```

#### 2. Services Don't Recover

**Symptoms**: Services remain unhealthy after experiment
**Causes**: 
- Docker restart policy issues
- Resource exhaustion
- Dependency failures

**Solutions**:
```bash
# Manual service restart
docker restart sutazai-SERVICE_NAME

# Check container logs
docker logs sutazai-SERVICE_NAME

# Verify health checks
docker ps --format "table {{.Names}}\t{{.Status}}"

# Run recovery script
python3 scripts/auto-recovery.py
```

#### 3. Chaos Monkey Won't Start

**Symptoms**: Chaos Monkey process exits
**Causes**: 
- Configuration errors
- Permission issues
- Missing dependencies

**Solutions**:
```bash
# Check configuration
python3 -c "import yaml; yaml.safe_load(open('config/chaos-config.yaml'))"

# Verify permissions
ls -la /var/run/docker.sock

# Install dependencies
pip3 install -r requirements.txt

# Check logs
tail -f /opt/sutazaiapp/logs/chaos_monkey.log
```

#### 4. No Metrics in Grafana

**Symptoms**: Empty chaos dashboard
**Causes**: 
- Prometheus not scraping chaos metrics
- Dashboard not imported
- Network connectivity issues

**Solutions**:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify chaos metrics endpoint
curl http://localhost:8200/metrics

# Import dashboard manually
# Copy from chaos/monitoring/grafana-chaos-dashboard.json

# Check Prometheus configuration
grep -A 10 "chaos-" /opt/sutazaiapp/monitoring/prometheus/prometheus.yml
```

### Debug Mode

Enable detailed logging:

```bash
# Run with debug logging
LOG_LEVEL=DEBUG ./scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Enable chaos engine debug
python3 scripts/chaos-engine.py --experiment basic-container-chaos --safe-mode --log-level DEBUG
```

### Validation Commands

```bash
# Validate complete framework
./scripts/init-chaos.sh

# Test experiment definitions
find experiments/ -name "*.yaml" -exec python3 -c "import yaml; yaml.safe_load(open('{}'))" \;

# Check Docker integration
python3 scripts/docker-integration.py --validate

# Verify monitoring setup
python3 scripts/monitoring-integration.py --validate
```

## Advanced Configuration

### Custom Experiments

Create new experiment definitions:

```yaml
# experiments/custom-experiment.yaml
apiVersion: chaos.sutazai.com/v1
kind: ChaosExperiment
metadata:
  name: custom-experiment
  description: "Custom chaos experiment"
spec:
  duration: "15m"
  schedule: "manual"
  safety:
    safe_mode: true
    max_affected_services: 1
  
  targets:
    - service: "sutazai-my-service"
      weight: 1.0
  
  scenarios:
    - name: "custom_scenario"
      probability: 0.5
      actions:
        - type: "custom_action"
          parameters:
            intensity: "medium"
            duration: "5m"
```

### Scheduling Configuration

Customize Chaos Monkey schedules:

```yaml
# In chaos-config.yaml
chaos_monkey:
  schedules:
    - name: "custom-schedule"
      experiment_type: "custom-experiment"
      frequency: "daily"
      schedule_pattern: "03:00"  # 3:00 AM
      target_services: ["sutazai-my-service"]
      safe_mode: true
      enabled: true
      probability: 0.2
```

### Integration Hooks

Add custom integrations:

```python
# Custom monitoring integration
class CustomMonitoring:
    def __init__(self):
        self.webhook_url = os.getenv('CUSTOM_WEBHOOK_URL')
    
    def send_alert(self, experiment_result):
        # Custom alerting logic
        pass
```

### Performance Tuning

Optimize for your environment:

```yaml
# In chaos-config.yaml
global:
  max_concurrent_experiments: 1  # Reduce for limited resources
  experiment_timeout: "30m"      # Increase for slow recovery
  health_check_interval: "60s"   # Adjust monitoring frequency

safety:
  min_healthy_services: 0.9      # Increase safety threshold
  max_cpu_usage: 70             # Reduce for constrained systems
```

---

## Support and Contributing

### Getting Help

1. **Check logs**: Always start with `/opt/sutazaiapp/logs/chaos*.log`
2. **Review configuration**: Validate YAML syntax and parameters
3. **Test incrementally**: Start with safe mode and simple experiments
4. **Monitor system**: Use Grafana dashboard for insights

### Best Practices

1. **Start Small**: Begin with safe mode and non-critical services
2. **Monitor Continuously**: Watch dashboards during experiments
3. **Document Results**: Keep track of findings and improvements
4. **Regular Testing**: Run experiments consistently
5. **Team Coordination**: Inform team of scheduled experiments

### Framework Evolution

The chaos engineering framework is designed to evolve with your needs:

- **Custom experiments** for specific scenarios
- **Enhanced monitoring** for deeper insights
- **Advanced scheduling** for complex environments
- **Integration hooks** for external systems

---

**Remember**: Chaos engineering is about building confidence in your system's resilience. Start conservatively, learn from each experiment, and gradually increase sophistication as your system and team mature.