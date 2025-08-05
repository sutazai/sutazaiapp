# Automation System Controller

## Overview

The Automation System Controller manages and coordinates system operations through automated monitoring and resource management. It implements concrete, production-ready automation capabilities to maintain system reliability and performance.

### Core Capabilities

- **System Monitoring**: Continuous monitoring of system metrics and health indicators
- **Resource Management**: Dynamic resource allocation based on actual usage patterns
- **State Management**: Maintains system state and configuration consistency
- **Fault Detection**: Identifies and logs system anomalies and failures
- **Performance Optimization**: Adjusts system parameters based on operational metrics
- **Load Balancing**: Distributes workload across available resources

## Quick Start

### Prerequisites
- Docker Engine 24.0+
- Python 3.9+
- PostgreSQL 15+
- Redis 7.0+

### 1. Configuration
```bash
# Set required environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start Services
```bash
# Start required services
docker-compose up -d redis postgres

# Start the controller
./scripts/start-controller.sh
```

### 3. Verify Installation
```bash
# Check controller status
curl http://localhost:8090/health

# View metrics
curl http://localhost:8090/metrics

# Run verification tests
./scripts/verify-installation.sh
```

## System Architecture

```
┌─────────────────────────────────────────┐
│         System Controller                │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────┐  ┌──────────────────┐   │
│  │ Metrics   │  │ Health Check      │   │
│  │ Collector │  │ Service           │   │
│  └───────────┘  └──────────────────┘   │
│                                         │
│  ┌───────────┐  ┌──────────────────┐   │
│  │ Resource  │  │ Configuration     │   │
│  │ Monitor   │  │ Manager          │   │
│  └───────────┘  └──────────────────┘   │
│                                         │
│  ┌───────────┐  ┌──────────────────┐   │
│  │ Alert     │  │ Load Balancer     │   │
│  │ Manager   │  │                   │   │
│  └───────────┘  └──────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│             Managed Services             │
└─────────────────────────────────────────┘
```

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/health` | GET | Health check endpoint |
| `/v1/metrics` | GET | System resource metrics |
| `/v1/config` | GET | Current configuration |
| `/v1/services` | GET | Service status list |
| `/v1/alerts` | GET | Active system alerts |
| `/v1/control` | POST | Control operations |
| `/v1/maintenance` | POST | Maintenance mode |
| `/v1/shutdown` | POST | Graceful shutdown |

### Example Usage

```bash
# Check system health
curl http://localhost:8090/v1/health

# Get current metrics
curl http://localhost:8090/v1/metrics

# Enable maintenance mode
curl -X POST http://localhost:8090/v1/maintenance \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "reason": "Scheduled maintenance"}'

# Graceful shutdown
curl -X POST http://localhost:8090/v1/shutdown \
  -H "Content-Type: application/json" \
  -d '{"reason": "Planned downtime"}'
```

## 🧠 Autonomous Capabilities

### 1. Self-Monitoring
- Collects metrics every 5 seconds
- Detects anomalies using statistical analysis
- Tracks system health score in real-time

### 2. Decision Making
- **Resource Allocation**: Optimally distributes resources
- **Auto-scaling**: Scales up/down based on load
- **Error Recovery**: Automatically recovers from failures
- **Performance Optimization**: Tunes system for best performance

### 3. Learning & Adaptation
- Learns from successful/failed decisions
- Identifies patterns in system behavior
- Improves strategies over time
- Shares knowledge across the system

### 4. Safety Mechanisms
- Validates all decisions against constraints
- Provides rollback plans for every action
- Allows human override at any time
- Maintains complete audit trail

## 📊 Monitoring Dashboard

The monitoring dashboard (`monitor_autonomous_controller.py`) provides:

- Real-time system state
- Resource usage graphs
- Recent decisions with impacts
- Learning patterns
- Active alerts

## 🔧 Configuration

Edit `agents/autonomous-system-controller/configs/autonomous_controller.yaml`:

```yaml
# Key settings
monitoring_interval: 5  # seconds
decision_threshold: 0.7  # minimum confidence
emergency_cpu_threshold: 95
emergency_memory_threshold: 90

safety_constraints:
  max_scaling_factor: 10
  min_resource_reserve: 0.1
  max_error_rate: 0.05
  max_response_time: 5000  # ms
```

## 🚨 Emergency Procedures

### Emergency Stop
```bash
curl -X POST http://localhost:8090/emergency/stop
```

### Manual Override
```bash
curl -X POST http://localhost:8090/override \
  -d '{"action": "pause_all", "reason": "Emergency maintenance"}'
```

### View Logs
```bash
docker logs -f sutazai-autonomous-system-controller
```

## 📈 Performance Metrics

The controller tracks:
- **CPU Usage**: Target < 80%
- **Memory Usage**: Target < 85%
- **Error Rate**: Target < 0.05
- **Response Time**: Target < 5000ms
- **Decision Success Rate**: Target > 90%

## 🔄 System States

| State | Description | Color |
|-------|-------------|-------|
| HEALTHY | System operating normally | Green |
| WARNING | Minor issues detected | Yellow |
| CRITICAL | Major issues, intervention may be needed | Orange |
| EMERGENCY | System in emergency mode | Red |
| RECOVERING | Recovering from failure | Yellow |
| MAINTENANCE | In maintenance mode | Blue |
| SHUTDOWN | System shutting down | Gray |

## 🛠️ Troubleshooting

### Controller Not Starting
```bash
# Check logs
docker logs sutazai-autonomous-system-controller

# Check dependencies
docker ps | grep -E "redis|postgres"

# Restart
docker-compose -f docker-compose-autonomous-controller.yml restart
```

### High Resource Usage
1. Check current metrics: `curl http://localhost:8090/metrics`
2. Review recent decisions: `curl http://localhost:8090/decisions`
3. Adjust thresholds in configuration
4. Manual override if needed

### No Autonomous Decisions
1. Check decision threshold in config
2. Verify metrics collection is working
3. Review safety constraints
4. Check system logs for errors

## 🚀 Advanced Features

### Predictive Scaling
The controller predicts future load and scales proactively:
- Analyzes historical patterns
- Forecasts resource needs
- Scales before bottlenecks occur

### Cost Optimization
Automatically optimizes for cost:
- Identifies underutilized resources
- Suggests consolidation opportunities
- Balances performance vs cost

### Strategic Planning
Plans for long-term system evolution:
- Capacity planning for growth
- Technology refresh cycles
- Architecture improvements

## 📝 Best Practices

1. **Start Conservative**: Begin with high safety thresholds
2. **Monitor Decisions**: Review autonomous decisions regularly
3. **Learn Patterns**: Allow time for ML to learn your workload
4. **Test Overrides**: Regularly test manual override procedures
5. **Review Logs**: Check logs for optimization opportunities

## 🔗 Integration

The controller integrates with:
- **Agent Registry**: Monitors all AI agents
- **Message Bus**: Coordinates system-wide events
- **Redis**: Fast state storage
- **PostgreSQL**: Persistent decision history
- **Prometheus**: Metrics collection
- **Coordinator System**: High-level coordination

## 📚 Further Reading

- [Architecture Documentation](docs/AUTONOMOUS_SYSTEM_CONTROLLER.md)
- [API Reference](docs/api/autonomous_controller_api.md)
- [Safety Guidelines](docs/safety/autonomous_safety.md)
- [ML Patterns](docs/ml/learning_patterns.md)

---

**Remember**: The automation system Controller is designed to make your life easier by handling routine operations automatically. Trust it to manage your system, but always maintain oversight through monitoring and regular reviews of its decisions.