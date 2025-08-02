# SutazAI automation system Controller

## 🚀 Overview

The automation system Controller is the master coordinator of the SutazAI automation system system. It operates 24/7 without human intervention, making intelligent decisions to keep your AI system running optimally.

### Key Features

- **🤖 Fully Autonomous**: Operates independently with self-healing capabilities
- **🧠 Machine Learning**: Learns from patterns and improves over time
- **⚡ Real-time Decisions**: Makes decisions in seconds based on system state
- **🛡️ Safety First**: Multiple safety mechanisms and human override capabilities
- **📊 Comprehensive Monitoring**: Tracks all metrics and decisions
- **🔄 Auto-scaling**: Scales resources based on demand predictions

## 🎯 Quick Start

### 1. Start the Controller

```bash
cd /opt/sutazaiapp
./scripts/start_autonomous_controller.sh
```

### 2. Monitor the System

```bash
# Real-time monitoring dashboard
python3 scripts/monitor_autonomous_controller.py

# Or check status via API
curl http://localhost:8090/status | python3 -m json.tool
```

### 3. Test the System

```bash
python3 tests/test_autonomous_controller.py
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│       automation system Controller       │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────┐  ┌──────────────────┐  │
│  │ Decision  │  │ Health Monitor    │  │
│  │ Engine    │  │ (5s interval)     │  │
│  └───────────┘  └──────────────────┘  │
│                                         │
│  ┌───────────┐  ┌──────────────────┐  │
│  │ Learning  │  │ Resource Manager  │  │
│  │ Engine    │  │                   │  │
│  └───────────┘  └──────────────────┘  │
│                                         │
│  ┌───────────┐  ┌──────────────────┐  │
│  │ Safety    │  │ Strategic Planner │  │
│  │ Controller│  │                   │  │
│  └───────────┘  └──────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   automation system System Agents │
        └─────────────────────┘
```

## 🎮 Control Interfaces

### Web API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `http://localhost:8090/health` | GET | System health check |
| `http://localhost:8090/status` | GET | Detailed system status |
| `http://localhost:8090/metrics` | GET | System metrics (CPU, memory, etc.) |
| `http://localhost:8090/decisions` | GET | Recent autonomous decisions |
| `http://localhost:8090/learning` | GET | Machine learning patterns |
| `http://localhost:8090/command` | POST | Send commands to controller |
| `http://localhost:8090/override` | POST | Manual override decisions |
| `http://localhost:8090/emergency/stop` | POST | Emergency shutdown |

### Example API Calls

```bash
# Check system status
curl http://localhost:8090/status

# Get last 20 metrics
curl http://localhost:8090/metrics?count=20

# Send a command
curl -X POST http://localhost:8090/command \
  -H "Content-Type: application/json" \
  -d '{"type": "optimize", "target": "memory"}'

# Manual override
curl -X POST http://localhost:8090/override \
  -H "Content-Type: application/json" \
  -d '{"action": "scale_down", "parameters": {"factor": 0.8}, "reason": "Reduce costs"}'
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