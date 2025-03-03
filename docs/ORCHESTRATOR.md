# Supreme AI Orchestrator Documentation

## Overview

The Supreme AI Orchestrator is a distributed system that manages AI agents and handles synchronization between primary and secondary servers. It provides robust task management, agent coordination, and failover capabilities.

## Architecture

### Components

1. **Supreme AI Orchestrator**
   - Core orchestration logic
   - Task distribution
   - Agent management
   - Server synchronization

2. **Agent Manager**
   - Agent lifecycle management
   - Health monitoring
   - Task assignment
   - Resource allocation

3. **Sync Manager**
   - Server synchronization
   - Data consistency
   - Conflict resolution
   - Failover handling

4. **Task Queue**
   - Priority-based task scheduling
   - Task status tracking
   - Queue management
   - Task retry logic

### System Flow

1. Tasks are submitted to the orchestrator
2. Tasks are prioritized and queued
3. Available agents are assigned tasks
4. Task results are synchronized between servers
5. System state is maintained consistently

## Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 14 or higher
- Redis 6 or higher
- Systemd-based Linux system

### Installation Steps

1. Clone the repository:
   ```bash
   cd /opt
   git clone https://github.com/sutazai/sutazaiapp.git
   cd sutazaiapp
   ```

2. Run the installation script:
   ```bash
   sudo ./scripts/install_orchestrator.sh
   ```

3. Verify the installation:
   ```bash
   systemctl status supreme-ai-orchestrator
   ```

## Configuration

### Main Configuration File

The main configuration file is located at `config/orchestrator.toml`. Key settings include:

```toml
[primary_server]
host = "localhost"
port = 8000

[secondary_server]
host = "localhost"
port = 8010

[orchestrator]
sync_interval = 60
max_agents = 10
task_timeout = 3600
```

### SSL Certificates

SSL certificates are stored in `config/certs/`:
- `server.crt`: Server certificate
- `server.key`: Private key

## Usage

### Service Management

Start the service:
```bash
sudo systemctl start supreme-ai-orchestrator
```

Stop the service:
```bash
sudo systemctl stop supreme-ai-orchestrator
```

Check status:
```bash
sudo systemctl status supreme-ai-orchestrator
```

View logs:
```bash
sudo journalctl -u supreme-ai-orchestrator
```

### Manual Control

The orchestrator can be managed manually using the management script:

```bash
./scripts/manage_orchestrator.sh {start|stop|restart|status|logs}
```

## Monitoring

### Log Files

- Main log: `/opt/sutazaiapp/logs/orchestrator.log`
- Error log: `/opt/sutazaiapp/logs/orchestrator.error.log`

### Metrics

Prometheus metrics are exposed on port 9090:
- Agent status
- Task queue size
- Sync status
- System health

## Security

### Authentication

- API endpoints require authentication
- JWT tokens are used for authorization
- SSL/TLS encryption for all communications

### File Permissions

- Configuration files: 640
- SSL certificates: 600
- Log files: 644
- Executables: 750

## Troubleshooting

### Common Issues

1. Service fails to start:
   - Check log files
   - Verify permissions
   - Ensure dependencies are installed

2. Synchronization fails:
   - Check network connectivity
   - Verify server configurations
   - Check SSL certificates

3. Agents not responding:
   - Check agent processes
   - Verify network connectivity
   - Check resource usage

### Debug Mode

Enable debug logging in `config/orchestrator.toml`:
```toml
[logging]
level = "DEBUG"
```

## Development

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

### Testing

Run tests:
```bash
python -m pytest tests/
```

### Code Style

Follow PEP 8 guidelines and use provided linting configuration:
```bash
pylint core_system/orchestrator/
```

## API Reference

### Task Management

```python
# Submit task
POST /api/v1/tasks
{
    "type": "text_processing",
    "parameters": {...},
    "priority": 1
}

# Get task status
GET /api/v1/tasks/{task_id}

# Cancel task
DELETE /api/v1/tasks/{task_id}
```

### Agent Management

```python
# Register agent
POST /api/v1/agents
{
    "type": "text_processing",
    "capabilities": [...]
}

# Update agent status
PUT /api/v1/agents/{agent_id}
{
    "status": "IDLE"
}
```

### System Status

```python
# Get system status
GET /api/v1/status

# Get metrics
GET /metrics
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License

Copyright © 2024 SutazAI. All rights reserved. 