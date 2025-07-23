# SutazAI Monitoring Dashboard

## Overview
The `monitor_dashboard.sh` script provides a comprehensive real-time monitoring dashboard for the SutazAI system. It displays system metrics, container health, API performance, and more in a clean terminal UI with color coding.

## Features

### 1. Real-time System Metrics
- **CPU Usage**: Shows current CPU utilization with color-coded progress bar
- **Memory Usage**: Displays memory usage percentage and actual values (used/total)
- **Disk Usage**: Shows disk space utilization for the root partition
- **Load Average**: System load average with core count

### 2. Container Health Status
Monitors all SutazAI Docker containers:
- PostgreSQL Database
- Redis Cache
- Ollama Model Server
- ChromaDB Vector Store
- Qdrant Vector Database
- Backend API
- Frontend UI

For each container, displays:
- Running status (Running/Stopped/Not Found)
- Health check status (Healthy/Unhealthy/Starting)
- CPU and Memory usage when running

### 3. API Response Times and Errors
Tracks API performance for:
- Backend API Health endpoint
- Frontend status
- Agents API
- Models API
- Ollama API

Shows HTTP status codes and response times for each endpoint.

### 4. Ollama Model Performance
- Total number of models available
- Number of currently running models
- Name of the currently loaded model
- Model inference performance metrics

### 5. Vector Database Statistics
Displays statistics for:
- **ChromaDB**: Number of collections
- **Qdrant**: Number of collections and points

### 6. Active Agent Status
- Shows count of active AI agents
- Monitors agent resource usage
- Tracks agent health status

### 7. Critical Issue Alerting
Automatic alerts for:
- CPU usage > 80% (warning) or > 90% (critical)
- Memory usage > 80% (warning) or > 90% (critical)
- Disk usage > 80% (warning) or > 90% (critical)
- API failures
- Container health issues

### 8. Metrics History Logging
- Saves all metrics to JSON format
- Maintains 24-hour rolling history
- Enables historical analysis and trending

## Usage

### Basic Usage
```bash
cd /opt/sutazaiapp
./scripts/monitor_dashboard.sh
```

### Running in Background
```bash
nohup ./scripts/monitor_dashboard.sh > /dev/null 2>&1 &
```

### Viewing Logs
Metrics are saved to:
- Real-time logs: `/opt/sutazaiapp/logs/monitoring/dashboard_metrics_[timestamp].log`
- Historical metrics: `/opt/sutazaiapp/logs/monitoring/metrics_history.json`

## Terminal UI

The dashboard uses a clean, color-coded terminal interface:

- 🟢 **Green**: Healthy/Normal status
- 🟡 **Yellow**: Warning conditions
- 🔴 **Red**: Critical issues
- ⚫ **Gray**: Inactive/Not found

Progress bars visually represent resource usage:
```
CPU Usage:    ████░░░░░░░░░░░░░░░░░░░░░░░░  15.2%
Memory:       ████████████░░░░░░░░░░░░░░░░  40.5% (9.4Gi / 23Gi)
```

## Update Interval

By default, the dashboard updates every 5 seconds. This can be modified by editing the `UPDATE_INTERVAL` variable in the script.

## Requirements

- Bash 4.0+
- Docker (for container monitoring)
- curl (for API checks)
- jq (for JSON parsing)
- bc (for calculations)
- Standard Linux utilities (free, df, top, etc.)

## Exit

Press `Ctrl+C` to exit the dashboard cleanly. The cursor will be restored and final metrics will be saved.

## Troubleshooting

### Dashboard shows "N/A" for containers
- Ensure Docker is installed and running
- Check if containers are named correctly (sutazai-*)
- Verify Docker permissions

### API checks fail
- Ensure services are running on expected ports
- Check firewall settings
- Verify curl is installed

### No Ollama model information
- Ensure Ollama is running on port 11434
- Check if models are downloaded
- Verify Ollama API is accessible

## Integration with Other Scripts

The dashboard can be used alongside:
- `health_check.sh`: For detailed health analysis
- `monitor_system.sh`: For long-term system monitoring
- `service_monitor.sh`: For automatic service restarts

## Customization

You can customize the dashboard by modifying:
- Color schemes (see color code variables)
- Alert thresholds (CPU, memory, disk percentages)
- Container names to monitor
- API endpoints to check
- Update interval