# Dynamic MCP Management System

A comprehensive orchestrator for Model Context Protocol (MCP) servers providing centralized management, health monitoring, and unified interfaces.

## Features

### 🚀 Core Capabilities
- **Dynamic Server Discovery**: Automatic discovery from `.mcp.json`, `pyproject.toml`, `package.json`, and YAML configs
- **Lifecycle Management**: Start, stop, restart, and monitor MCP servers with robust error handling
- **Health Monitoring**: Continuous health checks with automatic failure detection and recovery
- **Unified Interface**: Single API for all MCP operations with intelligent routing and load balancing
- **Connection Management**: Support for STDIO, HTTP, WebSocket, and TCP connections with retry logic

### 🔧 Management Features
- **Configuration Validation**: Comprehensive validation of server configurations
- **State Persistence**: Save and restore system state across restarts
- **Performance Monitoring**: Track request metrics, response times, and resource usage
- **Auto-Recovery**: Automatic restart of failed servers with configurable thresholds
- **Concurrent Operations**: Parallel server startup and health checking with rate limiting

### 🛠 Developer Experience
- **Rich CLI**: Comprehensive command-line interface with colored output and progress bars
- **Real-time Monitoring**: Live status updates and health dashboards
- **Detailed Logging**: Structured logging with configurable levels and file output
- **JSON API**: RESTful API for programmatic access and integration

## Quick Start

### Installation

```bash
# Install the MCP Manager
cd /opt/sutazaiapp/mcp-manager
pip install -e .

# Install additional dependencies for development
pip install -e .[dev]
```

### Basic Usage

```bash
# Start the MCP Manager (discovers and starts all enabled servers)
mcp-manager start

# Check system status
mcp-manager status

# List all servers
mcp-manager list-servers

# Check server health
mcp-manager health

# Discover new servers
mcp-manager discover

# Start a specific server
mcp-manager start-server claude-task-runner

# List available capabilities
mcp-manager capabilities
```

### Configuration

The system automatically discovers servers from:

1. **Official MCP configs**: `.mcp.json` files
2. **Python projects**: `pyproject.toml` with MCP server definitions
3. **Node.js projects**: `package.json` with MCP scripts
4. **YAML configs**: Custom `mcp-config.yaml` files
5. **Executable scripts**: Shell scripts in `scripts/mcp/` directories

Example `.mcp.json`:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_mcp_server"],
      "type": "stdio",
      "description": "My custom MCP server",
      "enabled": true
    }
  }
}
```

## Architecture

### Components

1. **MCPManager**: Central orchestrator coordinating all components
2. **ConnectionManager**: Handles connections using official Python MCP SDK
3. **ServerDiscoveryEngine**: Discovers servers from configuration files
4. **HealthMonitor**: Monitors server health and triggers recovery
5. **UnifiedMCPInterface**: Single interface for all MCP operations

### Connection Types

- **STDIO**: Standard input/output (most common)
- **HTTP**: RESTful API servers
- **WebSocket**: Real-time bidirectional communication
- **TCP**: Direct socket connections

### Health Monitoring

The system continuously monitors server health:

- ✅ **Healthy**: Server responding normally
- ⚠️ **Degraded**: Server responding but slowly
- ❌ **Unhealthy**: Server not responding
- 🚨 **Critical**: Server crashed or unreachable

## Fixed Task Runner

### Problem Solved

The original claude-task-runner used FastMCP which had compatibility issues. We've created a fixed version using the official Python MCP SDK:

**Before (FastMCP)**:
```python
from fastmcp import FastMCP  # Deprecated/problematic

mcp = FastMCP(name="Task Runner")
mcp.add_tool(handler_function)  # API changed
```

**After (Official SDK)**:
```python
from mcp.server import Server  # Official SDK
import mcp.server.stdio

server = Server("claude-task-runner")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    # Proper tool definitions

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]):
    # Proper tool handling
```

### Usage

```bash
# Use the fixed task runner directly
python -m mcp_manager.fixed_task_runner

# Or through the MCP Manager
mcp-manager start-server claude-task-runner
```

## Configuration Reference

### Server Configuration

```yaml
servers:
  my-server:
    # Basic information
    description: "My MCP server"
    command: "python"
    args: ["-m", "my_server"]
    connection_type: "stdio"  # stdio, http, websocket, tcp
    server_type: "python"     # python, node, go, rust, shell
    
    # Runtime settings
    working_directory: "/path/to/server"
    environment:
      ENV_VAR: "value"
    
    # Connection settings
    host: "localhost"         # For HTTP/WebSocket
    port: 3000               # For HTTP/WebSocket
    
    # Timeouts and limits
    startup_timeout: 30.0
    request_timeout: 60.0
    max_retries: 3
    retry_delay: 1.0
    
    # Health monitoring
    health_check_interval: 30.0
    health_check_timeout: 10.0
    failure_threshold: 3
    recovery_threshold: 2
    
    # Auto-restart settings
    auto_restart: true
    restart_delay: 5.0
    max_restart_attempts: 5
    
    # Performance limits
    max_concurrent_requests: 10
    request_rate_limit: 100  # requests per second
    
    # Metadata
    tags: ["production", "ai"]
    priority: 100
    enabled: true
```

### Manager Configuration

```yaml
# Discovery settings
discovery:
  config_directories:
    - "/opt/sutazaiapp/.mcp"
    - "~/.mcp"
  auto_discovery: true
  discovery_interval: 60.0

# Health monitoring
health_monitoring:
  global_health_check_interval: 30.0
  max_concurrent_health_checks: 5

# Recovery settings
recovery:
  global_auto_recovery: true
  recovery_check_interval: 10.0

# Performance
performance:
  max_startup_concurrency: 3

# Storage
storage:
  state_file: "/tmp/mcp_manager_state.json"
  log_level: "INFO"
  log_file: "/var/log/mcp-manager.log"

# API
api:
  enabled: true
  host: "localhost"
  port: 8080
```

## CLI Reference

### Server Management
```bash
mcp-manager start [--config PATH] [--daemon] [--no-monitoring]
mcp-manager stop
mcp-manager status [--detailed] [--json]
mcp-manager restart
```

### Individual Server Control
```bash
mcp-manager start-server <name>
mcp-manager stop-server <name>
mcp-manager restart-server <name>
mcp-manager list-servers [--status STATUS] [--config]
```

### Discovery and Configuration
```bash
mcp-manager discover [--rescan]
mcp-manager capabilities [--server NAME] [--json]
```

### Health and Monitoring
```bash
mcp-manager health [--server NAME] [--watch]
```

## Development

### Project Structure
```
mcp-manager/
├── src/mcp_manager/          # Main package
│   ├── __init__.py
│   ├── manager.py            # Main orchestrator
│   ├── connection.py         # Connection management
│   ├── discovery.py          # Server discovery
│   ├── health.py            # Health monitoring
│   ├── interface.py         # Unified interface
│   ├── models.py            # Data models
│   ├── cli.py               # Command line interface
│   └── fixed_task_runner.py # Fixed task runner
├── config/                   # Configuration files
├── tests/                   # Test suite
└── docs/                    # Documentation
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_manager

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Security

### Best Practices

- **Input Validation**: All server configurations are validated using Pydantic
- **Process Isolation**: Servers run in separate processes
- **Resource Limits**: Configurable memory and CPU limits
- **SSL/TLS**: Support for secure connections
- **Authentication**: Token-based authentication for HTTP servers

### Security Considerations

- Server commands are executed with current user privileges
- Configuration files should be protected with appropriate permissions
- Network-based servers should use authentication and encryption
- Regular security updates of dependencies are recommended

## Performance

### Optimization Features

- **Parallel Operations**: Concurrent server startup and health checks
- **Connection Pooling**: Efficient connection reuse
- **Request Caching**: Configurable response caching
- **Load Balancing**: Intelligent request routing
- **Resource Monitoring**: Track CPU and memory usage

### Scaling

The system is designed to handle:
- 50+ concurrent MCP servers
- 1000+ requests per second
- Multiple connection types simultaneously
- Automatic failover and recovery

## Troubleshooting

### Common Issues

1. **Server Won't Start**
   ```bash
   # Check server configuration
   mcp-manager list-servers --config
   
   # Check logs
   tail -f /var/log/mcp-manager.log
   
   # Test server command manually
   python -m my_server
   ```

2. **Health Check Failures**
   ```bash
   # Check specific server health
   mcp-manager health --server my-server
   
   # Increase timeout in configuration
   health_check_timeout: 30.0
   ```

3. **Discovery Issues**
   ```bash
   # Force rediscovery
   mcp-manager discover --rescan
   
   # Check configuration directories
   ls -la /opt/sutazaiapp/.mcp/
   ```

### Debug Mode

```bash
# Enable debug logging
export MCP_MANAGER_LOG_LEVEL=DEBUG
mcp-manager start

# Or in configuration
log_level: "DEBUG"
```

## License

MIT License - see LICENSE file for details.

## Support

- 📧 Email: admin@sutazai.com
- 🐛 Issues: GitHub Issues
- 📖 Documentation: [mcp-manager.sutazai.com](https://mcp-manager.sutazai.com)
- 💬 Discussions: GitHub Discussions