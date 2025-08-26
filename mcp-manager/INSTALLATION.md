# MCP Manager Installation Guide

Complete installation and setup instructions for the Dynamic MCP Management System.

## Quick Installation

```bash
cd /opt/sutazaiapp/mcp-manager
./scripts/install.sh
```

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: 3.10 or higher
- **Memory**: Minimum 512MB RAM, recommended 2GB+
- **Disk Space**: 500MB for installation, additional space for logs and state

### Dependencies
- Python 3.10+
- pip (Python package manager)
- git (recommended for development)

### Verification
```bash
# Check Python version
python3 --version

# Check pip
pip3 --version

# Check available memory
free -h  # Linux
vm_stat | head -5  # macOS
```

## Installation Methods

### Method 1: Automatic Installation (Recommended)

```bash
# Basic installation
cd /opt/sutazaiapp/mcp-manager
./scripts/install.sh

# Development installation with all features
./scripts/install.sh --dev --systemd --test
```

### Method 2: Manual Installation

```bash
# 1. Create directories
sudo mkdir -p /var/log/mcp-manager /var/lib/mcp-manager
sudo chown -R $USER:$USER /var/log/mcp-manager /var/lib/mcp-manager

# 2. Create virtual environment
cd /opt/sutazaiapp/mcp-manager
python3 -m venv venv
source venv/bin/activate

# 3. Install package
pip install --upgrade pip
pip install -e .

# 4. Copy configuration
cp config/default.yaml config/config.yaml

# 5. Test installation
mcp-manager --version
```

### Method 3: Docker Installation (Future)

```bash
# Pull image (when available)
docker pull sutazai/mcp-manager:latest

# Run container
docker run -d --name mcp-manager \
  -p 8080:8080 \
  -v /opt/sutazaiapp/.mcp:/config \
  sutazai/mcp-manager:latest
```

## Installation Options

### Development Installation

Includes additional tools for development and debugging:

```bash
./scripts/install.sh --dev
```

Additional packages installed:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- pytest-asyncio (async testing)
- black (code formatting)
- isort (import sorting)
- mypy (type checking)
- ruff (linting)

### Systemd Service Installation

Creates a systemd service for automatic startup:

```bash
./scripts/install.sh --systemd
```

Service management:
```bash
# Enable service
sudo systemctl enable mcp-manager

# Start service
sudo systemctl start mcp-manager

# Check status
sudo systemctl status mcp-manager

# View logs
journalctl -u mcp-manager -f
```

## Configuration

### Initial Configuration

1. **Copy default configuration**:
   ```bash
   cp config/default.yaml config/config.yaml
   ```

2. **Edit configuration**:
   ```bash
   nano config/config.yaml
   ```

3. **Key settings to review**:
   - `discovery.config_directories`: Directories to scan for MCP servers
   - `health_monitoring.global_health_check_interval`: Health check frequency
   - `storage.log_level`: Logging verbosity
   - `api.port`: Management API port

### Directory Structure After Installation

```
/opt/sutazaiapp/mcp-manager/
├── venv/                    # Virtual environment
├── src/mcp_manager/         # Source code
├── config/
│   ├── default.yaml         # Default configuration
│   ├── config.yaml          # Your configuration
│   └── mcp-servers.json     # Example server definitions
├── scripts/
│   ├── install.sh           # Installation script
│   └── fixed-task-runner.sh # Fixed task runner
├── tests/                   # Test suite
└── README.md               # Documentation

/var/log/mcp-manager/        # Log files
└── mcp-manager.log

/var/lib/mcp-manager/        # State and data
└── state.json
```

## Post-Installation Setup

### 1. Verify Installation

```bash
# Check version
mcp-manager --version

# Check system status
mcp-manager status

# Test discovery
mcp-manager discover
```

### 2. Fix Claude Task Runner

The installation automatically fixes the claude-task-runner to use the official MCP SDK:

```bash
# Test the fixed task runner
python -m mcp_manager.fixed_task_runner &
# Should start without FastMCP errors

# Test through manager
mcp-manager start-server claude-task-runner
```

### 3. Configure MCP Servers

Add your MCP servers to the configuration:

```yaml
# config/config.yaml
servers:
  my-custom-server:
    description: "My custom MCP server"
    command: "python"
    args: ["-m", "my_server"]
    connection_type: "stdio"
    server_type: "python"
    enabled: true
```

Or use the existing `.mcp.json` format:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_server"],
      "type": "stdio"
    }
  }
}
```

### 4. Start Services

```bash
# Start all enabled servers
mcp-manager start

# Or start as daemon
mcp-manager start --daemon
```

## Troubleshooting Installation

### Common Issues

#### 1. Python Version Issues
```bash
# Error: Python 3.10+ required
# Solution: Install newer Python
sudo apt update && sudo apt install python3.11 python3.11-venv
# Or on macOS:
brew install python@3.11
```

#### 2. Permission Issues
```bash
# Error: Permission denied creating directories
# Solution: Check ownership
sudo chown -R $USER:$USER /opt/sutazaiapp/mcp-manager
sudo chown -R $USER:$USER /var/log/mcp-manager /var/lib/mcp-manager
```

#### 3. Virtual Environment Issues
```bash
# Error: Cannot create virtual environment
# Solution: Install venv module
sudo apt install python3-venv  # Ubuntu/Debian
brew install python@3.11       # macOS
```

#### 4. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Activate virtual environment
source /opt/sutazaiapp/mcp-manager/venv/bin/activate
pip install -e .
```

#### 5. FastMCP Compatibility Issues
```bash
# Error: FastMCP API changes
# Solution: Use the fixed task runner
mcp-manager start-server claude-task-runner
# The installation script automatically configures this
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Environment variable
export MCP_MANAGER_LOG_LEVEL=DEBUG
mcp-manager start

# Or in configuration
echo "log_level: DEBUG" >> config/config.yaml
```

### Log Analysis

```bash
# View recent logs
tail -f /var/log/mcp-manager/mcp-manager.log

# Search for errors
grep -i error /var/log/mcp-manager/mcp-manager.log

# View systemd logs (if using systemd)
journalctl -u mcp-manager -f --since "1 hour ago"
```

### Reinstallation

If you need to reinstall:

```bash
# 1. Stop services
mcp-manager stop
sudo systemctl stop mcp-manager  # If using systemd

# 2. Remove virtual environment
rm -rf /opt/sutazaiapp/mcp-manager/venv

# 3. Clean state (optional)
rm -f /var/lib/mcp-manager/state.json

# 4. Reinstall
./scripts/install.sh
```

## Verification Tests

### Basic Functionality Test

```bash
# 1. Check installation
mcp-manager --version

# 2. Test discovery
mcp-manager discover

# 3. Check status
mcp-manager status

# 4. Test server management
mcp-manager list-servers
mcp-manager start-server claude-task-runner
mcp-manager health --server claude-task-runner
mcp-manager stop-server claude-task-runner
```

### Integration Test

```bash
# Run full integration test
cd /opt/sutazaiapp/mcp-manager
source venv/bin/activate
pytest tests/ -v
```

### Performance Test

```bash
# Start multiple servers simultaneously
mcp-manager start

# Check resource usage
htop
ps aux | grep mcp

# Monitor connections
mcp-manager status --detailed
```

## Upgrade Instructions

### Upgrading MCP Manager

```bash
# 1. Stop services
mcp-manager stop

# 2. Pull latest code (if using git)
cd /opt/sutazaiapp/mcp-manager
git pull origin main

# 3. Upgrade dependencies
source venv/bin/activate
pip install --upgrade -e .

# 4. Restart services
mcp-manager start
```

### Upgrading Python

```bash
# If upgrading Python version
# 1. Stop services
mcp-manager stop

# 2. Remove old virtual environment
rm -rf venv

# 3. Reinstall with new Python
./scripts/install.sh

# 4. Restore configuration
# (Configuration files are preserved)
```

## Uninstallation

### Complete Removal

```bash
# 1. Stop all services
mcp-manager stop
sudo systemctl stop mcp-manager
sudo systemctl disable mcp-manager

# 2. Remove systemd service
sudo rm -f /etc/systemd/system/mcp-manager.service
sudo systemctl daemon-reload

# 3. Remove installation
rm -rf /opt/sutazaiapp/mcp-manager

# 4. Remove data (optional)
sudo rm -rf /var/log/mcp-manager
sudo rm -rf /var/lib/mcp-manager

# 5. Remove shell completion (optional)
# Edit ~/.bashrc and ~/.zshrc to remove mcp-manager lines
```

### Keep Configuration

```bash
# To keep configuration for future use
mv config/config.yaml ~/mcp-manager-config-backup.yaml
```

## Support

If you encounter issues during installation:

1. **Check logs**: `/var/log/mcp-manager/mcp-manager.log`
2. **Run debug mode**: `MCP_MANAGER_LOG_LEVEL=DEBUG mcp-manager start`
3. **Check requirements**: Verify Python version and dependencies
4. **Review configuration**: Ensure paths and permissions are correct
5. **Test manually**: Try running MCP servers directly

For additional support:
- 📧 Email: admin@sutazai.com
- 🐛 Issues: GitHub Issues
- 📖 Documentation: README.md