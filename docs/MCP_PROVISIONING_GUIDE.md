# MCP Server Suite Provisioning Guide

## Overview

The MCP Server Suite Provisioning Script (`/opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh`) is a comprehensive automation tool that installs all dependencies and ensures every MCP (Model Context Protocol) server starts correctly without manual intervention.

## Features

✅ **Automatic Dependency Installation**
- System dependencies (Docker, Python, Node.js, Go, uv)
- Programming language runtimes and package managers
- Required npm packages and Python modules

✅ **Complete MCP Server Setup**
- 17 different MCP servers across multiple technologies
- Go-based servers (language-server)
- Python-based servers (UltimateCoder, extended-memory)
- Node.js-based servers (GitHub, files, context7, etc.)
- Docker-based servers (postgres, duckduckgo)

✅ **Intelligent Operation**
- Idempotent (safe to run multiple times)
- Detects existing installations
- Cross-platform support (Linux, macOS)
- Comprehensive error handling and logging

## Quick Start

### Basic Usage

```bash
# Run the provisioning script
/opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh
```

### Check Current Status

```bash
# Run MCP server self-checks
/opt/sutazaiapp/scripts/mcp/selfcheck_all.sh
```

## What Gets Installed

### System Dependencies

| Component | Purpose | Installation Method |
|-----------|---------|---------------------|
| **Docker** | Container runtime for postgres/ddg MCPs | Official install script |
| **Node.js 22.x** | Runtime for npm-based MCPs | NodeSource repository |
| **Python 3.12+** | Runtime for Python MCPs | System package manager |
| **Go 1.21+** | Runtime for Go-based MCPs | Official Go tarball |
| **uv** | Fast Python package manager | Official install script |

### MCP Servers by Category

#### 🔧 **Core Infrastructure (5 servers)**
- `language-server` - Code intelligence (Go binary)
- `files` - File system operations (npm)
- `postgres` - Database operations (Docker)
- `github` - GitHub integration (npm) 
- `http` - Web content fetching (Docker)

#### 🧠 **AI & Reasoning (3 servers)**
- `sequentialthinking` - Chain of thought reasoning (Docker)
- `context7` - Library documentation (npm)
- `extended-memory` - Persistent memory (Python)

#### 🔍 **Search & Data (4 servers)**
- `ddg` - DuckDuckGo web search (Docker)
- `nx-mcp` - Nx workspace support (npm)
- `knowledge-graph-mcp` - Graph operations (npm)
- `compass-mcp` - MCP discovery (npm)

#### 🔨 **Development Tools (3 servers)**
- `ultimatecoder` - Advanced file operations (Python)
- `mcp_ssh` - SSH operations (Python)
- `playwright-mcp` - Browser automation (npm)

#### 🎭 **Alternative Tools (2 servers)**
- `puppeteer-mcp` - Chrome automation (npm)
- `memory-bank-mcp` - Memory management (npm/Python)

## Script Execution Flow

### 1. System Analysis
```
✓ Detect operating system and architecture
✓ Check for root/sudo access
✓ Identify package manager (apt-get, yum, dnf, brew)
```

### 2. Dependency Installation
```
✓ Install system tools (curl, wget, jq, git)
✓ Install Docker (if not present)
✓ Install Python 3.12+ and pip
✓ Install Node.js 22.x and npm
✓ Install Go 1.21+
✓ Install uv Python package manager
```

### 3. MCP Server Setup
```
✓ Go servers: mcp-language-server + TypeScript LSP
✓ Python servers: Virtual environments + packages
✓ Node servers: Global npm package installation
✓ Docker servers: Image pulling + network setup
✓ Special servers: Repository cloning
```

### 4. Configuration & Validation
```
✓ Create environment files (.env)
✓ Set file permissions
✓ Run comprehensive self-checks
✓ Generate status report
```

## Output and Logging

### Console Output
The script provides real-time colored output:
- 🔵 **[INFO]** - General information
- 🟢 **[OK]** - Successful operations
- 🟡 **[WARN]** - Non-critical issues
- 🔴 **[ERROR]** - Critical failures

### Log Files
All operations are logged to timestamped files:
```
/opt/sutazaiapp/logs/mcp_provision_YYYYMMDD_HHMMSS.log
```

### Status Reports
Detailed reports are generated:
```
/opt/sutazaiapp/logs/mcp_provision_status_YYYYMMDD_HHMMSS.md
```

## Example Execution Results

### Successful Run (Existing Installation)
```
[INFO] MCP Server Suite Provisioning Started
[OK] Linux OS detected
[OK] Node.js already installed: v22.18.0
[OK] Docker already installed
[OK] All MCP server self-checks passed
[OK] MCP Server Suite Provisioning completed successfully!
Total errors: 0
```

### Fresh Installation
```
[INFO] Installing Node.js via NodeSource...
[INFO] Installing Go...
[INFO] Installing npm packages globally...
[INFO] Pulling Docker image: crystaldba/postgres-mcp
[OK] All dependencies installed and configured.
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors
```bash
# Run with sudo if needed
sudo /opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh
```

#### 2. Network Issues
- Check internet connectivity
- Verify Docker daemon is running
- Check npm registry access

#### 3. Package Conflicts
```bash
# Clear npm cache
npm cache clean --force

# Reset Python environments
rm -rf /opt/sutazaiapp/.venvs/extended-memory
```

### Validation Commands

```bash
# Check all system dependencies
docker --version && node --version && python3 --version && go version

# Verify MCP servers
/opt/sutazaiapp/scripts/mcp/selfcheck_all.sh

# Check Docker containers
docker ps | grep sutazai

# Test individual MCP wrapper
/opt/sutazaiapp/scripts/mcp/wrappers/files.sh --selfcheck
```

## Advanced Usage

### Environment Variables

```bash
# Customize installation paths
export PROJECT_ROOT="/custom/path"

# Skip certain components
export SKIP_DOCKER=1
export SKIP_PYTHON=1
```

### Selective Installation

The script is modular - you can comment out sections in the `main()` function to skip certain components:

```bash
# Edit the script to customize
vim /opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh

# Comment out unwanted sections:
# setup_docker_mcp_servers  # Skip Docker MCPs
# setup_nodejs_mcp_servers  # Skip npm MCPs
```

## Integration with Other Tools

### CI/CD Pipeline
```yaml
- name: Provision MCP Servers
  run: /opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh
```

### Development Workflow
```bash
# 1. Initial setup
./provision_mcps_suite.sh

# 2. Daily verification
./scripts/mcp/selfcheck_all.sh

# 3. After system updates
./provision_mcps_suite.sh  # Safe to re-run
```

## Security Considerations

- Script requires root/sudo for system package installation
- Downloads packages from official sources only
- Creates non-root users for container operations where possible
- All network communication uses HTTPS
- Environment variables are properly isolated

## Performance Notes

- **First run**: 5-10 minutes (downloads dependencies)
- **Subsequent runs**: 1-2 minutes (skips existing installations)
- **Disk space**: ~2GB for all dependencies and images
- **Network usage**: ~500MB for initial downloads

## Support and Maintenance

### Regular Maintenance
```bash
# Monthly: Update all MCP dependencies
/opt/sutazaiapp/scripts/deployment/provision_mcps_suite.sh

# Weekly: Verify all servers are working
/opt/sutazaiapp/scripts/mcp/selfcheck_all.sh
```

### Getting Help
1. Check the log files for detailed error information
2. Run individual MCP wrapper scripts with `--selfcheck`
3. Verify system requirements are met
4. Check network connectivity for downloads

---

## Quick Reference Card

| Command | Purpose |
|---------|---------|
| `./provision_mcps_suite.sh` | Install/update all MCP dependencies |
| `./scripts/mcp/selfcheck_all.sh` | Test all MCP servers |
| `./scripts/mcp/wrappers/[name].sh --selfcheck` | Test individual server |
| `tail -f logs/mcp_provision_*.log` | Monitor installation progress |
| `cat logs/mcp_provision_status_*.md` | View detailed status report |

**Remember**: The script is idempotent and safe to run multiple times! 🚀