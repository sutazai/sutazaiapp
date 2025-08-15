# MCP Automation System - Installation & Deployment Guide

**Version**: 3.0.0  
**Last Updated**: 2025-08-15 16:45:00 UTC  
**Target Audience**: System Administrators, DevOps Engineers

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Pre-Installation Checklist](#pre-installation-checklist)
4. [Installation Methods](#installation-methods)
5. [Step-by-Step Installation](#step-by-step-installation)
6. [Post-Installation Configuration](#post-installation-configuration)
7. [Deployment Strategies](#deployment-strategies)
8. [Verification & Testing](#verification--testing)
9. [Rollback Procedures](#rollback-procedures)
10. [Troubleshooting Installation](#troubleshooting-installation)

## Prerequisites

### Required Software

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|-------------------|--------|
| Docker | 20.0 | 24.0+ | Required for containerized services |
| Docker Compose | 2.0 | 2.20+ | For orchestration |
| Python | 3.11 | 3.11+ | Runtime environment |
| Git | 2.25 | 2.40+ | Version control |
| Make | 4.0 | 4.3+ | Build automation |
| curl | 7.68 | Latest | API testing |
| jq | 1.6 | Latest | JSON processing |

### Operating System Support

| OS | Version | Support Level | Notes |
|----|---------|--------------|--------|
| Ubuntu | 20.04 LTS, 22.04 LTS | Full | Primary development platform |
| Debian | 11, 12 | Full | Compatible with Ubuntu instructions |
| RHEL/CentOS | 8, 9 | Full | SELinux configuration required |
| macOS | 12+ | Full | Docker Desktop required |
| Windows | WSL2 | Partial | Requires WSL2 with Ubuntu |

### Network Requirements

- **Outbound Connectivity**: Required for pulling Docker images and MCP server updates
- **Internal Ports**: Range 10000-11000 must be available
- **Firewall Rules**: Configure as needed for your environment

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 4 cores (x86_64 or ARM64)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 100 Mbps

#### Recommended Configuration
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **Storage**: 100 GB NVMe SSD
- **Network**: 1 Gbps

#### Production Configuration
- **CPU**: 16+ cores
- **RAM**: 32 GB
- **Storage**: 500 GB NVMe SSD RAID 1
- **Network**: 10 Gbps redundant

### Storage Planning

```
/opt/sutazaiapp/
├── scripts/mcp/automation/  # 5 GB - Automation system
├── backups/                 # 20 GB - Backup storage
├── logs/                    # 10 GB - Log storage
├── staging/                 # 5 GB - Staging area
└── data/                    # 10 GB - Runtime data
```

## Pre-Installation Checklist

### System Preparation

- [ ] Operating system updated to latest patches
- [ ] Docker and Docker Compose installed
- [ ] Python 3.11+ installed with pip
- [ ] Git configured with credentials
- [ ] Sufficient disk space available (50GB minimum)
- [ ] Network connectivity verified
- [ ] Firewall rules configured
- [ ] SELinux/AppArmor configured (if applicable)
- [ ] Time synchronization (NTP) configured
- [ ] DNS resolution working

### Security Checklist

- [ ] Non-root user created for installation
- [ ] SSH key authentication configured
- [ ] Sudo privileges configured
- [ ] Security updates applied
- [ ] Antivirus exclusions configured (if applicable)

## Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Download installation script
curl -fsSL https://raw.githubusercontent.com/your-org/mcp-automation/main/install.sh -o install.sh

# Review the script
less install.sh

# Make executable and run
chmod +x install.sh
./install.sh --production
```

### Method 2: Manual Installation

Follow the step-by-step guide below for complete control over the installation process.

### Method 3: Container-Based Installation

```bash
# Pull and run installer container
docker run -it --rm \
  -v /opt/sutazaiapp:/opt/sutazaiapp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  mcp-automation/installer:latest
```

## Step-by-Step Installation

### Step 1: System Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install required system packages
sudo apt-get install -y \
  curl \
  wget \
  git \
  make \
  jq \
  htop \
  net-tools \
  software-properties-common \
  ca-certificates \
  gnupg \
  lsb-release

# Add Docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker --version
docker compose version
```

### Step 2: Python Environment Setup

```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Verify Python installation
python3 --version
pip3 --version
```

### Step 3: Create Directory Structure

```bash
# Create main directory
sudo mkdir -p /opt/sutazaiapp
sudo chown -R $USER:$USER /opt/sutazaiapp

# Create subdirectories
cd /opt/sutazaiapp
mkdir -p {scripts/mcp/automation,backups/mcp,staging/mcp,logs/mcp,data,config}

# Set permissions
chmod 755 scripts backups staging logs data config
```

### Step 4: Clone Repository

```bash
# Clone the repository (adjust URL as needed)
cd /opt/sutazaiapp
git clone https://github.com/your-org/sutazai.git .

# Checkout stable version
git checkout v3.0.0
```

### Step 5: Install MCP Automation System

```bash
# Navigate to automation directory
cd /opt/sutazaiapp/scripts/mcp/automation

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 6: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Set the following variables:
# MCP_CONFIG_PATH=/opt/sutazaiapp/.mcp.json
# MCP_BACKUP_DIR=/opt/sutazaiapp/backups/mcp
# MCP_STAGING_DIR=/opt/sutazaiapp/staging/mcp
# MCP_LOG_DIR=/opt/sutazaiapp/logs/mcp
# MCP_AUTO_UPDATE=false
# MCP_CLEANUP_ENABLED=true
# MCP_REQUIRE_AUTH=true

# Source environment
source .env
```

### Step 7: Initialize Database

```bash
# Initialize state database
python -m orchestration.state_manager --init-db

# Create initial configuration
python -m orchestration.state_manager --create-initial-state

# Verify database
python -m orchestration.state_manager --verify
```

### Step 8: Deploy Monitoring Stack

```bash
# Navigate to monitoring directory
cd /opt/sutazaiapp/scripts/mcp/automation/monitoring

# Deploy monitoring services
docker compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
sleep 30

# Verify monitoring services
docker compose -f docker-compose.monitoring.yml ps

# Check Prometheus
curl -s http://localhost:10200/-/healthy

# Check Grafana
curl -s http://localhost:10201/api/health
```

### Step 9: Start Core Services

```bash
# Return to automation directory
cd /opt/sutazaiapp/scripts/mcp/automation

# Start orchestrator
python -m orchestration.orchestrator --start --daemon

# Start health monitor
python -m monitoring.health_monitor --start --daemon

# Start cleanup scheduler
python -m cleanup.cleanup_scheduler --start --daemon

# Verify services
python -m orchestration.orchestrator --status
```

### Step 10: Verify MCP Servers

```bash
# Check MCP configuration
python -m mcp_update_manager --verify-config

# Test MCP server connectivity
python -m tests.test_mcp_health --quick-check

# List available MCP servers
python -m mcp_update_manager --list-servers
```

## Post-Installation Configuration

### Configure API Access

```bash
# Generate API token
python -m orchestration.api_gateway --generate-token

# Save token securely
echo "API_TOKEN=<generated-token>" >> ~/.mcp_credentials
chmod 600 ~/.mcp_credentials
```

### Configure Alerting

```bash
# Edit alert configuration
nano monitoring/config/alert_rules.yml

# Configure notification channels
python -m monitoring.alert_manager --configure-channels

# Test alerting
python -m monitoring.alert_manager --test-alert
```

### Configure Backup Schedule

```bash
# Set up automated backups
crontab -e

# Add the following line for daily backups at 2 AM
0 2 * * * /opt/sutazaiapp/scripts/mcp/automation/scripts/backup.sh
```

### Configure Log Rotation

```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/mcp-automation

# Add configuration:
/opt/sutazaiapp/logs/mcp/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 $USER $USER
    sharedscripts
    postrotate
        systemctl reload mcp-automation 2>/dev/null || true
    endscript
}
```

## Deployment Strategies

### Development Deployment

```bash
# Use development configuration
export MCP_ENV=development

# Start with verbose logging
python -m orchestration.orchestrator --start --log-level=DEBUG

# Enable hot reload
python -m orchestration.orchestrator --start --reload
```

### Staging Deployment

```bash
# Use staging configuration
export MCP_ENV=staging

# Deploy with limited resources
docker compose -f docker-compose.staging.yml up -d

# Run integration tests
python -m tests.test_mcp_integration --staging
```

### Production Deployment

```bash
# Use production configuration
export MCP_ENV=production

# Pre-deployment checks
./scripts/pre-deploy-check.sh

# Deploy with zero downtime
python -m orchestration.orchestrator --deploy --zero-downtime

# Post-deployment verification
python -m tests.test_mcp_health --production-check
```

### Blue-Green Deployment

```bash
# Prepare green environment
python -m orchestration.orchestrator --prepare-green

# Deploy to green
python -m orchestration.orchestrator --deploy-green

# Test green environment
python -m tests.test_mcp_integration --target=green

# Switch traffic to green
python -m orchestration.orchestrator --switch-to-green

# Cleanup blue environment (after verification)
python -m orchestration.orchestrator --cleanup-blue
```

## Verification & Testing

### System Health Check

```bash
# Run comprehensive health check
python -m monitoring.health_monitor --full-check

# Expected output:
# ✓ Database connection: OK
# ✓ MCP servers: 17/17 healthy
# ✓ Monitoring stack: OK
# ✓ API gateway: OK
# ✓ Cleanup service: OK
```

### Functional Testing

```bash
# Run test suite
pytest tests/ -v

# Run specific test categories
pytest tests/test_mcp_health.py -v
pytest tests/test_mcp_integration.py -v
pytest tests/test_mcp_performance.py -v

# Generate test report
pytest tests/ --html=report.html --self-contained-html
```

### Performance Testing

```bash
# Run performance benchmarks
python -m tests.test_mcp_performance --benchmark

# Load testing
python -m tests.test_mcp_performance --load-test --users=100 --duration=60
```

### Security Testing

```bash
# Security scan
python -m tests.test_mcp_security --full-scan

# Vulnerability assessment
bandit -r . -f json -o security-report.json
safety check --json > vulnerabilities.json
```

## Rollback Procedures

### Automated Rollback

```bash
# Rollback to previous version
python -m orchestration.orchestrator --rollback

# Rollback specific component
python -m mcp_update_manager --rollback <server-name>
```

### Manual Rollback

```bash
# Stop services
python -m orchestration.orchestrator --stop

# Restore from backup
python -m orchestration.state_manager --restore --backup-id=<backup-id>

# Restart services
python -m orchestration.orchestrator --start

# Verify rollback
python -m monitoring.health_monitor --verify-rollback
```

## Troubleshooting Installation

### Common Installation Issues

#### Docker Permission Denied

**Problem**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
sudo usermod -aG docker $USER
newgrp docker
# Log out and back in if necessary
```

#### Python Version Conflicts

**Problem**: Wrong Python version being used

**Solution**:
```bash
# Use explicit Python version
python3.11 -m venv venv
source venv/bin/activate
which python  # Should show venv path
```

#### Port Already in Use

**Problem**: `bind: address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :10200
# Kill process or change port in configuration
```

#### Insufficient Disk Space

**Problem**: `no space left on device`

**Solution**:
```bash
# Check disk usage
df -h
# Clean Docker resources
docker system prune -a
# Remove old logs
find /opt/sutazaiapp/logs -name "*.log" -mtime +30 -delete
```

#### Network Connectivity Issues

**Problem**: Cannot pull Docker images or updates

**Solution**:
```bash
# Check DNS
nslookup github.com
# Check proxy settings
echo $HTTP_PROXY
# Test connectivity
curl -I https://github.com
```

### Installation Validation Script

Create `validate_installation.sh`:

```bash
#!/bin/bash

echo "MCP Automation System - Installation Validation"
echo "================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Check Docker
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker installed: $(docker --version)"
else
    echo -e "${RED}✗${NC} Docker not found"
fi

# Check Python
if command -v python3.11 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python installed: $(python3.11 --version)"
else
    echo -e "${RED}✗${NC} Python 3.11 not found"
fi

# Check directories
for dir in /opt/sutazaiapp/{scripts,backups,logs,staging,data}; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} Directory exists: $dir"
    else
        echo -e "${RED}✗${NC} Directory missing: $dir"
    fi
done

# Check services
services=("prometheus" "grafana" "loki")
for service in "${services[@]}"; do
    if docker ps | grep -q $service; then
        echo -e "${GREEN}✓${NC} Service running: $service"
    else
        echo -e "${RED}✗${NC} Service not running: $service"
    fi
done

# Check API
if curl -s http://localhost:8080/api/v1/status > /dev/null; then
    echo -e "${GREEN}✓${NC} API responding"
else
    echo -e "${RED}✗${NC} API not responding"
fi

echo "================================================"
echo "Validation complete"
```

## Next Steps

After successful installation:

1. **Review Security Settings**: Check `SECURITY.md` for hardening guidelines
2. **Configure Monitoring**: Set up dashboards and alerts in Grafana
3. **Schedule Maintenance**: Configure automated backups and cleanup
4. **Documentation Review**: Read the API documentation and user guides
5. **Training**: Complete operator training modules

## Support

For installation support:
- Documentation: `/opt/sutazaiapp/scripts/mcp/automation/docs/`
- Issue Tracker: GitHub Issues
- Support Email: mcp-support@example.com
- Emergency Hotline: Available for production deployments

---

**Installation Guide Version**: 3.0.0  
**Last Updated**: 2025-08-15  
**Next Review**: 2025-09-15