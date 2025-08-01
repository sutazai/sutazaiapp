# 📚 SutazAI Scripts Guide

This guide provides comprehensive documentation for all essential scripts in the SutazAI system.

## 📋 Table of Contents

1. [Quick Reference](#quick-reference)
2. [System Management Scripts](#system-management-scripts)
3. [Deployment Scripts](#deployment-scripts)
4. [Monitoring Scripts](#monitoring-scripts)
5. [Agent Management Scripts](#agent-management-scripts)
6. [Utility Scripts](#utility-scripts)
7. [Database Scripts](#database-scripts)

---

## 🚀 Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `start.sh` | Quick start core system | Initial system startup |
| `status.sh` | Check system status | Regular health checks |
| `deploy_complete_system.sh` | Full AGI deployment | Production deployment |
| `agent_status_dashboard.sh` | Monitor all agents | Real-time monitoring |
| `monitor` | Live system monitoring | Continuous monitoring |
| `check_system_status.sh` | Detailed status check | Troubleshooting |
| `fix_docker_deployment_issues.sh` | Fix Docker issues | Docker problems |
| `validate_integrated_system.py` | Validate deployment | Post-deployment check |

---

## 🖥️ System Management Scripts

### start.sh
**Purpose:** Quick start script for core SutazAI system
**Usage:** `./scripts/start.sh`
**Description:** 
- Checks Docker availability
- Starts core services using docker-compose
- Displays access points for services

**Example:**
```bash
cd /opt/sutazaiapp
./scripts/start.sh
```

### status.sh
**Purpose:** Display current system status
**Usage:** `./scripts/status.sh`
**Description:**
- Shows running Docker containers
- Checks backend health endpoint
- Displays file statistics
- Shows disk usage

**Output:**
- Container status table
- Health check results
- File counts (Python files, Docker compose files)
- Disk usage for core and archive directories

### check_system_status.sh
**Purpose:** Comprehensive system status check
**Usage:** `./scripts/check_system_status.sh`
**Description:**
- More detailed than status.sh
- Checks all components
- Reports on service health
- Memory and resource usage

---

## 🚀 Deployment Scripts

### deploy_complete_system.sh
**Purpose:** Deploy the complete AGI/ASI system with all services
**Usage:** `./scripts/deploy_complete_system.sh [options]`
**Description:**
- Enterprise-grade deployment script
- Deploys 50+ AI services
- BuildKit optimization
- Advanced error handling
- WSL2 compatibility
- Rollback capabilities

**Features:**
- Parallel deployment orchestration
- Health checks for all services
- Intelligent recovery system
- Deployment state tracking
- Comprehensive logging

**Options:**
- `--rollback`: Rollback to previous state
- `--dry-run`: Test deployment without executing
- `--verbose`: Enable verbose logging

### deploy_essential_ai.sh
**Purpose:** Deploy only essential AI components
**Usage:** `./scripts/deploy_essential_ai.sh`
**Description:**
- Lightweight deployment
- Core services only
- Faster startup
- Minimal resource usage

### deploy_autonomous_agi.sh
**Purpose:** Deploy autonomous AGI system
**Usage:** `./scripts/deploy_autonomous_agi.sh`
**Description:**
- Focuses on autonomous agents
- Self-improving capabilities
- Advanced orchestration

---

## 📊 Monitoring Scripts

### agent_status_dashboard.sh
**Purpose:** Real-time agent coordination dashboard
**Usage:** `./scripts/agent_status_dashboard.sh`
**Description:**
- Visual dashboard for all agents
- Color-coded status indicators
- Shows agent communication
- Real-time updates
- Resource utilization

**Display Sections:**
- System Overview
- Core Services Status
- AI Agents Status
- Active Tasks
- Agent Communication
- Resource Usage

### monitor
**Purpose:** Launch system monitoring
**Usage:** `./scripts/monitor`
**Description:**
- Wrapper for monitoring tools
- Launches static monitor for flicker-free display
- Continuous system monitoring
- Resource tracking

### sutazai_status.sh
**Purpose:** Comprehensive SutazAI status report
**Usage:** `./scripts/sutazai_status.sh`
**Description:**
- Detailed status information
- All components checked
- Performance metrics
- Deployment verification

---

## 🤖 Agent Management Scripts

### validate_integrated_system.py
**Purpose:** Validate agent integration and deployment
**Usage:** `python scripts/validate_integrated_system.py`
**Description:**
- Post-deployment validation
- Agent health checks
- Integration testing
- Configuration validation

**Checks:**
- Agent availability
- Communication channels
- Resource allocation
- Configuration consistency

---

## 🛠️ Utility Scripts

### fix_docker_deployment_issues.sh
**Purpose:** Resolve common Docker deployment problems
**Usage:** `./scripts/fix_docker_deployment_issues.sh`
**Description:**
- Fixes permission issues
- Resolves network conflicts
- Cleans up stale containers
- Rebuilds problematic images

**Common Fixes:**
- Docker socket permissions
- Port conflicts
- Volume mount issues
- Network configuration

### create_archive.sh
**Purpose:** Archive old files and clean up
**Usage:** `./scripts/create_archive.sh`
**Description:**
- Moves old files to archive
- Maintains system cleanliness
- Preserves important data
- Reduces clutter

### apply_wsl2_config.ps1
**Purpose:** Apply WSL2 configuration (Windows only)
**Usage:** Run in PowerShell as Administrator
**Description:**
- Configures WSL2 settings
- Optimizes memory usage
- Sets processor allocation
- Network configuration

---

## 🗄️ Database Scripts

### init_db.sql
**Purpose:** Initialize database schema
**Usage:** Automatically executed during deployment
**Description:**
- Creates required tables
- Sets up indexes
- Configures permissions
- Initial data population

### init-postgres.sql
**Purpose:** PostgreSQL specific initialization
**Usage:** Executed by PostgreSQL container
**Description:**
- PostgreSQL database setup
- User creation
- Schema initialization
- Performance tuning

---

## 📝 Usage Examples

### Starting the System
```bash
# Basic startup
cd /opt/sutazaiapp
./scripts/start.sh

# Check status
./scripts/status.sh
```

### Full Deployment
```bash
# Deploy complete system
./scripts/deploy_complete_system.sh

# Monitor deployment
./scripts/agent_status_dashboard.sh
```

### Troubleshooting
```bash
# Check detailed status
./scripts/check_system_status.sh

# Fix Docker issues
./scripts/fix_docker_deployment_issues.sh

# Validate deployment
python scripts/validate_integrated_system.py
```

### Monitoring
```bash
# Real-time agent dashboard
./scripts/agent_status_dashboard.sh

# Continuous monitoring
./scripts/monitor
```

---

## ⚠️ Important Notes

1. **Always run scripts from the project root** (`/opt/sutazaiapp`)
2. **Check Docker is running** before deployment scripts
3. **Review logs** in `/opt/sutazaiapp/logs` for troubleshooting
4. **Use status scripts** before and after deployments
5. **Monitor resource usage** during heavy deployments

## 🔧 Script Maintenance

- Scripts are version controlled in Git
- Test changes in development first
- Document any modifications
- Keep scripts modular and focused
- Use consistent error handling

## 📞 Support

For issues with scripts:
1. Check the logs directory
2. Run status checks
3. Review this documentation
4. Check Git history for recent changes

---

*Last Updated: 2025-07-31*