# Maintenance Scripts

Maintenance and cleanup scripts for SutazaiApp infrastructure.

## Scripts in this directory:

### fix-compliance-violations.py
Comprehensive script to fix all compliance violations and bring the project to 90%+ compliance score.
```bash
python3 fix-compliance-violations.py
```

### fix-mcp-bridge.py
Fixes MCP Bridge server configuration and connectivity issues.
```bash
python3 fix-mcp-bridge.py
```

### check-compliance.sh
Checks project compliance against 20 Professional Codebase Standards.
```bash
./check-compliance.sh
```

### compliance-checker.py
Python-based compliance checker with detailed reporting.
```bash
python3 compliance-checker.py
```

### auto-maintain.sh
Automated maintenance script that runs all cleanup and optimization tasks.
```bash
./auto-maintain.sh
```

## Maintenance Tasks

### Daily Maintenance
```bash
# Check compliance
./check-compliance.sh

# Clean Docker resources
docker system prune -f

# Check disk usage
df -h
```

### Weekly Maintenance
```bash
# Full system audit
../monitoring/comprehensive_system_audit.sh

# Update and optimize
./auto-maintain.sh

# Backup critical data
./backup.sh  # If exists
```

### Emergency Recovery
```bash
# Fix all compliance violations
python3 fix-compliance-violations.py

# Fix unhealthy services
../monitoring/fix-unhealthy-services.sh

# Restart infrastructure
../deploy/stop-infrastructure.sh
../deploy/start-infrastructure.sh
```

## Compliance Standards

The project aims for 90%+ compliance across:
- CHANGELOG.md in all directories
- Healthy Docker services
- Proper network configuration
- Organized script structure
- Complete documentation
- Testing infrastructure
- Monitoring capabilities
- Backup systems