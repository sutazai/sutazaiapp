# System Cleanup Standard Operating Procedure

**Version:** 1.0.0  
**Created:** 2025-08-17 03:55:00 UTC  
**Purpose:** Standardized procedure for system cleanup and optimization

## Overview

This document provides a standardized procedure for cleaning up system resources, consolidating configurations, and optimizing infrastructure performance while maintaining full functionality.

## Pre-Cleanup Checklist

### Required Validations
- [ ] Load and review `/opt/sutazaiapp/CLAUDE.md`
- [ ] Check `/opt/sutazaiapp/IMPORTANT/` for policies
- [ ] Review enforcement rules if present
- [ ] Verify CHANGELOG.md exists
- [ ] Create system backup if needed
- [ ] Document current service status

## Cleanup Procedure

### Step 1: Process Analysis and Cleanup

#### Identify Running Processes
```bash
# Count MCP-related processes
ps aux | grep -E "(mcp|claude-flow|ruv-swarm)" | grep -v grep | wc -l

# List process details
ps aux | grep -E "(mcp|claude-flow)" | grep -v grep
```

#### Clean Host Processes
```bash
#!/bin/bash
# Kill zombies first
ps aux | grep defunct | grep -E "(mcp|claude-flow)" | awk '{print $2}' | xargs -r kill -9

# Stop monitoring scripts
pkill -f "mcp_conflict_monitoring.sh"
pkill -f "cleanup_containers.sh"

# Graceful termination
ps aux | grep -E "(mcp-server|extended_memory_mcp)" | grep -v grep | awk '{print $2}' | xargs -r kill -TERM
sleep 2

# Force kill remaining
ps aux | grep -E "(mcp|claude-flow)" | grep -v grep | grep -v docker | awk '{print $2}' | xargs -r kill -9
```

### Step 2: Docker Configuration Consolidation

#### Identify Configuration Files
```bash
# Count docker-compose files
find /opt/sutazaiapp/docker -name "docker-compose*.yml" -o -name "compose*.yml" | wc -l

# List all files
find /opt/sutazaiapp/docker -name "docker-compose*.yml" -o -name "compose*.yml"
```

#### Archive Old Configurations
```bash
# Create archive directory
mkdir -p /opt/sutazaiapp/docker/archived_configs_$(date +%Y%m%d)

# Move old files (preserve consolidated)
find /opt/sutazaiapp/docker -name "docker-compose*.yml" -o -name "compose*.yml" | \
  grep -v "docker-compose.consolidated.yml" | \
  xargs -I {} mv {} /opt/sutazaiapp/docker/archived_configs_$(date +%Y%m%d)/
```

### Step 3: Container Management

#### Handle Failing Containers
```bash
# Check container status
docker ps -a --format "table {{.Names}}\t{{.Status}}"

# Stop and remove failing containers
docker stop <container-name>
docker rm <container-name>

# Example: Remove mcp-manager if failing
docker stop sutazai-mcp-manager 2>/dev/null
docker rm sutazai-mcp-manager 2>/dev/null
```

#### Clean Orphaned Resources
```bash
# Remove orphaned containers
docker container prune -f

# Clean unused volumes
docker volume prune -f

# Remove unused images
docker image prune -f

# Complete system prune (use carefully)
# docker system prune -a -f --volumes
```

### Step 4: Service Validation

#### Verify Critical Services
```bash
# Check backend API
curl -s http://localhost:10010/health | jq .

# Check MCP orchestrator
docker ps | grep mcp-orchestrator

# Test MCP API endpoint
curl -s http://localhost:10010/api/v1/mcp/servers

# Count running containers
docker ps -q | wc -l
```

### Step 5: Documentation Updates

#### Update CHANGELOG.md
Include:
- Timestamp (UTC)
- Agent/person performing cleanup
- What was cleaned
- Why cleanup was needed
- Impact and metrics
- Files modified/archived

#### Create Validation Report
Document:
- Before/after metrics
- Services verified
- Space recovered
- Performance improvements
- Compliance verification

## Monitoring Commands

### Real-time Process Monitoring
```bash
# Watch MCP processes
watch -n 2 'ps aux | grep -E "(mcp|claude-flow)" | grep -v grep | wc -l'

# Monitor container restarts
watch -n 5 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep -i restart'
```

### Resource Usage Checks
```bash
# Disk usage
df -h /opt/sutazaiapp

# Docker space usage
docker system df

# Memory usage
free -h
```

## Common Issues and Solutions

### Issue: Zombie Processes
**Symptom:** Processes marked as <defunct>  
**Solution:** Kill parent process or use kill -9 on zombie PIDs

### Issue: Container Restart Loops
**Symptom:** Container constantly restarting  
**Solution:** Check logs, fix configuration, or remove if not needed

### Issue: Docker Socket Errors
**Symptom:** "Error while fetching server API version"  
**Solution:** Check Docker socket permissions or remove container needing host socket

### Issue: Configuration Conflicts
**Symptom:** Multiple docker-compose files with conflicting settings  
**Solution:** Consolidate to single authoritative file

## Best Practices

### DO:
- Always backup before major cleanup
- Test services after each cleanup phase
- Document all changes in CHANGELOG
- Use graceful termination before force kill
- Verify functionality preservation
- Archive rather than delete configurations

### DON'T:
- Don't delete files without understanding purpose
- Don't kill processes without checking dependencies
- Don't remove containers without checking if needed
- Don't skip validation steps
- Don't cleanup during peak usage
- Don't modify MCP servers without authorization

## Automation Scripts

### Weekly Cleanup Script
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/maintenance/weekly_cleanup.sh

echo "=== Weekly System Cleanup - $(date) ==="

# Process cleanup
ps aux | grep defunct | awk '{print $2}' | xargs -r kill -9

# Docker cleanup
docker container prune -f
docker volume prune -f
docker image prune -f

# Report
echo "Processes: $(ps aux | grep -E '(mcp|claude-flow)' | grep -v grep | wc -l)"
echo "Containers: $(docker ps -q | wc -l)"
echo "API Status: $(curl -s http://localhost:10010/health | jq -r .status)"
```

## Metrics to Track

### Success Metrics
- Host process count < 10
- Single docker-compose file
- No restarting containers
- All services healthy
- Disk space recovered > 100MB

### Performance Metrics
- CPU usage reduction
- Memory freed
- Disk I/O reduction
- Network traffic optimization
- Response time improvements

## Emergency Procedures

### System Not Responding
1. Check critical services first
2. Review recent changes
3. Restore from backup if needed
4. Restart Docker daemon if necessary
5. Reboot system as last resort

### Service Degradation After Cleanup
1. Check service logs
2. Verify configuration files
3. Restore archived configs if needed
4. Roll back cleanup changes
5. Document issue for investigation

## Compliance Requirements

- Follow all 20 base rules
- Apply enforcement rules if present
- Update CHANGELOG with timestamps
- Preserve MCP server integrity
- Maintain audit trail
- Document all changes

## Support and Escalation

### Level 1: Self-Service
- Use this documentation
- Check system logs
- Review CHANGELOG

### Level 2: Team Support
- Escalate to DevOps team
- Provide cleanup report
- Include error logs

### Level 3: Emergency
- Contact system architect
- Prepare incident report
- Initiate recovery procedures

---

**Document Status:** Active  
**Review Schedule:** Monthly  
**Next Review:** 2025-09-17