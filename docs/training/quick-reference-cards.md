# SutazAI Quick Reference Cards

## Overview

These quick reference cards provide instant access to common SutazAI operations, commands, and workflows. Print these cards or keep them bookmarked for quick access during daily operations.

---

## Card 1: System Health & Status

### Quick Health Check
```bash
# System health
curl http://localhost:8000/health

# Agent status
curl http://localhost:8000/health/agents

# Resource usage
curl http://localhost:8000/health/resources
```

### System Commands
```bash
# Start system
./deploy.sh --environment development

# Stop system
docker-compose down

# Restart specific service
docker-compose restart [service-name]

# View logs
docker-compose logs -f [service-name]
```

### Key Endpoints
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Monitoring Dashboard**: `http://localhost:3000`
- **Agent Registry**: `http://localhost:8000/agents`

### Emergency Commands
```bash
# Emergency stop
./scripts/emergency-shutdown-coordinator.py

# System validation
./scripts/validate-complete-system.py

# Quick health fix
./scripts/fix-container-health-immediate.sh
```

---

## Card 2: Agent Operations

### Agent Status Check
```bash
# List all agents
curl http://localhost:8000/agents/list

# Check specific agent
curl http://localhost:8000/agents/[agent-name]/health

# Agent performance
curl http://localhost:8000/agents/[agent-name]/metrics
```

### Common Agents & Ports
| Agent | Port | Purpose |
|-------|------|---------|
| api-gateway | 8000 | Main API |
| senior-ai-engineer | 8001 | AI/ML tasks |
| code-generation-improver | 8002 | Code quality |
| testing-qa-validator | 8003 | Testing |
| security-pentesting-specialist | 8004 | Security |
| deployment-automation-master | 8005 | Deployment |

### Agent Commands
```bash
# Restart agent
docker-compose restart [agent-name]

# Scale agent
docker-compose up --scale [agent-name]=3

# Agent logs
docker-compose logs [agent-name]
```

### Task Submission
```bash
# Submit task to specific agent
curl -X POST http://localhost:8000/agents/[agent-name]/task \
  -H "Content-Type: application/json" \
  -d '{"action": "analyze", "data": {...}}'
```

---

## Card 3: Development Workflows

### Code Review Workflow
```bash
# Basic code review
python workflows/simple_code_review.py --file src/main.py

# Comprehensive review
python workflows/code_improvement_workflow.py --path ./src

# Security review
python workflows/security_scan_workflow.py --target ./src
```

### Testing Workflows
```bash
# Generate tests
curl -X POST http://localhost:8000/agents/testing-qa-validator/generate \
  -d '{"source_file": "src/main.py", "test_type": "unit"}'

# Run test validation
python workflows/test_validation.py --path ./tests
```

### Deployment Workflows
```bash
# Development deployment
./deploy.sh --environment development

# Production deployment
./deploy.sh --environment production --validate

# Rollback deployment
./scripts/rollback-deployment.sh --version previous
```

### Common File Locations
- **Workflows**: `/opt/sutazaiapp/workflows/`
- **Agent Configs**: `/opt/sutazaiapp/agents/configs/`
- **System Logs**: `/opt/sutazaiapp/logs/`
- **Documentation**: `/opt/sutazaiapp/docs/`

---

## Card 4: Security Operations

### Security Scanning
```bash
# Basic security scan
python workflows/security_scan_workflow.py --target ./app

# Vulnerability scan
curl -X POST http://localhost:8000/agents/semgrep-security-analyzer/scan \
  -d '{"path": "./src", "rules": ["owasp-top-10"]}'

# Penetration test
curl -X POST http://localhost:8000/agents/security-pentesting-specialist/test \
  -d '{"target": "http://localhost:3000"}'
```

### Security Commands
```bash
# Security validation
./scripts/validate-security.sh

# Compliance check
./scripts/compliance-audit.py --standard soc2

# Security hardening
./scripts/security-hardening.sh
```

### Security Endpoints
- **Security Dashboard**: `http://localhost:3000/security`
- **Vulnerability Reports**: `http://localhost:8000/security/reports`
- **Compliance Status**: `http://localhost:8000/security/compliance`

### Emergency Security
```bash
# Emergency security scan
./scripts/emergency-security-scan.sh

# Lock down system
./scripts/security-lockdown.sh

# Security incident response
./scripts/incident-response.py --severity critical
```

---

## Card 5: Monitoring & Troubleshooting

### Monitoring Commands
```bash
# Start monitoring
./scripts/start-monitoring-stack.sh

# Performance check
./scripts/performance-profiler-suite.py

# Resource usage
./scripts/system-resource-analyzer.py
```

### Log Analysis
```bash
# Recent errors
grep -i error /opt/sutazaiapp/logs/*.log | tail -20

# Agent performance
tail -f /opt/sutazaiapp/logs/agent-orchestration.log

# System metrics
cat /opt/sutazaiapp/logs/performance_metrics.json | jq .
```

### Troubleshooting Steps
1. **Check System Health**: `curl http://localhost:8000/health`
2. **Review Logs**: `docker-compose logs --tail=50`
3. **Validate Configuration**: `./scripts/validate-containers.sh`
4. **Restart Services**: `docker-compose restart`
5. **Full System Check**: `./scripts/validate-complete-system.py`

### Common Issues & Solutions
| Issue | Quick Fix |
|-------|-----------|
| Agent not responding | `docker-compose restart [agent-name]` |
| High memory usage | `./scripts/garbage-collection-system.py` |
| Port conflicts | `./scripts/fix-container-conflicts.sh` |
| Database connection | `docker-compose restart postgres` |
| Ollama not working | `./scripts/restart-ollama.sh` |

---

## Card 6: Configuration Management

### Configuration Files
```bash
# Main config
/opt/sutazaiapp/config/services.yaml

# Agent config
/opt/sutazaiapp/config/agent_orchestration.yaml

# Ollama config
/opt/sutazaiapp/config/ollama.yaml

# Load balancer
/opt/sutazaiapp/config/load_balancer.json
```

### Configuration Validation
```bash
# Validate all configs
./scripts/validate-configuration.sh

# Test configuration changes
./scripts/test-configuration.py --dry-run

# Apply configuration
./scripts/apply-configuration.sh
```

### Environment Variables
```bash
# Set environment
export SUTAZAI_ENV=production
export SUTAZAI_SCALE=medium
export OLLAMA_HOST=http://ollama:11434

# View current config
./scripts/show-environment.sh
```

### Backup & Restore
```bash
# Backup configuration
./scripts/backup-configuration.sh

# Restore configuration
./scripts/restore-configuration.sh --backup [timestamp]

# Configuration diff
./scripts/config-diff.sh --compare current previous
```

---

## Card 7: Performance Optimization

### Performance Commands
```bash
# Performance analysis
./scripts/performance-optimization.py

# Resource optimization
./scripts/hardware-optimization-master.py

# Agent optimization
./scripts/optimize-agent-utilization.py
```

### Scaling Operations
```bash
# Horizontal scaling
docker-compose up --scale api-gateway=3

# Resource scaling
docker-compose -f docker-compose.production.yml up

# Auto-scaling setup
./scripts/setup-autoscaling.sh
```

### Performance Metrics
```bash
# Current metrics
curl http://localhost:8000/metrics

# Performance report
./scripts/generate-performance-report.py

# Capacity analysis
./scripts/capacity-analysis.py
```

### Optimization Targets
- **Response Time**: < 5 seconds
- **Memory Usage**: < 80%
- **CPU Usage**: < 75%
- **Error Rate**: < 0.1%
- **Uptime**: > 99.9%

---

## Card 8: Emergency Procedures

### System Emergency
```bash
# Emergency shutdown
./scripts/emergency-shutdown-coordinator.py

# Emergency restart
./scripts/emergency-restart.sh

# System recovery
./scripts/disaster-recovery.py
```

### Service Recovery
```bash
# Service health check
./scripts/service-health-checker.py

# Auto-healing trigger
./scripts/self-healing-system.py

# Manual recovery
./scripts/manual-recovery.sh --service [name]
```

### Data Recovery
```bash
# Database backup
./scripts/backup-database.sh

# Point-in-time recovery
./scripts/point-in-time-recovery.py --timestamp [time]

# Data validation
./scripts/validate-data-integrity.sh
```

### Emergency Contacts & Escalation
1. **Level 1**: System Administrator
2. **Level 2**: DevOps Team Lead  
3. **Level 3**: Infrastructure Manager
4. **Level 4**: CTO/Technical Director

### Recovery Time Objectives
- **Critical Services**: 15 minutes
- **Standard Services**: 1 hour
- **Development Services**: 4 hours
- **Full System**: 8 hours

---

## Card 9: Integration & API

### API Authentication
```bash
# Get API token
curl -X POST http://localhost:8000/auth/token \
  -d '{"username": "admin", "password": "secure_password"}'

# Use token
curl -H "Authorization: Bearer [token]" \
  http://localhost:8000/agents/list
```

### Webhook Integration
```bash
# Register webhook
curl -X POST http://localhost:8000/webhooks/register \
  -d '{"url": "https://your-app.com/webhook", "events": ["task.completed"]}'

# Test webhook
curl -X POST http://localhost:8000/webhooks/test \
  -d '{"webhook_id": "12345"}'
```

### API Rate Limits
- **Standard**: 1000 requests/hour
- **Authenticated**: 10000 requests/hour
- **Admin**: Unlimited

### Common API Endpoints
```bash
# Task submission
POST /api/v1/tasks

# Task status
GET /api/v1/tasks/{task_id}

# Agent interaction
POST /api/v1/agents/{agent_name}/execute

# System metrics
GET /api/v1/metrics
```

---

## Card 10: Maintenance & Updates

### Regular Maintenance
```bash
# Daily maintenance
./scripts/daily-maintenance.sh

# Weekly cleanup
./scripts/weekly-cleanup.sh

# Monthly optimization
./scripts/monthly-optimization.sh
```

### System Updates
```bash
# Check for updates
./scripts/check-updates.sh

# Update system
./scripts/update-system.sh --version latest

# Rollback update
./scripts/rollback-update.sh --version previous
```

### Maintenance Windows
- **Daily**: 02:00-02:30 (automated cleanup)
- **Weekly**: Sunday 01:00-03:00 (optimization)
- **Monthly**: First Sunday 00:00-04:00 (major updates)

### Pre-Maintenance Checklist
- [ ] Backup system configuration
- [ ] Backup critical data
- [ ] Notify users of maintenance window
- [ ] Prepare rollback procedures
- [ ] Validate test environment

### Post-Maintenance Checklist
- [ ] Verify all services running
- [ ] Check system health
- [ ] Validate core functionality
- [ ] Monitor performance metrics
- [ ] Document any issues

---

## Usage Tips

### Quick Tips
1. **Bookmark** the monitoring dashboard for quick access
2. **Use aliases** for frequently used commands
3. **Monitor logs** during deployments
4. **Keep backups** current and tested
5. **Document changes** for team awareness

### Keyboard Shortcuts
- **Ctrl+C**: Stop current process
- **Ctrl+Z**: Background process  
- **Ctrl+R**: Search command history
- **Tab**: Auto-complete commands
- **↑/↓**: Navigate command history

### Best Practices
- Always check system health before operations
- Use dry-run mode for risky operations
- Keep configuration in version control
- Test in development before production
- Monitor system after any changes

---

**Print these cards and keep them handy for quick reference during daily operations!**