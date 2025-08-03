# Sutazai Hygiene Enforcement System - Complete Guide

> **Version:** 1.0.0  
> **Last Updated:** 2025-01-03  
> **Maintainer:** AI Observability and Monitoring Engineer  

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Rule Enforcement Guide](#rule-enforcement-guide)
4. [Agent Reference](#agent-reference)
5. [Monitoring Dashboard](#monitoring-dashboard)
6. [Deployment Guide](#deployment-guide)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [FAQ](#faq)
10. [Appendices](#appendices)

---

## System Overview

The Sutazai Hygiene Enforcement System is a comprehensive, automated codebase health management solution designed to maintain the highest standards of code quality, organization, and operational excellence across the entire project ecosystem.

### Core Objectives

- **Automated Compliance**: Continuous enforcement of 16 CLAUDE.md hygiene rules
- **Real-time Monitoring**: Live visibility into codebase health and violations
- **Proactive Prevention**: Prevent hygiene violations before they impact operations
- **Comprehensive Reporting**: Detailed insights into compliance trends and agent performance
- **Zero-Tolerance Policy**: No exceptions for cleanliness and organization standards

### Key Features

- **16-Rule Enforcement Framework**: Complete coverage of all CLAUDE.md hygiene standards
- **Multi-Agent Coordination**: Specialized AI agents for each rule category
- **Real-time Dashboard**: Web-based monitoring interface with live updates
- **Automated Remediation**: Self-healing capabilities for common violations
- **Comprehensive Logging**: Full audit trail of all enforcement actions
- **One-Command Deployment**: Complete system setup with single script execution

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hygiene Enforcement System                   │
├─────────────────────────┬───────────────────────────────────────┤
│   Monitoring Dashboard  │            Agent Ecosystem            │
│   ┌─────────────────┐  │  ┌─────────────────────────────────┐  │
│   │ Web Interface   │  │  │ Hygiene Coordinator Agent       │  │
│   │ Real-time Data  │  │  │ ┌─────────────────────────────┐ │  │
│   │ Control Panel   │  │  │ │ Rule 1-16 Enforcement       │ │  │
│   └─────────────────┘  │  │ │ Violation Detection         │ │  │
│                        │  │ │ Automated Remediation       │ │  │
│   ┌─────────────────┐  │  │ └─────────────────────────────┘ │  │
│   │ API Endpoints   │  │  └─────────────────────────────────┐  │
│   │ System Metrics  │  │                                    │  │
│   │ Report Export   │  │  ┌─────────────────────────────────┐  │
│   └─────────────────┘  │  │ Specialized Agents              │  │
└─────────────────────────┼──┤ • Garbage Collector             │  │
                          │  │ • Deploy Automation Master      │  │
┌─────────────────────────┼──┤ • Script Organizer              │  │
│     Data Storage        │  │ • Docker Optimizer              │  │
│ ┌─────────────────────┐ │  │ • Documentation Manager         │  │
│ │ Violation Logs      │ │  │ • Python Validator              │  │
│ │ Compliance History  │ │  │ • Compliance Monitor            │  │
│ │ Agent Metrics       │ │  └─────────────────────────────────┘  │
│ │ System Reports      │ │                                       │
│ └─────────────────────┘ │  ┌─────────────────────────────────┐  │
└─────────────────────────┼──┤ Automation Framework            │  │
                          │  │ • Continuous Monitoring         │  │
                          │  │ • Scheduled Audits              │  │
                          │  │ • Auto-remediation              │  │
                          │  │ • Alert Management              │  │
                          │  └─────────────────────────────────┘  │
                          └───────────────────────────────────────┘
```

### Data Flow

1. **Continuous Monitoring**: Agents continuously scan codebase for violations
2. **Violation Detection**: Rule-specific patterns trigger enforcement actions
3. **Automated Remediation**: Safe cleanup and organization operations
4. **Real-time Updates**: Dashboard receives live status updates via WebSocket
5. **Logging & Reporting**: All actions logged for audit and analysis
6. **Human Intervention**: Manual controls available through dashboard interface

---

## Rule Enforcement Guide

### Rule Categories

The 16 CLAUDE.md rules are organized into four priority categories:

#### Critical Rules (Immediate Enforcement)
- **Rule 1**: No Fantasy Elements
- **Rule 2**: No Breaking Changes  
- **Rule 10**: Verify Before Cleanup
- **Rule 12**: Single Deployment Script
- **Rule 13**: No Garbage Files

#### High Priority Rules
- **Rule 3**: Analyze Everything
- **Rule 5**: Professional Standards
- **Rule 6**: Centralized Documentation
- **Rule 9**: No Code Duplication
- **Rule 11**: Clean Docker Structure

#### Medium Priority Rules
- **Rule 4**: Reuse Before Creating
- **Rule 7**: Script Organization
- **Rule 8**: Python Script Standards
- **Rule 14**: Correct AI Agent Usage
- **Rule 15**: Clean Documentation

#### Low Priority Rules
- **Rule 16**: Ollama/TinyLlama Standard

### Rule-Specific Enforcement

#### Rule 1: No Fantasy Elements
**Agent**: `fantasy-element-detector`
**Patterns**: Magic terms, hypothetical constructs, speculative code
**Actions**: 
- Scan for banned keywords (magic, wizard, teleport, etc.)
- Flag speculative comments and TODOs
- Validate all external dependencies exist

#### Rule 2: No Breaking Changes
**Agent**: `functionality-preservation-validator`
**Patterns**: API changes, dependency modifications, configuration updates
**Actions**:
- Pre-change functionality verification
- Regression test execution
- Rollback capability validation

#### Rule 3: Analyze Everything
**Agent**: `comprehensive-analyzer`
**Patterns**: Unreviewed files, undocumented changes, missing tests
**Actions**:
- Complete codebase analysis before changes
- Dependency verification
- Test coverage validation

#### Rule 4: Reuse Before Creating
**Agent**: `duplicate-detection-specialist`
**Patterns**: Similar functions, duplicate logic, redundant components
**Actions**:
- Scan for existing solutions
- Recommend reuse over creation
- Consolidate duplicate implementations

#### Rule 5: Professional Standards
**Agent**: `professional-standards-enforcer`
**Patterns**: Sloppy code, inconsistent formatting, poor practices
**Actions**:
- Code quality analysis
- Standards compliance verification
- Professional review process enforcement

#### Rule 6: Centralized Documentation
**Agent**: `documentation-manager`
**Patterns**: Scattered docs, duplicate content, outdated information
**Actions**:
- Consolidate documentation to `/docs/`
- Remove duplicate content
- Validate document freshness

#### Rule 7: Script Organization
**Agent**: `script-organizer`
**Patterns**: Script duplication, poor naming, scattered locations
**Actions**:
- Centralize scripts to `/scripts/`
- Standardize naming conventions
- Remove duplicate scripts

#### Rule 8: Python Script Standards
**Agent**: `python-validator`
**Patterns**: Missing docstrings, poor structure, hardcoded values
**Actions**:
- Validate script documentation
- Enforce coding standards
- Check for best practices

#### Rule 9: No Code Duplication
**Agent**: `duplication-eliminator`
**Patterns**: Multiple versions of same functionality
**Actions**:
- Detect duplicate code blocks
- Consolidate similar implementations
- Enforce single source of truth

#### Rule 10: Verify Before Cleanup
**Agent**: `safe-cleanup-coordinator`
**Patterns**: Unsafe deletions, missing verification, blind removal
**Actions**:
- Reference verification before deletion
- Safety checks for file removal
- Archive before permanent deletion

#### Rule 11: Clean Docker Structure
**Agent**: `docker-optimizer`
**Patterns**: Messy Dockerfiles, redundant containers, poor structure
**Actions**:
- Standardize Docker structure
- Optimize container definitions
- Remove redundant configurations

#### Rule 12: Single Deployment Script
**Agent**: `deploy-automation-master`
**Patterns**: Multiple deployment scripts, inconsistent processes
**Actions**:
- Consolidate to single `deploy.sh`
- Standardize deployment process
- Ensure idempotent operations

#### Rule 13: No Garbage Files
**Agent**: `garbage-collector`
**Patterns**: Backup files, temporary files, abandoned code
**Actions**:
- Identify and remove junk files
- Clean up development artifacts
- Maintain pristine repository

#### Rule 14: Correct AI Agent Usage
**Agent**: `agent-coordination-monitor`
**Patterns**: Wrong agent for task, inefficient routing
**Actions**:
- Validate agent selection
- Optimize task routing
- Ensure proper agent utilization

#### Rule 15: Clean Documentation
**Agent**: `documentation-hygiene-enforcer`
**Patterns**: Duplicate docs, outdated content, poor organization
**Actions**:
- Remove duplicate documentation
- Update outdated content
- Maintain documentation structure

#### Rule 16: Ollama/TinyLlama Standard
**Agent**: `llm-standards-enforcer`
**Patterns**: Non-standard LLM usage, configuration drift
**Actions**:
- Enforce Ollama framework usage
- Standardize on TinyLlama default
- Validate LLM configurations

---

## Agent Reference

### Core Coordination Agent

#### `hygiene-enforcement-coordinator`
**Purpose**: Master coordinator for all hygiene enforcement activities  
**Location**: `/opt/sutazaiapp/scripts/hygiene-enforcement-coordinator.py`  
**Key Features**:
- Multi-phase enforcement execution
- Agent orchestration and task delegation
- Comprehensive violation detection
- Safe file archival before deletion
- Detailed logging and reporting

**Usage**:
```bash
# Run critical violations phase
python3 hygiene-enforcement-coordinator.py --phase 1

# Dry run mode
python3 hygiene-enforcement-coordinator.py --phase 1 --dry-run

# Full enforcement cycle
python3 hygiene-enforcement-coordinator.py --phase 1
python3 hygiene-enforcement-coordinator.py --phase 2  
python3 hygiene-enforcement-coordinator.py --phase 3
```

### Specialized Enforcement Agents

#### `garbage-collector-coordinator`
**Rules**: Rule 13 (No Garbage Files)  
**Location**: `/opt/sutazaiapp/agents/garbage-collector-coordinator/`  
**Capabilities**:
- Pattern-based junk file detection
- Safe archival of removed files
- Development artifact cleanup
- Backup file elimination

#### `deploy-automation-master`
**Rules**: Rule 12 (Single Deployment Script)  
**Location**: `/opt/sutazaiapp/agents/deployment-automation-master/`  
**Capabilities**:
- Deployment script consolidation
- Idempotent operation validation
- Multi-environment support
- Rollback capability

#### `documentation-manager`
**Rules**: Rule 6 (Centralized Documentation), Rule 15 (Clean Documentation)  
**Location**: `/opt/sutazaiapp/agents/document-knowledge-manager/`  
**Capabilities**:
- Documentation centralization
- Duplicate content detection
- Freshness validation
- Structure standardization

#### `script-organizer`
**Rules**: Rule 7 (Script Organization)  
**Location**: `/opt/sutazaiapp/scripts/`  
**Capabilities**:
- Script location standardization
- Naming convention enforcement
- Duplicate script elimination
- Executable permission management

#### `python-validator`
**Rules**: Rule 8 (Python Script Standards)  
**Location**: `/opt/sutazaiapp/agents/senior-backend-developer/`  
**Capabilities**:
- Python code quality analysis
- Docstring validation
- Standards compliance checking
- Best practices enforcement

#### `docker-optimizer`
**Rules**: Rule 11 (Clean Docker Structure)  
**Location**: `/opt/sutazaiapp/docker/`  
**Capabilities**:
- Dockerfile optimization
- Container structure validation
- Base image standardization
- Resource efficiency analysis

### Health Monitoring

All agents implement standardized health monitoring:

```python
class AgentHealthMonitor:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.task_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def record_task_completion(self, success=True):
        self.task_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.last_heartbeat = datetime.now()
    
    def get_health_metrics(self):
        uptime = (datetime.now() - self.start_time).total_seconds()
        success_rate = (self.success_count / max(self.task_count, 1)) * 100
        
        return {
            'agent_name': self.agent_name,
            'status': self.get_status(),
            'uptime_seconds': uptime,
            'task_count': self.task_count,
            'success_rate': success_rate,
            'last_heartbeat': self.last_heartbeat.isoformat()
        }
```

---

## Monitoring Dashboard

### Dashboard Overview

The Hygiene Enforcement Monitoring Dashboard provides real-time visibility into system health, rule compliance, and agent performance through a modern web interface.

**Access**: `http://localhost:8080` (default)  
**Location**: `/opt/sutazaiapp/dashboard/hygiene-monitor/`

### Key Features

#### Real-time Status Overview
- **System Health**: Overall enforcement system status
- **Compliance Score**: Percentage of rules in compliance
- **Active Violations**: Count of current rule violations
- **Agent Status**: Health of all enforcement agents

#### Rule Compliance Matrix
- **Individual Rule Status**: Per-rule compliance visualization
- **Violation Trends**: Historical compliance data
- **Priority Indicators**: Color-coded rule priority levels
- **Last Check Timestamps**: Freshness of compliance data

#### Agent Health Dashboard
- **Agent Status Monitoring**: Real-time agent health metrics
- **Performance Metrics**: Task completion rates and success ratios
- **Resource Utilization**: CPU, memory, and disk usage per agent
- **Communication Status**: Agent connectivity and responsiveness

#### Interactive Charts
- **Violation Trends**: 24-hour rolling violation history
- **Rule Distribution**: Breakdown of violations by rule category
- **Agent Performance**: Comparative agent efficiency metrics
- **System Metrics**: Infrastructure health indicators

#### Control Panel
- **Manual Actions**: 
  - Run full audit
  - Force cleanup
  - Generate reports
  - Export data
- **Automation Settings**:
  - Toggle auto-enforcement
  - Configure refresh rates
  - Set monitoring intervals

### API Endpoints

The dashboard communicates with the backend through RESTful APIs:

#### GET Endpoints
```
GET /api/hygiene/status          # Current system status
GET /api/hygiene/report          # Generate compliance report  
GET /api/system/metrics          # System performance metrics
GET /api/agents/health           # Agent health status
```

#### POST Endpoints
```
POST /api/hygiene/audit          # Trigger full system audit
POST /api/hygiene/cleanup        # Force cleanup operations
POST /api/agents/restart         # Restart specific agent
POST /api/config/update          # Update system configuration
```

### Usage Examples

#### Starting the Dashboard
```bash
# Default start (port 8080)
./scripts/start-hygiene-dashboard.sh

# Custom port and host
./scripts/start-hygiene-dashboard.sh --port 3000 --host localhost

# Development mode with auto-reload
./scripts/start-hygiene-dashboard.sh --dev --verbose
```

#### Dashboard Features

1. **Status Monitoring**: Live updates every 10 seconds (configurable)
2. **Interactive Elements**: Click any item for detailed information
3. **Filtering**: Filter actions by severity level
4. **Export Capabilities**: Download reports in JSON format
5. **Responsive Design**: Works on desktop, tablet, and mobile devices

---

## Deployment Guide

### Prerequisites

- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.8 or higher
- **Git**: For repository management
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Network**: Outbound internet access for dependency installation

### One-Command Deployment

Use the master deployment script for complete system setup:

```bash
# Deploy to production environment
./scripts/deploy-hygiene-system.sh --env production

# Deploy to staging with verbose output
./scripts/deploy-hygiene-system.sh --env staging --verbose

# Dry run (show what would be done)
./scripts/deploy-hygiene-system.sh --env production --dry-run
```

### Manual Installation Steps

#### 1. Repository Setup
```bash
# Clone repository
git clone <repository-url> /opt/sutazaiapp
cd /opt/sutazaiapp

# Set proper permissions
sudo chown -R $(whoami):$(whoami) /opt/sutazaiapp
chmod +x scripts/*.sh
```

#### 2. Python Environment
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. System Configuration
```bash
# Create necessary directories
mkdir -p logs data/monitoring archive

# Set up log rotation
sudo cp config/logrotate/hygiene-enforcement /etc/logrotate.d/

# Install systemd service (optional)
sudo cp config/systemd/hygiene-enforcement.service /etc/systemd/system/
sudo systemctl enable hygiene-enforcement
```

#### 4. Agent Configuration
```bash
# Validate agent registry
python3 -c "import agents.agent_registry; print('Agent registry valid')"

# Test agent communication
./scripts/validate-agents.py --test-communication

# Initialize agent workspaces
./scripts/setup-agent-workspaces.sh
```

#### 5. Dashboard Setup
```bash
# Install dashboard dependencies
cd dashboard/hygiene-monitor
npm install  # if using Node.js development server

# Test dashboard
./scripts/start-hygiene-dashboard.sh --port 8080 --host localhost
```

### Environment-Specific Configurations

#### Development Environment
```bash
# Enable development mode
export HYGIENE_ENV=development
export HYGIENE_DEBUG=true
export HYGIENE_LOG_LEVEL=DEBUG

# Start with hot reload
./scripts/start-hygiene-dashboard.sh --dev
```

#### Staging Environment
```bash
# Configure staging
export HYGIENE_ENV=staging
export HYGIENE_API_ENDPOINT=http://staging-api:8081
export HYGIENE_REFRESH_INTERVAL=5000

# Deploy with validation
./scripts/deploy-hygiene-system.sh --env staging --validate
```

#### Production Environment
```bash
# Production configuration
export HYGIENE_ENV=production
export HYGIENE_AUTO_ENFORCEMENT=true
export HYGIENE_LOG_RETENTION_DAYS=90

# Secure deployment
./scripts/deploy-hygiene-system.sh --env production --secure
```

### Verification Steps

After deployment, verify system functionality:

```bash
# 1. Check system status
./scripts/hygiene-enforcement-coordinator.py --dry-run

# 2. Verify agent health
curl http://localhost:8080/api/agents/health

# 3. Test dashboard access
curl http://localhost:8080/api/hygiene/status

# 4. Run compliance audit
./scripts/hygiene-enforcement-coordinator.py --phase 1

# 5. Check logs
tail -f logs/hygiene-enforcement.log
```

---

## Troubleshooting

### Common Issues

#### Dashboard Won't Start
**Symptoms**: Server fails to start on specified port
**Diagnosis**:
```bash
# Check port availability
netstat -tuln | grep :8080

# Check Python installation
python3 --version

# Verify file permissions
ls -la scripts/start-hygiene-dashboard.sh
```
**Resolution**:
```bash
# Kill processes using the port
sudo fuser -k 8080/tcp

# Fix permissions
chmod +x scripts/start-hygiene-dashboard.sh

# Try alternative port
./scripts/start-hygiene-dashboard.sh --port 8081
```

#### Agent Communication Failures
**Symptoms**: Agents showing as "OFFLINE" or "ERROR" status
**Diagnosis**:
```bash
# Check agent processes
ps aux | grep -E "(hygiene|agent)"

# Test agent connectivity
python3 -c "
import agents.agent_registry as registry
print(registry.test_agent_communication())
"

# Check log files
tail -f logs/agent-*.log
```
**Resolution**:
```bash
# Restart agent services
./scripts/restart-agents.sh

# Clear agent state
rm -rf data/agent-states/*

# Rebuild agent registry
python3 agents/rebuild_registry.py
```

#### High Resource Usage
**Symptoms**: System running slowly, high CPU/memory usage
**Diagnosis**:
```bash
# Check system resources
top -p $(pgrep -d',' -f hygiene)

# Monitor agent resource usage
./scripts/monitor-agent-resources.sh

# Check for memory leaks
valgrind --tool=memcheck python3 scripts/hygiene-enforcement-coordinator.py
```
**Resolution**:
```bash
# Adjust refresh intervals
export HYGIENE_REFRESH_INTERVAL=30000

# Limit concurrent agents
export HYGIENE_MAX_CONCURRENT_AGENTS=4

# Optimize monitoring frequency
./scripts/optimize-monitoring-frequency.sh
```

#### Rule Enforcement Failures
**Symptoms**: Rules showing violations but not being enforced
**Diagnosis**:
```bash
# Check rule configuration
cat logs/rule-enforcement-*.log

# Validate file permissions
ls -la scripts/ | head -20

# Test individual rule enforcement
python3 scripts/test-rule-enforcement.py --rule 13
```
**Resolution**:
```bash
# Reset rule states
rm -rf data/rule-states/*

# Rebuild rule patterns
python3 scripts/rebuild-rule-patterns.py

# Force rule re-evaluation
./scripts/force-rule-reevaluation.sh
```

### Log Analysis

#### Key Log Files
- **Main System**: `/opt/sutazaiapp/logs/hygiene-enforcement.log`
- **Dashboard**: `/opt/sutazaiapp/logs/dashboard.log`
- **Agents**: `/opt/sutazaiapp/logs/agent-*.log`
- **Rules**: `/opt/sutazaiapp/logs/rule-*.log`

#### Log Analysis Commands
```bash
# View recent enforcement actions
tail -100 logs/hygiene-enforcement.log | grep -E "(VIOLATION|ENFORCEMENT|ERROR)"

# Analyze agent performance
grep "HEALTH_CHECK" logs/agent-*.log | awk '{print $3, $NF}' | sort | uniq -c

# Find rule violations
grep "RULE_VIOLATION" logs/rule-*.log | sort | uniq -c

# Monitor real-time activity
tail -f logs/hygiene-enforcement.log | grep --color=always -E "(CRITICAL|ERROR|VIOLATION)"
```

### Performance Optimization

#### Monitoring Optimization
```bash
# Reduce monitoring frequency for stable systems
export HYGIENE_MONITORING_INTERVAL=60  # seconds

# Disable verbose logging in production
export HYGIENE_LOG_LEVEL=INFO

# Optimize agent polling
export HYGIENE_AGENT_POLL_INTERVAL=30
```

#### Resource Management
```bash
# Limit concurrent operations
export HYGIENE_MAX_CONCURRENT_OPERATIONS=8

# Set memory limits for agents
export HYGIENE_AGENT_MEMORY_LIMIT=512M

# Configure garbage collection
export HYGIENE_GC_INTERVAL=300
```

---

## Best Practices

### System Administration

#### Regular Maintenance
1. **Daily**: Monitor dashboard for critical violations
2. **Weekly**: Review compliance trends and agent performance
3. **Monthly**: Audit system configuration and update rules
4. **Quarterly**: Performance analysis and optimization review

#### Security Considerations
- **Access Control**: Limit dashboard access to authorized personnel
- **Log Security**: Ensure logs don't contain sensitive information
- **Network Security**: Use HTTPS for production dashboard access
- **API Security**: Implement authentication for API endpoints

#### Backup and Recovery
```bash
# Backup configuration
tar -czf hygiene-config-backup-$(date +%Y%m%d).tar.gz \
  config/ scripts/ docs/HYGIENE_ENFORCEMENT_COMPLETE_GUIDE.md

# Backup logs and data
tar -czf hygiene-data-backup-$(date +%Y%m%d).tar.gz \
  logs/ data/ archive/

# Test recovery procedure
./scripts/test-disaster-recovery.sh
```

### Development Workflow

#### Pre-commit Integration
```bash
# Install pre-commit hooks
./scripts/install-hygiene-hooks.sh

# Test hooks before commit
pre-commit run --all-files

# Configure automatic enforcement
git config core.hooksPath .githooks/
```

#### Continuous Integration
```yaml
# .github/workflows/hygiene-check.yml
name: Hygiene Enforcement Check
on: [push, pull_request]
jobs:
  hygiene-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Hygiene Audit
        run: |
          ./scripts/hygiene-enforcement-coordinator.py --phase 1 --dry-run
          ./scripts/validate-compliance.sh
```

#### Code Review Guidelines
1. **Rule Compliance**: Verify all changes maintain rule compliance
2. **Agent Impact**: Consider impact on enforcement agents
3. **Documentation**: Update docs for any hygiene-related changes
4. **Testing**: Include hygiene tests in PR validation

### Monitoring Best Practices

#### Alert Configuration
```bash
# Set up critical alerts
export HYGIENE_ALERT_CRITICAL_VIOLATIONS=5
export HYGIENE_ALERT_AGENT_FAILURES=3
export HYGIENE_ALERT_COMPLIANCE_DROP=10  # percentage

# Configure notification channels
export HYGIENE_SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export HYGIENE_EMAIL_ALERTS="admin@company.com"
```

#### Performance Monitoring
- **Response Times**: Monitor API endpoint response times
- **Resource Usage**: Track CPU, memory, and disk usage trends
- **Agent Health**: Maintain agent uptime above 99%
- **Compliance Scores**: Target 95%+ compliance rate

#### Capacity Planning
- **Storage**: Plan for log growth (approximately 100MB/day)
- **Processing**: Monitor enforcement operation duration
- **Network**: Consider bandwidth for real-time updates
- **Scalability**: Plan for codebase growth impact

---

## FAQ

### General Questions

**Q: How often does the system check for violations?**
A: By default, the system performs continuous monitoring with checks every 10 seconds for critical rules and every 60 seconds for lower-priority rules. This is configurable.

**Q: Can I disable specific rules?**
A: While all 16 rules are mandatory per CLAUDE.md standards, you can adjust enforcement priority and frequency through the configuration files.

**Q: What happens if an agent fails?**
A: The system includes automatic failover and recovery mechanisms. Failed agents are automatically restarted, and their tasks are redistributed to healthy agents.

**Q: How much storage does the system require?**
A: Base installation requires ~500MB. Logs and archives grow by approximately 100MB per day depending on codebase activity.

### Technical Questions

**Q: Can I run the system in Docker?**
A: Yes, Docker configurations are available in `/docker/`. Use `docker-compose up hygiene-monitor` to start the containerized version.

**Q: How do I add custom rules?**
A: The system is designed for the specific 16 CLAUDE.md rules. Custom rules require modification of the core coordination logic and are not recommended.

**Q: What's the performance impact on development?**
A: Minimal. The system uses asynchronous processing and smart caching to avoid impacting development workflows.

**Q: Can I integrate with existing CI/CD pipelines?**
A: Yes, the system provides CLI interfaces and API endpoints for integration with Jenkins, GitHub Actions, GitLab CI, and other platforms.

### Troubleshooting Questions

**Q: Dashboard shows all agents as "OFFLINE"**
A: Check if the hygiene coordinator service is running: `ps aux | grep hygiene-enforcement-coordinator`. Restart with `./scripts/restart-hygiene-services.sh`.

**Q: Rules keep showing violations despite cleanup**
A: Some rules require manual intervention. Check the detailed violation reports in the dashboard and follow the recommended remediation steps.

**Q: System using too much CPU**
A: Reduce monitoring frequency: `export HYGIENE_MONITORING_INTERVAL=120` and restart the services.

**Q: Can't access dashboard remotely**
A: Ensure the dashboard is bound to the correct interface: `./scripts/start-hygiene-dashboard.sh --host 0.0.0.0 --port 8080`

---

## Appendices

### Appendix A: Configuration Reference

#### Environment Variables
```bash
# Core System
HYGIENE_ENV=production                    # Environment: development, staging, production
HYGIENE_PROJECT_ROOT=/opt/sutazaiapp     # Project root directory
HYGIENE_LOG_LEVEL=INFO                   # Logging level: DEBUG, INFO, WARN, ERROR
HYGIENE_LOG_RETENTION_DAYS=30            # Log retention period

# Monitoring
HYGIENE_MONITORING_INTERVAL=10           # Monitoring frequency (seconds)
HYGIENE_DASHBOARD_PORT=8080              # Dashboard port
HYGIENE_DASHBOARD_HOST=0.0.0.0           # Dashboard host
HYGIENE_API_ENDPOINT=/api/hygiene        # API base endpoint
HYGIENE_WS_ENDPOINT=ws://localhost:8081/ws # WebSocket endpoint

# Enforcement
HYGIENE_AUTO_ENFORCEMENT=true            # Enable automatic enforcement
HYGIENE_MAX_CONCURRENT_AGENTS=8          # Maximum concurrent agents
HYGIENE_ENFORCEMENT_TIMEOUT=300          # Operation timeout (seconds)
HYGIENE_SAFE_MODE=false                  # Enable extra safety checks

# Performance
HYGIENE_CACHE_SIZE=1000                  # Cache size for violations
HYGIENE_BATCH_SIZE=50                    # Batch processing size
HYGIENE_WORKER_THREADS=4                 # Number of worker threads
HYGIENE_MEMORY_LIMIT=2G                  # Memory limit per agent

# Notifications
HYGIENE_ENABLE_ALERTS=true               # Enable alert system
HYGIENE_SLACK_WEBHOOK_URL=""             # Slack webhook for alerts
HYGIENE_EMAIL_ALERTS=""                  # Email addresses for alerts
HYGIENE_ALERT_THRESHOLD_CRITICAL=5       # Critical violation threshold
```

### Appendix B: API Documentation

#### Status Endpoints
```http
GET /api/hygiene/status
Response: {
  "timestamp": "2025-01-03T10:30:00Z",
  "systemStatus": "MONITORING",
  "complianceScore": 87,
  "totalViolations": 23,
  "criticalViolations": 3,
  "activeAgents": 8,
  "rules": { ... },
  "agents": { ... }
}

GET /api/agents/health
Response: {
  "agents": [
    {
      "name": "hygiene-coordinator",
      "status": "ACTIVE",
      "health": 95,
      "uptime": 86400,
      "lastHeartbeat": "2025-01-03T10:29:55Z"
    }
  ]
}
```

#### Action Endpoints
```http
POST /api/hygiene/audit
Request: {
  "phase": 1,
  "dryRun": false,
  "rules": ["rule_13", "rule_12"]
}
Response: {
  "success": true,
  "taskId": "audit-20250103-103000",
  "estimatedDuration": 120
}

POST /api/hygiene/cleanup
Request: {
  "force": false,
  "archiveMode": true,
  "targets": ["*.bak", "*.tmp"]
}
Response: {
  "success": true,
  "itemsProcessed": 45,
  "itemsRemoved": 23,
  "archiveLocation": "/archive/cleanup-20250103"
}
```

### Appendix C: Agent Development Guide

#### Creating Custom Agents
While the system is designed for the specific 16 CLAUDE.md rules, developers can extend agent capabilities:

```python
from agents.agent_base import BaseAgent
from agents.agent_with_health import AgentWithHealth

class CustomHygieneAgent(BaseAgent, AgentWithHealth):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.supported_rules = ['rule_custom']
        
    def process_violation(self, violation):
        """Process a detected violation"""
        try:
            # Custom processing logic
            result = self.handle_violation_custom(violation)
            self.record_task_completion(success=True)
            return result
        except Exception as e:
            self.record_task_completion(success=False)
            raise
    
    def handle_violation_custom(self, violation):
        """Custom violation handling logic"""
        # Implementation specific to custom rule
        pass
```

#### Agent Communication Protocol
```python
# Message format for inter-agent communication
{
  "sender": "agent-name",
  "recipient": "target-agent",
  "timestamp": "2025-01-03T10:30:00Z",
  "messageType": "TASK_REQUEST|HEALTH_CHECK|STATUS_UPDATE",
  "payload": {
    "task": "enforce_rule",
    "rule": "rule_13",
    "priority": "HIGH",
    "data": { ... }
  }
}
```

### Appendix D: Deployment Checklist

#### Pre-Deployment
- [ ] Verify system requirements
- [ ] Check network connectivity
- [ ] Validate Python version (3.8+)
- [ ] Ensure sufficient disk space (2GB minimum)
- [ ] Review security requirements
- [ ] Backup existing configuration

#### Deployment
- [ ] Run deployment script
- [ ] Verify agent registration
- [ ] Test dashboard accessibility
- [ ] Validate API endpoints
- [ ] Check log file creation
- [ ] Confirm monitoring startup

#### Post-Deployment
- [ ] Run initial compliance audit
- [ ] Monitor system performance
- [ ] Validate alert configurations
- [ ] Test backup procedures
- [ ] Document any customizations
- [ ] Schedule maintenance windows

#### Production Readiness
- [ ] Enable HTTPS for dashboard
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Document access procedures
- [ ] Train operations team
- [ ] Establish incident response procedures

---

## Conclusion

The Sutazai Hygiene Enforcement System represents a comprehensive solution for maintaining exceptional codebase quality through automated monitoring, enforcement, and reporting. By following this guide, teams can ensure consistent adherence to the 16 CLAUDE.md hygiene rules while maintaining high operational standards.

For additional support or questions, refer to the troubleshooting section or contact the system maintainers through the established channels.

---

**Document Information**
- **Version**: 1.0.0
- **Last Updated**: 2025-01-03
- **Next Review**: 2025-04-03
- **Maintainer**: AI Observability and Monitoring Engineer
- **Status**: Production Ready