# SutazAI System Automation Guide

## Overview

This comprehensive automation system provides autonomous operation capabilities for the SutazAI system, minimizing the need for manual intervention while maintaining high reliability and security standards.

## Table of Contents

1. [Architecture](#architecture)
2. [Automation Components](#automation-components)
3. [Installation and Setup](#installation-and-setup)
4. [Configuration](#configuration)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)
8. [Security Considerations](#security-considerations)

## Architecture

The SutazAI automation system consists of eight core components working together to provide comprehensive system management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SutazAI Automation System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Health Monitor │  │  Log Manager    │  │ Database Maint  │  │
│  │  (Every 5 min)  │  │  (Daily 2 AM)   │  │  (Daily 3 AM)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │ 
│  │ Cert Renewal    │  │ Agent Monitor   │  │ Performance     │  │
│  │ (Daily 4 AM)    │  │ (Every 5 min)   │  │ (Daily 7 AM)    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ Security Scan   │  │ Backup Verify   │                       │
│  │ (Daily 1 AM)    │  │ (Daily 5 AM)    │                       │
│  └─────────────────┘  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Core Principles

- **Autonomous Operation**: Minimal human intervention required
- **Fail-Safe Design**: Graceful degradation and recovery mechanisms
- **Comprehensive Logging**: All actions are logged with detailed context
- **Configurable Alerting**: Email notifications for critical issues
- **Performance Monitoring**: Continuous system health assessment
- **Security First**: Regular security scanning and vulnerability assessment

## Automation Components

### 1. Daily Health Check (`daily-health-check.sh`)

**Purpose**: Comprehensive system health assessment and reporting

**Schedule**: Daily at 6:00 AM

**Features**:
- Docker daemon health verification
- Core service status checking (PostgreSQL, Redis, Ollama)
- API endpoint availability testing
- Database connectivity validation
- AI agent health monitoring
- Model availability verification
- System resource monitoring (CPU, memory, disk)
- Log file size analysis
- Email notifications for critical issues

**Configuration**:
```bash
# Critical threshold for failed checks
CRITICAL_THRESHOLD=5

# Renewal threshold for alerts
RENEWAL_THRESHOLD_DAYS=30

# Email recipient (optional)
EMAIL_RECIPIENT="admin@domain.com"
```

**Output Files**:
- `$BASE_DIR/reports/daily_health_report_TIMESTAMP.json`
- `$BASE_DIR/logs/daily_health_check_TIMESTAMP.log`
- Latest report symlink: `$BASE_DIR/reports/latest_health_report.json`

### 2. Log Rotation and Cleanup (`log-rotation-cleanup.sh`)

**Purpose**: Automated log management and storage optimization

**Schedule**: Daily at 2:00 AM

**Features**:
- Large log file rotation (>100MB)
- Old log archival (>7 days)
- Docker container log cleanup
- Archive cleanup (>30 days)
- Total log directory size monitoring
- Compressed archive storage

**Configuration**:
```bash
MAX_LOG_SIZE_MB=100
MAX_LOG_AGE_DAYS=7
DELETE_ARCHIVES_DAYS=30
MAX_TOTAL_LOG_SIZE_GB=5
```

**Output Files**:
- `$BASE_DIR/archive/logs/` - Archived log files
- `$BASE_DIR/logs/log_cleanup_report_TIMESTAMP.json`

### 3. Database Maintenance (`database-maintenance.sh`)

**Purpose**: Automated database optimization and backup management

**Schedule**: Daily at 3:00 AM

**Features**:
- PostgreSQL VACUUM and ANALYZE operations
- Index reindexing (configurable)
- Redis memory defragmentation
- Database statistics collection
- Backup creation and verification
- Old backup cleanup
- Connection monitoring
- Performance metrics collection

**Configuration**:
```bash
VACUUM_THRESHOLD_DAYS=7
REINDEX_THRESHOLD_DAYS=30
BACKUP_RETENTION_DAYS=30
MAX_CONNECTION_THRESHOLD=80
```

**Output Files**:
- `$BASE_DIR/backups/database/sutazai_backup_TIMESTAMP.sql.gz`
- `$BASE_DIR/logs/database_maintenance_report_TIMESTAMP.json`

### 4. Certificate Renewal (`certificate-renewal.sh`)

**Purpose**: Automated SSL certificate management and renewal

**Schedule**: Daily at 4:00 AM

**Features**:
- Certificate expiration monitoring
- Automatic self-signed certificate generation
- Certificate backup and archival
- Service restart coordination
- Certificate validation and testing
- Multi-domain SAN support

**Configuration**:
```bash
RENEWAL_THRESHOLD_DAYS=30
CERT_VALIDITY_DAYS=365
KEY_SIZE=2048
DOMAIN_NAME="${SUTAZAI_DOMAIN:-localhost}"
```

**Output Files**:
- `$BASE_DIR/ssl/cert.pem` - Active certificate
- `$BASE_DIR/ssl/key.pem` - Private key
- `$BASE_DIR/backups/certificates/backup_TIMESTAMP/`

### 5. Agent Restart Monitor (`agent-restart-monitor.sh`)

**Purpose**: AI agent health monitoring and automatic recovery

**Schedule**: Every 5 minutes

**Features**:
- Individual agent health checking
- Automatic restart on consecutive failures
- Restart rate limiting
- Agent performance metrics collection
- State persistence between runs
- Configurable failure thresholds

**Configuration**:
```bash
MAX_RESTARTS_PER_HOUR=3
HEALTH_CHECK_INTERVAL=60
MAX_RESPONSE_TIME=30
CONSECUTIVE_FAILURES=3
```

**Monitored Agents**:
- `sutazai-senior-ai-engineer:8001`
- `sutazai-infrastructure-devops-manager:8002`
- `sutazai-testing-qa-validator:8003`
- `sutazai-agent-orchestrator:8004`
- `sutazai-ai-system-architect:8005`

**Output Files**:
- `$BASE_DIR/data/agent-monitor/AGENT_NAME.json` - Agent state
- `$BASE_DIR/logs/agent_monitoring_report_TIMESTAMP.json`

### 6. Performance Report Generator (`performance-report-generator.sh`)

**Purpose**: Comprehensive system performance analysis and reporting

**Schedule**: 
- Daily reports: 7:00 AM
- Weekly reports: Sundays at 8:00 AM

**Features**:
- System resource utilization tracking
- Docker container performance metrics
- AI agent response time monitoring
- Database performance analysis
- Ollama model performance testing
- JSON and HTML report generation
- Performance scoring and recommendations

**Configuration**:
```bash
REPORT_FORMAT="both"  # json, html, or both
REPORT_PERIOD="daily" # daily, weekly, monthly
RETENTION_DAYS=90
```

**Output Files**:
- `$BASE_DIR/reports/performance/performance_report_PERIOD_TIMESTAMP.json`
- `$BASE_DIR/reports/performance/performance_report_PERIOD_TIMESTAMP.html`

### 7. Security Scanner (`security-scanner.sh`)

**Purpose**: Automated security vulnerability assessment

**Schedule**: 
- Daily scan: 1:00 AM
- Full scan: Saturdays at 11:00 PM

**Features**:
- Container vulnerability scanning (Trivy)
- Filesystem security assessment
- Network security configuration review
- Code security analysis (Semgrep)
- Configuration security validation
- Risk assessment and scoring
- JSON and HTML security reports

**Configuration**:
```bash
SEVERITY_THRESHOLD="MEDIUM"
MAX_SCAN_TIME=3600
REPORT_FORMAT="both"
```

**Output Files**:
- `$BASE_DIR/reports/security/security_scan_report_TIMESTAMP.json`
- `$BASE_DIR/reports/security/security_scan_report_TIMESTAMP.html`

### 8. Backup Verification (`backup-verification.sh`)

**Purpose**: Automated backup integrity verification and testing

**Schedule**: 
- Standard verification: Daily at 5:00 AM
- Comprehensive verification: Sundays at 5:30 AM

**Features**:
- Backup file integrity checking
- Database backup restoration testing
- Configuration backup validation
- Certificate backup verification
- Backup completeness assessment
- Storage capacity monitoring

**Configuration**:
```bash
MAX_BACKUP_AGE_DAYS=7
MAX_VERIFICATION_TIME=300
VERIFY_ALL=false  # Set to true for comprehensive checks
```

**Output Files**:
- `$BASE_DIR/logs/backup_verification_report_TIMESTAMP.json`

## Installation and Setup

### Prerequisites

- Ubuntu 20.04+ or compatible Linux distribution
- Docker and Docker Compose installed
- Root or sudo access for systemd setup
- At least 2GB free disk space for logs and reports

### Quick Installation

1. **Navigate to the automation directory**:
   ```bash
   cd /opt/sutazaiapp/scripts/automation
   ```

2. **Make the setup script executable**:
   ```bash
   chmod +x setup-automation-cron.sh
   ```

3. **Install with cron jobs** (recommended for most setups):
   ```bash
   ./setup-automation-cron.sh
   ```

4. **Or install with systemd timers** (recommended for production):
   ```bash
   sudo ./setup-automation-cron.sh --systemd
   ```

5. **Enable email notifications** (optional):
   ```bash
   ./setup-automation-cron.sh --email admin@yourdomain.com
   ```

### Advanced Installation Options

**Systemd with email notifications**:
```bash
sudo ./setup-automation-cron.sh --systemd --email admin@yourdomain.com
```

**Remove existing automation**:
```bash
./setup-automation-cron.sh --remove
# or for systemd
sudo ./setup-automation-cron.sh --systemd --remove
```

### Manual Script Installation

If you prefer to set up individual components:

1. **Make all scripts executable**:
   ```bash
   chmod +x /opt/sutazaiapp/scripts/automation/*.sh
   ```

2. **Create required directories**:
   ```bash
   mkdir -p /opt/sutazaiapp/{logs,reports,backups,data/agent-monitor}
   ```

3. **Test individual scripts**:
   ```bash
   # Test health check
   /opt/sutazaiapp/scripts/automation/daily-health-check.sh --verbose
   
   # Test log cleanup (dry run)
   /opt/sutazaiapp/scripts/automation/log-rotation-cleanup.sh --dry-run
   ```

## Configuration

### Environment Variables

Set these environment variables to customize the automation system:

```bash
# Email notifications
export SUTAZAI_ADMIN_EMAIL="admin@yourdomain.com"

# SSL Certificate configuration
export SUTAZAI_DOMAIN="your-domain.com"
export SUTAZAI_ORG="Your Organization"
export SUTAZAI_CITY="Your City"
export SUTAZAI_STATE="Your State"
export SUTAZAI_COUNTRY="US"

# Performance thresholds
export SUTAZAI_MAX_MEMORY_USAGE="80"
export SUTAZAI_MAX_DISK_USAGE="85"
export SUTAZAI_MAX_LOAD_THRESHOLD="1.5"
```

### Configuration Files

Key configuration locations:

- **Main configuration**: `/opt/sutazaiapp/config/`
- **SSL certificates**: `/opt/sutazaiapp/ssl/`
- **Secrets**: `/opt/sutazaiapp/secrets_secure/`
- **Agent monitoring state**: `/opt/sutazaiapp/data/agent-monitor/`

### Customizing Schedules

#### For Cron Jobs

Edit your crontab:
```bash
crontab -e
```

Example custom schedule:
```bash
# Custom schedule - run health check every 2 hours
0 */2 * * * /opt/sutazaiapp/scripts/automation/daily-health-check.sh
```

#### For Systemd Timers

Edit the timer file:
```bash
sudo systemctl edit sutazai-health-check.timer
```

Add custom schedule:
```ini
[Timer]
OnCalendar=*:0/30:0  # Every 30 minutes
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart sutazai-health-check.timer
```

## Monitoring and Alerting

### Monitoring Dashboard

Access the automation status dashboard:
```
file:///opt/sutazaiapp/dashboard/automation-status.html
```

Or serve it via HTTP:
```bash
cd /opt/sutazaiapp/dashboard
python3 -m http.server 8080
# Then access http://localhost:8080/automation-status.html
```

### Log Monitoring

Monitor automation logs in real-time:
```bash
# Follow all automation logs
tail -f /opt/sutazaiapp/logs/*automation* /opt/sutazaiapp/logs/*health* /opt/sutazaiapp/logs/*performance*

# Monitor specific component
tail -f /opt/sutazaiapp/logs/daily_health_check_*.log

# View latest reports
ls -la /opt/sutazaiapp/reports/*/latest_*
```

### Email Alerting

Email notifications are sent for:
- Critical health check failures
- Security vulnerabilities (HIGH/CRITICAL)
- Backup verification failures
- Certificate expiration warnings
- Agent restart failures exceeding thresholds

### Status Checking Commands

#### For Cron Jobs
```bash
# List all SutazAI automation jobs
crontab -l | grep sutazai-automation

# Check cron service status
systemctl status cron
```

#### For Systemd Timers
```bash
# List all SutazAI timers
systemctl list-timers | grep sutazai

# Check specific timer
systemctl status sutazai-health-check.timer

# View timer logs
journalctl -u sutazai-health-check.service -f
```

### Performance Metrics

Key metrics monitored:
- **System Health Score**: 0-100 based on all health checks
- **Performance Score**: 0-100 based on resource usage and response times
- **Security Score**: 0-100 based on vulnerability assessment
- **Agent Uptime**: Percentage of time agents are healthy
- **Backup Success Rate**: Percentage of successful backup verifications

## Troubleshooting

### Common Issues

#### 1. Automation Scripts Not Running

**Symptoms**: No new log files or reports generated

**Diagnosis**:
```bash
# Check cron service
systemctl status cron

# Check systemd timers
systemctl list-timers | grep sutazai

# Check script permissions
ls -la /opt/sutazaiapp/scripts/automation/
```

**Solutions**:
```bash
# Restart cron service
sudo systemctl restart cron

# Reload systemd timers
sudo systemctl daemon-reload

# Fix permissions
sudo chmod +x /opt/sutazaiapp/scripts/automation/*.sh
```

#### 2. Email Notifications Not Working

**Symptoms**: No email alerts received despite critical issues

**Diagnosis**:
```bash
# Check mail command
which mail

# Test email sending
echo "Test" | mail -s "Test Subject" admin@yourdomain.com
```

**Solutions**:
```bash
# Install mail utilities
sudo apt-get install mailutils

# Configure postfix for local delivery
sudo dpkg-reconfigure postfix
```

#### 3. High Resource Usage

**Symptoms**: System performance degradation during automation runs

**Diagnosis**:
```bash
# Monitor resource usage during automation
top -p $(pgrep -f "automation")

# Check log file sizes
du -sh /opt/sutazaiapp/logs/
```

**Solutions**:
```bash
# Adjust automation schedules to off-peak hours
# Increase log cleanup frequency
# Add resource limits to scripts
```

#### 4. Failed Security Scans

**Symptoms**: Security scanner reports failures or timeouts

**Diagnosis**:
```bash
# Check Trivy installation
trivy --version

# Test manual scan
trivy image --format json sutazai/backend
```

**Solutions**:
```bash
# Install/update Trivy
sudo apt-get update && sudo apt-get install trivy

# Increase scan timeout
# Check available disk space for scan cache
```

### Debug Mode

Run any automation script in debug mode:
```bash
# Enable verbose output
/opt/sutazaiapp/scripts/automation/daily-health-check.sh --verbose

# Dry run mode (no changes made)
/opt/sutazaiapp/scripts/automation/log-rotation-cleanup.sh --dry-run

# Manual execution with full logging
bash -x /opt/sutazaiapp/scripts/automation/database-maintenance.sh
```

### Log Analysis

Analyze automation logs:
```bash
# Search for errors across all logs
grep -r "ERROR" /opt/sutazaiapp/logs/ | grep "$(date +%Y-%m-%d)"

# Find recent failures
find /opt/sutazaiapp/logs/ -name "*.log" -mtime -1 -exec grep -l "FAIL\|ERROR" {} \;

# Performance monitoring
grep "performance_score" /opt/sutazaiapp/reports/performance/latest_performance_report.json
```

## Maintenance

### Regular Maintenance Tasks

#### Weekly Tasks
- Review automation logs for errors or warnings
- Check disk space usage in `/opt/sutazaiapp/`
- Verify email notifications are working
- Review security scan reports

#### Monthly Tasks
- Update security scanning tools (Trivy, Semgrep)
- Review and adjust automation schedules if needed
- Clean up old report files manually if needed
- Test backup restoration procedures

#### Quarterly Tasks
- Review and update automation scripts
- Performance optimization based on historical data
- Security audit of automation system
- Disaster recovery testing

### Updating Automation Scripts

1. **Backup current scripts**:
   ```bash
   cp -r /opt/sutazaiapp/scripts/automation /opt/sutazaiapp/scripts/automation.backup.$(date +%Y%m%d)
   ```

2. **Update scripts** (replace with new versions)

3. **Test new scripts**:
   ```bash
   # Test in dry-run mode
   /opt/sutazaiapp/scripts/automation/daily-health-check.sh --verbose
   ```

4. **Reinstall automation**:
   ```bash
   # Remove old automation
   ./setup-automation-cron.sh --remove
   
   # Install updated version
   ./setup-automation-cron.sh
   ```

### Performance Optimization

#### Resource Usage Optimization

Monitor and optimize resource usage:
```bash
# Check disk usage trends
df -h /opt/sutazaiapp/
du -sh /opt/sutazaiapp/logs/ /opt/sutazaiapp/reports/

# Monitor memory usage during automation
free -h

# Check I/O usage
iostat -x 1 5
```

#### Schedule Optimization

Distribute automation tasks to minimize resource conflicts:
```bash
# Example optimized schedule
# 1:00 AM - Security scanning (CPU intensive)
# 2:00 AM - Log cleanup (I/O intensive)
# 3:00 AM - Database maintenance (I/O intensive)
# 4:00 AM - Certificate renewal (minimal resources)
# 5:00 AM - Backup verification (I/O intensive)
# 6:00 AM - Health check (minimal resources)
# 7:00 AM - Performance reporting (CPU intensive)
```

## Security Considerations

### Access Control

- Automation scripts run with appropriate user privileges only
- Sensitive operations require root access through sudo
- Log files have restricted permissions (640)
- Report files exclude sensitive information

### Secrets Management

- No hardcoded credentials in scripts
- Environment variables for configuration
- Secure storage in `/opt/sutazaiapp/secrets_secure/`
- Regular rotation of service passwords

### Network Security

- Automation only accesses local services by default
- External connectivity limited to necessary operations
- SSL/TLS verification for external communications
- Firewall rules maintained for automation ports

### Audit Trail

- All automation activities are logged with timestamps
- Failed operations are logged with detailed error information
- Successful operations include metadata and results
- Log rotation preserves audit history

### Backup Security

- Backup files are compressed and stored securely
- Database backups exclude sensitive user data when possible
- Certificate backups maintain appropriate permissions
- Backup verification includes integrity checking

## Advanced Configuration

### Custom Notification Handlers

Create custom notification scripts:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/automation/custom-alert.sh
# Custom alerting script for integration with external systems

ALERT_TYPE="$1"
ALERT_MESSAGE="$2"
ALERT_SEVERITY="$3"

case $ALERT_TYPE in
    "health")
        # Send to monitoring system
        curl -X POST "https://monitoring.yourdomain.com/api/alerts" \
             -H "Content-Type: application/json" \
             -d "{\"type\":\"$ALERT_TYPE\",\"message\":\"$ALERT_MESSAGE\",\"severity\":\"$ALERT_SEVERITY\"}"
        ;;
    "security")
        # Send to security information system
        # ... implementation
        ;;
esac
```

### Integration with External Systems

#### Prometheus Metrics Export

Export metrics for Prometheus monitoring:
```bash
# Add to performance report generator
echo "sutazai_health_score $HEALTH_SCORE" >> /var/lib/prometheus/node-exporter/sutazai.prom
echo "sutazai_agent_uptime $AGENT_UPTIME" >> /var/lib/prometheus/node-exporter/sutazai.prom
```

#### Grafana Dashboard Integration

Create Grafana dashboard using exported metrics:
- System health trends
- Performance score over time
- Agent availability statistics
- Security vulnerability trends

### High Availability Setup

For production environments requiring high availability:

1. **Distributed Automation**:
   - Run automation on multiple nodes
   - Use distributed locking to prevent conflicts
   - Failover mechanisms for critical tasks

2. **Backup Automation Nodes**:
   - Secondary automation node with delayed schedules
   - Automatic failover if primary node fails
   - Shared storage for reports and logs

3. **Load Balancing**:
   - Distribute automation load across multiple nodes
   - Use message queues for task distribution
   - Monitor and balance resource usage

## API Integration

### REST API for Automation Control

Create API endpoints for automation control:
```bash
# Start/stop automation
POST /api/automation/health-check/start
POST /api/automation/health-check/stop

# Get automation status
GET /api/automation/status

# Get latest reports
GET /api/automation/reports/health
GET /api/automation/reports/performance
GET /api/automation/reports/security
```

### Webhook Integration

Configure webhooks for external system integration:
```bash
# Webhook configuration in automation scripts
WEBHOOK_URL="https://yourdomain.com/api/webhooks/sutazai"
WEBHOOK_TOKEN="your-secure-token"

# Send webhook notification
curl -X POST "$WEBHOOK_URL" \
     -H "Authorization: Bearer $WEBHOOK_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{\"event\":\"health_check_completed\",\"status\":\"$STATUS\",\"timestamp\":\"$TIMESTAMP\"}"
```

## Conclusion

The SutazAI automation system provides comprehensive autonomous operation capabilities while maintaining security, reliability, and observability. Regular monitoring and maintenance ensure optimal performance and early detection of issues.

For additional support or customization needs, refer to the individual script documentation or contact the system administrator.

---

**Document Version**: 1.0  
**Last Updated**: $(date +%Y-%m-%d)  
**Automation System Version**: 1.0  
**Compatible SutazAI Versions**: v40+