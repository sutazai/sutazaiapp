# SutazAI Comprehensive Backup Automation System

A complete backup automation solution implementing the 3-2-1 backup strategy (3 copies, 2 different media, 1 offsite) for the SutazAI platform.

## Overview

This backup system provides:
- **Automated daily/weekly/monthly backups** of all critical data
- **Database backups** (PostgreSQL, SQLite, Loki time-series data)
- **Configuration file backups** (Docker, agent configs, environment settings)
- **Agent state backups** (runtime data, collective intelligence state)
- **AI model backups** (Ollama models, configurations)
- **Monitoring data retention** (Prometheus, Grafana, health reports)
- **Log archival** with intelligent compression and retention
- **Backup verification** and integrity checking
- **Offsite replication** (local drive, rsync, S3, SFTP, NFS)
- **Restore testing** automation
- **Monitoring and alerting** (email, Slack, webhooks, SMS)

## Architecture

```
/opt/sutazaiapp/scripts/backup-automation/
├── sutazai-backup-orchestrator.py    # Master orchestration system
├── core/
│   └── database-backup-system.py     # Database backups
├── config/
│   └── config-backup-system.py       # Configuration backups
├── agents/
│   └── agent-state-backup-system.py  # Agent state backups
├── models/
│   └── ollama-model-backup-system.py # AI model backups
├── monitoring/
│   └── monitoring-data-retention-system.py # Monitoring data
├── logs/
│   └── log-archival-system.py        # Log archival
├── verification/
│   └── backup-verification-system.py # Backup verification
├── offsite/
│   └── offsite-backup-replication-system.py # Offsite replication
├── restore/
│   └── restore-testing-system.py     # Restore testing
├── alerts/
│   └── backup-monitoring-alerting-system.py # Monitoring & alerts
└── utils/
    ├── backup-status-checker.py      # Quick status checks
    └── setup-backup-automation.sh    # Setup script
```

## Quick Start

### 1. Setup and Installation

```bash
# Run the setup script
sudo /opt/sutazaiapp/scripts/backup-automation/setup-backup-automation.sh

# Enable backup services
sudo backup-control enable
sudo backup-control start
```

### 2. Check Status

```bash
# Quick status check
backup-status

# Detailed JSON status
backup-status --json

# Service status
backup-control status
```

### 3. Manual Backup

```bash
# Run daily backup now
backup-control run-now

# Or use orchestrator directly
python3 /opt/sutazaiapp/scripts/backup-automation/sutazai-backup-orchestrator.py daily
```

## Configuration

### Main Configuration Files

1. **`/opt/sutazaiapp/config/backup-config.json`** - Core backup settings
2. **`/opt/sutazaiapp/config/backup-orchestration-config.json`** - Orchestration settings
3. **`/opt/sutazaiapp/config/backup-monitoring-config.json`** - Monitoring & alerting

### Key Settings

```json
{
  "backup_root": "/opt/sutazaiapp/data/backups",
  "retention_days": 30,
  "compression": true,
  "encryption": false,
  "offsite": {
    "enabled": true,
    "methods": ["local_drive", "s3"],
    "retention_days": 90
  }
}
```

## Backup Types and Schedules

| Type | Frequency | Content | Retention |
|------|-----------|---------|-----------|
| **Daily** | 2:00 AM | Databases, configs, agent states, monitoring, logs | 30 days |
| **Weekly** | Sunday 3:00 AM | All daily + AI models | 90 days |
| **Monthly** | 1st 4:00 AM | Complete system snapshot | 365 days |

## 3-2-1 Strategy Implementation

### 3 Copies
1. **Original data** (production system)
2. **Local backup** (daily/weekly/monthly on local storage)
3. **Offsite backup** (replicated to external location)

### 2 Different Media
- **Local disk** (primary backup storage)
- **Network/Cloud storage** (offsite replication)

### 1 Offsite Copy
- **External drive** (mounted locally)
- **Remote server** (via rsync/SFTP)
- **Cloud storage** (AWS S3, etc.)

## Backup Components

### 1. Database Backups
- **PostgreSQL**: Full dumps with compression
- **SQLite**: Binary backup with integrity checks
- **Loki**: Time-series data archival
- **Automatic verification** and restoration testing

### 2. Configuration Backups
- Docker Compose files
- Agent configurations
- Environment variables (sanitized)
- Service configurations
- SSL certificates and keys

### 3. Agent State Backups
- Runtime state data
- Collective intelligence state
- Agent registry and status
- Workflow reports and history

### 4. AI Model Backups
- Ollama model files
- Model configurations
- Model registry and metadata
- Version tracking

### 5. Monitoring Data Retention
- Prometheus metrics (90-day retention)
- Grafana dashboards export
- Health reports and compliance data
- Performance metrics and trends

### 6. Log Archival
- Application logs with rotation
- System logs (filtered)
- Deployment and build logs
- Automatic compression and cleanup

## Verification and Testing

### Backup Verification
- **Checksum validation** for all backup files
- **Archive integrity** testing
- **Database restoration** validation
- **Configuration parsing** verification

### Restore Testing
- **Automated restore tests** (weekly)
- **Database restoration** to test environment
- **Configuration restoration** validation
- **End-to-end recovery** testing

## Monitoring and Alerting

### Health Checks
- Backup freshness (< 26 hours)
- Storage usage (< 85% full)
- Verification status
- Offsite replication status
- Restore test results

### Alert Methods
- **System logs** (always enabled)
- **Email notifications** (SMTP)
- **Slack webhooks**
- **Generic webhooks**
- **SMS** (via Twilio)

### Alert Conditions
- Backup failures
- Missing backups
- Storage space low
- Verification failures
- Offsite replication issues
- Restore test failures

## Command Line Usage

### Orchestrator Commands

```bash
# Run different backup types
python3 sutazai-backup-orchestrator.py daily
python3 sutazai-backup-orchestrator.py weekly
python3 sutazai-backup-orchestrator.py monthly

# Run with specific options
python3 sutazai-backup-orchestrator.py daily --parallel
python3 sutazai-backup-orchestrator.py daily --sequential

# Start scheduler (runs continuously)
python3 sutazai-backup-orchestrator.py schedule

# Check 3-2-1 strategy compliance
python3 sutazai-backup-orchestrator.py status
```

### Individual System Commands

```bash
# Database backups
python3 core/database-backup-system.py

# Configuration backups
python3 config/config-backup-system.py

# Agent state backups
python3 agents/agent-state-backup-system.py

# Model backups
python3 models/ollama-model-backup-system.py

# Verification
python3 verification/backup-verification-system.py

# Offsite replication
python3 offsite/offsite-backup-replication-system.py

# Restore testing
python3 restore/restore-testing-system.py

# Monitoring
python3 alerts/backup-monitoring-alerting-system.py
```

## Directory Structure

```
/opt/sutazaiapp/data/backups/
├── daily/           # Daily backup files
├── weekly/          # Weekly backup files
├── monthly/         # Monthly backup files
├── postgres/        # PostgreSQL dumps
├── sqlite/          # SQLite backups
├── config/          # Configuration backups
├── agents/          # Agent state backups
├── models/          # AI model backups
├── monitoring/      # Monitoring data
├── logs/            # Log archives
├── verification/    # Verification reports
├── offsite/         # Offsite replication status
└── restore_tests/   # Restore test results
```

## Environment Variables

```bash
# Database credentials
export POSTGRES_PASSWORD="your_postgres_password"

# Backup encryption (optional)
export BACKUP_ENCRYPTION_PASSPHRASE="your_encryption_key"

# AWS S3 (if using S3 offsite)
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"

# Email notifications (optional)
export BACKUP_SMTP_PASSWORD="your_smtp_password"

# Slack notifications (optional)
export BACKUP_SLACK_WEBHOOK="your_slack_webhook_url"

# SMS notifications (optional)
export TWILIO_ACCOUNT_SID="your_twilio_sid"
export TWILIO_AUTH_TOKEN="your_twilio_token"
```

## Troubleshooting

### Common Issues

1. **Permission errors**
   ```bash
   sudo chown -R root:root /opt/sutazaiapp/data/backups
   sudo chmod -R 755 /opt/sutazaiapp/scripts/backup-automation
   ```

2. **Storage space issues**
   ```bash
   # Check storage usage
   backup-status
   
   # Clean up old backups manually
   find /opt/sutazaiapp/data/backups -type f -mtime +30 -delete
   ```

3. **Service not starting**
   ```bash
   # Check service logs
   journalctl -u sutazai-backup.service -f
   
   # Check configuration
   python3 -c "import json; json.load(open('/opt/sutazaiapp/config/backup-config.json'))"
   ```

### Log Files

- **Service logs**: `/opt/sutazaiapp/logs/backup-service.log`
- **Monitor logs**: `/opt/sutazaiapp/logs/backup-monitor.log`
- **Cron logs**: `/opt/sutazaiapp/logs/backup-cron.log`
- **Individual system logs**: `/opt/sutazaiapp/logs/*-backup.log`

## Security Considerations

### Data Protection
- **Encryption at rest** (optional GPG encryption)
- **Secure credential storage** (environment variables)
- **Access control** (file permissions)
- **Network security** (encrypted transfers)

### Best Practices
- Regular restore testing
- Offsite backup verification
- Monitor backup integrity
- Keep multiple retention periods
- Document recovery procedures

## Performance Optimization

### Parallel Execution
- Multiple backup operations run concurrently
- Configurable concurrency limits
- Priority-based execution order

### Compression and Deduplication
- Automatic compression for large files
- Incremental backups where possible
- Storage usage optimization

### Resource Management
- Configurable timeout limits
- Memory usage monitoring
- Disk I/O optimization

## Recovery Procedures

### Database Recovery
1. Stop application services
2. Restore database from backup
3. Verify data integrity
4. Restart services

### Full System Recovery
1. Install base system
2. Restore configuration files
3. Restore databases
4. Restore agent states and models
5. Verify system functionality

### Partial Recovery
- Individual database restoration
- Configuration rollback
- Specific agent state recovery
- Model restoration

## Maintenance

### Regular Tasks
- Monitor backup success rates
- Review storage usage trends
- Test restore procedures
- Update retention policies
- Verify offsite replication

### Monthly Reviews
- Backup strategy effectiveness
- Storage capacity planning
- Recovery time objectives
- Compliance requirements

## Support and Monitoring

### Health Dashboard
The backup system provides comprehensive health monitoring through:
- Real-time status checks
- Historical trend analysis
- Performance metrics
- Compliance reporting

### Integration Points
- Prometheus metrics export
- Grafana dashboard templates
- REST API endpoints
- Webhook notifications

## Contributing

When extending the backup system:
1. Follow the existing modular architecture
2. Implement proper error handling
3. Add comprehensive logging
4. Include verification methods
5. Update configuration schemas
6. Document new features

## License

This backup automation system is part of the SutazAI platform and follows the same licensing terms.