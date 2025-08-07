# SutazAI Human Oversight Interface

A comprehensive human oversight system for monitoring, controlling, and ensuring safe operation of 69 AI agents in the SutazAI system.

## 🛡️ Overview

The SutazAI Human Oversight Interface provides robust human control and monitoring capabilities for autonomous AI systems, ensuring safety, compliance, and accountability through:

- **Real-time Monitoring**: Live dashboard showing status of all 69 AI agents
- **Control Mechanisms**: Pause, resume, override, and emergency stop capabilities
- **Alert System**: Intelligent notification and escalation system
- **Compliance Reporting**: Automated compliance monitoring for multiple frameworks
- **Audit Trail**: Complete logging and tracking of all human interventions
- **Performance Analytics**: System health and performance monitoring

## 🏗️ Architecture

The system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                 Human Oversight Orchestrator                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Dashboard     │  │  Alert System   │  │  Compliance  │ │
│  │   Interface     │  │                 │  │   Reporter   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Agent Control  │  │  Audit Logger   │  │   Database   │ │
│  │    Manager      │  │                 │  │   Storage    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: aiohttp, sqlite3, jinja2, matplotlib, pandas, plotly
- 69 AI agents running in SutazAI system

### Installation

1. **Start the Oversight System**:
   ```bash
   cd /opt/sutazaiapp
   ./scripts/start-oversight-system.sh
   ```

2. **Access the Dashboard**:
   Open your browser and navigate to: `http://localhost:8095`

3. **Stop the System**:
   ```bash
   ./scripts/stop-oversight-system.sh
   ```

## 📊 Dashboard Features

### Real-time Overview
- **Agent Status Grid**: Visual status of all 69 agents
- **System Metrics**: CPU, memory, and performance indicators
- **Active Alerts**: Current system alerts requiring attention
- **Intervention History**: Recent human actions and overrides

### Agent Control Panel
- **Individual Agent Control**: Pause, resume, or stop specific agents
- **Bulk Operations**: Control multiple agents simultaneously
- **Parameter Override**: Modify agent parameters in real-time
- **Emergency Stop**: Immediate system-wide shutdown capability

### Approval Workflows
- **Pending Requests**: Tasks requiring human approval
- **Risk Assessment**: Automatic risk level evaluation
- **Decision Tracking**: Complete audit trail of approvals/rejections
- **Escalation Management**: Automatic escalation for urgent requests

### Alert Management
- **Severity Levels**: Critical/High/Medium/Low alert classification
- **Multi-channel Notifications**: Email, Slack, Teams, PagerDuty integration
- **Escalation Policies**: Automatic escalation based on response time
- **Alert Acknowledgment**: Track which alerts have been addressed

### Compliance Monitoring
- **Multiple Frameworks**: GDPR, HIPAA, AI Ethics, ISO27001, NIST
- **Automated Reports**: Daily, weekly, monthly compliance reports
- **Violation Detection**: Real-time compliance violation alerts
- **Remediation Tracking**: Monitor resolution of compliance issues

## 🔧 Configuration

### Main Configuration (`config.json`)

```json
{
  "oversight_port": 8095,
  "enable_compliance_reporting": true,
  "enable_alert_monitoring": true,
  "compliance_frameworks": ["ai_ethics", "gdpr", "hipaa"],
  "alert_thresholds": {
    "agent_failure_threshold": 3,
    "memory_usage_threshold": 90,
    "cpu_usage_threshold": 95
  },
  "monitoring_intervals": {
    "agent_health_check": 60,
    "system_metrics": 300,
    "compliance_check": 1800
  }
}
```

### Alert Configuration (`alert_config.json`)

```json
{
  "email": {
    "smtp_server": "localhost",
    "smtp_port": 587,
    "from_address": "sutazai-alerts@localhost"
  },
  "slack": {
    "webhook_url": "https://hooks.slack.com/...",
    "default_channel": "#sutazai-alerts"
  },
  "escalation": {
    "level_1_timeout": 300,
    "level_2_timeout": 900,
    "level_3_timeout": 1800
  }
}
```

## 🎯 Key Capabilities

### 1. Agent Control Mechanisms

**Pause Agent**:
```python
# Via API or dashboard
POST /api/agents/{agent_id}/pause
{
  "operator_id": "human_operator",
  "reason": "Maintenance required"
}
```

**Emergency Stop**:
```python
# System-wide emergency stop
POST /api/system/emergency_stop
{
  "operator_id": "emergency_responder",
  "reason": "Critical system issue detected" 
}
```

**Parameter Override**:
```python
# Override agent parameters
POST /api/agents/{agent_id}/override
{
  "override_type": "parameter_change",
  "previous_value": "old_setting",
  "new_value": "new_setting",
  "operator_id": "system_admin"
}
```

### 2. Alert System

**Create Alert**:
```python
await alert_system.create_alert(
    title="High Memory Usage",
    description="System memory usage exceeded 90%",
    severity=AlertSeverity.HIGH,
    category=AlertCategory.RESOURCE_EXHAUSTION,
    metadata={"memory_usage": 92.5}
)
```

**Notification Channels**:
- Email with HTML templates
- Slack integration with rich formatting
- Microsoft Teams webhooks
- PagerDuty incident management
- Custom webhooks for third-party systems

### 3. Compliance Reporting

**Generate Report**:
```python
report = await compliance_reporter.generate_compliance_report(
    framework=ComplianceFramework.GDPR,
    report_type=ReportType.MONTHLY
)
```

**Supported Frameworks**:
- **AI Ethics**: Algorithmic transparency, bias detection, human oversight
- **GDPR**: Data processing consent, right to erasure, privacy by design
- **HIPAA**: PHI encryption, access controls, audit logging
- **SOX**: Financial data accuracy, internal controls
- **ISO27001**: Information security policy, risk assessment
- **NIST**: Cybersecurity framework implementation

### 4. Audit Trail

All actions are automatically logged with:
- **Event Type**: What action was performed
- **Operator ID**: Who performed the action
- **Timestamp**: When the action occurred
- **Before/After State**: What changed
- **Compliance Tags**: Relevant compliance frameworks
- **Metadata**: Additional context and details

## 📈 Monitoring and Analytics

### System Health Metrics
- **Agent Status**: Health, performance, and availability
- **Resource Usage**: CPU, memory, disk, network utilization
- **Response Times**: Agent and system response performance
- **Error Rates**: Failure rates and error patterns
- **Queue Depths**: Task queue monitoring

### Performance Analytics
- **Trend Analysis**: Historical performance trends
- **Anomaly Detection**: Unusual behavior identification
- **Capacity Planning**: Resource usage forecasting
- **SLA Monitoring**: Service level agreement tracking

### Compliance Metrics
- **Compliance Percentage**: Overall compliance score
- **Violation Trends**: Compliance violation patterns
- **Remediation Time**: Time to resolve violations
- **Framework Coverage**: Coverage across compliance frameworks

## 🔒 Security Features

### Access Control
- **Role-based Access**: Different permission levels for operators
- **Session Management**: Secure session handling
- **Authentication**: Configurable authentication mechanisms
- **Audit Logging**: Complete access audit trail

### Data Protection
- **Encryption**: All sensitive data encrypted at rest
- **Secure Communications**: TLS/SSL for all network communications
- **Data Retention**: Configurable data retention policies
- **Privacy Controls**: GDPR-compliant data handling

## 🚨 Emergency Procedures

### Emergency Stop Protocol
1. **Immediate Action**: All agents stopped within 5 seconds
2. **Safety State**: System enters safe mode
3. **Notification**: Critical alerts sent to all escalation levels
4. **Audit Logging**: Complete record of emergency stop action
5. **Recovery Plan**: Structured recovery procedure

### Escalation Procedures
1. **Level 1** (5 minutes): First responders notified
2. **Level 2** (15 minutes): Team leads notified
3. **Level 3** (30 minutes): Management notified
4. **Level 4** (60 minutes): Executive escalation

## 📋 API Reference

### Agent Control Endpoints
- `GET /api/agents/status` - Get all agent statuses
- `POST /api/agents/{id}/pause` - Pause specific agent
- `POST /api/agents/{id}/resume` - Resume paused agent
- `POST /api/agents/{id}/emergency_stop` - Emergency stop agent
- `POST /api/agents/{id}/override` - Create parameter override

### Alert Management Endpoints
- `GET /api/alerts` - Get active alerts
- `POST /api/alerts/{id}/acknowledge` - Acknowledge alert
- `POST /api/alerts/{id}/resolve` - Resolve alert

### Approval Workflow Endpoints
- `GET /api/approvals` - Get pending approvals
- `POST /api/approvals/{id}/approve` - Approve request
- `POST /api/approvals/{id}/reject` - Reject request

### Compliance Endpoints
- `GET /api/compliance/reports` - Get compliance reports
- `POST /api/compliance/generate` - Generate new report
- `GET /api/compliance/violations` - Get active violations

## 🔧 Troubleshooting

### Common Issues

**System Won't Start**:
```bash
# Check configuration
python3 -m json.tool /opt/sutazaiapp/backend/oversight/config.json

# Check permissions
ls -la /opt/sutazaiapp/backend/oversight/

# Check logs
tail -f /opt/sutazaiapp/logs/oversight/orchestrator.log
```

**Dashboard Not Loading**:
- Verify port 8095 is not blocked by firewall
- Check if process is running: `ps aux | grep oversight`
- Examine browser console for errors

**Alerts Not Being Sent**:
- Verify notification configuration in `alert_config.json`
- Check SMTP/Slack/Teams credentials
- Review alert logs for error messages

**Database Issues**:
```bash
# Check database integrity
sqlite3 /opt/sutazaiapp/backend/oversight/oversight.db ".schema"

# Reset database (WARNING: This will delete all data)
rm /opt/sutazaiapp/backend/oversight/oversight.db
./scripts/start-oversight-system.sh
```

### Log Files
- **Main System**: `/opt/sutazaiapp/logs/oversight/orchestrator.log`
- **Dashboard**: `/opt/sutazaiapp/logs/oversight/dashboard.log`
- **Alerts**: `/opt/sutazaiapp/logs/oversight/alerts.log`
- **Compliance**: `/opt/sutazaiapp/logs/oversight/compliance.log`

## 📞 Support and Contact

For support with the Human Oversight Interface:

1. **Check Documentation**: Review this README and inline code documentation
2. **Check Logs**: Examine system logs for error messages
3. **Configuration Review**: Verify all configuration files are valid
4. **System Status**: Use `./scripts/check-oversight-status.sh` to check system health

## 🔄 Updates and Maintenance

### Regular Maintenance
- **Weekly**: Review compliance reports and violations
- **Monthly**: Analyze system performance and capacity
- **Quarterly**: Update compliance framework requirements
- **Annually**: Security audit and penetration testing

### Backup Procedures
```bash
# Backup database
cp /opt/sutazaiapp/backend/oversight/oversight.db /backup/oversight-$(date +%Y%m%d).db

# Backup configuration
tar -czf /backup/oversight-config-$(date +%Y%m%d).tar.gz \
    /opt/sutazaiapp/backend/oversight/*.json

# Backup reports
tar -czf /backup/oversight-reports-$(date +%Y%m%d).tar.gz \
    /opt/sutazaiapp/backend/oversight/reports/
```

## 📜 License and Compliance

This system is designed to help ensure compliance with various regulatory frameworks including GDPR, HIPAA, SOX, and AI ethics guidelines. Regular audits and updates are recommended to maintain compliance status.

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Compatibility**: SutazAI System v40+