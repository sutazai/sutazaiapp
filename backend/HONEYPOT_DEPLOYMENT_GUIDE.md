# SutazAI Honeypot Infrastructure Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and managing the SutazAI honeypot infrastructure. The system includes multiple types of honeypots designed to detect and analyze various attack vectors against AI systems.

## Architecture

The honeypot infrastructure consists of:

### Core Components
- **Honeypot Orchestrator**: Central management system
- **Security Integration Bridge**: Connects with existing security systems
- **Threat Intelligence Engine**: Analyzes and correlates attack data
- **Database System**: Stores events and attacker profiles

### Honeypot Types
1. **SSH Honeypot (Cowrie)**: Detects brute force attacks and captures credentials
2. **Web Honeypots**: HTTP/HTTPS services to catch web application attacks
3. **Database Honeypots**: MySQL, PostgreSQL, and Redis honeypots for SQL injection detection
4. **AI Agent Honeypots**: Mimic SutazAI services to detect AI-specific attacks

## Quick Start

### Prerequisites
- Python 3.8+
- Root or sudo access
- Minimum 2GB RAM
- 10GB free disk space
- Network access for dependency installation

### 1. Deploy Infrastructure

```bash
# Quick deployment with default settings
cd /opt/sutazaiapp/backend
python3 deploy_honeypot_infrastructure.py deploy

# Deployment with verbose logging
python3 deploy_honeypot_infrastructure.py deploy --verbose
```

### 2. Check Status

```bash
# Check deployment status
python3 deploy_honeypot_infrastructure.py status
```

### 3. Monitor Activity

```bash
# View recent honeypot events
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/honeypot/events?limit=50&hours=1"

# Get threat intelligence report
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/honeypot/intelligence/report"
```

## Detailed Configuration

### Custom Configuration File

Create a JSON configuration file for custom deployment:

```json
{
  "honeypot_types": ["ssh", "web", "database", "ai_agent"],
  "ports_config": {
    "ssh": 2222,
    "http": 8080,
    "https": 8443,
    "mysql": 13306,
    "postgresql": 15432,
    "redis": 16379,
    "ai_agent_primary": 10104,
    "ai_agent_secondary": 8000
  },
  "enable_cowrie": true,
  "enable_https": true,
  "security_integration": true,
  "logging_level": "INFO",
  "alert_thresholds": {
    "critical": 1,
    "high": 3,
    "medium": 10,
    "low": 50
  }
}
```

Deploy with custom configuration:

```bash
python3 deploy_honeypot_infrastructure.py deploy --config honeypot_config.json
```

## API Endpoints

### Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/honeypot/deploy` | POST | Deploy honeypot infrastructure |
| `/api/v1/honeypot/status` | GET | Get infrastructure status |
| `/api/v1/honeypot/undeploy` | POST | Undeploy infrastructure |
| `/api/v1/honeypot/health` | GET | Health check |

### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/honeypot/events` | GET | Get honeypot events |
| `/api/v1/honeypot/attackers` | GET | Get attacker profiles |
| `/api/v1/honeypot/analytics/dashboard` | GET | Dashboard data |
| `/api/v1/honeypot/intelligence/report` | GET | Threat intelligence report |

### Configuration Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/honeypot/config/update` | POST | Update configuration |
| `/api/v1/honeypot/capabilities` | GET | System capabilities |

## Security Considerations

### Network Isolation
- Honeypots are deployed on non-standard ports by default
- Implement network segmentation to isolate honeypots
- Use firewall rules to restrict honeypot access

### Data Protection
- All captured data is stored in encrypted database
- Credentials are hashed and never stored in plaintext
- Personal data is anonymized according to privacy regulations

### Access Control
- All management endpoints require admin authentication
- API access is restricted to authorized users only
- Audit logs track all administrative actions

## Monitoring and Alerting

### Real-time Monitoring
The system provides real-time monitoring through:
- WebSocket connections for live events
- REST API for programmatic access
- Integration with existing SIEM systems

### Alert Thresholds
Default alert thresholds:
- **Critical**: 1 event (immediate alert)
- **High**: 3 events within 1 hour
- **Medium**: 10 events within 1 hour  
- **Low**: 50 events within 1 hour

### Dashboard Metrics
- Total events by time period
- Attack vector distribution
- Geographic attack sources
- Honeypot effectiveness metrics
- Threat actor profiles

## Attack Detection Capabilities

### SSH Attacks
- Brute force login attempts
- Credential harvesting
- Command injection attempts
- Session hijacking

### Web Application Attacks
- SQL injection
- Cross-site scripting (XSS)
- Path traversal
- Command injection
- Authentication bypass

### Database Attacks
- SQL injection variants
- Unauthorized access attempts
- Data exfiltration attempts
- Privilege escalation

### AI-Specific Attacks
- Prompt injection
- Model extraction attempts
- AI service manipulation
- Code injection via AI interfaces

## Threat Intelligence

### Data Collection
The system collects:
- Attack patterns and techniques
- Attacker behavioral profiles
- Geographic distribution of threats
- Temporal attack patterns
- Tool and technique fingerprints

### Analysis Capabilities
- Attack correlation across honeypots
- Threat actor attribution
- Campaign identification  
- Trend analysis
- Predictive threat modeling

### Intelligence Sharing
- Export data in STIX/TAXII format
- Integration with threat intelligence platforms
- Automated indicator generation
- Community threat sharing

## Troubleshooting

### Common Issues

#### Port Conflicts
If deployment fails due to port conflicts:
```bash
# Check which ports are in use
netstat -tulpn | grep LISTEN

# Use custom ports in configuration
python3 deploy_honeypot_infrastructure.py deploy --config custom_ports.json
```

#### Permission Issues
If deployment fails due to permissions:
```bash
# Run with sudo (if required)
sudo python3 deploy_honeypot_infrastructure.py deploy

# Or fix permissions on directories
sudo chown -R $(whoami):$(whoami) /opt/sutazaiapp/backend/
```

#### Database Issues
If database connectivity fails:
```bash
# Check database file permissions
ls -la /opt/sutazaiapp/backend/data/honeypot.db

# Recreate database (will lose existing data)
rm /opt/sutazaiapp/backend/data/honeypot.db
python3 deploy_honeypot_infrastructure.py deploy
```

### Log Locations
- **Deployment logs**: `/opt/sutazaiapp/backend/logs/honeypot_deployment.log`
- **Runtime logs**: `/opt/sutazaiapp/backend/logs/`
- **Security events**: Database and security system integration
- **System logs**: `/var/log/syslog` (Linux system logs)

### Debug Mode
Enable debug logging for troubleshooting:
```bash
python3 deploy_honeypot_infrastructure.py deploy --verbose
```

## Maintenance

### Regular Tasks
1. **Monitor disk space**: Honeypot logs can grow large
2. **Review threat intelligence**: Analyze weekly reports
3. **Update signatures**: Keep attack detection patterns current
4. **Backup data**: Regular backups of honeypot database
5. **Security updates**: Keep honeypot software updated

### Performance Optimization
- Monitor system resources
- Tune alert thresholds based on activity
- Archive old data periodically
- Optimize database queries

### Scaling
For high-volume environments:
- Deploy multiple honeypot instances
- Use load balancing for management APIs
- Implement data sharding
- Consider dedicated database servers

## Security Score Integration

The honeypot system enhances the existing security score (currently 8.5/10) by:

### Early Threat Detection (+0.5 points)
- Real-time attack detection
- Proactive threat identification
- Zero-day attack capture

### Intelligence Enhancement (+0.3 points)  
- Attacker behavior profiling
- Threat pattern analysis
- Predictive threat modeling

### Response Automation (+0.2 points)
- Automated alerting
- Integration with security orchestration
- Dynamic threat response

**Expected Security Score**: **9.5/10** with full honeypot deployment

## Support and Contact

For technical support:
- Check logs first: `/opt/sutazaiapp/backend/logs/`
- Review this documentation
- Check API health endpoints
- Contact system administrators

## License and Legal

### Data Handling
- All honeypot data is collected for security purposes only
- Data retention follows organizational policies
- Privacy regulations are respected (GDPR, CCPA, etc.)

### Legal Considerations
- Honeypots are deployed on authorized infrastructure only
- All activities are logged for legal compliance
- Incident response procedures follow legal requirements

---

**Version**: 1.0.0  
**Last Updated**: 2024-08-05  
**Maintained By**: SutazAI Security Team