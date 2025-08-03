# 🚀 COMPLIANCE MONITORING SYSTEM - USAGE GUIDE

## Quick Start Commands

### 1. 📊 View Current Compliance Status
```bash
# Check the dashboard (RECOMMENDED - START HERE)
cd /opt/sutazaiapp
./scripts/monitoring/compliance-dashboard.sh

# View latest compliance report
cat /opt/sutazaiapp/compliance-reports/latest.json | jq '.'

# Get just the compliance score
cat /opt/sutazaiapp/compliance-reports/latest.json | jq '.compliance_score'
```

### 2. 🔍 Run Manual Compliance Check
```bash
# Quick check (report only, no fixes)
cd /opt/sutazaiapp
python scripts/monitoring/continuous-compliance-monitor.py --report-only

# Check with auto-fix enabled
python scripts/monitoring/continuous-compliance-monitor.py --fix

# Run specific rule check (example: Rule 1 - No Fantasy Elements)
python scripts/monitoring/continuous-compliance-monitor.py --rule 1
```

### 3. 🛡️ Check Monitoring Service Status
```bash
# Check if the monitoring daemon is running
sudo systemctl status sutazai-compliance-monitor

# View real-time logs
sudo journalctl -u sutazai-compliance-monitor -f

# Start/Stop/Restart the service
sudo systemctl start sutazai-compliance-monitor
sudo systemctl stop sutazai-compliance-monitor
sudo systemctl restart sutazai-compliance-monitor
```

### 4. 📈 View Historical Reports
```bash
# List all compliance reports
ls -la /opt/sutazaiapp/compliance-reports/

# View a specific report
cat /opt/sutazaiapp/compliance-reports/report_20250804_000924.json | jq '.'

# Compare compliance scores over time
for report in /opt/sutazaiapp/compliance-reports/report_*.json; do
    echo -n "$(basename $report): "
    jq '.compliance_score' "$report"
done
```

## 📋 Detailed Usage Scenarios

### Scenario 1: Daily Compliance Check
```bash
# Morning routine - check overnight compliance
cd /opt/sutazaiapp

# 1. View dashboard summary
./scripts/monitoring/compliance-dashboard.sh

# 2. If violations found, see details
cat compliance-reports/latest.json | jq '.violations_by_rule'

# 3. Auto-fix safe violations
python scripts/monitoring/continuous-compliance-monitor.py --fix
```

### Scenario 2: Before Making Code Changes
```bash
# Run pre-commit validation manually
cd /opt/sutazaiapp

# Test all pre-commit hooks
pre-commit run --all-files

# Test specific hook
pre-commit run enforce-no-fantasy-elements --all-files
```

### Scenario 3: After Major Changes
```bash
# Run comprehensive analysis
cd /opt/sutazaiapp

# 1. Full hygiene check with all agents
python scripts/hygiene-enforcement-coordinator.py --all-phases

# 2. Generate detailed report
python scripts/monitoring/continuous-compliance-monitor.py --report-only

# 3. Review violations by severity
cat compliance-reports/latest.json | jq '.violations_by_rule | to_entries | sort_by(.value | length) | reverse'
```

### Scenario 4: Monthly Deep Cleanup
```bash
# Run monthly cleanup manually (normally automated)
cd /opt/sutazaiapp
python scripts/monitoring/monthly-cleanup.py --force
```

## 🔧 Configuration & Customization

### View Current Schedule
```bash
# Check cron jobs
crontab -l | grep sutazai

# Output shows:
# 0 * * * * - Hourly quick check
# 0 2 * * * - Daily full check with auto-fix
# 0 3 * * 0 - Weekly deep analysis
# 0 4 1 * * - Monthly cleanup
```

### Modify Monitoring Settings
```bash
# Edit monitoring configuration
nano /opt/sutazaiapp/scripts/monitoring/continuous-compliance-monitor.py

# Key settings:
# - rule_cache_ttl = 60  # Cache rule states for 60 seconds
# - forbidden_terms = ["magic", "wizard", "black-box", "teleport"]
# - garbage_patterns = ["*.backup", "*.bak", "*.old", "*.tmp"]
```

### Disable/Enable Specific Rules
```bash
# Edit rules configuration
nano /opt/sutazaiapp/config/hygiene-agents.json

# Set rule enabled/disabled status
```

## 📊 Understanding Reports

### Report Structure
```json
{
  "timestamp": "2025-08-04T00:09:24.874974",
  "compliance_score": 87.5,        // Overall percentage
  "total_violations": 158,         // Total issues found
  "rules_violated": 4,            // Number of rules with violations
  "violations_by_rule": {         // Detailed violations per rule
    "1": [...],                  // Rule 1 violations
    "7": [...],                  // Rule 7 violations
    "12": [...],                 // Rule 12 violations
    "13": [...]                  // Rule 13 violations
  },
  "auto_fixable_count": 158,     // Violations that can be auto-fixed
  "critical_violations": 0       // High-severity issues
}
```

### Violation Details
```json
{
  "rule_number": 1,
  "rule_name": "No Fantasy Elements",
  "severity": "high",
  "file_path": "/opt/sutazaiapp/scripts/test.py",
  "line_number": 42,
  "description": "Found forbidden term 'magic' in code",
  "timestamp": "2025-08-04T00:09:24.066437",
  "auto_fixable": true
}
```

## 🚨 Troubleshooting

### Issue: Monitoring Service Not Running
```bash
# Check service logs
sudo journalctl -u sutazai-compliance-monitor --since "1 hour ago"

# Reinstall service
cd /opt/sutazaiapp
./scripts/monitoring/setup-compliance-monitoring.sh
```

### Issue: Pre-commit Hooks Not Working
```bash
# Reinstall hooks
cd /opt/sutazaiapp
pre-commit uninstall
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

### Issue: Reports Not Generating
```bash
# Check permissions
ls -la /opt/sutazaiapp/compliance-reports/
mkdir -p /opt/sutazaiapp/compliance-reports
chmod 755 /opt/sutazaiapp/compliance-reports

# Run manual report
python scripts/monitoring/continuous-compliance-monitor.py --report-only
```

## 🎯 Quick Reference Card

```bash
# MOST COMMON COMMANDS - SAVE THIS!

# 1. CHECK STATUS (use this first)
./scripts/monitoring/compliance-dashboard.sh

# 2. RUN MANUAL CHECK
python scripts/monitoring/continuous-compliance-monitor.py --report-only

# 3. FIX VIOLATIONS
python scripts/monitoring/continuous-compliance-monitor.py --fix

# 4. VIEW LOGS
tail -f /opt/sutazaiapp/logs/compliance-monitor.log

# 5. CHECK SERVICE
sudo systemctl status sutazai-compliance-monitor
```

## 📧 Alerts & Notifications

Currently, alerts are logged to:
- `/opt/sutazaiapp/logs/compliance-monitor.log`
- `/opt/sutazaiapp/logs/compliance-hourly.log`
- `/opt/sutazaiapp/logs/compliance-daily.log`

To add email/Slack alerts, edit:
```bash
nano /opt/sutazaiapp/scripts/monitoring/continuous-compliance-monitor.py
# Add notification logic in generate_report() method
```

## 🔗 Integration Points

### CI/CD Integration
```yaml
# Add to your CI/CD pipeline
- name: Run Compliance Check
  run: |
    cd /opt/sutazaiapp
    python scripts/monitoring/continuous-compliance-monitor.py --report-only
    if [ $? -ne 0 ]; then
      echo "Compliance check failed"
      exit 1
    fi
```

### Git Hook Integration
Already installed! Every commit is validated automatically.

### API Access (Future)
The coordinator API is available at:
- `http://localhost:8100/api/v1/coordinator/status`
- `http://localhost:8100/api/v1/coordinator/tasks`

---

## Need Help?

1. Check the dashboard first: `./scripts/monitoring/compliance-dashboard.sh`
2. View detailed logs: `tail -f logs/compliance-monitor.log`
3. Run manual check: `python scripts/monitoring/continuous-compliance-monitor.py`

The system is designed to be self-monitoring and self-healing. Most issues will be automatically detected and fixed!