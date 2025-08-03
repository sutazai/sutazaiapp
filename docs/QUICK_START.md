# 🚀 SUTAZAI COMPLIANCE MONITOR - QUICK START

## Step 1: Check Current Status (START HERE!)
```bash
cd /opt/sutazaiapp
./scripts/monitoring/compliance-dashboard.sh
```

This shows:
- Current compliance score (target: 85%+)
- Number of violations
- Recent monitoring activity

## Step 2: If You See Violations
```bash
# View details of what's wrong
cat compliance-reports/latest.json | jq '.violations_by_rule'

# Auto-fix safe violations
python scripts/monitoring/continuous-compliance-monitor.py --fix
```

## Step 3: Manual Checks (Optional)
```bash
# Quick compliance check (no changes)
python scripts/monitoring/continuous-compliance-monitor.py --report-only

# Check if monitoring service is running
sudo systemctl status sutazai-compliance-monitor
```

## 📊 What The Numbers Mean
- **Compliance Score**: Percentage of rules followed (87.5% = B+ grade)
- **Total Violations**: Issues found (lower is better)
- **Auto-fixable**: Violations that can be fixed automatically

## 🔧 Most Common Issues & Fixes

### "Fantasy Elements" (Rule 1)
- Found words like "magic", "wizard" in code
- Usually in test files (safe to ignore)

### "Scripts Outside /scripts/" (Rule 7)
- Scripts not in the /scripts/ directory
- Move them or add to exceptions

### "Garbage Files" (Rule 13)
- Backup files (*.bak, *.old, *.tmp)
- Run auto-fix to clean them

## 🚨 If Something's Wrong
```bash
# Check logs
tail -f /opt/sutazaiapp/logs/compliance-monitor.log

# Restart monitoring
sudo systemctl restart sutazai-compliance-monitor

# Run full system check
python scripts/hygiene-enforcement-coordinator.py --all-phases
```

## ⏰ Automatic Schedule
- **Every Hour**: Quick compliance check
- **2 AM Daily**: Full check with auto-fix
- **Sunday 3 AM**: Deep analysis
- **1st of Month 4 AM**: Major cleanup

## 💡 Remember
- The system auto-fixes most issues
- Check dashboard daily: `./scripts/monitoring/compliance-dashboard.sh`
- Commits are automatically validated (pre-commit hooks)
- Monitoring runs 24/7 in background

---
**Need the full guide?** See `/opt/sutazaiapp/COMPLIANCE_SYSTEM_USAGE_GUIDE.md`