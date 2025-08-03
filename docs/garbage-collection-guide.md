# Garbage Collection System - Complete Guide

## Overview

The Sutazai Garbage Collection System is a sophisticated, automated solution that manages all report files, logs, and temporary files with zero manual intervention required.

## Features

- **Automatic Report Archival**: Archives important reports before deletion
- **Log Rotation & Compression**: Compresses large logs and rotates based on age
- **Temporary File Cleanup**: Removes temp files, caches, and build artifacts
- **Configurable Retention Policies**: Customizable retention for different file types
- **Real-time Disk Monitoring**: Tracks disk usage and prevents overflow
- **Integration with Hygiene System**: Works seamlessly with existing compliance monitoring
- **Dashboard & Reporting**: Web-based dashboard for monitoring status
- **Database Tracking**: SQLite database for historical tracking

## Quick Start

### 1. Installation

```bash
# Run the setup script (requires root)
sudo /opt/sutazaiapp/scripts/setup-garbage-collection.sh
```

### 2. Check Status

```bash
# View service status
sudo systemctl status sutazai-garbage-collection

# Check current statistics
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py --status

# View logs
sudo journalctl -u sutazai-garbage-collection -f
```

### 3. Access Dashboard

Open in browser: `/opt/sutazaiapp/dashboard/garbage-collection.html`

The dashboard auto-refreshes every 60 seconds.

## Configuration

Edit `/opt/sutazaiapp/config/garbage-collection.json`:

```json
{
  "logs": {
    "deployment_*.log": 7,      // Keep deployment logs for 7 days
    "health_*.log": 3,          // Keep health logs for 3 days
    "compliance-*.log": 14,     // Keep compliance logs for 14 days
    "*.log": 30,               // Keep other logs for 30 days
    "max_size_mb": 100         // Compress logs larger than 100MB
  },
  "reports": {
    "*_report_*.json": 7,      // Keep timestamped reports for 7 days
    "latest.json": -1,         // Never delete files named 'latest.json'
    "*.json": 14              // Keep other JSON files for 14 days
  },
  "temporary": {
    "*.tmp": 1,               // Delete .tmp files after 1 day
    "*.bak": 3,               // Delete .bak files after 3 days
    "__pycache__": 7          // Delete Python cache after 7 days
  },
  "archives": {
    "retention_days": 90      // Delete archives after 90 days
  }
}
```

## Usage Commands

### Manual Collection

```bash
# Dry run (see what would be deleted without making changes)
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py --dry-run

# Run actual collection
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py

# Generate dashboard
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py --dashboard
```

### Integration with Compliance System

```bash
# Check garbage collection health and update compliance report
python3 /opt/sutazaiapp/scripts/garbage-collection-monitor.py

# View integration status
cat /opt/sutazaiapp/compliance-reports/latest.json | jq '.services.garbage_collection'
```

## Monitoring & Alerts

### Health Checks

The system monitors:
- Last run time (warns if > 2 hours)
- Error count in last run
- Disk usage percentage
- Archive directory size

### Compliance Integration

```bash
# The garbage collection status appears in compliance reports
cat /opt/sutazaiapp/compliance-reports/latest.json | jq '{
  gc_status: .services.garbage_collection.status,
  gc_health: .services.garbage_collection.health_score,
  disk_usage: .services.garbage_collection.details.disk_usage
}'
```

## File Lifecycle

1. **Active Phase**: Files are in use
2. **Compression**: Large log files are gzipped when they exceed size limits
3. **Archival**: Important files are moved to `/opt/sutazaiapp/archive/garbage-collection/`
4. **Deletion**: Files are removed based on retention policies
5. **Archive Cleanup**: Archives older than 90 days are deleted

## Database Schema

The system uses SQLite to track all operations:

- `file_tracking`: Tracks all processed files
- `collection_runs`: Statistics for each collection run
- `disk_usage`: Historical disk usage data

Query examples:

```sql
-- View recent collection runs
sqlite3 /opt/sutazaiapp/data/garbage-collection.db \
  "SELECT * FROM collection_runs ORDER BY run_time DESC LIMIT 10"

-- Check disk usage trend
sqlite3 /opt/sutazaiapp/data/garbage-collection.db \
  "SELECT timestamp, percent_used FROM disk_usage ORDER BY timestamp DESC LIMIT 24"
```

## Troubleshooting

### Service Not Running

```bash
# Check service status
sudo systemctl status sutazai-garbage-collection

# Check for errors
sudo journalctl -u sutazai-garbage-collection -n 50

# Restart service
sudo systemctl restart sutazai-garbage-collection
```

### High Disk Usage

```bash
# Run immediate collection
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py

# Check what's using space
du -sh /opt/sutazaiapp/* | sort -h
```

### Configuration Issues

```bash
# Validate configuration
python3 -c "import json; json.load(open('/opt/sutazaiapp/config/garbage-collection.json'))"

# Reset to defaults
rm /opt/sutazaiapp/config/garbage-collection.json
python3 /opt/sutazaiapp/scripts/garbage-collection-system.py --dry-run
```

## Best Practices

1. **Monitor Dashboard Daily**: Check `/opt/sutazaiapp/dashboard/garbage-collection.html`
2. **Review Retention Policies**: Adjust based on your needs
3. **Check Compliance Integration**: Ensure GC health stays above 70%
4. **Archive Important Data**: Move critical data outside the managed directories
5. **Test Configuration Changes**: Always use `--dry-run` first

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Scheduler     │────▶│   Collector  │────▶│   Archive   │
│  (every hour)   │     │   (Python)   │     │  Directory  │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Systemd       │     │   SQLite     │     │  Dashboard  │
│   Service       │     │   Database   │     │   (HTML)    │
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Performance Impact

- CPU Usage: < 20% (limited by systemd)
- Memory Usage: < 512MB (limited by systemd)
- Disk I/O: Throttled to prevent system impact
- Run Duration: Typically 10-60 seconds per cycle

## Security

- Runs with limited privileges
- No network access required
- Logs sensitive operations
- Preserves file permissions in archives

## Support

For issues or questions:
1. Check logs: `journalctl -u sutazai-garbage-collection`
2. Review dashboard: `/opt/sutazaiapp/dashboard/garbage-collection.html`
3. Check database: `sqlite3 /opt/sutazaiapp/data/garbage-collection.db`