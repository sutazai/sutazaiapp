# Memory Bank Cleanup Documentation

## Overview

The Memory Bank system's `activeContext.md` file was experiencing severe bloat, growing to 142MB with 1.8 million lines due to accumulation of duplicate entries without rotation. This documentation describes the cleanup solution implemented on 2025-08-20.

## Problem Analysis

### Original Issue
- **File Size**: 142MB (target: <1MB)
- **Line Count**: 1,824,532 lines
- **Entry Count**: 133 duplicate "Code Changes" sections
- **Date Range**: 8 days of accumulated data
- **Problem**: No automatic rotation or deduplication

### Root Cause
The memory-bank system was appending new context entries without:
1. Removing old entries
2. Detecting and removing duplicates
3. Implementing size limits or rotation
4. Archiving historical data

## Solution Implementation

### 1. Cleanup Scripts Created

#### `/opt/sutazaiapp/scripts/maintenance/memory_cleanup.py`
Comprehensive cleanup system with:
- Automatic backup creation with compression (92.5% compression ratio achieved)
- Entry-based archival by age (configurable retention period)
- Archive indexing for searchability
- Dry-run mode for safety
- Command-line interface for various operations

**Key Features:**
- Archives old entries to compressed files
- Creates searchable indexes
- Maintains forensic backups
- Provides detailed logging
- Supports multiple operation modes

**Usage:**
```bash
# Analyze file structure
python3 memory_cleanup.py --analyze

# Perform cleanup (keeps last 7 days by default)
python3 memory_cleanup.py --cleanup --max-age 7

# Monitor and auto-cleanup if needed
python3 memory_cleanup.py --monitor

# Search archives
python3 memory_cleanup.py --search "keyword"

# Dry run mode
python3 memory_cleanup.py --cleanup --dry-run
```

#### `/opt/sutazaiapp/scripts/maintenance/memory_deduplicator.py`
Aggressive deduplication for removing duplicate entries:
- Content-based hash comparison
- Preserves most recent unique entries
- Falls back to keeping only latest entry if file still too large

**Usage:**
```bash
python3 memory_deduplicator.py
```

#### `/opt/sutazaiapp/scripts/maintenance/memory_monitor.sh`
Bash monitoring script for automated cleanup:
- Checks file size against threshold
- Triggers cleanup when needed
- Falls back to deduplication if necessary
- Logs all operations

### 2. Automated Monitoring

#### Systemd Service and Timer
- **Service**: `memory-monitor.service` - Runs cleanup checks
- **Timer**: `memory-monitor.timer` - Schedules hourly checks

**Installation:**
```bash
# Copy service files to systemd directory
sudo cp /opt/sutazaiapp/scripts/maintenance/memory-monitor.* /etc/systemd/system/

# Reload systemd and enable timer
sudo systemctl daemon-reload
sudo systemctl enable memory-monitor.timer
sudo systemctl start memory-monitor.timer

# Check status
sudo systemctl status memory-monitor.timer
sudo systemctl list-timers memory-monitor.timer
```

### 3. Archive Structure

Archives are stored in `/opt/sutazaiapp/memory-bank/archives/` with:
- **Backup files**: `activeContext_backup_YYYYMMDD_HHMMSS.md.gz`
- **Archive files**: `archive_YYYYMMDD_HHMMSS.md.gz`
- **Index files**: `*.index.json` for searchability
- **Checksum files**: `*.sha256` for integrity verification

## Cleanup Results

### Before Cleanup
- File Size: 142.24MB
- Entries: 133 duplicate sections
- Lines: 1,824,532

### After Cleanup
- File Size: 0.78MB (803KB)
- Entries: 1 (most recent only)
- Space Saved: 141.46MB (99.5% reduction)
- Backup Created: 10.69MB compressed (92.5% compression)

### Performance Impact
- Reduced memory usage for applications reading the file
- Faster file operations
- Improved system responsiveness
- Maintained data integrity with full backups

## Maintenance Procedures

### Daily Operations
No manual intervention required - automated monitoring handles routine cleanup.

### Weekly Tasks
1. Check archive directory size:
```bash
du -sh /opt/sutazaiapp/memory-bank/archives/
```

2. Verify monitoring is active:
```bash
sudo systemctl status memory-monitor.timer
```

### Monthly Tasks
1. Review old archives (>30 days):
```bash
find /opt/sutazaiapp/memory-bank/archives -name "*.gz" -mtime +30 -ls
```

2. Clean up old archives if needed:
```bash
find /opt/sutazaiapp/memory-bank/archives -name "*.gz" -mtime +90 -delete
```

### Emergency Recovery

If activeContext.md is corrupted or lost:

1. List available backups:
```bash
ls -lah /opt/sutazaiapp/memory-bank/archives/*.gz
```

2. Restore from most recent backup:
```bash
# Decompress backup
gunzip -c /opt/sutazaiapp/memory-bank/archives/activeContext_backup_YYYYMMDD_HHMMSS.md.gz > /opt/sutazaiapp/memory-bank/activeContext.md
```

3. Verify restoration:
```bash
head -20 /opt/sutazaiapp/memory-bank/activeContext.md
```

## Prevention Strategies

### Implemented Safeguards
1. **Hourly monitoring** via systemd timer
2. **Automatic cleanup** when size exceeds 1MB
3. **Deduplication** removes duplicate entries
4. **Archive rotation** preserves old data compressed
5. **Backup creation** before any cleanup operation

### Best Practices
1. Keep activeContext.md under 1MB
2. Archive entries older than 7 days
3. Maintain compressed backups for 90 days
4. Monitor log files for cleanup issues
5. Test recovery procedures quarterly

## Troubleshooting

### Common Issues

#### Issue: Cleanup fails with permission error
**Solution**: Ensure script runs with appropriate permissions
```bash
sudo chown root:opt-admins /opt/sutazaiapp/memory-bank/activeContext.md
sudo chmod 664 /opt/sutazaiapp/memory-bank/activeContext.md
```

#### Issue: Archives consuming too much space
**Solution**: Adjust retention period or compression level
```bash
# Remove archives older than 30 days
find /opt/sutazaiapp/memory-bank/archives -name "*.gz" -mtime +30 -delete
```

#### Issue: Memory-bank not updating after cleanup
**Solution**: Restart dependent services
```bash
# Restart any services that depend on memory-bank
sudo systemctl restart your-service-name
```

## Monitoring and Alerts

### Log Files
- Cleanup logs: `/var/log/memory_cleanup_YYYYMMDD.log`
- Monitor logs: `/var/log/memory_monitor.log`
- Systemd logs: `journalctl -u memory-monitor.service`

### Key Metrics to Monitor
1. File size growth rate
2. Cleanup frequency
3. Archive directory size
4. Compression ratios
5. Cleanup failures

### Alert Thresholds
- WARN: activeContext.md > 800KB
- CRITICAL: activeContext.md > 1MB
- CRITICAL: Cleanup failure
- WARN: Archive directory > 1GB

## Related Systems

### Dependencies
- Python 3.8+
- Standard libraries only (no external dependencies)
- Systemd for scheduling
- Sufficient disk space for archives

### Integrated Systems
- Memory Bank feature system
- Any applications reading activeContext.md
- Backup and recovery systems

## Version History

### v1.0.0 (2025-08-20)
- Initial implementation
- Reduced 142MB file to 0.78MB
- Implemented automated monitoring
- Created comprehensive cleanup tooling
- Added archive search capability

## Contact and Support

For issues or questions about the memory cleanup system:
1. Check this documentation
2. Review log files for errors
3. Test with dry-run mode first
4. Create detailed issue report with logs

---

*Last Updated: 2025-08-20*
*Document Version: 1.0.0*