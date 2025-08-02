# SutazAI Codebase Reorganization System

A comprehensive, safe reorganization system for the SutazAI codebase that moves redundant files while preserving system stability.

## Overview

This system safely reorganizes the SutazAI codebase by:
- Moving ~150+ redundant/duplicate files to organized archive
- Preserving all critical system files
- Providing complete backup and restoration capabilities
- Testing system health after each phase
- Maintaining full operational stability

## Quick Start

To reorganize the entire codebase safely:

```bash
cd /opt/sutazaiapp/scripts/reorganization
./reorganize_codebase.sh
```

## Files Protected (Never Moved)

These critical files are NEVER moved during reorganization:

- `backend/app/main.py` - Active backend application
- `frontend/app.py` - Active frontend application  
- `backend/app/working_main.py` - Backup backend
- `docker-compose.minimal.yml` - Current deployment configuration
- `scripts/live_logs.sh` - Essential monitoring script
- `health_check.sh` - System health monitoring
- `requirements.txt` - Dependencies
- Environment files (`.env*`)

## Scripts Overview

### Master Script
- **`reorganize_codebase.sh`** - Main orchestration script that runs all phases

### Individual Phase Scripts
1. **`01_backup_system.sh`** - Creates complete system backup
2. **`02_create_archive_structure.sh`** - Creates organized archive directories
3. **`03_identify_files_to_move.sh`** - Analyzes and identifies files for movement
4. **`04_move_files_safely.sh`** - Moves files in phases with health checks
5. **`05_test_system_health.sh`** - Comprehensive system health testing

## Safety Features

### Complete Backup System
- Full system state backup before any changes
- Docker container states preserved
- All files backed up with timestamps
- Automatic restoration scripts generated

### Incremental Health Testing
- System health checked after each phase
- Automatic rollback on critical failures
- Container status monitoring
- API endpoint testing

### Emergency Recovery
- Automatic emergency rollback on failures
- Manual restoration capabilities
- Multiple recovery points
- Full system state restoration

## Archive Structure

Files are moved to organized archive directories:

```
archive/reorganization_YYYYMMDD_HHMMSS/
├── duplicates/
│   ├── monitoring/          # Duplicate monitoring scripts
│   ├── deployment/          # Duplicate deployment scripts
│   ├── testing/             # Duplicate test scripts
│   └── utilities/           # Redundant utility scripts
├── obsolete/
│   ├── old_versions/        # Previous versions
│   ├── deprecated/          # Deprecated functionality
│   └── unused/              # Unused scripts
├── redundant/
│   ├── scripts/             # Scripts with overlapping purposes
│   └── configs/             # Duplicate configurations
└── legacy/
    └── old_implementations/ # Historical implementations
```

## Usage Examples

### Full Reorganization (Recommended)
```bash
./reorganize_codebase.sh
```

### Individual Phases (Advanced)
```bash
# Phase 1: Backup
./01_backup_system.sh

# Phase 2: Create Archive Structure  
./02_create_archive_structure.sh

# Phase 3: Identify Files
./03_identify_files_to_move.sh

# Phase 4: Move Files
./04_move_files_safely.sh

# Phase 5: Health Test
./05_test_system_health.sh
```

### Restore from Archive
```bash
# Find latest backup
ls /opt/sutazaiapp/backups/reorganization_backup_*

# Restore specific file
/path/to/archive/restore_file.sh duplicates/monitoring/old_script.sh /opt/sutazaiapp/scripts/

# Full system restoration
/path/to/backup/restore.sh
```

## What Gets Moved

### Duplicate Monitoring Scripts (~20 files)
- Multiple system monitors (keeping `monitor_system.sh`)
- Duplicate health checks (keeping `health_check.sh`)
- Alternative log processors (keeping `live_logs.sh`)

### Duplicate Deployment Scripts (~25 files)
- Multiple deployment versions
- Backup deployment scripts
- Specific deployment cases
- Fixed versions that are now redundant

### Duplicate Testing Scripts (~30 files)
- Multiple coordinator test scripts
- Duplicate deployment tests
- Obsolete test versions
- Specialized test scripts

### Obsolete Configurations (~10 files)
- Old Docker Compose files
- Unused configuration files
- Deprecated settings

### Redundant Utilities (~50 files)
- Cleanup scripts with similar functionality
- Fix scripts for resolved issues
- Optimization scripts with overlapping purposes

## Pre-Flight Checks

The system performs comprehensive checks before starting:

- ✅ Sufficient disk space (1GB+ required)
- ✅ Docker daemon running
- ✅ Essential containers operational
- ✅ Critical files present
- ✅ Proper permissions

## Post-Reorganization Health Tests

Comprehensive testing includes:

- 🐳 Docker infrastructure
- 🔌 Backend API endpoints
- 🌐 Frontend accessibility
- 🤖 AI model availability
- 💾 Database connectivity
- 💻 System resources
- 📜 Essential scripts
- 🔄 End-to-end functionality

## Monitoring and Logs

All operations are logged to:
- `/opt/sutazaiapp/logs/reorganization.log` - Main log
- `/opt/sutazaiapp/logs/reorganization_master.log` - Master script log
- `/opt/sutazaiapp/logs/file_movements.log` - File movement tracking
- `/opt/sutazaiapp/logs/health_report_*.md` - Health test reports

## Recovery Procedures

### If Reorganization Fails
The system automatically attempts emergency rollback. If that fails:

1. Find latest backup: `ls /opt/sutazaiapp/backups/reorganization_backup_*`
2. Run restoration: `bash /path/to/backup/restore.sh`
3. Verify system health: `./05_test_system_health.sh`

### If Files Are Needed Later
1. List archived files: `/path/to/archive/list_archived.sh`
2. Restore specific file: `/path/to/archive/restore_file.sh`
3. Update any references in code

## Expected Results

After successful reorganization:

- **Scripts directory**: Reduced from 366+ files to ~50 essential files
- **Disk space**: ~200MB moved to organized archive
- **Maintenance**: Significantly easier navigation and maintenance  
- **Performance**: No impact on system performance
- **Stability**: All critical functionality preserved

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
sudo chown -R $USER:$USER /opt/sutazaiapp/scripts/reorganization
chmod +x /opt/sutazaiapp/scripts/reorganization/*.sh
```

**Docker Not Running**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

**Insufficient Disk Space**
Clean up before reorganization:
```bash
docker system prune -f
```

**Health Tests Failing**
Check specific component:
```bash
docker ps
curl http://localhost:8000/health
```

## Support

For issues or questions:

1. Check logs in `/opt/sutazaiapp/logs/`
2. Review health reports
3. Use backup restoration if needed
4. System provides detailed error messages and recovery instructions

## Advanced Usage

### Custom File Selection
Edit the identification script to modify which files are moved:
```bash
nano 03_identify_files_to_move.sh
```

### Archive Cleanup
After 7+ days of stable operation:
```bash
# Review archive contents
/path/to/archive/archive_stats.sh

# Remove archive if confident
rm -rf /path/to/archive/
```

### Integration with CI/CD
The scripts can be integrated into automated deployment pipelines with appropriate error handling and notification systems.

---

**Safety First**: This system prioritizes safety over speed. Every operation includes multiple safety checks, backups, and recovery options to ensure your SutazAI system remains operational throughout the reorganization process.