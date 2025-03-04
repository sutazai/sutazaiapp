# Two-Way Server Synchronization System

## Overview

This system provides bidirectional synchronization between the code server (192.168.100.28) and the deployment/test server (192.168.100.100).

## Components

1. **SSH Key Exchange**: Handles authentication between servers
2. **Two-Way Sync Script**: Performs the actual file synchronization
3. **Sync Monitor**: Continuously monitors for changes
4. **Git Hook**: Triggers synchronization after Git commits
5. **Systemd Service**: Ensures the monitor runs continuously
6. **Cron Job**: Performs regular synchronization as a backup

## Usage

### Manual Synchronization

To manually trigger synchronization:

```bash
# Sync from code server to deployment server
/opt/sutazaiapp/scripts/two_way_sync.sh --to-deploy

# Sync from deployment server to code server
/opt/sutazaiapp/scripts/two_way_sync.sh --to-code

# Perform a dry run (no actual changes)
/opt/sutazaiapp/scripts/two_way_sync.sh --dry-run
```

### Conflict Resolution

By default, conflicts are resolved by taking the newer file. This behavior can be configured in:

```
/opt/sutazaiapp/scripts/config/sync_config.sh
```

Options for `CONFLICT_RESOLUTION` are:
- `newer`: Use the file with the most recent modification time
- `code-server`: Always prefer the code server version
- `deploy-server`: Always prefer the deployment server version

### Logs

Logs are stored in the following locations:

- SSH Setup: `/opt/sutazaiapp/logs/ssh_setup.log`
- Sync Operations: `/opt/sutazaiapp/logs/sync/sync_*.log`
- Git Hook: `/opt/sutazaiapp/logs/git_hooks.log`
- Sync Monitor: `/opt/sutazaiapp/logs/sync_monitor.log`
- Cron Job: `/opt/sutazaiapp/logs/sync/cron_sync_*.log`

### Service Management

The monitoring service can be managed with standard systemd commands:

```bash
# Start the service
systemctl start sutazai-sync-monitor.service

# Stop the service
systemctl stop sutazai-sync-monitor.service

# Check status
systemctl status sutazai-sync-monitor.service

# View logs
journalctl -u sutazai-sync-monitor.service
```

## Troubleshooting

1. **SSH Authentication Issues**: Run `/opt/sutazaiapp/scripts/ssh_key_exchange.sh` to regenerate and exchange SSH keys.
2. **Permission Issues**: Ensure scripts are executable with `chmod +x /opt/sutazaiapp/scripts/*.sh`
3. **Service Not Running**: Check service status with `systemctl status sutazai-sync-monitor.service`
