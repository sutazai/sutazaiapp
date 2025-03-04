#!/bin/bash

# Master setup script for two-way synchronization

# Change to the project root directory
cd /opt/sutazaiapp || exit 1

# Set up logging
LOG_FILE="logs/setup.log"
mkdir -p "$(dirname $LOG_FILE)"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Starting two-way sync setup"

# Create required directories
mkdir -p scripts/config logs/sync

# Check if scripts exist, if not create them
for script in scripts/ssh_key_exchange.sh scripts/two_way_sync.sh scripts/sync_monitor.sh scripts/sync_exclude.txt scripts/config/sync_config.sh; do
    if [ ! -f "$script" ]; then
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $script does not exist, will be created during setup"
    else
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $script exists, will be skipped"
    fi
done

# Make sure all scripts are executable
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Making scripts executable"
chmod +x scripts/ssh_key_exchange.sh scripts/two_way_sync.sh scripts/sync_monitor.sh

# Run SSH key exchange
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setting up SSH key exchange"
./scripts/ssh_key_exchange.sh

# Copy systemd service file if needed
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Installing systemd service"
if [ ! -f "/etc/systemd/system/sutazai-sync-monitor.service" ]; then
    if [ -f "/tmp/sutazai-sync-monitor.service" ]; then
        cp /tmp/sutazai-sync-monitor.service /etc/systemd/system/
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Copied service file from /tmp"
    else
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Creating service file directly"
        cat > /etc/systemd/system/sutazai-sync-monitor.service << 'EOL'
[Unit]
Description=SutazAI Two-Way Server Sync Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/scripts/sync_monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL
    fi

    systemctl daemon-reload
    systemctl enable sutazai-sync-monitor.service
    systemctl start sutazai-sync-monitor.service
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Systemd service installed and started"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Systemd service already exists. Restarting..."
    systemctl restart sutazai-sync-monitor.service
fi

# Set up Git hook if we're on the code server
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "192.168.100.28" ]]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setting up Git hook on code server"
    
    # Ensure .git/hooks directory exists
    mkdir -p .git/hooks
    
    # Check if post-commit hook exists, if not create it
    if [ ! -f ".git/hooks/post-commit" ]; then
        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Creating Git post-commit hook"
        cat > .git/hooks/post-commit << 'EOL'
#!/bin/bash

# Git post-commit hook to trigger synchronization
LOG_FILE="/opt/sutazaiapp/logs/git_hooks.log"
SYNC_SCRIPT="/opt/sutazaiapp/scripts/two_way_sync.sh"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Log the commit
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Post-commit hook triggered" >> "$LOG_FILE"

# Run the sync script to deploy to the deployment server
$SYNC_SCRIPT --to-deploy >> "$LOG_FILE" 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Sync completed successfully" >> "$LOG_FILE"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: Sync failed" >> "$LOG_FILE"
fi
EOL
    fi
    
    # Make post-commit hook executable
    chmod +x .git/hooks/post-commit
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Git hook set up successfully"
fi

# Set up cron job for regular sync
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setting up cron job for regular sync"
CRON_ENTRY="0 * * * * /opt/sutazaiapp/scripts/two_way_sync.sh >> /opt/sutazaiapp/logs/sync/cron_sync_\$(date +\%Y\%m\%d).log 2>&1"
if ! (crontab -l 2>/dev/null | grep -q "two_way_sync.sh"); then
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Cron job added"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Cron job already exists, skipping"
fi

# Create a README file for the sync system
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Creating README file for the sync system"
mkdir -p docs
cat > docs/SYNC_SYSTEM_README.md << 'EOL'
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
EOL

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setup completed successfully"
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] You can now use ./scripts/two_way_sync.sh to manually trigger synchronization"
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Documentation is available at docs/SYNC_SYSTEM_README.md" 