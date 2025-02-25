#!/bin/bash
# Create a system snapshot
tar -czf /var/backups/system_snapshot_$(date +%Y%m%d).tar.gz /etc /var/log
echo "System snapshot created successfully!" 