#!/bin/bash
# Backup system files
tar -czf /var/backups/system_backup_$(date +%Y%m%d).tar.gz /etc /var/log
echo "System backup completed successfully!" 