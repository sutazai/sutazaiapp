#!/bin/bash
# Clean up old backups
find /var/backups/sutazai -type f -mtime +30 -exec rm -f {} \;
echo "Old backups cleaned up successfully!" 