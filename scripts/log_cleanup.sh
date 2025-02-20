#!/bin/bash
# Automatically clean up old logs
find /var/log/sutazai -type f -mtime +30 -exec rm -f {} \;
echo "Old logs cleaned up successfully!" 