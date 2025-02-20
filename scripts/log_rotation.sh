#!/bin/bash
# Rotate system logs
logrotate -f /etc/logrotate.conf
echo "System logs rotated successfully!" 