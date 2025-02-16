#!/bin/bash
# Monitor system health
top -bn1 > /var/log/system_monitor.log
echo "System monitoring completed successfully!" 