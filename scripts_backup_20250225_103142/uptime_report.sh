#!/bin/bash
# Generate system uptime report
uptime > /var/log/uptime_report.log
echo "Uptime report generated successfully!" 