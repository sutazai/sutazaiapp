#!/bin/bash
# Generate system resource report
free -h > /var/log/resource_report.log
df -h >> /var/log/resource_report.log
echo "Resource report generated successfully!" 