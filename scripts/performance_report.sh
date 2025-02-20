#!/bin/bash
# Generate system performance report
top -bn1 > /var/log/performance_report.log
echo "Performance report generated successfully!" 