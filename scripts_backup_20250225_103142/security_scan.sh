#!/bin/bash
# Run a security scan
lynis audit system --quick > /var/log/security_scan.log
echo "Security scan completed successfully!" 