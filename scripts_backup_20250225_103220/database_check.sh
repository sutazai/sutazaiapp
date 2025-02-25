#!/bin/bash

# Database Health Check Script

log_message "=== Starting Database Health Check ==="

# Check MySQL/MariaDB
if command -v mysql &> /dev/null; then
    log_message "MySQL/MariaDB status:"
    mysql -e "SHOW STATUS LIKE 'Uptime';" | while read -r line; do
        log_message "$line"
    done
else
    log_message "MySQL/MariaDB not installed, skipping check"
fi

# Check PostgreSQL
if command -v psql &> /dev/null; then
    log_message "PostgreSQL status:"
    psql -c "SELECT version();" | while read -r line; do
        log_message "$line"
    done
else
    log_message "PostgreSQL not installed, skipping check"
fi

log_message "=== Database Health Check Completed ===" 