#!/bin/bash

# Cron Job Audit Script

log_message "=== Starting Cron Job Audit ==="

# System cron jobs
log_message "System cron jobs:"
ls /etc/cron.* | while read -r file; do
    log_message "Cron file: $file"
    cat "$file" | while read -r line; do
        log_message "$line"
    done
done

# User cron jobs
log_message "User cron jobs:"
cut -d: -f1 /etc/passwd | while read -r user; do
    crontab -l -u "$user" 2>/dev/null | while read -r line; do
        log_message "User $user: $line"
    done
done

log_message "=== Cron Job Audit Completed ===" 