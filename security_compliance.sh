#!/bin/bash

# Security Compliance Check Script

log_message "=== Starting Security Compliance Check ==="

# Check password policies
log_message "Password Policies:"
grep "^PASS_MAX_DAYS" /etc/login.defs | while read -r line; do
    log_message "$line"
done

# Check SSH configuration
log_message "SSH Configuration:"
grep -E "^PermitRootLogin|^PasswordAuthentication" /etc/ssh/sshd_config | while read -r line; do
    log_message "$line"
done

# Check SELinux/AppArmor status
if command -v sestatus &> /dev/null; then
    log_message "SELinux Status:"
    sestatus | while read -r line; do
        log_message "$line"
    done
elif command -v aa-status &> /dev/null; then
    log_message "AppArmor Status:"
    aa-status | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: No mandatory access control system found"
fi

log_message "=== Security Compliance Check Completed ===" 