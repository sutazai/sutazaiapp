#!/bin/bash

# User Account Audit Script

log_message "=== Starting User Account Audit ==="

# Check for users with UID 0
log_message "Users with UID 0:"
awk -F: '($3 == "0") {print}' /etc/passwd | while read -r line; do
    log_message "$line"
done

# Check for empty passwords
log_message "Accounts with empty passwords:"
awk -F: '($2 == "") {print}' /etc/shadow | while read -r line; do
    log_message "$line"
done

# Check for password expiration
log_message "Password expiration information:"
chage -l root | while read -r line; do
    log_message "$line"
done

log_message "=== User Account Audit Completed ===" 