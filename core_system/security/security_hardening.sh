#!/bin/bash

# Add password verification
source "${SCRIPT_DIR}/password_manager.sh"

if ! verify_password; then
    echo "Access denied"
    exit 1
fi

# System Security Hardening Script

log_message "=== Starting System Security Hardening ==="

# Harden system security
log_message "Hardening System Security:"
chmod 600 /etc/ssh/sshd_config
chmod 700 /root
chmod 750 /home/*

log_message "=== System Security Hardening Completed ===" 