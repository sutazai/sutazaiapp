#!/bin/bash

# Backup configurations
backup_configs() {
    local timestamp=$(date +%Y%m%d%H%M%S)
    mkdir -p "$BACKUP_DIR"
    tar -czf "$BACKUP_DIR/config_$timestamp.tar.gz" /etc/sutazai
}