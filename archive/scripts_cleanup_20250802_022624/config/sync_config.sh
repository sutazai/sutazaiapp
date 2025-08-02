#!/bin/bash
# Central configuration for SutazAI sync system

# Server configuration
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"

# Project paths
PROJECT_ROOT="/opt/sutazaiapp"

# Sync settings
SYNC_INTERVAL=600  # In seconds (5 minutes)
MONITORING_INTERVAL=120  # In seconds (2 minutes)
AUTO_SYNC=true
CONFLICT_RESOLUTION="newer"  # Options: newer, code-server, deploy-server, interactive
LARGE_FILES_OPTIMIZATION=true

# Performance settings
MAX_BANDWIDTH="5000"  # 0 for unlimited, otherwise specify in KB/s
COMPRESSION_LEVEL=9
MAX_SYNC_RETRIES=3
SYNC_TIMEOUT=600  # In seconds (10 minutes)

# Email notifications
ENABLE_EMAIL_NOTIFICATIONS=false
ADMIN_EMAIL="admin@example.com"

# Advanced settings
DEBUG_MODE=false
SSH_DIR="/root/.ssh"
