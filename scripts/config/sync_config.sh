#!/bin/bash

# Server configuration
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"

# Project paths
PROJECT_ROOT="/opt/sutazaiapp"

# Sync settings
SYNC_INTERVAL=300  # In seconds (5 minutes)
AUTO_SYNC=true
CONFLICT_RESOLUTION="newer"  # Options: newer, code-server, deploy-server

# Email notifications
ENABLE_EMAIL_NOTIFICATIONS=false
ADMIN_EMAIL="admin@example.com"

# Advanced settings
MAX_SYNC_RETRIES=3
SYNC_TIMEOUT=600  # In seconds (10 minutes)
DEBUG_MODE=false 