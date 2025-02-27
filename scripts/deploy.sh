#!/bin/bash

# Sutazaiapp Comprehensive Deployment Script
# Manages offline-first deployment with OTP validation

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
VENV_PATH="$SUTAZAIAPP_HOME/venv"
DEPLOY_LOG="/opt/sutazaiapp/logs/deploy.log"
CONFIG_DIR="$SUTAZAIAPP_HOME/config"
BACKUP_DIR="$SUTAZAIAPP_HOME/backups"
WHEELS_DIR="$SUTAZAIAPP_HOME/wheels"

# Source OTP validation functions
source /opt/sutazaiapp/scripts/otp_override.py

# Logging configuration
LOG_FILE="/var/log/sutazaiapp/deployment.log"
BLOCKED_ATTEMPTS_LOG="/var/log/sutazaiapp/blocked_attempts.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DEPLOY_LOG"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    rollback
    exit 1
}

trap 'handle_error $LINENO' ERR

# Validate OTP
validate_otp() {
    log "Validating OTP for deployment"
    local otp="$1"
    
    # Use existing OTP manager for validation
    if python3 -c "
import sys
sys.path.append('$SUTAZAIAPP_HOME/scripts')
from otp_manager import OTPManager

otp_manager = OTPManager()
# Retrieve last generated OTP info from a secure storage mechanism
last_otp_info = {}  # This would be replaced with actual secure storage retrieval
result = otp_manager.validate_otp('$otp', last_otp_info)
sys.exit(0) if result else sys.exit(1)
    "; then
        log "✅ OTP Validation Successful"
        return 0
    else
        log "❌ OTP Validation Failed"
        return 1
    fi
}

# Create backup of current deployment
create_backup() {
    log "Creating deployment backup"
    mkdir -p "$BACKUP_DIR"
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    
    cp -r "$SUTAZAIAPP_HOME" "$BACKUP_DIR/$backup_name"
    log "Backup created: $BACKUP_DIR/$backup_name"
}

# Rollback to previous stable version
rollback() {
    log "Initiating rollback procedure"
    
    # Find most recent backup
    local latest_backup=$(ls -td "$BACKUP_DIR"/backup_* | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log "No backup available for rollback"
        return 1
    fi
    
    log "Rolling back to $latest_backup"
    rm -rf "$SUTAZAIAPP_HOME"
    cp -r "$latest_backup" "$SUTAZAIAPP_HOME"
    
    log "Rollback completed successfully"
}

# Install dependencies with offline fallback
install_dependencies() {
    log "Installing project dependencies"
    
    source "$VENV_PATH/bin/activate"
    
    # Try online installation first
    if [[ -n "${OTP:-}" ]]; then
        log "Attempting online dependency installation"
        if pip install -r "$SUTAZAIAPP_HOME/requirements.txt"; then
            log "✅ Online dependency installation successful"
            return 0
        fi
    fi
    
    # Fallback to offline wheel installation
    log "Falling back to offline wheel installation"
    pip install --no-index --find-links="$WHEELS_DIR" -r "$SUTAZAIAPP_HOME/requirements.txt"
    
    # Node.js dependencies
    if command -v npm &> /dev/null; then
        npm ci --offline || log "Warning: Node.js dependency installation partial"
    fi
}

# Verify model files
verify_models() {
    log "Verifying AI model files"
    
    local models_dir="$SUTAZAIAPP_HOME/model_management"
    local required_models=(
        "gpt4all-model.bin"
        "deepseek-coder-model.bin"
    )
    
    for model in "${required_models[@]}"; do
        if [[ ! -f "$models_dir/$model" ]]; then
            log "❌ Required model missing: $model"
            return 1
        fi
    done
    
    log "✅ All required models verified"
}

# Launch services
launch_services() {
    log "Launching Sutazaiapp services"
    
    # Backend (FastAPI) service
    nohup python3 -m backend.main \
        --config "$CONFIG_DIR/backend_config.toml" \
        > "$SUTAZAIAPP_HOME/logs/backend.log" 2>&1 &
    local backend_pid=$!
    
    # Web UI service
    nohup npm start --prefix "$SUTAZAIAPP_HOME/web_ui" \
        > "$SUTAZAIAPP_HOME/logs/webui.log" 2>&1 &
    local webui_pid=$!
    
    # Wait and verify services
    sleep 10
    
    # Health checks
    if kill -0 "$backend_pid" && kill -0 "$webui_pid"; then
        log "✅ Services launched successfully"
    else
        log "❌ Service launch failed"
        kill "$backend_pid" "$webui_pid" 2>/dev/null || true
        return 1
    fi
}

# OTP validation function
validate_external_call() {
    local otp="$1"
    
    # Call Python OTP validation
    python3 -c "
from scripts.otp_override import OTPManager

otp_manager = OTPManager()
is_valid = otp_manager.validate_otp('$otp')

if is_valid:
    print('VALID')
else:
    print('INVALID')
"
}

# Deployment stages with OTP validation
deploy_stage() {
    local stage_name="$1"
    local otp="$2"
    
    log() {
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
    }
    
    log_message "Starting deployment stage: $stage_name"
    
    # Validate OTP for external calls
    validation_result=$(validate_external_call "$otp")
    
    if [ "$validation_result" != "VALID" ]; then
        log_message "OTP validation failed for $stage_name" | tee -a "$BLOCKED_ATTEMPTS_LOG"
        echo "Error: Invalid OTP. External call blocked."
        exit 1
    fi
    
    # Perform deployment stage
    case "$stage_name" in
        "git_pull")
            git pull origin main
            ;;
        "pip_install")
            pip install -r requirements.txt
            ;;
        "database_migration")
            alembic upgrade head
            ;;
        *)
            log_message "Unknown deployment stage: $stage_name"
            exit 1
            ;;
    esac
    
    log_message "Deployment stage $stage_name completed successfully"
}

# Main deployment function
main() {
    # Check if OTP is provided
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <stage_name> <otp>"
        exit 1
    fi
    
    local stage_name="$1"
    local otp="$2"
    
    # Perform deployment with OTP validation
    deploy_stage "$stage_name" "$otp"
}

# Execute main function with arguments
main "$@" 