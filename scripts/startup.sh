#!/usr/bin/env bash

# SutazAI Comprehensive System Initialization and Optimization Script

# Strict error handling and debugging
set -euo pipefail
export PYTHONUNBUFFERED=1

# Logging Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/opt/sutazai_project/SutazAI"
LOG_DIR="${BASE_PATH}/logs/startup"
STARTUP_LOG="${LOG_DIR}/startup_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/startup_errors_${TIMESTAMP}.log"

# Create log directories
mkdir -p "${LOG_DIR}"

# Redirect all output to log files
exec > >(tee -a "${STARTUP_LOG}") 2> >(tee -a "${ERROR_LOG}" >&2)

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Error handling function
handle_error() {
    log "ERROR: Startup process failed at stage: $1"
    exit 1
}

# Trap any errors
trap 'handle_error "Unknown Error"' ERR

# Start of initialization process
log "Starting SutazAI Comprehensive System Initialization"

# 1. Environment Preparation
log "Preparing System Environment"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Create and activate virtual environment
VENV_PATH="${BASE_PATH}/venv"
if [ ! -d "${VENV_PATH}" ]; then
    log "Creating Python Virtual Environment"
    python3 -m venv "${VENV_PATH}" || handle_error "Virtual Environment Creation"
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate" || handle_error "Virtual Environment Activation"

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel || handle_error "Pip Upgrade"

# 2. Dependency Management
log "Installing Project Dependencies"
pip install -r "${BASE_PATH}/requirements.txt" || handle_error "Dependency Installation"

# 3. System Diagnostic and Optimization
log "Running Comprehensive System Diagnostic"
python3 "${BASE_PATH}/scripts/system_diagnostic.py" || handle_error "System Diagnostic"

# 4. Autonomous Monitor Initialization
log "Starting Autonomous Monitoring System"
python3 "${BASE_PATH}/scripts/autonomous_monitor.py" &
MONITOR_PID=$!


# 6. Model Initialization
log "Initializing AI Models"
python3 "${BASE_PATH}/scripts/model_initializer.py" || handle_error "Model Initialization"

# 7. Comprehensive System Review
log "Performing Comprehensive System Review"
python3 "${BASE_PATH}/scripts/system_comprehensive_review.py" || handle_error "System Review"

# Final Confirmation
log "SutazAI System Initialization Completed Successfully"

# Save monitor PID for potential later use
echo "${MONITOR_PID}" > "${BASE_PATH}/logs/monitor.pid"

# Deactivate virtual environment
deactivate

exit 0