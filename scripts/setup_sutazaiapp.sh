#!/bin/bash

# Sutazaiapp System Setup Script
# Version: 1.0.0
# Author: AI Assistant for Florin Cristian Suta

set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/sutazaiapp_setup.log
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Configuration Variables
SUTAZAIAPP_USER="sutazaiapp_dev"
SUTAZAIAPP_HOME="/opt/sutazaiapp"
GITHUB_REPO="https://github.com/chrissuta/sutazaiapp.git"
PYTHON_VERSION="3.11"

# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
   log "This script must be run as root" 
   exit 1
fi

# 1. Create non-root user with sudo permissions
create_user() {
    log "Creating non-root user: $SUTAZAIAPP_USER"
    if ! id "$SUTAZAIAPP_USER" &>/dev/null; then
        useradd -m -s /bin/bash -G sudo "$SUTAZAIAPP_USER"
        echo "$SUTAZAIAPP_USER ALL=(ALL) NOPASSWD:ALL" | tee "/etc/sudoers.d/$SUTAZAIAPP_USER"
        chmod 440 "/etc/sudoers.d/$SUTAZAIAPP_USER"
    else
        log "User $SUTAZAIAPP_USER already exists"
    fi
}

# 2. Configure Git pull strategy
configure_git() {
    log "Configuring Git pull strategy"
    su - "$SUTAZAIAPP_USER" -c "
        git config --global pull.rebase true
        git config --global fetch.prune true
        git config --global merge.conflictstyle diff3
    "
}

# 3. Clone repository with proper ownership
clone_repository() {
    log "Cloning repository to $SUTAZAIAPP_HOME"
    if [[ ! -d "$SUTAZAIAPP_HOME/.git" ]]; then
        git clone "$GITHUB_REPO" "$SUTAZAIAPP_HOME"
    else
        log "Repository already exists, pulling latest changes"
        cd "$SUTAZAIAPP_HOME" && git pull origin main
    fi
    
    chown -R "$SUTAZAIAPP_USER:$SUTAZAIAPP_USER" "$SUTAZAIAPP_HOME"
}

# 4. Ensure directory structure
ensure_directory_structure() {
    log "Ensuring directory structure"
    mkdir -p "$SUTAZAIAPP_HOME"/{ai_agents,model_management,backend,web_ui,scripts,packages,logs,doc_data,docs}
    chown -R "$SUTAZAIAPP_USER:$SUTAZAIAPP_USER" "$SUTAZAIAPP_HOME"
}

# 5. Set up Python virtual environment
setup_python_venv() {
    log "Setting up Python virtual environment"
    if [[ ! -d "$SUTAZAIAPP_HOME/venv" ]]; then
        su - "$SUTAZAIAPP_USER" -c "
            python$PYTHON_VERSION -m venv '$SUTAZAIAPP_HOME/venv'
            source '$SUTAZAIAPP_HOME/venv/bin/activate'
            pip install --upgrade pip setuptools wheel
        "
    else
        log "Virtual environment already exists"
    fi
}

# 6. Install dependencies with offline fallback
install_dependencies() {
    log "Installing dependencies"
    su - "$SUTAZAIAPP_USER" -c "
        source '$SUTAZAIAPP_HOME/venv/bin/activate'
        pip install -r '$SUTAZAIAPP_HOME/requirements.txt' || \
        pip install --no-index --find-links='$SUTAZAIAPP_HOME/packages' -r '$SUTAZAIAPP_HOME/requirements.txt'
    "
}

# 7. Create automated code audit script
create_code_audit_script() {
    log "Creating code audit script"
    cat > "$SUTAZAIAPP_HOME/scripts/code_audit.sh" << 'EOL'
#!/bin/bash

set -euo pipefail

# Activate virtual environment
source /opt/sutazaiapp/venv/bin/activate

# Code Audit Script
echo "Starting Comprehensive Code Audit"

# Semgrep Scan
echo "Running Semgrep..."
semgrep scan --config=auto /opt/sutazaiapp

# Pylint Scan
echo "Running Pylint..."
find /opt/sutazaiapp -name "*.py" | xargs pylint

# Mypy Type Checking
echo "Running Mypy Type Checking..."
mypy /opt/sutazaiapp

# Bandit Security Scan
echo "Running Bandit Security Scan..."
bandit -r /opt/sutazaiapp -f custom

echo "Code Audit Complete"
EOL

    chmod +x "$SUTAZAIAPP_HOME/scripts/code_audit.sh"
    chown "$SUTAZAIAPP_USER:$SUTAZAIAPP_USER" "$SUTAZAIAPP_HOME/scripts/code_audit.sh"
}

# Main execution
main() {
    log "Starting Sutazaiapp System Setup"
    create_user
    configure_git
    clone_repository
    ensure_directory_structure
    setup_python_venv
    install_dependencies
    create_code_audit_script
    log "Sutazaiapp System Setup Complete"
}

main 