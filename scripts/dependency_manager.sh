#!/bin/bash

# Sutazaiapp Dependency Management Script
# Handles package installation, updates, and offline fallback

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
PACKAGES_DIR="$SUTAZAIAPP_HOME/packages"
REQUIREMENTS_FILE="$SUTAZAIAPP_HOME/requirements.txt"
VENV_PATH="$SUTAZAIAPP_HOME/venv"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/sutazaiapp_dependency_manager.log
}

# Activate virtual environment
activate_venv() {
    source "$VENV_PATH/bin/activate"
}

# Download wheel files for offline installation
download_wheel_files() {
    log "Downloading wheel files for offline installation"
    mkdir -p "$PACKAGES_DIR/wheels"
    pip download \
        -r "$REQUIREMENTS_FILE" \
        -d "$PACKAGES_DIR/wheels" \
        --only-binary=:all: \
        --platform manylinux2014_x86_64 \
        --python-version 311 \
        --implementation cp
}

# Install dependencies with online/offline fallback
install_dependencies() {
    log "Installing dependencies"
    activate_venv

    # Try online installation first
    if pip install -r "$REQUIREMENTS_FILE"; then
        log "Dependencies installed successfully online"
        return 0
    fi

    # Fallback to offline installation
    log "Online installation failed. Attempting offline installation"
    pip install \
        --no-index \
        --find-links="$PACKAGES_DIR/wheels" \
        -r "$REQUIREMENTS_FILE"
}

# Update dependencies
update_dependencies() {
    log "Updating dependencies"
    activate_venv

    # Update pip and setuptools first
    pip install --upgrade pip setuptools wheel

    # Update all packages
    pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
}

# Validate dependencies
validate_dependencies() {
    log "Validating dependencies"
    activate_venv

    # Run safety check for vulnerabilities
    safety check -r "$REQUIREMENTS_FILE"

    # Check for any incompatible packages
    pip check
}

# Main execution
main() {
    case "${1:-install}" in
        "download")
            download_wheel_files
            ;;
        "install")
            install_dependencies
            ;;
        "update")
            update_dependencies
            ;;
        "validate")
            validate_dependencies
            ;;
        *)
            log "Invalid command. Use download, install, update, or validate."
            exit 1
            ;;
    esac
}

main "$@" 