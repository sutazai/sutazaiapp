#!/usr/bin/env bash
# SutazAI Advanced Project Analysis Tools Setup Script

set -euo pipefail

# Logging configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/opt/sutazaiapp/logs/setup"
SETUP_LOG="${LOG_DIR}/analysis_tools_setup_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SETUP_LOG}"
}

# Error handling function
handle_error() {
    log "ERROR: Setup failed at stage: $1"
    exit 1
}

# Python version verification
verify_python_version() {
    log "Verifying Python 3.11"
    
    if command -v python3.11 >/dev/null 2>&1; then
        log "Python 3.11 is installed"
    else
        log "Python 3.11 not found. Installing..."
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    fi
    
    PYTHON_VERSION=$(python3.11 --version)
    log "Using ${PYTHON_VERSION}"
}

# Main setup workflow
main() {
    log "Starting SutazAI Advanced Project Analysis Tools Setup"

    # 0. Verify Python version
    verify_python_version \
        || handle_error "Python Version Verification"
        
    # 1. Activate virtual environment
    log "Stage 1: Activating Virtual Environment"
    source /opt/sutazaiapp/venv/bin/activate \
        || handle_error "Virtual Environment Activation"

    # 2. Install system dependencies
    log "Stage 2: Installing System Dependencies"
    sudo apt-get update
    sudo apt-get install -y \
        graphviz \
        python3.11-dev \
        build-essential \
        || handle_error "System Dependency Installation"

    # 3. Install Python analysis requirements
    log "Stage 3: Installing Python Analysis Requirements"
    pip install --upgrade pip
    pip install -r /opt/sutazaiapp/requirements-analysis.txt \
        || handle_error "Python Analysis Requirements Installation"

    # 4. Configure analysis tools
    log "Stage 4: Configuring Analysis Tools"
    
    # Pylint configuration
    mkdir -p ~/.config/pylint
    cat > ~/.config/pylint/pylintrc << EOL
[MASTER]
ignore=CVS
ignore-patterns=
persistent=yes
load-plugins=
jobs=1
unsafe-load-any-extension=no
extension-pkg-whitelist=

[MESSAGES CONTROL]
confidence=
disable=C0111

[REPORTS]
output-format=text
files-output=no
reports=yes
evaluation=10.0 - ((float(5 * error + warning + refactor + convention) / statement) * 10)

[LOGGING]
logging-modules=logging

[MISCELLANEOUS]
notes=FIXME,XXX,TODO

[SIMILARITIES]
min-similarity-lines=4
ignore-comments=yes
ignore-docstrings=yes
ignore-imports=no
EOL

    # Bandit configuration
    mkdir -p ~/.config/bandit
    cat > ~/.config/bandit/config.yml << EOL
---
# Bandit configuration file
# https://bandit.readthedocs.io/en/latest/config.html

# Define which tests are run
tests:
  - B201
  - B301
  - B302

# Paths to exclude from scanning
exclude_paths:
  - venv/
  - node_modules/
  - build/
  - dist/
EOL

    # 5. Install pre-commit hooks
    log "Stage 5: Installing Pre-Commit Hooks"
    pip install pre-commit
    cat > /opt/sutazaiapp/.pre-commit-config.yaml << EOL
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/PyCQA/pylint
    rev: v2.17.4
    hooks:
    -   id: pylint
        args: [--rcfile=~/.config/pylint/pylintrc]

-   repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
    -   id: bandit
        args: [-c, ~/.config/bandit/config.yml]
EOL

    pre-commit install

    log "SutazAI Advanced Project Analysis Tools Setup Completed Successfully"
}

# Execute main setup function
main 2>&1 | tee -a "${SETUP_LOG}" 