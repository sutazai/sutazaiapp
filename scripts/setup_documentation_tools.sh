#!/usr/bin/env bash
# SutazAI Documentation Analysis Tools Setup Script

set -euo pipefail

# Logging configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/opt/sutazai_project/SutazAI/logs/setup"
SETUP_LOG="${LOG_DIR}/documentation_tools_setup_${TIMESTAMP}.log"

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

# Main setup workflow
main() {
    log "Starting SutazAI Documentation Analysis Tools Setup"

    # 1. Activate virtual environment
    log "Stage 1: Activating Virtual Environment"
    source /opt/sutazai_project/SutazAI/venv/bin/activate \
        || handle_error "Virtual Environment Activation"

    # 2. Install system dependencies
    log "Stage 2: Installing System Dependencies"
    sudo apt-get update
    sudo apt-get install -y \
        graphviz \
        python3-dev \
        build-essential \
        || handle_error "System Dependency Installation"

    # 3. Install Python documentation requirements
    log "Stage 3: Installing Documentation Requirements"
    pip install --upgrade pip
    pip install -r /opt/sutazai_project/SutazAI/requirements-documentation.txt \
        || handle_error "Documentation Requirements Installation"

    # 4. Download NLP models
    log "Stage 4: Downloading NLP Models"
    python3 -m spacy download en_core_web_sm \
        || handle_error "SpaCy Model Download"
    
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')" \
        || handle_error "NLTK Resource Download"

    # 5. Configure documentation tools
    log "Stage 5: Configuring Documentation Tools"
    
    # Sphinx configuration
    mkdir -p ~/.config/sphinx
    cat > ~/.config/sphinx/conf.py << EOL
# Sphinx configuration for SutazAI documentation

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SutazAI'
copyright = '2023, Florin Cristian Suta'
author = 'Florin Cristian Suta'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
EOL

    # 6. Install pre-commit hooks for documentation
    log "Stage 6: Installing Documentation Pre-Commit Hooks"
    pip install pre-commit
    cat > /opt/sutazai_project/SutazAI/.pre-commit-config.yaml << EOL
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: local
    hooks:
    -   id: documentation-check
        name: Documentation Validation
        entry: python3 /opt/sutazai_project/SutazAI/core_system/advanced_documentation_analyzer.py
        language: system
        pass_filenames: false
EOL

    pre-commit install

    log "SutazAI Documentation Analysis Tools Setup Completed Successfully"
}

# Execute main setup function
main 2>&1 | tee -a "${SETUP_LOG}" 