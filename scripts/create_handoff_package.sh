#!/bin/bash
set -euo pipefail

# Sutazaiapp Handoff Package Generator

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
HANDOFF_DIR="/tmp/sutazaiapp_handoff"
VERSION=$(git describe --tags --always)
PACKAGE_NAME="sutazaiapp_handoff_${VERSION}.tar.gz"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Prepare handoff directory
prepare_handoff_dir() {
    log "Preparing handoff directory"
    mkdir -p "$HANDOFF_DIR"
    
    # Copy essential directories
    cp -r "$SUTAZAIAPP_HOME"/{backend,web_ui,scripts,config,docs} "$HANDOFF_DIR"
    
    # Generate offline dependencies
    pip wheel -r "$SUTAZAIAPP_HOME/requirements.txt" -w "$HANDOFF_DIR/wheels"
    
    # Copy documentation
    cp "$SUTAZAIAPP_HOME"/README.md "$HANDOFF_DIR"
    cp "$SUTAZAIAPP_HOME"/LICENSE "$HANDOFF_DIR"
}

# Create archive
create_archive() {
    log "Creating handoff package"
    tar -czvf "/tmp/$PACKAGE_NAME" -C /tmp sutazaiapp_handoff \
        --exclude="**/venv" \
        --exclude="**/node_modules" \
        --exclude="**/*.pyc" \
        --exclude="**/__pycache__"
}

# Generate git tag
generate_tag() {
    log "Generating handoff version tag"
    git tag -a "handoff-${VERSION}" -m "Handoff package version"
    git push origin "handoff-${VERSION}"
}

# Cleanup
cleanup() {
    log "Cleaning up temporary files"
    rm -rf "$HANDOFF_DIR"
}

# Main function
main() {
    log "Starting Sutazaiapp Handoff Package Generation"
    
    prepare_handoff_dir
    create_archive
    generate_tag
    cleanup
    
    log "Handoff package created: /tmp/$PACKAGE_NAME"
}

# Execute main function
main 