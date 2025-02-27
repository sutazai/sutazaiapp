#!/bin/bash
set -euo pipefail

# Sutazaiapp Final Handoff Package Generator

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
HANDOFF_DIR="/tmp/sutazaiapp_final_handoff"
VERSION=$(git describe --tags --always)
PACKAGE_NAME="sutazaiapp_final_handoff_${VERSION}.tar.gz"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "/var/log/sutazaiapp/handoff.log"
}

# Validate OTP for handoff
validate_otp() {
    python3 -c "
from scripts.otp_override import OTPManager

otp_manager = OTPManager()
is_valid = otp_manager.validate_otp('$1')
exit(0) if is_valid else exit(1)
"
}

# Prepare handoff directory
prepare_handoff_dir() {
    log "Preparing comprehensive handoff directory"
    mkdir -p "$HANDOFF_DIR"
    
    # Copy essential directories with comprehensive content
    cp -r "$SUTAZAIAPP_HOME"/{backend,web_ui,scripts,config,docs} "$HANDOFF_DIR"
    
    # Generate offline dependencies
    pip wheel -r "$SUTAZAIAPP_HOME/requirements.txt" -w "$HANDOFF_DIR/wheels"
    
    # Copy training and documentation materials
    cp -r "$SUTAZAIAPP_HOME/docs/guides" "$HANDOFF_DIR/training_materials"
    
    # Generate comprehensive README
    generate_readme
}

# Generate comprehensive README
generate_readme() {
    cat > "$HANDOFF_DIR/README.md" << EOL
# Sutazaiapp Final Handoff Package

## Version: ${VERSION}
## Timestamp: $(date)

### Package Contents
- Complete source code
- Offline dependencies
- Training materials
- Configuration files
- Documentation

### Restoration Procedure
1. Extract package
2. Verify OTP configuration
3. Install offline dependencies
   \`\`\`bash
   pip install --no-index --find-links=wheels -r requirements.txt
   \`\`\`
4. Configure OTP secrets
5. Run initialization script

### Security Notes
- All external interactions require OTP validation
- Offline-first deployment recommended
- Rotate secrets before production use

### Support
Contact: sutazai-support@organization.com
Emergency Hotline: +1-SUPPORT-HOTLINE

### Licensing
See LICENSE file for details
EOL
}

# Create comprehensive archive
create_archive() {
    log "Creating final handoff package"
    tar -czvf "/tmp/$PACKAGE_NAME" -C /tmp sutazaiapp_final_handoff \
        --exclude="**/venv" \
        --exclude="**/node_modules" \
        --exclude="**/*.pyc" \
        --exclude="**/__pycache__" \
        --exclude="**/.git"
}

# Generate git tag
generate_tag() {
    log "Generating final handoff version tag"
    git tag -a "final-handoff-${VERSION}" -m "Final comprehensive handoff package"
    git push origin "final-handoff-${VERSION}"
}

# Cleanup
cleanup() {
    log "Cleaning up temporary files"
    rm -rf "$HANDOFF_DIR"
}

# Main function
main() {
    # Require OTP for handoff generation
    if [ $# -ne 1 ]; then
        log "OTP required for handoff package generation"
        exit 1
    fi
    
    validate_otp "$1"
    
    log "Starting Sutazaiapp Final Handoff Package Generation"
    
    prepare_handoff_dir
    create_archive
    generate_tag
    cleanup
    
    log "Final handoff package created: /tmp/$PACKAGE_NAME"
}

# Execute main function
main "$@" 