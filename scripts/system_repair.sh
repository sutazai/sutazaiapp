#!/bin/bash
# SutazAI Comprehensive System Repair and Optimization Script

# Ensure script is run with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo" 
   exit 1
fi

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/var/log/sutazai/system_repair.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Dependency Repair
repair_dependencies() {
    log "${YELLOW}🔧 Repairing Python Dependencies${NC}"
    
    # Update pip and setuptools
    pip3 install --upgrade pip setuptools wheel
    
    # Check and repair dependencies
    pip3 check || {
        log "${RED}Dependency issues detected. Attempting repair...${NC}"
        pip3 list --outdated
        pip3 install --upgrade --upgrade-strategy eager $(pip3 list --outdated --format=freeze | cut -d = -f 1)
    }
    
    # Install safety for vulnerability checks
    pip3 install safety
    safety check
}

# Syntax Warning Repair
repair_syntax_warnings() {
    log "${YELLOW}🔍 Checking and Fixing Syntax Warnings${NC}"
    python3 scripts/fix_syntax_warnings.py
}

# System Optimization
optimize_system() {
    log "${YELLOW}🚀 Optimizing System Performance${NC}"
    bash scripts/system_tune.sh
}

# Security Review
review_security() {
    log "${YELLOW}🔒 Performing Security Review${NC}"
    python3 scripts/security_review.py
}

# Main Repair Process
main() {
    log "${GREEN}🌟 Starting SutazAI System Repair and Optimization${NC}"
    
    # Repair stages
    repair_dependencies
    repair_syntax_warnings
    optimize_system
    review_security
    
    log "${GREEN}✅ System Repair and Optimization Complete!${NC}"
}

# Execute main function
main

# Optional: Display final report
echo -e "\n${GREEN}🔍 Review the detailed logs at $LOG_FILE${NC}" 