#!/bin/bash
# SutazAI Final System Verification and Reboot Script

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/var/log/sutazai/final_verification.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# System Checks
check_firewall() {
    log "${YELLOW}🔥 Checking Firewall Configuration${NC}"
    sudo ufw status | tee -a "$LOG_FILE"
}

check_selinux() {
    log "${YELLOW}🛡️ Checking SELinux Status${NC}"
    sestatus | tee -a "$LOG_FILE"
}

    echo "PYTHONSAFEPATH: ${PYTHONSAFEPATH:-Not Set}"
    echo "PYTHONDONTWRITEBYTECODE: ${PYTHONDONTWRITEBYTECODE:-Not Set}"
    echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE:-Not Set}"
}

check_open_ports() {
    log "${YELLOW}🌐 Checking Open Ports${NC}"
    sudo netstat -tuln | grep LISTEN | tee -a "$LOG_FILE"
}

run_system_tests() {
    log "${YELLOW}🧪 Running System Tests${NC}"
    
    # Python version check
    python3 --version
    
    # Pip dependency check
    pip3 check || log "${RED}Dependency issues detected${NC}"
    
    # Docker check (if installed)
    if command -v docker &> /dev/null; then
        docker info
    fi
}

    
    # Run ClamAV if installed
    if command -v clamscan &> /dev/null; then
        sudo clamscan -r /
    fi
    
    # Check for any pending system updates
    sudo apt-get update
    sudo apt list --upgradable
}

main() {
    log "${GREEN}🌟 Starting SutazAI Final System Verification${NC}"
    
    # Run all checks
    check_firewall
    check_selinux
    check_open_ports
    run_system_tests
    
    # Prompt for reboot
    log "${YELLOW}🔄 System verification complete. Preparing for reboot...${NC}"
    echo -e "\n${YELLOW}The system will reboot in 30 seconds. Press Ctrl+C to cancel.${NC}"
    sleep 30
    
    # Reboot
    log "${GREEN}✅ Rebooting system to apply all changes${NC}"
    sudo reboot
}

# Execute main function
main 