#!/bin/bash
# SutazAI Post-Verification Security Fixes

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
LOG_FILE="/var/log/sutazai/post_verification.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Enable SELinux
enable_selinux() {
    log "${YELLOW}🛡️ Configuring SELinux${NC}"
    
    # Install SELinux if not present
    sudo apt-get update
    sudo apt-get install -y selinux-basics selinux-policy-default
    
    # Configure SELinux to enforcing mode
    sudo sed -i 's/SELINUX=disabled/SELINUX=enforcing/g' /etc/selinux/config
    sudo selinux-activate
    
    log "${GREEN}✅ SELinux configured in enforcing mode${NC}"
}

# Set Python Security Environment Variables
configure_python_security() {
    log "${YELLOW}🐍 Configuring Python Security Environment${NC}"
    
    # Create a system-wide Python security configuration
    sudo tee /etc/profile.d/python_security.sh << EOF
# SutazAI Python Security Configuration
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=$(od -An -N4 -i /dev/urandom)
EOF
    
    # Make the script executable
    sudo chmod +x /etc/profile.d/python_security.sh
    
    log "${GREEN}✅ Python security environment configured${NC}"
}

# Close Unnecessary Ports
secure_ports() {
    log "${YELLOW}🌐 Securing Open Ports${NC}"
    
    # Close port 8080
    sudo ufw deny 8080/tcp
    sudo ufw deny 8080/udp
    
    # Disable unnecessary services
    sudo systemctl disable cups
    
    log "${GREEN}✅ Unnecessary ports and services disabled${NC}"
}

# Main execution
main() {
    log "${GREEN}🌟 Starting SutazAI Post-Verification Security Fixes${NC}"
    
    # Run security fixes
    enable_selinux
    configure_python_security
    secure_ports
    
    # Reload firewall
    sudo ufw reload
    
    log "${GREEN}✅ Post-verification security fixes complete${NC}"
    
    # Optional: Prompt for reboot
    echo -e "\n${YELLOW}Reboot recommended to apply all changes. Reboot now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
}

# Execute main function
main 