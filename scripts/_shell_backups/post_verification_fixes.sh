#!/bin/bash

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

    
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=$(od -An -N4 -i /dev/urandom)
EOF
    
    # Make the script executable
    
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
    
    enable_selinux
    secure_ports
    
    # Reload firewall
    sudo ufw reload
    
    
    # Optional: Prompt for reboot
    echo -e "\n${YELLOW}Reboot recommended to apply all changes. Reboot now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
}

# Execute main function
main 