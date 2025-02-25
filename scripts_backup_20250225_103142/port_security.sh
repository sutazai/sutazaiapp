#!/bin/bash
# SutazAI Port Security Management

# Logging
LOG_FILE="/var/log/sutazai/port_security.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Close unnecessary ports
close_ports() {
    log "Closing unnecessary ports..."
    
    # List of ports to close
    PORTS_TO_CLOSE=(8080)
    
    for port in "${PORTS_TO_CLOSE[@]}"; do
        # UFW rules
        sudo ufw deny "$port/tcp"
        sudo ufw deny "$port/udp"
        
        log "Closed port $port"
    done
}

# Whitelist specific ports for SutazAI
whitelist_ports() {
    log "Whitelisting SutazAI required ports..."
    
    # Ports to keep open
    PORTS_TO_WHITELIST=(8000 22 443)
    
    for port in "${PORTS_TO_WHITELIST[@]}"; do
        sudo ufw allow "$port/tcp"
        log "Whitelisted port $port"
    done
}

# Main execution
main() {
    log "🔒 Starting SutazAI Port Security Configuration"
    
    # Close unnecessary ports
    close_ports
    
    # Whitelist required ports
    whitelist_ports
    
    # Reload firewall
    sudo ufw reload
    
    log "✅ Port security configuration complete"
}

# Execute main function
main 