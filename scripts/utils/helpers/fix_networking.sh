#\!/bin/bash
# 🌐 SutazAI Network Connectivity and Container Communication Fix
# This script fixes Docker networking issues in WSL2 environments

# Enable error handling
set -euo pipefail

# Logging functions
log_info() { echo -e "\033[0;34m[INFO]\033[0m $*"; }
log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $*"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }

# Check if running as root
check_root() {
    if [ "$(id -u)" \!= "0" ]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Fix WSL2 networking issues
fix_wsl2_networking() {
    log_info "🌐 Fixing WSL2 network connectivity..."
    
    # Fix WSL configuration
    cat > /etc/wsl.conf << 'EOL'
[boot]
systemd=true

[network]
hostname = sutazai-wsl
generateHosts = false
generateResolvConf = false

[interop]
enabled = true
appendWindowsPath = true
EOL
    
    # Fix DNS resolution
    cat > /etc/resolv.conf << 'EOL'
# Fixed DNS configuration for SutazAI
nameserver 8.8.8.8
nameserver 1.1.1.1
nameserver 8.8.4.4
options edns0 trust-ad
search .
EOL
    
    # Make resolv.conf immutable
    chattr +i /etc/resolv.conf 2>/dev/null || true
    
    log_success "✅ WSL2 networking configuration fixed"
}

# Configure Docker daemon for optimal WSL2 networking
configure_docker_daemon() {
    log_info "🐳 Configuring Docker daemon for optimal networking..."
    
    mkdir -p /etc/docker
    
    cat > /etc/docker/daemon.json << 'EOL'
{
    "dns": ["8.8.8.8", "1.1.1.1", "8.8.4.4"],
    "dns-opts": ["ndots:0"],
    "dns-search": ["."],
    "bip": "172.18.0.1/16",
    "default-address-pools": [
        {
            "base": "172.80.0.0/12",
            "size": 24
        }
    ],
    "max-concurrent-downloads": 6,
    "max-concurrent-uploads": 6,
    "iptables": true,
    "userland-proxy": false,
    "icc": true,
    "live-restore": false,
    "storage-driver": "overlay2",
    "exec-opts": ["native.cgroupdriver=systemd"],
    "features": {
        "buildkit": true
    },
    "builder": {
        "gc": {
            "enabled": true,
            "defaultKeepStorage": "20GB"
        }
    },
    "experimental": false,
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-ulimits": {
        "memlock": {
            "Hard": -1,
            "Name": "memlock",
            "Soft": -1
        },
        "nofile": {
            "Hard": 65536,
            "Name": "nofile", 
            "Soft": 65536
        }
    }
}
EOL
    
    log_success "✅ Docker daemon configuration optimized"
}

# Test container communication
test_container_communication() {
    log_info "🔍 Testing container communication..."
    
    # Check if containers can resolve each other
    if docker ps --format "{{.Names}}" | grep -q "sutazai-backend"; then
        if docker ps --format "{{.Names}}" | grep -q "sutazai-chromadb"; then
            log_info "Testing ChromaDB connectivity from backend..."
            if docker exec sutazai-backend bash -c "curl -s http://sutazai-chromadb:8000/api/v1/heartbeat" >/dev/null 2>&1; then
                log_success "✅ Backend -> ChromaDB communication working"
            else
                log_warn "⚠️ Backend -> ChromaDB communication failed"
                return 1
            fi
        fi
    fi
    
    return 0
}

# Main function
main() {
    log_info "🚀 Starting SutazAI Network Fix..."
    
    check_root
    
    # Stop Docker to apply changes
    systemctl stop docker 2>/dev/null || true
    
    # Apply fixes
    if grep -qi microsoft /proc/version || grep -qi wsl /proc/version; then
        fix_wsl2_networking
    fi
    
    configure_docker_daemon
    
    # Restart Docker
    log_info "🔄 Restarting Docker daemon..."
    systemctl start docker
    sleep 5
    
    # Test connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        log_success "✅ External network connectivity verified"
    else
        log_error "❌ External network connectivity failed"
        exit 1
    fi
    
    # Test Docker
    if docker info >/dev/null 2>&1; then
        log_success "✅ Docker daemon is working"
    else
        log_error "❌ Docker daemon is not working"
        exit 1
    fi
    
    log_success "🎉 SutazAI Network Fix completed successfully\!"
}

# Run main function
main "$@"
EOF < /dev/null
