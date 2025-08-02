#!/bin/bash
# Fix deployment issues for SutazAI v28 - 100% Perfect Deployment

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  [$(date +'%H:%M:%S')] $1${NC}"; }
log_success() { echo -e "${GREEN}✅ [$(date +'%H:%M:%S')] $1${NC}"; }
log_error() { echo -e "${RED}❌ [$(date +'%H:%M:%S')] $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  [$(date +'%H:%M:%S')] $1${NC}"; }

# Main fix function
main() {
    log_info "🔧 Starting SutazAI Deployment Fix Script"
    
    # 1. Fix Docker in WSL2
    log_info "🐋 Fixing Docker for WSL2 environment..."
    
    # Check if we're in WSL2
    if grep -q -E "(WSL|Microsoft)" /proc/version 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
        log_info "WSL2 detected - applying specialized fixes"
        
        # Fix iptables for WSL2
        log_info "Configuring iptables for WSL2..."
        sudo update-alternatives --set iptables /usr/sbin/iptables-legacy 2>/dev/null || true
        sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy 2>/dev/null || true
        
        # Check if Docker is already running
        if docker version >/dev/null 2>&1; then
            log_success "Docker is already running!"
        else
            # Try multiple Docker startup methods
            log_info "Starting Docker with WSL2 compatibility..."
            
            # Method 1: Check for Docker Desktop
            if [ -S /var/run/docker.sock ]; then
                log_info "Docker socket found - testing connection..."
                if docker version >/dev/null 2>&1; then
                    log_success "Connected to Docker Desktop!"
                    sudo chmod 666 /var/run/docker.sock 2>/dev/null || true
                else
                    log_warn "Docker socket exists but not responding"
                fi
            fi
            
            # Method 2: Try systemd if available
            if [ -d /run/systemd/system ] && ! docker version >/dev/null 2>&1; then
                log_info "Attempting systemd start..."
                sudo systemctl start docker 2>/dev/null || true
                sleep 3
                if docker version >/dev/null 2>&1; then
                    log_success "Docker started with systemd!"
                fi
            fi
            
            # Method 3: Try service command
            if ! docker version >/dev/null 2>&1; then
                log_info "Attempting service start..."
                sudo service docker start 2>/dev/null || true
                sleep 5
                if docker version >/dev/null 2>&1; then
                    log_success "Docker started with service command!"
                fi
            fi
            
            # Method 4: Direct dockerd startup
            if ! docker version >/dev/null 2>&1; then
                log_info "Starting dockerd directly..."
                
                # Kill any existing dockerd
                sudo pkill -f dockerd 2>/dev/null || true
                sleep 2
                
                # Start dockerd with minimal config
                sudo dockerd \
                    --host=unix:///var/run/docker.sock \
                    --storage-driver=overlay2 \
                    --iptables=false \
                    >/tmp/dockerd_fix.log 2>&1 &
                
                # Wait for startup
                local count=0
                while [ $count -lt 20 ]; do
                    if docker version >/dev/null 2>&1; then
                        log_success "Docker started directly!"
                        sudo chmod 666 /var/run/docker.sock 2>/dev/null || true
                        break
                    fi
                    sleep 1
                    count=$((count + 1))
                done
            fi
        fi
        
        # Final check
        if docker version >/dev/null 2>&1; then
            log_success "Docker is now running successfully!"
        else
            log_error "Failed to start Docker. Please check /tmp/dockerd_fix.log"
            echo "Try running: sudo dockerd"
            exit 1
        fi
    fi
    
    # 2. Create optimized Docker daemon.json
    log_info "Creating optimized Docker configuration..."
    sudo mkdir -p /etc/docker
    
    if [ -f /etc/docker/daemon.json ]; then
        sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "log-level": "warn",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "live-restore": true,
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "dns": ["8.8.8.8", "1.1.1.1", "8.8.4.4"],
  "features": {
    "buildkit": true
  }
}
EOF
    
    log_success "Docker configuration optimized"
    
    # 3. Fix permissions
    log_info "Fixing file permissions..."
    sudo chmod +x /opt/sutazaiapp/scripts/*.sh 2>/dev/null || true
    
    # 4. Clean up any failed containers
    log_info "Cleaning up Docker environment..."
    docker system prune -f >/dev/null 2>&1 || true
    
    log_success "All fixes applied successfully!"
    log_info "You can now run: sudo ./scripts/deploy_complete_system.sh"
}

# Run main function
main "$@"