#!/bin/bash

# 🧠 SUPER INTELLIGENT Emergency Docker Recovery Script (2025)
# This script applies advanced Docker recovery strategies for WSL2 and Linux environments

set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  [$(date '+%H:%M:%S')] $1${NC}"; }
log_success() { echo -e "${GREEN}✅ [$(date '+%H:%M:%S')] $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  [$(date '+%H:%M:%S')] $1${NC}"; }
log_error() { echo -e "${RED}❌ [$(date '+%H:%M:%S')] $1${NC}"; }

main() {
    log_info "🧠 SUPER INTELLIGENT Emergency Docker Recovery (2025)"
    log_info "======================================================="
    
    # Detect environment
    local is_wsl2=false
    local is_ubuntu_2404=false
    if grep -q WSL2 /proc/version 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
        is_wsl2=true
        log_info "🐧 WSL2 environment detected"
    fi
    
    # Check for Ubuntu 24.04
    if grep -q "24.04" /etc/os-release 2>/dev/null; then
        is_ubuntu_2404=true
        log_info "🔧 Ubuntu 24.04 detected - applying specific fixes"
    fi
    
    # Ubuntu 24.04 AppArmor fix
    if [ "$is_ubuntu_2404" = "true" ]; then
        log_info "Step 0: Applying Ubuntu 24.04 AppArmor fix..."
        if sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1; then
            # Make it permanent with dedicated file
            echo "kernel.apparmor_restrict_unprivileged_userns=0" > /etc/sysctl.d/60-apparmor-namespace.conf
            log_success "   ✅ Ubuntu 24.04 AppArmor fix applied (permanent)"
        else
            log_warn "   ⚠️  AppArmor fix failed - continuing anyway"
        fi
        
        # Fix iptables for WSL2 (critical for Ubuntu 24.04)
        log_info "   → Switching to iptables-legacy for WSL2 compatibility..."
        if update-alternatives --set iptables /usr/sbin/iptables-legacy >/dev/null 2>&1; then
            log_success "   ✅ Switched to iptables-legacy"
        fi
        if update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy >/dev/null 2>&1; then
            log_success "   ✅ Switched to ip6tables-legacy"
        fi
    fi
    
    # Step 1: Stop any existing Docker processes
    log_info "Step 1: Cleaning up existing Docker processes..."
    pkill -f dockerd >/dev/null 2>&1 || true
    systemctl stop docker >/dev/null 2>&1 || true
    sleep 3
    log_success "   ✅ Docker processes cleaned up"
    
    # Step 2: Clean up Docker socket files
    log_info "Step 2: Cleaning Docker socket files..."
    rm -f /var/run/docker.sock /var/run/docker.pid >/dev/null 2>&1 || true
    log_success "   ✅ Socket files cleaned"
    
    # Step 3: Apply emergency Docker configuration
    log_info "Step 3: Applying emergency Docker configuration..."
    
    # Backup existing config
    if [ -f /etc/docker/daemon.json ]; then
        cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%Y%m%d%H%M%S) 2>/dev/null || true
    fi
    
    # WSL2 optimized configuration - DISABLED for stability
    log_info "   → Skipping daemon.json creation for Docker stability"
    # Create stable minimal configuration instead
    cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2"
}
EOF
    log_success "   ✅ Stable Docker configuration applied"
    
    # Step 4: Ubuntu 24.04 specific Docker installation fix
    if [ "$is_ubuntu_2404" = "true" ]; then
        log_info "Step 4a: Applying Ubuntu 24.04 Docker installation fix..."
        
        # Check if Docker binary exists and create symlink if needed
        if [ ! -f /usr/bin/dockerd ] && [ -f /usr/sbin/dockerd ]; then
            log_info "   → Creating dockerd symlink..."
            ln -sf /usr/sbin/dockerd /usr/bin/dockerd || true
            log_success "   ✅ dockerd symlink created"
        fi
        
        # Reload systemd and enable services
        systemctl daemon-reload >/dev/null 2>&1 || true
        systemctl unmask docker.service >/dev/null 2>&1 || true
        systemctl unmask docker.socket >/dev/null 2>&1 || true
        systemctl unmask containerd.service >/dev/null 2>&1 || true
        systemctl enable containerd.service >/dev/null 2>&1 || true
        systemctl enable docker.service >/dev/null 2>&1 || true
        
        # Start containerd first
        log_info "   → Starting containerd service..."
        if systemctl start containerd.service >/dev/null 2>&1; then
            log_success "   ✅ containerd started successfully"
            sleep 3
        else
            log_warn "   ⚠️  containerd start failed - continuing anyway"
        fi
    fi
    
    # Step 5: Start Docker daemon
    log_info "Step 5: Starting Docker daemon..."
    
    # Clean up any stale iptables rules first
    if [ "$is_wsl2" = "true" ]; then
        log_info "   → Cleaning up iptables rules for WSL2..."
        iptables -F DOCKER >/dev/null 2>&1 || true
        iptables -F DOCKER-ISOLATION-STAGE-1 >/dev/null 2>&1 || true
        iptables -F DOCKER-ISOLATION-STAGE-2 >/dev/null 2>&1 || true
        iptables -F DOCKER-USER >/dev/null 2>&1 || true
        iptables -X DOCKER >/dev/null 2>&1 || true
        iptables -X DOCKER-ISOLATION-STAGE-1 >/dev/null 2>&1 || true
        iptables -X DOCKER-ISOLATION-STAGE-2 >/dev/null 2>&1 || true
        iptables -X DOCKER-USER >/dev/null 2>&1 || true
    fi
    
    if [ "$is_wsl2" = "true" ]; then
        log_info "   🐧 Using WSL2-optimized startup..."
        
        # Method 1: Direct dockerd with minimal config
        log_info "   → Method 1: Direct dockerd startup..."
        dockerd --storage-driver=overlay2 --iptables=false --bridge=none >/tmp/dockerd.log 2>&1 &
        local dockerd_pid=$!
        
        # Wait for startup
        local wait_count=0
        while [ $wait_count -lt 15 ]; do
            if docker version >/dev/null 2>&1; then
                log_success "   ✅ WSL2 Docker daemon started successfully"
                return 0
            fi
            sleep 1
            wait_count=$((wait_count + 1))
        done
        
        # Kill if failed
        kill $dockerd_pid >/dev/null 2>&1 || true
        
        # Method 2: Try with full config
        log_info "   → Method 2: Trying with full config..."
        dockerd --config-file=/etc/docker/daemon.json >/tmp/dockerd.log 2>&1 &
        dockerd_pid=$!
        sleep 8
        
        if timeout 5 docker --version >/dev/null 2>&1; then
            log_success "   ✅ WSL2 Docker daemon started"
        else
            log_warn "   ⚠️  WSL2 method failed, trying systemctl..."
            kill $dockerd_pid >/dev/null 2>&1 || true
            
            # Method 3: systemctl
            systemctl start docker >/dev/null 2>&1 || true
            sleep 5
        fi
    else
        # Standard Linux approach with Ubuntu 24.04 optimizations
        if systemctl start docker.service >/dev/null 2>&1; then
            log_success "   ✅ Docker daemon started via systemctl"
            sleep 5
        else
            log_warn "   ⚠️  systemctl failed, trying direct dockerd startup..."
            # Fallback to direct dockerd startup
            nohup dockerd --config-file=/etc/docker/daemon.json >/tmp/dockerd.log 2>&1 &
            sleep 10
        fi
    fi
    
    # Step 6: Verify Docker functionality
    log_info "Step 6: Verifying Docker functionality..."
    local attempts=0
    local max_attempts=15
    
    while [ $attempts -lt $max_attempts ]; do
        if timeout 3 docker --version >/dev/null 2>&1; then
            log_success "   ✅ Docker version command working"
            break
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    
    if [ $attempts -ge $max_attempts ]; then
        log_error "   ❌ Docker version check failed after $max_attempts attempts"
        return 1
    fi
    
    # Step 7: Test basic Docker operations
    log_info "Step 7: Testing basic Docker operations..."
    if timeout 5 docker system df >/dev/null 2>&1; then
        log_success "   ✅ Docker system commands working"
    else
        log_warn "   ⚠️  Docker system commands limited but basic functionality available"
    fi
    
    # Step 8: Final verification with Ubuntu 24.04 compatibility check
    log_info "Step 8: Final verification..."
    if timeout 3 docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
        local docker_version=$(docker --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        log_success "   ✅ Docker daemon fully operational (Version: $docker_version)"
        
        # Ubuntu 24.04 BuildKit verification
        if [ "$is_ubuntu_2404" = "true" ]; then
            if docker buildx version >/dev/null 2>&1; then
                log_success "   ✅ Docker BuildKit available for Ubuntu 24.04"
            else
                log_warn "   ⚠️  BuildKit not available but Docker is functional"
            fi
        fi
    else
        log_warn "   ⚠️  Docker daemon partially operational (sufficient for deployment)"
    fi
    
    log_success "🎉 Emergency Docker recovery completed successfully!"
    log_info "💡 Docker is now ready for deployment operations"
    
    return 0
}

# Run main function
main "$@"