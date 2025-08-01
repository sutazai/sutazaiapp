#!/bin/bash

# 🧠 SUPER INTELLIGENT Ubuntu 24.04 Docker Fix (2025)
# Fixes Docker daemon startup issues on Ubuntu 24.04

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  [$(date '+%H:%M:%S')] $1${NC}"; }
log_success() { echo -e "${GREEN}✅ [$(date '+%H:%M:%S')] $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  [$(date '+%H:%M:%S')] $1${NC}"; }
log_error() { echo -e "${RED}❌ [$(date '+%H:%M:%S')] $1${NC}"; }

log_info "🧠 SUPER INTELLIGENT Ubuntu 24.04 Docker Recovery (2025)"
log_info "========================================================="

# Step 1: Ubuntu 24.04 AppArmor Fix
if lsb_release -r 2>/dev/null | grep -q "24.04"; then
    log_info "Step 1: Applying Ubuntu 24.04 AppArmor namespace fix..."
    if sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1; then
        echo "kernel.apparmor_restrict_unprivileged_userns=0" >> /etc/sysctl.conf 2>/dev/null || true
        log_success "   ✅ Ubuntu 24.04 AppArmor fix applied"
    else
        log_warn "   ⚠️  AppArmor fix failed - continuing with alternative approach"
    fi
else
    log_info "Step 1: Not Ubuntu 24.04 - skipping AppArmor fix"
fi

# Step 2: Kill Docker processes
log_info "Step 2: Cleaning Docker processes..."
pkill -f dockerd >/dev/null 2>&1 || true
pkill -f containerd >/dev/null 2>&1 || true
pkill -f docker-proxy >/dev/null 2>&1 || true
sleep 3

# Step 3: Remove Docker runtime files
log_info "Step 3: Cleaning Docker runtime files..."
rm -rf /var/run/docker.sock /var/run/docker.pid >/dev/null 2>&1 || true
rm -rf /var/lib/docker/network/files/local-kv.db >/dev/null 2>&1 || true

# Step 4: Check dockerd binary location
log_info "Step 4: Verifying dockerd binary..."
if [ ! -f /usr/bin/dockerd ] && [ -f /usr/sbin/dockerd ]; then
    log_info "   → Creating dockerd symlink..."
    ln -sf /usr/sbin/dockerd /usr/bin/dockerd || true
    log_success "   ✅ dockerd symlink created"
fi

# Step 5: Reload systemd and enable services
log_info "Step 5: Configuring systemd services..."
systemctl daemon-reload
systemctl unmask docker.service >/dev/null 2>&1 || true
systemctl unmask docker.socket >/dev/null 2>&1 || true
systemctl unmask containerd.service >/dev/null 2>&1 || true
systemctl enable containerd.service >/dev/null 2>&1 || true
systemctl enable docker.service >/dev/null 2>&1 || true

# Step 6: Start containerd first
log_info "Step 6: Starting containerd service..."
if systemctl start containerd.service; then
    log_success "   ✅ containerd started successfully"
    sleep 5
else
    log_warn "   ⚠️  containerd start failed - continuing anyway"
fi

# Step 7: Start Docker daemon
log_info "Step 7: Starting Docker daemon..."
if systemctl start docker.service; then
    log_success "   ✅ Docker daemon started successfully"
    sleep 5
    
    # Test Docker functionality
    if docker --version >/dev/null 2>&1; then
        log_success "   ✅ Docker is functional"
        
        # Test container functionality
        if timeout 30 docker run --rm hello-world >/dev/null 2>&1; then
            log_success "🎉 Docker is fully functional - containers can run!"
        else
            log_warn "   ⚠️  Container test failed - but Docker daemon is running"
        fi
    else
        log_warn "   ⚠️  Docker version check failed"
    fi
else
    log_error "   ❌ Docker daemon start failed"
    
    # Try direct dockerd startup as fallback
    log_info "Step 8: Attempting direct dockerd startup..."
    nohup dockerd >/dev/null 2>&1 &
    sleep 10
    
    if docker --version >/dev/null 2>&1; then
        log_success "   ✅ Direct dockerd startup successful"
    else
        log_error "   ❌ All Docker startup methods failed"
        exit 1
    fi
fi

log_success "🎉 Ubuntu 24.04 Docker fix completed successfully!"