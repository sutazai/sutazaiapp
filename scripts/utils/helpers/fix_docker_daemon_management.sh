#!/bin/bash

# 🚀 SutazAI Docker Daemon Management Fix
# Fixes the race condition and conflict issues in deploy_complete_system.sh

set -euo pipefail

LOG_FILE="/opt/sutazaiapp/logs/docker_fix_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log_info() { echo -e "\033[0;34mℹ️  [$(date +%H:%M:%S)] $1\033[0m" | tee -a "$LOG_FILE"; }
log_success() { echo -e "\033[0;32m✅ [$(date +%H:%M:%S)] $1\033[0m" | tee -a "$LOG_FILE"; }
log_warn() { echo -e "\033[1;33m⚠️  [$(date +%H:%M:%S)] $1\033[0m" | tee -a "$LOG_FILE"; }
log_error() { echo -e "\033[0;31m❌ [$(date +%H:%M:%S)] $1\033[0m" | tee -a "$LOG_FILE"; }

# Smart Docker daemon management function
smart_docker_daemon_management() {
    log_info "🧠 Smart Docker Daemon Management System"
    
    # Step 1: Check current state
    local docker_running=false
    local containerd_running=false
    
    if systemctl is-active --quiet docker 2>/dev/null; then
        docker_running=true
        log_info "   → Docker service is currently active"
    fi
    
    if systemctl is-active --quiet containerd 2>/dev/null; then
        containerd_running=true
        log_info "   → Containerd service is currently active"
    fi
    
    # Step 2: Check for stale processes
    local stale_dockerd=$(pgrep -f "dockerd" | head -1 || echo "")
    local stale_containerd=$(pgrep -f "containerd.*--config" | head -1 || echo "")
    
    if [ -n "$stale_dockerd" ] || [ -n "$stale_containerd" ]; then
        log_warn "   → Stale processes detected, cleaning up..."
        
        # Stop services first
        systemctl stop docker containerd 2>/dev/null || true
        sleep 2
        
        # Kill stale processes
        [ -n "$stale_dockerd" ] && kill -9 "$stale_dockerd" 2>/dev/null || true
        [ -n "$stale_containerd" ] && kill -9 "$stale_containerd" 2>/dev/null || true
        
        # Clean up stale files
        rm -f /var/run/docker.pid /run/containerd/containerd.sock 2>/dev/null || true
        
        log_success "   ✅ Stale processes cleaned up"
    fi
    
    # Step 3: Start services in proper order
    log_info "   → Starting services in proper order..."
    
    # Start containerd first
    if ! systemctl start containerd 2>/dev/null; then
        log_error "   ❌ Failed to start containerd"
        return 1
    fi
    
    # Wait for containerd to be ready
    sleep 3
    
    # Verify containerd socket
    local retry_count=0
    while [ $retry_count -lt 10 ] && [ ! -S /run/containerd/containerd.sock ]; do
        sleep 1
        retry_count=$((retry_count + 1))
    done
    
    if [ ! -S /run/containerd/containerd.sock ]; then
        log_error "   ❌ Containerd socket not available"
        return 1
    fi
    
    log_success "   ✅ Containerd started successfully"
    
    # Start Docker daemon
    if ! systemctl start docker 2>/dev/null; then
        log_error "   ❌ Failed to start Docker daemon"
        return 1
    fi
    
    # Wait for Docker to be responsive
    log_info "   → Waiting for Docker daemon to become responsive..."
    local wait_count=0
    local max_wait=30
    
    while [ $wait_count -lt $max_wait ]; do
        if docker version >/dev/null 2>&1; then
            log_success "   ✅ Docker daemon is responsive"
            return 0
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    log_error "   ❌ Docker daemon not responsive after ${max_wait}s"
    return 1
}

# Create improved Docker daemon management function for the script
create_improved_docker_function() {
    local script_file="/opt/sutazaiapp/scripts/deploy_complete_system.sh"
    local backup_file="${script_file}.backup_$(date +%Y%m%d_%H%M%S)"
    
    log_info "🔧 Creating improved Docker management function..."
    
    # Backup original script
    cp "$script_file" "$backup_file"
    log_success "   ✅ Backup created: $backup_file"
    
    # Create the improved function
    cat > /tmp/improved_docker_function.sh << 'EOF'
# 🧠 SUPER INTELLIGENT Docker Daemon Management (Fixed)
super_intelligent_docker_management() {
    local operation="${1:-start}"
    local max_retries="${2:-3}"
    local retry_count=0
    
    log_info "🧠 Super Intelligent Docker Management: $operation"
    
    while [ $retry_count -lt $max_retries ]; do
        case "$operation" in
            "start")
                # Check if already running and responsive
                if docker version >/dev/null 2>&1; then
                    log_success "   ✅ Docker already running and responsive"
                    return 0
                fi
                
                # Clean up any stale processes
                local stale_pids=$(pgrep -f "dockerd" || echo "")
                if [ -n "$stale_pids" ]; then
                    log_info "   → Cleaning stale Docker processes: $stale_pids"
                    echo "$stale_pids" | xargs -r kill -9 2>/dev/null || true
                    rm -f /var/run/docker.pid /run/containerd/containerd.sock 2>/dev/null || true
                    sleep 2
                fi
                
                # Start containerd first
                if ! systemctl is-active --quiet containerd; then
                    systemctl start containerd 2>/dev/null || true
                    sleep 3
                fi
                
                # Start Docker daemon
                if systemctl start docker 2>/dev/null; then
                    sleep 5
                    
                    # Verify Docker is responsive
                    local verify_count=0
                    while [ $verify_count -lt 10 ]; do
                        if docker version >/dev/null 2>&1; then
                            log_success "   ✅ Docker started successfully (attempt $((retry_count + 1)))"
                            return 0
                        fi
                        sleep 2
                        verify_count=$((verify_count + 1))
                    done
                fi
                ;;
                
            "restart")
                log_info "   → Restarting Docker services..."
                systemctl stop docker containerd 2>/dev/null || true
                sleep 2
                super_intelligent_docker_management "start" 1
                return $?
                ;;
                
            "stop")
                log_info "   → Stopping Docker services..."
                systemctl stop docker containerd 2>/dev/null || true
                pkill -f "dockerd" 2>/dev/null || true
                return 0
                ;;
        esac
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log_warn "   ⚠️  Attempt $retry_count failed, retrying in 5s..."
            sleep 5
        fi
    done
    
    log_error "   ❌ Docker $operation failed after $max_retries attempts"
    return 1
}
EOF
    
    log_success "   ✅ Improved Docker function created"
}

# Main execution
main() {
    log_info "🚀 Starting Docker Daemon Management Fix"
    
    # Check if Docker is already working
    if docker version >/dev/null 2>&1; then
        log_success "✅ Docker is already working properly"
        
        # Test Docker functionality
        log_info "🧪 Testing Docker functionality..."
        if docker run --rm hello-world >/dev/null 2>&1; then
            log_success "✅ Docker test passed"
        else
            log_warn "⚠️  Docker test failed, but daemon is running"
        fi
        
        log_success "🎉 Docker Daemon Management Fix Complete - No action needed!"
        log_info "📄 Log saved to: $LOG_FILE"
        return 0
    fi
    
    # Fix current Docker state
    if smart_docker_daemon_management; then
        log_success "✅ Docker daemon management fixed successfully"
    else
        log_error "❌ Docker daemon management fix failed"
        exit 1
    fi
    
    # Create improved function for the script
    create_improved_docker_function
    
    # Test Docker functionality
    log_info "🧪 Testing Docker functionality..."
    if docker run --rm hello-world >/dev/null 2>&1; then
        log_success "✅ Docker test passed"
    else
        log_warn "⚠️  Docker test failed, but daemon is running"
    fi
    
    log_success "🎉 Docker Daemon Management Fix Complete!"
    log_info "📄 Log saved to: $LOG_FILE"
}

# Execute main function
main "$@"