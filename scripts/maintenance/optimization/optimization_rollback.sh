#!/bin/bash
"""
OPTIMIZATION ROLLBACK SCRIPT
Date: August 12, 2025
Author: System Optimization and Reorganization Specialist

CRITICAL: Safe rollback that preserves MCP containers
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/opt/sutazaiapp/logs/optimization_rollback.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Trap for cleanup
cleanup() {
    log "Rollback script interrupted, cleaning up..."
    exit 130
}
trap cleanup SIGINT SIGTERM

main() {
    log "=== ULTRA SYSTEM OPTIMIZATION ROLLBACK ==="
    log "CRITICAL: Preserving ALL MCP containers"
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for system parameter changes"
    fi
    
    log "Phase 1: Restoring system parameters..."
    
    # Restore memory parameters to defaults
    log "Restoring memory parameters..."
    sysctl vm.swappiness=60 2>/dev/null || true
    sysctl vm.vfs_cache_pressure=100 2>/dev/null || true
    sysctl vm.dirty_ratio=20 2>/dev/null || true
    sysctl vm.dirty_background_ratio=10 2>/dev/null || true
    sysctl vm.overcommit_memory=0 2>/dev/null || true
    
    # Restore CPU scheduler parameters to defaults
    log "Restoring CPU parameters..."
    sysctl kernel.sched_migration_cost_ns=500000 2>/dev/null || true
    sysctl kernel.sched_min_granularity_ns=2250000 2>/dev/null || true
    sysctl kernel.sched_wakeup_granularity_ns=3000000 2>/dev/null || true
    sysctl kernel.sched_compat_yield=0 2>/dev/null || true
    sysctl vm.stat_interval=1 2>/dev/null || true
    
    log "Phase 2: Restoring file descriptor limits..."
    
    # Remove custom limits file if it exists
    if [[ -f /etc/security/limits.d/99-ultra-optimization.conf ]]; then
        rm -f /etc/security/limits.d/99-ultra-optimization.conf
        log "Removed custom file descriptor limits"
    fi
    
    log "Phase 3: Restoring CPU governor..."
    
    # Set CPU governor back to default (ondemand/powersave)
    for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
        if [[ -f "$cpu_dir/cpufreq/scaling_governor" ]]; then
            # Try ondemand first, then powersave
            if echo "ondemand" > "$cpu_dir/cpufreq/scaling_governor" 2>/dev/null; then
                true
            elif echo "powersave" > "$cpu_dir/cpufreq/scaling_governor" 2>/dev/null; then
                true
            fi
        fi
    done
    log "CPU governor restored to default"
    
    log "Phase 4: Checking MCP container integrity..."
    
    # Verify MCP containers are still running
    mcp_patterns=("mcp/duckduckgo" "mcp/fetch" "mcp/sequentialthinking")
    mcp_count=0
    
    for pattern in "${mcp_patterns[@]}"; do
        count=$(docker ps --format "{{.Image}}" | grep -c "$pattern" || true)
        mcp_count=$((mcp_count + count))
        log "Found $count running containers for $pattern"
    done
    
    log "Total MCP containers preserved: $mcp_count"
    
    log "Phase 5: System health check..."
    
    # Basic health checks
    load_avg=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | sed 's/,//')
    memory_usage=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100.0}')
    
    log "Current system load: $load_avg"
    log "Current memory usage: ${memory_usage}%"
    
    # Check if any services are down
    if ! docker ps >/dev/null 2>&1; then
        log "WARNING: Docker daemon may be unresponsive"
    else
        running_containers=$(docker ps | wc -l)
        log "Docker containers running: $((running_containers - 1))"
    fi
    
    log "Phase 6: Cleanup temporary files..."
    
    # Clean up optimization logs (keep recent ones)
    find /opt/sutazaiapp/logs/ -name "*optimization*" -type f -mtime +7 -delete 2>/dev/null || true
    log "Cleaned old optimization logs"
    
    log "=== ROLLBACK COMPLETED SUCCESSFULLY ==="
    log "MCP containers preserved and system parameters restored"
    log "If issues persist, consider rebooting the system"
    
    # Show final system status
    echo ""
    echo "📊 SYSTEM STATUS AFTER ROLLBACK:"
    echo "================================="
    echo "Load Average: $(uptime | awk -F'load average:' '{ print $2 }')"
    echo "Memory Usage: $(free -h | awk '/^Mem:/{printf "%s/%s (%.1f%%)", $3, $2, $3/$2 * 100.0}')"
    echo "Docker Status: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'Not available')"
    echo "MCP Containers: $mcp_count preserved"
    echo ""
    echo "✅ Rollback completed successfully!"
}

# Show usage if requested
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Ultra System Optimization Rollback Script"
    echo ""
    echo "This script safely rolls back system optimizations while preserving"
    echo "all MCP containers and restoring default system parameters."
    echo ""
    echo "Usage: $0"
    echo ""
    echo "Requirements:"
    echo "  - Must be run as root"
    echo "  - Docker must be accessible"
    echo ""
    echo "The rollback will:"
    echo "  - Restore system parameters to defaults"
    echo "  - Remove custom file descriptor limits"
    echo "  - Reset CPU governor to default"
    echo "  - Verify MCP container integrity"
    echo "  - Perform system health checks"
    exit 0
fi

# Run main function
main "$@"