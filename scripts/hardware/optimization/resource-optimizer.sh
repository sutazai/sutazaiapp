#!/bin/bash
# Resource Optimization Script - Enterprise Hardware Management
# Created: 2025-08-19
# Purpose: Optimize system resources and manage container memory limits

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check system resources
check_resources() {
    log_info "Checking system resources..."
    
    # Memory usage
    MEM_TOTAL=$(free -b | grep Mem | awk '{print $2}')
    MEM_USED=$(free -b | grep Mem | awk '{print $3}')
    MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))
    
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' || echo "0")
    
    echo "Memory Usage: ${MEM_PERCENT}%"
    echo "CPU Usage: ${CPU_USAGE}%"
    
    if [ $MEM_PERCENT -gt 80 ]; then
        log_warning "High memory usage detected: ${MEM_PERCENT}%"
        return 1
    fi
    
    return 0
}

# Function to optimize Docker containers
optimize_docker() {
    log_info "Optimizing Docker containers..."
    
    # Set memory limits for containers
    declare -A CONTAINER_LIMITS=(
        ["sutazai-neo4j"]="512m"
        ["sutazai-backend"]="1g"
        ["sutazai-frontend"]="1g"
        ["sutazai-postgres"]="512m"
        ["sutazai-redis"]="256m"
        ["sutazai-rabbitmq"]="512m"
        ["sutazai-prometheus"]="512m"
        ["sutazai-grafana"]="512m"
        ["sutazai-consul"]="256m"
    )
    
    for container in "${!CONTAINER_LIMITS[@]}"; do
        if docker ps -q -f name="$container" &>/dev/null; then
            log_info "Setting memory limit for $container to ${CONTAINER_LIMITS[$container]}"
            docker update --memory="${CONTAINER_LIMITS[$container]}" "$container" 2>/dev/null || \
                log_warning "Could not update memory limit for $container"
        fi
    done
    
    # Clean up stopped containers
    STOPPED_CONTAINERS=$(docker ps -aq -f status=exited | wc -l)
    if [ "$STOPPED_CONTAINERS" -gt 0 ]; then
        log_info "Removing $STOPPED_CONTAINERS stopped containers..."
        docker container prune -f
    fi
    
    # Clean up unused images
    log_info "Cleaning unused Docker images..."
    docker image prune -af --filter "until=24h"
}

# Function to clean system caches
clean_caches() {
    log_info "Cleaning system caches..."
    
    # NPM cache
    if command -v npm &>/dev/null; then
        npm cache clean --force 2>/dev/null || log_warning "Could not clean npm cache"
    fi
    
    # Clear temp files
    rm -rf /tmp/npm-* /tmp/v8-* /tmp/tsc* 2>/dev/null || true
    
    # Clear old log files
    find /var/log -type f -name "*.log" -mtime +7 -exec rm {} \; 2>/dev/null || true
    
    # Sync and drop caches (requires root)
    if [ "$EUID" -eq 0 ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        log_info "System caches dropped"
    else
        log_warning "Not running as root, skipping system cache drop"
    fi
}

# Function to kill resource-intensive processes
kill_heavy_processes() {
    log_info "Checking for resource-intensive processes..."
    
    # Kill orphaned TypeScript servers
    TSSERVER_COUNT=$(pgrep -f tsserver | wc -l)
    if [ "$TSSERVER_COUNT" -gt 2 ]; then
        log_warning "Found $TSSERVER_COUNT TypeScript servers, killing excess..."
        pkill -f tsserver || true
    fi
    
    # Kill zombie processes
    ZOMBIES=$(ps aux | grep defunct | grep -v grep | wc -l)
    if [ "$ZOMBIES" -gt 0 ]; then
        log_warning "Found $ZOMBIES zombie processes"
        ps aux | grep defunct | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    fi
}

# Function to optimize running services
optimize_services() {
    log_info "Optimizing running services..."
    
    # Restart memory-heavy services if needed
    if check_resources; then
        log_info "Resources within acceptable limits"
    else
        log_warning "High resource usage detected, optimizing services..."
        
        # Restart specific services if they're using too much memory
        for service in neo4j grafana prometheus; do
            CONTAINER="sutazai-$service"
            if docker ps -q -f name="$CONTAINER" &>/dev/null; then
                MEM_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" "$CONTAINER" 2>/dev/null | sed 's/%//' || echo "0")
                if [ ! -z "$MEM_USAGE" ] && (( $(echo "$MEM_USAGE > 50" | bc -l 2>/dev/null || echo 0) )); then
                    log_warning "$CONTAINER using ${MEM_USAGE}% memory, restarting..."
                    docker restart "$CONTAINER"
                    sleep 5
                fi
            fi
        done
    fi
}

# Function to generate optimization report
generate_report() {
    log_info "Generating optimization report..."
    
    REPORT_FILE="/opt/sutazaiapp/reports/resource-optimization-$(date +%Y%m%d-%H%M%S).md"
    mkdir -p "$(dirname "$REPORT_FILE")"
    
    cat > "$REPORT_FILE" << EOF
# Resource Optimization Report
Generated: $(date)

## System Resources
\`\`\`
$(free -h)
\`\`\`

## Docker Containers
\`\`\`
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | head -20)
\`\`\`

## Top Processes by Memory
\`\`\`
$(ps aux --sort=-%mem | head -10)
\`\`\`

## Top Processes by CPU
\`\`\`
$(ps aux --sort=-%cpu | head -10)
\`\`\`

## Disk Usage
\`\`\`
$(df -h)
\`\`\`

## Optimization Actions Taken
- Cleaned Docker resources
- Set container memory limits
- Cleared system caches
- Removed orphaned processes
EOF
    
    log_info "Report saved to: $REPORT_FILE"
}

# Main execution
main() {
    log_info "Starting resource optimization..."
    
    # Check initial state
    check_resources
    
    # Perform optimizations
    kill_heavy_processes
    optimize_docker
    clean_caches
    optimize_services
    
    # Generate report
    generate_report
    
    # Final check
    check_resources
    
    log_info "Resource optimization completed!"
}

# Run main function
main "$@"