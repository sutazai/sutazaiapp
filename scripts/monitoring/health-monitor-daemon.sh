#!/bin/bash

# Sutazai Health Monitor Daemon
# Continuously monitors service health and auto-recovers unhealthy services

set -e

# Configuration
LOG_FILE="/opt/sutazaiapp/logs/health-monitor.log"
MONITOR_INTERVAL=30
AUTO_RESTART_THRESHOLD=3  # Restart after N consecutive unhealthy checks
RESOURCE_WARNING_THRESHOLD=80  # Warn when memory usage exceeds this percentage

# Ensure log directory exists
mkdir -p $(dirname "$LOG_FILE")

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Service health tracking
declare -A unhealthy_counts
declare -A last_restart_time

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case $level in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
    esac
}

# Function to get service health
get_service_health() {
    local service=$1
    docker inspect $service --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-health-check"
}

# Function to get service memory usage
get_memory_stats() {
    local service=$1
    local stats=$(docker stats $service --no-stream --format "{{.MemUsage}} ({{.MemPerc}})" 2>/dev/null || echo "N/A")
    echo "$stats"
}

# Function to check if service needs restart
should_restart_service() {
    local service=$1
    local current_time=$(date +%s)
    local last_restart=${last_restart_time[$service]:-0}
    local time_since_restart=$((current_time - last_restart))
    
    # Don't restart if we just restarted less than 5 minutes ago
    if [ $time_since_restart -lt 300 ]; then
        return 1
    fi
    
    # Check unhealthy count threshold
    local count=${unhealthy_counts[$service]:-0}
    if [ $count -ge $AUTO_RESTART_THRESHOLD ]; then
        return 0
    fi
    
    return 1
}

# Function to restart service
restart_service() {
    local service=$1
    log_message "INFO" "Restarting $service..."
    
    if docker restart $service > /dev/null 2>&1; then
        log_message "SUCCESS" "$service restarted successfully"
        unhealthy_counts[$service]=0
        last_restart_time[$service]=$(date +%s)
        return 0
    else
        log_message "ERROR" "Failed to restart $service"
        return 1
    fi
}

# Function to check and fix Ollama specifically
check_ollama() {
    local service="sutazai-ollama"
    
    # Check if Ollama API is responding
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        return 0
    else
        log_message "WARNING" "Ollama API not responding, checking container..."
        
        # Try to list models using the ollama command
        if docker exec $service ollama list > /dev/null 2>&1; then
            log_message "INFO" "Ollama container is functional but API may be slow"
            return 0
        else
            log_message "ERROR" "Ollama container is not functional"
            return 1
        fi
    fi
}

# Function to monitor single service
monitor_service() {
    local service=$1
    local health=$(get_service_health $service)
    local memory=$(get_memory_stats $service)
    
    # Special handling for Ollama
    if [ "$service" = "sutazai-ollama" ]; then
        if check_ollama; then
            health="healthy"
        else
            health="unhealthy"
        fi
    fi
    
    # Update unhealthy count
    if [ "$health" = "unhealthy" ]; then
        unhealthy_counts[$service]=$((${unhealthy_counts[$service]:-0} + 1))
        
        # Check if we should restart
        if should_restart_service $service; then
            restart_service $service
        else
            log_message "WARNING" "$service is unhealthy (count: ${unhealthy_counts[$service]})"
        fi
    elif [ "$health" = "healthy" ]; then
        # Reset unhealthy count if service is healthy
        if [ "${unhealthy_counts[$service]:-0}" -gt 0 ]; then
            log_message "SUCCESS" "$service recovered to healthy state"
            unhealthy_counts[$service]=0
        fi
    fi
    
    # Return status for display
    echo "$service|$health|$memory"
}

# Function to display dashboard
display_dashboard() {
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo "                 Sutazai Infrastructure Monitor                 "
    echo "════════════════════════════════════════════════════════════════"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Service Health Status:"
    echo "─────────────────────────────────────────────────────────────────"
    printf "%-25s %-15s %-30s\n" "Service" "Health" "Memory Usage"
    echo "─────────────────────────────────────────────────────────────────"
}

# Function to monitor all services
monitor_all_services() {
    local services=(
        "sutazai-ollama"
        "sutazai-semgrep"
        "sutazai-documind"
        "sutazai-finrobot"
        "sutazai-backend"
        "sutazai-frontend"
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-neo4j"
        "sutazai-rabbitmq"
    )
    
    display_dashboard
    
    for service in "${services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            local status=$(monitor_service $service)
            IFS='|' read -r name health memory <<< "$status"
            
            # Color code the output
            if [ "$health" = "healthy" ]; then
                printf "${GREEN}%-25s${NC} ${GREEN}%-15s${NC} %-30s\n" "$name" "✓ $health" "$memory"
            elif [ "$health" = "unhealthy" ]; then
                local count=${unhealthy_counts[$service]:-0}
                printf "${RED}%-25s${NC} ${RED}%-15s${NC} %-30s\n" "$name" "✗ $health ($count)" "$memory"
            else
                printf "${YELLOW}%-25s${NC} ${YELLOW}%-15s${NC} %-30s\n" "$name" "? $health" "$memory"
            fi
        fi
    done
    
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "Auto-restart threshold: $AUTO_RESTART_THRESHOLD consecutive failures"
    echo "Next check in: $MONITOR_INTERVAL seconds"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
}

# Function to cleanup on exit
cleanup() {
    log_message "INFO" "Health monitor daemon stopped"
    exit 0
}

# Trap signals for clean exit
trap cleanup SIGINT SIGTERM

# Main monitoring loop
main() {
    log_message "INFO" "Starting Sutazai Health Monitor Daemon"
    log_message "INFO" "Monitor interval: ${MONITOR_INTERVAL}s, Auto-restart threshold: $AUTO_RESTART_THRESHOLD"
    
    while true; do
        monitor_all_services
        sleep $MONITOR_INTERVAL
    done
}

# Check if running in background mode
if [ "$1" = "--daemon" ]; then
    log_message "INFO" "Running in daemon mode"
    while true; do
        for service in sutazai-ollama sutazai-semgrep sutazai-documind sutazai-finrobot; do
            if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
                monitor_service $service > /dev/null 2>&1
            fi
        done
        sleep $MONITOR_INTERVAL
    done
else
    # Run in interactive mode
    main
fi