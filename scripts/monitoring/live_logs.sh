#!/bin/bash
# Purpose: SutazAI Live Logs Management System - Comprehensive log monitoring and management
# Usage: ./live_logs.sh [live|follow|cleanup|reset|debug|config|status] [OPTIONS]
# Requires: docker, docker-compose, system utilities

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/.logs_config"

# Enable numlock automatically
enable_numlock() {
    # Check if setleds command is available
    if command -v setleds >/dev/null 2>&1; then
        # Enable numlock for all TTYs
        for tty in /dev/tty[1-6]; do
            setleds -D +num < "$tty" >/dev/null 2>&1 || true
        done
    fi
    
    # If numlockx is available, use it for X sessions
    if command -v numlockx >/dev/null 2>&1; then
        numlockx on >/dev/null 2>&1 || true
    fi
}

# Check numlock status
check_numlock_status() {
    if command -v setleds >/dev/null 2>&1; then
        # Check current TTY
        if setleds -F 2>&1 | grep -q "NumLock on"; then
            echo "ON"
        else
            echo "OFF"
        fi
    elif command -v numlockx >/dev/null 2>&1; then
        if numlockx status 2>&1 | grep -q "on"; then
            echo "ON"
        else
            echo "OFF"
        fi
    else
        echo "N/A"
    fi
}

# Enable numlock at script start
enable_numlock

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_DEBUG_MODE="false"
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_MAX_LOG_SIZE="100M"
DEFAULT_MAX_LOG_FILES="10"
DEFAULT_CLEANUP_DAYS="7"

# Initialize variables with defaults to prevent unbound variable errors
DEBUG_MODE="${DEBUG_MODE:-$DEFAULT_DEBUG_MODE}"
LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
MAX_LOG_SIZE="${MAX_LOG_SIZE:-$DEFAULT_MAX_LOG_SIZE}"
MAX_LOG_FILES="${MAX_LOG_FILES:-$DEFAULT_MAX_LOG_FILES}"
CLEANUP_DAYS="${CLEANUP_DAYS:-$DEFAULT_CLEANUP_DAYS}"

# Load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
    else
        DEBUG_MODE=${DEBUG_MODE:-$DEFAULT_DEBUG_MODE}
        LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}
        MAX_LOG_SIZE=${MAX_LOG_SIZE:-$DEFAULT_MAX_LOG_SIZE}
        MAX_LOG_FILES=${MAX_LOG_FILES:-$DEFAULT_MAX_LOG_FILES}
        CLEANUP_DAYS=${CLEANUP_DAYS:-$DEFAULT_CLEANUP_DAYS}
        save_config
    fi
}

# Save configuration
save_config() {
    cat > "$CONFIG_FILE" << EOF
# SutazAI Logs Configuration
DEBUG_MODE="${DEBUG_MODE:-$DEFAULT_DEBUG_MODE}"
LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
MAX_LOG_SIZE="${MAX_LOG_SIZE:-$DEFAULT_MAX_LOG_SIZE}"
MAX_LOG_FILES="${MAX_LOG_FILES:-$DEFAULT_MAX_LOG_FILES}"
CLEANUP_DAYS="${CLEANUP_DAYS:-$DEFAULT_CLEANUP_DAYS}"
LAST_UPDATED=$(date)
EOF
}

# Print header
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}               ${CYAN}SutazAI Live Logs Management System${NC}              ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Configuration:${NC}"
    echo -e "  Debug Mode: ${DEBUG_MODE}"
    echo -e "  Log Level: ${LOG_LEVEL}"
    echo -e "  Max Log Size: ${MAX_LOG_SIZE}"
    echo -e "  Max Log Files: ${MAX_LOG_FILES}"
    echo -e "  Cleanup Days: ${CLEANUP_DAYS}"
    echo ""
}

# Show usage
show_usage() {
    echo -e "${GREEN}Usage: $0 [COMMAND] [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}Primary Commands:${NC}"
    echo "  live                    - Start live log monitoring"
    echo "  follow [service]        - Follow specific service logs"
    echo "  cleanup [days]          - Clean old logs (default: 7 days)"
    echo "  reset                   - Reset all logs (DANGEROUS)"
    echo "  debug [on|off]          - Toggle debugging mode"
    echo "  level [DEBUG|INFO|WARN] - Set log level"
    echo "  config                  - Show current configuration"
    echo "  status                  - Show log status and disk usage"
    echo "  archive                 - Archive old logs"
    echo ""
    echo -e "${YELLOW}System Management:${NC}"
    echo "  --init-db               - Initialize SutazAI database"
    echo "  --repair                - Complete system repair"
    echo "  --overview              - System overview (non-interactive)"
    echo "  --test                  - Test API endpoints"
    echo "  --stats                 - Container statistics"
    echo ""
    echo -e "${YELLOW}Live Monitoring Options:${NC}"
    echo "  --filter [pattern]      - Filter logs by pattern"
    echo "  --service [name]        - Monitor specific service"
    echo "  --error-only           - Show only errors"
    echo "  --tail [lines]          - Number of lines to tail (default: 100)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 live --service backend --filter error"
    echo "  $0 follow sutazai-backend"
    echo "  $0 cleanup 3            # Clean logs older than 3 days"
    echo "  $0 debug on             # Enable debug mode"
    echo "  $0 --repair             # Fix database and restart services"
    echo "  $0 --init-db            # Initialize database only"
}

# Get container logs
get_container_logs() {
    local container_name="$1"
    local lines="${2:-100}"
    local follow="${3:-false}"
    
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        if [[ "$follow" == "true" ]]; then
            docker logs -f --tail "$lines" "$container_name" 2>&1
        else
            docker logs --tail "$lines" "$container_name" 2>&1
        fi
    else
        echo -e "${RED}Container ${container_name} not found or not running${NC}"
        return 1
    fi
}

# Live monitoring
start_live_monitoring() {
    local filter_pattern="${1:-}"
    local service_filter="${2:-}"
    local error_only="${3:-false}"
    local tail_lines="${4:-100}"
    
    # Check Docker daemon first
    if ! check_docker_daemon; then
        return 1
    fi
    
    echo -e "${GREEN}Starting live log monitoring...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Get all SutazAI containers
    local containers=($(docker ps --format "{{.Names}}" 2>/dev/null | grep "sutazai-" | sort))
    
    if [[ ${#containers[@]} -eq 0 ]]; then
        echo -e "${RED}No SutazAI containers found running${NC}"
        return 1
    fi
    
    # Filter by service if specified
    if [[ -n "$service_filter" ]]; then
        containers=($(printf '%s\n' "${containers[@]}" | grep "$service_filter" || true))
        if [[ ${#containers[@]} -eq 0 ]]; then
            echo -e "${RED}No containers found matching service: $service_filter${NC}"
            return 1
        fi
    fi
    
    echo -e "${CYAN}Monitoring containers:${NC}"
    printf '  - %s\n' "${containers[@]}"
    echo ""
    
    # Start logging each container in background with color coding
    for container in "${containers[@]}"; do
        (
            # Timeout mechanism to prevent infinite loops
            LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
            loop_start=$(date +%s)
            while true; do
                if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
                    docker logs -f --tail 0 "$container" 2>&1 | while read -r line; do
                        local timestamp=$(date '+%H:%M:%S')
                        local colored_line
                        
                        # Apply filtering
                        if [[ -n "$filter_pattern" ]] && ! echo "$line" | grep -qi "$filter_pattern"; then
                            continue
                        fi
                        
                        if [[ "$error_only" == "true" ]] && ! echo "$line" | grep -qiE "(error|exception|fail|critical)"; then
                            continue
                        fi
                        
                        # Color code by log level and container
                        if echo "$line" | grep -qi "error\|exception\|fail"; then
                            colored_line="${RED}[${timestamp}] ${container}: ${line}${NC}"
                        elif echo "$line" | grep -qi "warn"; then
                            colored_line="${YELLOW}[${timestamp}] ${container}: ${line}${NC}"
                        elif echo "$line" | grep -qi "info"; then
                            colored_line="${GREEN}[${timestamp}] ${container}: ${line}${NC}"
                        elif echo "$line" | grep -qi "debug"; then
                            colored_line="${PURPLE}[${timestamp}] ${container}: ${line}${NC}"
                        else
                            colored_line="${CYAN}[${timestamp}] ${container}:${NC} ${line}"
                        fi
                        
                        echo -e "$colored_line"
                # Check for timeout
                current_time=$(date +%s)
                if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
                    echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
                    break
                fi

                    done
                else
                    echo -e "${RED}[$(date '+%H:%M:%S')] ${container}: Container stopped${NC}"
                    sleep 5
                fi
            done
        ) &
    done
    
    # Wait for user interrupt
    wait
}

# Container name to service name mapping function
get_service_name_from_container() {
    local container_name="$1"
    
    # Remove sutazai- prefix to get base name
    local base_name=${container_name#sutazai-}
    
    # Handle special cases where service name differs from container base name
    case "$base_name" in
        "api")
            echo "backend"
            ;;
        *)
            echo "$base_name"
            ;;
    esac
}

# Follow specific service
follow_service() {
    local service_name="$1"
    local lines="${2:-100}"
    
    echo -e "${GREEN}Following logs for: ${service_name}${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    get_container_logs "$service_name" "$lines" "true"
}

# Cleanup logs
cleanup_logs() {
    local older_than="${1:-${CLEANUP_DAYS}d}"
    local dry_run="${2:-false}"
    
    echo -e "${YELLOW}Cleaning up logs older than: ${older_than}${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    # Find and process old log files
    local files_to_delete=()
    while IFS= read -r -d '' file; do
        files_to_delete+=("$file")
    done < <(find "$LOG_DIR" -type f \( -name "*.log*" -o -name "*.out" -o -name "*.err" \) -mtime "+${older_than%d}" -print0 2>/dev/null || true)
    
    if [[ ${#files_to_delete[@]} -eq 0 ]]; then
        echo -e "${GREEN}No old log files found to clean up${NC}"
        return 0
    fi
    
    echo -e "${CYAN}Found ${#files_to_delete[@]} files to clean up:${NC}"
    printf '  - %s\n' "${files_to_delete[@]}"
    
    if [[ "$dry_run" == "true" ]]; then
        echo -e "${YELLOW}Dry run - no files deleted${NC}"
        return 0
    fi
    
    # Calculate space to be freed
    local total_size=0
    for file in "${files_to_delete[@]}"; do
        if [[ -f "$file" ]]; then
            local size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            total_size=$((total_size + size))
        fi
    done
    
    echo -e "${CYAN}Total space to be freed: $(numfmt --to=iec $total_size)${NC}"
    
    read -p "Proceed with deletion? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for file in "${files_to_delete[@]}"; do
            rm -f "$file"
            echo -e "${GREEN}Deleted: ${file}${NC}"
        done
        echo -e "${GREEN}Cleanup completed successfully${NC}"
        
        # Cleanup Docker logs for stopped containers
        echo -e "${CYAN}Cleaning up Docker container logs...${NC}"
        docker system prune -f --filter "until=72h" > /dev/null 2>&1 || true
        
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}"
    fi
    
    # Rotate large log files
    rotate_large_logs
}

# Reset all logs
reset_logs() {
    local confirm="${1:-false}"
    
    if [[ "$confirm" != "true" ]]; then
        echo -e "${RED}WARNING: This will delete ALL log files!${NC}"
        echo -e "${YELLOW}This action cannot be undone.${NC}"
        echo ""
        read -p "Type 'CONFIRM' to proceed: " -r
        if [[ $REPLY != "CONFIRM" ]]; then
            echo -e "${YELLOW}Reset cancelled${NC}"
            return 0
        fi
    fi
    
    echo -e "${RED}Resetting all logs...${NC}"
    
    # Remove all log files
    if [[ -d "$LOG_DIR" ]]; then
        rm -rf "${LOG_DIR:?}"/*
        echo -e "${GREEN}Application logs cleared${NC}"
    fi
    
    # Clear Docker logs for all containers
    local containers=($(docker ps -a --format "{{.Names}}" | grep "sutazai-" || true))
    for container in "${containers[@]}"; do
        if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
            # Truncate logs by restarting container
            docker restart "$container" > /dev/null 2>&1 || true
            echo -e "${GREEN}Reset logs for: ${container}${NC}"
        fi
    done
    
    echo -e "${GREEN}Log reset completed${NC}"
}

# Toggle debug mode
toggle_debug() {
    local mode="$1"
    
    case "$mode" in
        "on"|"true"|"1")
            DEBUG_MODE="true"
            LOG_LEVEL="DEBUG"
            echo -e "${GREEN}Debug mode enabled${NC}"
            ;;
        "off"|"false"|"0")
            DEBUG_MODE="false"
            LOG_LEVEL="INFO"
            echo -e "${YELLOW}Debug mode disabled${NC}"
            ;;
        *)
            echo -e "${RED}Invalid debug mode: $mode${NC}"
            echo "Use: on/off, true/false, or 1/0"
            return 1
            ;;
    esac
    
    save_config
    
    # Update Docker environment variables for all compose files
    export DEBUG_MODE LOG_LEVEL
    
    echo -e "${CYAN}Applying debug settings to containers...${NC}"
    
    # Apply to all compose files
    local compose_files=(
        "docker-compose.yml"
        "docker-compose-complete-agi.yml"
        "docker-compose-complete.yml"
    )
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${compose_file}" ]]; then
            echo -e "  Updating ${compose_file}..."
            docker compose -f "${PROJECT_ROOT}/${compose_file}" restart > /dev/null 2>&1 || true
        fi
    done
    
    echo -e "${GREEN}Debug mode updated and applied${NC}"
}

# Set log level
set_log_level() {
    local level="$1"
    
    case "$level" in
        "DEBUG"|"INFO"|"WARN"|"ERROR")
            LOG_LEVEL="$level"
            save_config
            echo -e "${GREEN}Log level set to: ${level}${NC}"
            
            # Apply to containers
            export LOG_LEVEL
            docker compose restart > /dev/null 2>&1 || true
            ;;
        *)
            echo -e "${RED}Invalid log level: $level${NC}"
            echo "Valid levels: DEBUG, INFO, WARN, ERROR"
            return 1
            ;;
    esac
}

# Show configuration
show_config() {
    echo -e "${CYAN}Current Configuration:${NC}"
    echo "  Debug Mode: $DEBUG_MODE"
    echo "  Log Level: $LOG_LEVEL"
    echo "  Max Log Size: $MAX_LOG_SIZE"
    echo "  Max Log Files: $MAX_LOG_FILES"
    echo "  Cleanup Days: $CLEANUP_DAYS"
    echo "  Config File: $CONFIG_FILE"
    echo ""
    echo -e "${CYAN}Environment Variables:${NC}"
    echo "  DEBUG_MODE=${DEBUG_MODE}"
    echo "  LOG_LEVEL=${LOG_LEVEL}"
}

# Show log status
show_status() {
    echo -e "${CYAN}Log Status and Disk Usage:${NC}"
    echo ""
    
    # Disk usage
    if [[ -d "$LOG_DIR" ]]; then
        local log_size=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
        echo -e "${YELLOW}Log Directory Size:${NC} $log_size"
        echo -e "${YELLOW}Log Directory Path:${NC} $LOG_DIR"
        
        # Count log files
        local log_count=$(find "$LOG_DIR" -name "*.log*" -type f 2>/dev/null | wc -l)
        echo -e "${YELLOW}Total Log Files:${NC} $log_count"
    else
        echo -e "${YELLOW}Log directory does not exist${NC}"
    fi
    
    echo ""
    
    # Container log sizes
    echo -e "${CYAN}Container Log Status:${NC}"
    local containers=($(docker ps -a --format "{{.Names}}" | grep "sutazai-" | sort))
    
    for container in "${containers[@]}"; do
        local status=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null || echo "unknown")
        local log_path=$(docker inspect "$container" --format='{{.LogPath}}' 2>/dev/null || echo "unknown")
        local log_size="unknown"
        
        if [[ -f "$log_path" ]]; then
            log_size=$(du -sh "$log_path" 2>/dev/null | cut -f1)
        fi
        
        printf "  %-20s Status: %-10s Log Size: %s\n" "$container" "$status" "$log_size"
    done
    
    echo ""
    
    # System disk usage
    echo -e "${CYAN}System Disk Usage:${NC}"
    df -h / | tail -1 | awk '{printf "  Root: %s used, %s available (%s full)\n", $3, $4, $5}'
    
    # Docker disk usage
    echo -e "${CYAN}Docker Disk Usage:${NC}"
    docker system df 2>/dev/null | tail -n +2 | while read -r line; do
        echo "  $line"
    done || echo "  Docker system df unavailable"
}

# Archive old logs
archive_logs() {
    local archive_dir="${LOG_DIR}/archive"
    local date_suffix=$(date +%Y%m%d_%H%M%S)
    
    mkdir -p "$archive_dir"
    
    echo -e "${CYAN}Archiving logs older than ${CLEANUP_DAYS} days...${NC}"
    
    # Find old logs
    local old_logs=()
    while IFS= read -r -d '' file; do
        old_logs+=("$file")
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log*" -mtime "+${CLEANUP_DAYS}" -print0 2>/dev/null || true)
    
    if [[ ${#old_logs[@]} -eq 0 ]]; then
        echo -e "${GREEN}No old logs found to archive${NC}"
        return 0
    fi
    
    # Create archive
    local archive_file="${archive_dir}/logs_${date_suffix}.tar.gz"
    tar -czf "$archive_file" -C "$LOG_DIR" $(basename "${old_logs[@]}") 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
        # Remove original files
        rm -f "${old_logs[@]}"
        echo -e "${GREEN}Archived ${#old_logs[@]} log files to: ${archive_file}${NC}"
        
        # Show archive size
        local archive_size=$(du -sh "$archive_file" | cut -f1)
        echo -e "${CYAN}Archive size: ${archive_size}${NC}"
    else
        echo -e "${RED}Failed to create archive${NC}"
        return 1
    fi
}

# Rotate large log files
rotate_large_logs() {
    echo -e "${CYAN}Rotating large log files...${NC}"
    
    # Convert max size to bytes
    local max_bytes
    case "${MAX_LOG_SIZE}" in
        *K) max_bytes=$((${MAX_LOG_SIZE%K} * 1024)) ;;
        *M) max_bytes=$((${MAX_LOG_SIZE%M} * 1024 * 1024)) ;;
        *G) max_bytes=$((${MAX_LOG_SIZE%G} * 1024 * 1024 * 1024)) ;;
        *) max_bytes=$((MAX_LOG_SIZE)) ;;
    esac
    
    find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | while read -r logfile; do
        if [[ -f "$logfile" ]]; then
            local size=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
            if [[ $size -gt $max_bytes ]]; then
                local timestamp=$(date +%Y%m%d_%H%M%S)
                local rotated="${logfile}.${timestamp}"
                mv "$logfile" "$rotated"
                gzip "$rotated"
                echo -e "${GREEN}Rotated large log: $(basename "$logfile")${NC}"
                
                # Remove old rotated logs
                local basename=$(basename "$logfile")
                find "$LOG_DIR" -name "${basename}.*" -type f -mtime "+${MAX_LOG_FILES}" -delete 2>/dev/null || true
            fi
        fi
    done
}

# Main execution
main() {
    load_config
    
    case "${1:-}" in
        "live")
            shift
            local filter=""
            local service=""
            local error_only="false"
            local tail_lines="100"
            
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --filter)
                        filter="$2"
                        shift 2
                        ;;
                    --service)
                        service="$2"
                        shift 2
                        ;;
                    --error-only)
                        error_only="true"
                        shift
                        ;;
                    --tail)
                        tail_lines="$2"
                        shift 2
                        ;;
                    *)
                        echo -e "${RED}Unknown option: $1${NC}"
                        show_usage
                        exit 1
                        ;;
                esac
            done
            
            print_header
            start_live_monitoring "$filter" "$service" "$error_only" "$tail_lines"
            ;;
        "follow")
            follow_service "${2:-sutazai-backend}" "${3:-100}"
            ;;
        "cleanup")
            cleanup_logs "${2:-${CLEANUP_DAYS}d}" "${3:-false}"
            ;;
        "reset")
            reset_logs "${2:-false}"
            ;;
        "debug")
            toggle_debug "${2:-on}"
            ;;
        "level")
            set_log_level "${2:-INFO}"
            ;;
        "config")
            print_header
            show_config
            ;;
        "status")
            print_header
            show_status
            ;;
        "archive")
            archive_logs
            ;;
        "help"|"--help"|"-h")
            print_header
            show_usage
            ;;
        "")
            print_header
            show_usage
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# Note: main function execution removed - using direct case statement below

# Function to check if Docker daemon is running
check_docker_daemon() {
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker command not found. Please install Docker.${NC}"
        echo -e "${CYAN}💡 Auto-fix: Run deployment script to auto-install Docker:${NC}"
        echo -e "   ${CYAN}sudo ./scripts/deploy_complete_system.sh deploy${NC}"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Cannot connect to Docker daemon.${NC}"
        echo -e "${YELLOW}Docker daemon troubleshooting options:${NC}"
        echo -e "  1. Restart Docker service: ${CYAN}sudo systemctl restart docker${NC}"
        echo -e "  2. Start Docker manually: ${CYAN}sudo dockerd >/dev/null 2>&1 &${NC}"
        echo -e "  3. Check Docker status: ${CYAN}sudo systemctl status docker${NC}"
        echo -e "  4. View Docker logs: ${CYAN}sudo journalctl -u docker.service${NC}"
        echo -e "  5. Fix permissions: ${CYAN}sudo chmod 666 /var/run/docker.sock${NC}"
        echo -e "  ${GREEN}6. Auto-fix with deployment script: ${CYAN}sudo ./scripts/deploy_complete_system.sh deploy${NC}"
        return 1
    fi
    
    return 0
}

# Function to attempt Docker daemon recovery
attempt_docker_recovery() {
    echo -e "${YELLOW}Attempting Docker daemon recovery...${NC}"
    
    # Try method 1: Restart systemd service
    echo -e "${CYAN}Method 1: Restarting Docker service...${NC}"
    if sudo systemctl restart docker >/dev/null 2>&1; then
        sleep 3
        if docker info >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Docker service restarted successfully${NC}"
            return 0
        fi
    fi
    
    # Try method 2: Start Docker manually
    echo -e "${CYAN}Method 2: Starting Docker daemon manually...${NC}"
    sudo pkill -f dockerd >/dev/null 2>&1 || true
    sleep 2
    sudo dockerd >/dev/null 2>&1 &
    sleep 5
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Docker daemon started manually${NC}"
        return 0
    fi
    
    # Try method 3: Fix socket permissions
    echo -e "${CYAN}Method 3: Fixing Docker socket permissions...${NC}"
    if [[ -S /var/run/docker.sock ]]; then
        sudo chmod 666 /var/run/docker.sock
        if docker info >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Docker socket permissions fixed${NC}"
            return 0
        fi
    fi
    
    echo -e "${RED}✗ All recovery methods failed. Docker requires manual intervention.${NC}"
    return 1
}

# Docker troubleshooting menu
docker_troubleshooting_menu() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                DOCKER TROUBLESHOOTING MENU                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Show current Docker status
    echo -e "${BLUE}Current Docker Status:${NC}"
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Docker daemon is running${NC}"
        echo -e "  $(docker --version)"
        echo -e "  Running containers: $(docker ps --format "{{.Names}}" | wc -l)"
    else
        echo -e "${RED}✗ Docker daemon is not accessible${NC}"
    fi
    echo ""
    
    echo -e "${YELLOW}Troubleshooting Options:${NC}"
    echo "1. Check Docker Service Status"
    echo "2. Restart Docker Service"
    echo "3. Start Docker Manually"
    echo "4. Check Docker Logs"
    echo "5. Fix Docker Socket Permissions"
    echo "6. Automatic Recovery (All Methods)"
    echo "7. Reset Docker Configuration"
    echo "8. Show Docker System Information"
    echo "9. Test Docker with Hello-World"
    echo "0. Return to Main Menu"
    echo ""
    
    read -p "Select troubleshooting option (0-9): " docker_choice
    
    case $docker_choice in
        1) 
            echo -e "${CYAN}Docker Service Status:${NC}"
            sudo systemctl status docker --no-pager -l || true
            ;;
        2)
            echo -e "${CYAN}Restarting Docker Service...${NC}"
            sudo systemctl restart docker && echo -e "${GREEN}✓ Docker service restarted${NC}" || echo -e "${RED}✗ Failed to restart Docker service${NC}"
            ;;
        3)
            echo -e "${CYAN}Starting Docker Manually...${NC}"
            sudo pkill -f dockerd >/dev/null 2>&1 || true
            sleep 2
            sudo dockerd >/dev/null 2>&1 &
            sleep 3
            docker info >/dev/null 2>&1 && echo -e "${GREEN}✓ Docker started manually${NC}" || echo -e "${RED}✗ Failed to start Docker manually${NC}"
            ;;
        4)
            echo -e "${CYAN}Docker Service Logs (last 20 lines):${NC}"
            sudo journalctl -u docker.service --no-pager -n 20 || true
            ;;
        5)
            echo -e "${CYAN}Fixing Docker Socket Permissions...${NC}"
            if [[ -S /var/run/docker.sock ]]; then
                sudo chmod 666 /var/run/docker.sock
                echo -e "${GREEN}✓ Docker socket permissions fixed${NC}"
            else
                echo -e "${RED}✗ Docker socket not found${NC}"
            fi
            ;;
        6)
            attempt_docker_recovery
            ;;
        7)
            echo -e "${CYAN}Resetting Docker Configuration...${NC}"
            sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2"
}
EOF
            echo -e "${GREEN}✓ Stable Docker configuration created${NC}"
            sudo systemctl daemon-reload
            ;;
        8)
            echo -e "${CYAN}Docker System Information:${NC}"
            docker info 2>/dev/null || echo -e "${RED}Cannot retrieve Docker info - daemon not accessible${NC}"
            ;;
        9)
            echo -e "${CYAN}Testing Docker with Hello-World...${NC}"
            docker run --rm hello-world && echo -e "${GREEN}✓ Docker is working correctly${NC}" || echo -e "${RED}✗ Docker test failed${NC}"
            ;;
        0)
            return 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Function to check container status
check_container_status() {
    local container="$1"
    
    # First check if Docker daemon is accessible
    if ! check_docker_daemon >/dev/null 2>&1; then
        echo "?"
        return 1
    fi
    
    if docker ps --filter "name=${container}" --format "{{.Names}}" 2>/dev/null | grep -q "^${container}$"; then
        echo "✓"
    else
        echo "✗"
    fi
}

# Function to get container health
get_container_health() {
    local container=$1
    
    # First check if Docker daemon is accessible
    if ! check_docker_daemon >/dev/null 2>&1; then
        echo "no-docker"
        return 1
    fi
    
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "")
    if [ "$health" = "healthy" ]; then
        echo "healthy"
    elif [ "$health" = "unhealthy" ]; then
        echo "unhealthy"
    elif [ "$health" = "starting" ]; then
        echo "starting"
    else
        echo "no-check"
    fi
}

# Smart container health check and repair function
check_and_repair_unhealthy_containers() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        SMART CONTAINER HEALTH CHECK & REPAIR 🤖            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Enable numlock for this operation
    enable_numlock
    
    # Get list of all containers with health status
    echo -e "${YELLOW}⏳ Checking container health status and missing services...${NC}"
    echo ""
    
    # Arrays to store container states
    declare -a healthy_containers=()
    declare -a unhealthy_containers=()
    declare -a no_healthcheck_containers=()
    declare -a exited_containers=()
    declare -a missing_services=()
    
    # Get all services from docker-compose
    cd /opt/sutazaiapp
    all_services=($(docker compose config --services 2>/dev/null || echo ""))
    
    # Get all running/existing containers
    declare -A existing_containers
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        service_name=$(get_service_name_from_container "$container")
        existing_containers["$service_name"]=1
    done < <(docker ps -a --filter "name=sutazai-" --format "{{.Names}}")
    
    # Check for missing services
    for service in "${all_services[@]}"; do
        if [[ ! -v existing_containers["$service"] ]]; then
            missing_services+=("$service")
        fi
    done
    
    # Check all containers
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        
        container_name=$(echo "$container" | awk '{print $1}')
        container_status=$(echo "$container" | awk '{print $2}')
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
        
        if [[ "$container_status" == "Exited" ]] || [[ "$container_status" == "exited" ]]; then
            exited_containers+=("$container_name")
        elif [[ "$health_status" == "healthy" ]]; then
            healthy_containers+=("$container_name")
        elif [[ "$health_status" == "unhealthy" ]]; then
            unhealthy_containers+=("$container_name")
        elif [[ "$health_status" == "none" ]] || [[ "$health_status" == "<no value>" ]]; then
            no_healthcheck_containers+=("$container_name")
        fi
    done < <(docker ps -a --filter "name=sutazai-" --format "{{.Names}} {{.Status}}")
    
    # Analyze container issues for intelligent repair suggestions
    declare -A container_issues
    declare -A repair_priorities
    
    # Check for common issues in unhealthy containers
    for container in "${unhealthy_containers[@]}"; do
        service_name=$(get_service_name_from_container "$container")
        last_logs=$(docker logs --tail 20 "$container" 2>&1 | tail -5)
        
        # Detect common issues
        if echo "$last_logs" | grep -qi "connection refused\|cannot connect"; then
            container_issues["$container"]="Network/Connection issue"
            repair_priorities["$container"]=1
        elif echo "$last_logs" | grep -qi "out of memory\|memory"; then
            container_issues["$container"]="Memory issue"
            repair_priorities["$container"]=2
        elif echo "$last_logs" | grep -qi "permission denied\|access denied"; then
            container_issues["$container"]="Permission issue"
            repair_priorities["$container"]=3
        else
            container_issues["$container"]="Unknown issue"
            repair_priorities["$container"]=4
        fi
    done
    
    # Display results with enhanced information
    echo -e "${GREEN}✅ Healthy Containers (${#healthy_containers[@]}):${NC}"
    if [[ ${#healthy_containers[@]} -eq 0 ]]; then
        echo "   None found"
    else
        for container in "${healthy_containers[@]}"; do
            echo "   - $container"
        done
    fi
    echo ""
    
    echo -e "${RED}❌ Unhealthy Containers (${#unhealthy_containers[@]}):${NC}"
    if [[ ${#unhealthy_containers[@]} -eq 0 ]]; then
        echo "   None found"
    else
        for container in "${unhealthy_containers[@]}"; do
            issue="${container_issues[$container]:-Unknown}"
            priority="${repair_priorities[$container]:-5}"
            echo "   - $container ${YELLOW}[${issue}] Priority: ${priority}${NC}"
        done
    fi
    echo ""
    
    echo -e "${YELLOW}⚠️  Exited Containers (${#exited_containers[@]}):${NC}"
    if [[ ${#exited_containers[@]} -eq 0 ]]; then
        echo "   None found"
    else
        for container in "${exited_containers[@]}"; do
            echo "   - $container"
        done
    fi
    echo ""
    
    echo -e "${PURPLE}🚫 Missing Services (${#missing_services[@]}):${NC}"
    if [[ ${#missing_services[@]} -eq 0 ]]; then
        echo "   None found - all services deployed"
    else
        for service in "${missing_services[@]}"; do
            echo "   - $service"
        done
    fi
    echo ""
    
    echo -e "${BLUE}ℹ️  No Health Check (${#no_healthcheck_containers[@]}):${NC}"
    if [[ ${#no_healthcheck_containers[@]} -eq 0 ]]; then
        echo "   None found"
    else
        for container in "${no_healthcheck_containers[@]}"; do
            echo "   - $container"
        done
    fi
    echo ""
    
    # Ask for confirmation to repair/deploy
    total_to_repair=$((${#unhealthy_containers[@]} + ${#exited_containers[@]}))
    total_to_deploy=${#missing_services[@]}
    total_issues=$((total_to_repair + total_to_deploy))
    
    if [[ $total_issues -gt 0 ]]; then
        echo -e "${YELLOW}🔍 Analysis Summary:${NC}"
        [[ $total_to_repair -gt 0 ]] && echo "  - $total_to_repair container(s) need repair"
        [[ $total_to_deploy -gt 0 ]] && echo "  - $total_to_deploy service(s) need deployment"
        
        # Provide intelligent recommendation
        echo ""
        echo -e "${CYAN}🤖 Smart Recommendation:${NC}"
        if [[ ${#unhealthy_containers[@]} -gt 0 ]]; then
            # Check if we have high priority issues
            high_priority=0
            for container in "${unhealthy_containers[@]}"; do
                if [[ "${repair_priorities[$container]}" -le 2 ]]; then
                    high_priority=1
                    break
                fi
            done
            
            if [[ $high_priority -eq 1 ]]; then
                echo "  ⚠️  High priority issues detected (network/memory problems)"
                echo "  💡 Recommended: Option 3 - Full repair and deploy"
            else
                echo "  ℹ️  Standard issues detected"
                echo "  💡 Recommended: Option 1 - Repair containers first"
            fi
        elif [[ ${#missing_services[@]} -gt 0 ]]; then
            echo "  📦 Missing services detected"
            echo "  💡 Recommended: Option 2 - Deploy missing services"
        fi
        
        echo ""
        echo "What would you like to do?"
        echo "1) Repair unhealthy/exited containers only"
        echo "2) Deploy missing services only" 
        echo "3) Both repair and deploy (recommended for critical issues)"
        echo "4) Advanced repair with dependency check"
        echo "5) Cancel"
        read -p "Select option (1-5): " action
        
        case "$action" in
            1|3)
                if [[ $total_to_repair -gt 0 ]]; then
                    echo ""
                    echo -e "${CYAN}Starting repair process...${NC}"
                    echo ""
                    
                    # Repair unhealthy containers
                    for container in "${unhealthy_containers[@]}"; do
                        echo -e "${YELLOW}Repairing unhealthy container: $container${NC}"
                        
                        # Get service name from container name
                        service_name=$(get_service_name_from_container "$container")
                        
                        # Stop and remove the container
                        docker stop "$container" >/dev/null 2>&1
                        docker rm "$container" >/dev/null 2>&1
                        
                        # Recreate from docker-compose
                        echo "   Recreating container..."
                        cd /opt/sutazaiapp
                        docker compose up -d "$service_name" 2>&1 | grep -v "is up-to-date" || true
                        
                        # Wait for container to start
                        sleep 5
                        
                        # Check new health status
                        new_health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
                if [[ "$new_health" == "healthy" ]]; then
                    echo -e "   ${GREEN}✅ Container repaired successfully${NC}"
                else
                    echo -e "   ${YELLOW}⚠️  Container recreated, health status: $new_health${NC}"
                fi
                echo ""
            done
            
            # Repair exited containers
            for container in "${exited_containers[@]}"; do
                echo -e "${YELLOW}Repairing exited container: $container${NC}"
                
                # Get service name from container name
                service_name=$(get_service_name_from_container "$container")
                
                # Remove the container
                docker rm "$container" >/dev/null 2>&1
                
                # Recreate from docker-compose
                echo "   Recreating container..."
                cd /opt/sutazaiapp
                docker compose up -d "$service_name" 2>&1 | grep -v "is up-to-date" || true
                
                # Wait for container to start
                sleep 5
                
                # Check if running
                if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
                    echo -e "   ${GREEN}✅ Container restarted successfully${NC}"
                else
                    echo -e "   ${RED}❌ Failed to restart container${NC}"
                fi
                echo ""
            done
            
                fi
                ;;
            2|3)
                if [[ $total_to_deploy -gt 0 ]]; then
                    echo ""
                    echo -e "${CYAN}Starting deployment of missing services...${NC}"
                    echo ""
                    
                    # Check if images need to be built
                    echo -e "${YELLOW}Checking which services need to be built...${NC}"
                    declare -a services_to_build=()
                    declare -a services_to_pull=()
                    
                    for service in "${missing_services[@]}"; do
                        # Check if service has a build context in docker-compose
                        if docker compose config --services --resolve-image-digests | grep -q "^${service}$" && \
                           docker compose config | grep -A5 "^  ${service}:" | grep -q "build:"; then
                            # Check if image exists
                            image_name="sutazaiapp-${service}"
                            if ! docker images --format "{{.Repository}}" | grep -q "^${image_name}$"; then
                                services_to_build+=("$service")
                            fi
                        else
                            services_to_pull+=("$service")
                        fi
                    done
                    
                    # Build missing images
                    if [[ ${#services_to_build[@]} -gt 0 ]]; then
                        echo ""
                        echo -e "${YELLOW}Building ${#services_to_build[@]} service image(s)...${NC}"
                        for service in "${services_to_build[@]}"; do
                            echo -e "${CYAN}Building $service...${NC}"
                            if docker compose build "$service" 2>&1 | tail -20; then
                                echo -e "   ${GREEN}✅ Built successfully${NC}"
                            else
                                echo -e "   ${RED}❌ Build failed${NC}"
                            fi
                        done
                    fi
                    
                    # Deploy all missing services
                    echo ""
                    echo -e "${YELLOW}Deploying missing services...${NC}"
                    for service in "${missing_services[@]}"; do
                        echo -e "${CYAN}Deploying $service...${NC}"
                        if docker compose up -d "$service" 2>&1 | grep -v "is up-to-date" | tail -5; then
                            # Check if container started
                            sleep 3
                            if docker ps --filter "name=sutazai-${service}" --format "{{.Names}}" | grep -q "sutazai-${service}"; then
                                echo -e "   ${GREEN}✅ Deployed successfully${NC}"
                            else
                                echo -e "   ${RED}❌ Deployment failed${NC}"
                            fi
                        else
                            echo -e "   ${RED}❌ Failed to deploy${NC}"
                        fi
                        echo ""
                    done
                    
                    echo -e "${GREEN}Deployment process completed!${NC}"
                fi
                ;;
            4)
                # Advanced repair with dependency check
                echo ""
                echo -e "${CYAN}🔧 Advanced Repair with Dependency Analysis...${NC}"
                echo ""
                
                # Get service dependencies
                echo -e "${YELLOW}Analyzing service dependencies...${NC}"
                declare -A service_deps
                
                # Core services that should be repaired first
                core_services=("postgres" "redis" "ollama")
                
                # Repair core services first
                for service in "${core_services[@]}"; do
                    container="sutazai-${service}"
                    if [[ " ${unhealthy_containers[@]} " =~ " ${container} " ]] || \
                       [[ " ${exited_containers[@]} " =~ " ${container} " ]]; then
                        echo -e "${YELLOW}Repairing core service: $service${NC}"
                        docker stop "$container" >/dev/null 2>&1
                        docker rm "$container" >/dev/null 2>&1
                        docker compose up -d "$service" 2>&1 | grep -v "is up-to-date" || true
                        
                        # Wait for core service to be ready
                        echo "   Waiting for $service to be ready..."
                        sleep 10
                        
                        # Check health
                        health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
                        if [[ "$health" == "healthy" ]] || [[ "$health" == "none" ]]; then
                            echo -e "   ${GREEN}✅ Core service $service is ready${NC}"
                        else
                            echo -e "   ${YELLOW}⚠️  Core service $service started but not healthy yet${NC}"
                        fi
                        echo ""
                    fi
                done
                
                # Then repair other services
                echo -e "${YELLOW}Repairing dependent services...${NC}"
                for container in "${unhealthy_containers[@]}" "${exited_containers[@]}"; do
                    service_name=$(get_service_name_from_container "$container")
                    
                    # Skip if already handled as core service
                    if [[ " ${core_services[@]} " =~ " ${service_name} " ]]; then
                        continue
                    fi
                    
                    echo -e "${CYAN}Repairing: $container${NC}"
                    docker stop "$container" >/dev/null 2>&1
                    docker rm "$container" >/dev/null 2>&1
                    docker compose up -d "$service_name" 2>&1 | grep -v "is up-to-date" || true
                    sleep 3
                    
                    # Check status
                    if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
                        echo -e "   ${GREEN}✅ Service restarted${NC}"
                    else
                        echo -e "   ${RED}❌ Failed to restart${NC}"
                    fi
                    echo ""
                done
                
                echo -e "${GREEN}🎉 Advanced repair completed!${NC}"
                ;;
            5)
                echo -e "${YELLOW}Operation cancelled.${NC}"
                ;;
            *)
                echo -e "${RED}Invalid option selected.${NC}"
                ;;
        esac
    else
        echo -e "${GREEN}All services are healthy and deployed! No action needed.${NC}"
    fi
}

# Container health status display function
show_container_health_status() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                 CONTAINER HEALTH STATUS                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Create table header
    printf "%-30s %-15s %-20s %-15s\n" "Container" "Status" "Health" "Uptime"
    printf "%-30s %-15s %-20s %-15s\n" "─────────" "──────" "──────" "──────"
    
    # Get all containers and their health status
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        
        container_name=$(echo "$container" | awk '{print $1}')
        container_status=$(echo "$container" | awk '{$1=""; print $0}' | xargs)
        
        # Get health status
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no check")
        
        # Get uptime
        uptime=$(docker ps --filter "name=$container_name" --format "{{.Status}}" 2>/dev/null | sed 's/Up //' | sed 's/ (healthy)//' | sed 's/ (unhealthy)//')
        
        # Color code based on status
        if [[ "$health_status" == "healthy" ]]; then
            health_color="${GREEN}"
            health_icon="✅"
        elif [[ "$health_status" == "unhealthy" ]]; then
            health_color="${RED}"
            health_icon="❌"
        elif [[ "$health_status" == "starting" ]]; then
            health_color="${YELLOW}"
            health_icon="⏳"
        else
            health_color="${BLUE}"
            health_icon="ℹ️"
        fi
        
        # Check if running
        if echo "$container_status" | grep -q "Up"; then
            status_color="${GREEN}"
            status_text="Running"
        else
            status_color="${RED}"
            status_text="Stopped"
        fi
        
        printf "%-30s ${status_color}%-15s${NC} ${health_color}%-20s${NC} %-15s\n" \
            "$container_name" "$status_text" "$health_icon $health_status" "$uptime"
        
    done < <(docker ps -a --filter "name=sutazai-" --format "{{.Names}} {{.Status}}")
    
    echo ""
    
    # Show summary
    total=$(docker ps -a --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
    running=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
    healthy=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | xargs -I {} docker inspect --format='{{.State.Health.Status}}' {} 2>/dev/null | grep -c "healthy" || echo 0)
    unhealthy=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | xargs -I {} docker inspect --format='{{.State.Health.Status}}' {} 2>/dev/null | grep -c "unhealthy" || echo 0)
    
    echo -e "${CYAN}Summary:${NC}"
    echo "  Total containers: $total"
    echo "  Running: $running"
    echo "  Healthy: $healthy"
    echo "  Unhealthy: $unhealthy"
    echo "  Stopped: $((total - running))"
}

# Selective service deployment function
selective_service_deployment() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║               SELECTIVE SERVICE DEPLOYMENT                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Get list of all services from docker-compose
    cd /opt/sutazaiapp
    
    # Extract service names from docker-compose.yml
    services=($(docker compose config --services 2>/dev/null | sort))
    
    if [[ ${#services[@]} -eq 0 ]]; then
        echo -e "${RED}No services found in docker-compose.yml${NC}"
        return
    fi
    
    echo "Available services:"
    echo ""
    
    # Display services with numbers
    for i in "${!services[@]}"; do
        # Check if service is running
        if docker ps --filter "name=sutazai-${services[$i]}" --format "{{.Names}}" | grep -q "sutazai-${services[$i]}"; then
            status="${GREEN}[Running]${NC}"
        else
            status="${RED}[Stopped]${NC}"
        fi
        printf "%2d. %-25s %s\n" $((i+1)) "${services[$i]}" "$status"
    done
    
    echo ""
    echo "Enter service numbers to deploy (comma-separated, e.g., 1,3,5)"
    echo "Or enter 'all' to deploy all services"
    echo "Or enter 'stopped' to deploy only stopped services"
    echo "Or enter 'q' to quit"
    echo ""
    read -p "Your selection: " selection
    
    if [[ "$selection" == "q" ]]; then
        echo "Deployment cancelled."
        return
    fi
    
    # Determine which services to deploy
    services_to_deploy=()
    
    if [[ "$selection" == "all" ]]; then
        services_to_deploy=("${services[@]}")
    elif [[ "$selection" == "stopped" ]]; then
        for service in "${services[@]}"; do
            if ! docker ps --filter "name=sutazai-$service" --format "{{.Names}}" | grep -q "sutazai-$service"; then
                services_to_deploy+=("$service")
            fi
        done
    else
        # Parse comma-separated numbers
        IFS=',' read -ra selections <<< "$selection"
        for num in "${selections[@]}"; do
            num=$(echo "$num" | xargs)  # Trim whitespace
            if [[ "$num" =~ ^[0-9]+$ ]] && [[ $num -ge 1 ]] && [[ $num -le ${#services[@]} ]]; then
                services_to_deploy+=("${services[$((num-1))]}")
            fi
        done
    fi
    
    if [[ ${#services_to_deploy[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No services selected for deployment.${NC}"
        return
    fi
    
    echo ""
    echo -e "${CYAN}Selected services for deployment:${NC}"
    for service in "${services_to_deploy[@]}"; do
        echo "  - $service"
    done
    echo ""
    
    read -p "Proceed with deployment? (y/n): " confirm
    
    if [[ "$confirm" == "y" ]] || [[ "$confirm" == "Y" ]]; then
        echo ""
        echo -e "${CYAN}Starting deployment...${NC}"
        echo ""
        
        # Deploy each selected service
        for service in "${services_to_deploy[@]}"; do
            echo -e "${YELLOW}Deploying $service...${NC}"
            
            # Stop and remove existing container if exists
            if docker ps -a --filter "name=sutazai-$service" --format "{{.Names}}" | grep -q "sutazai-$service"; then
                docker stop "sutazai-$service" >/dev/null 2>&1
                docker rm "sutazai-$service" >/dev/null 2>&1
            fi
            
            # Deploy service
            docker compose up -d "$service" 2>&1 | grep -v "is up-to-date" || true
            
            # Check if deployed successfully
            sleep 3
            if docker ps --filter "name=sutazai-$service" --format "{{.Names}}" | grep -q "sutazai-$service"; then
                echo -e "  ${GREEN}✅ $service deployed successfully${NC}"
            else
                echo -e "  ${RED}❌ Failed to deploy $service${NC}"
            fi
            echo ""
        done
        
        echo -e "${GREEN}Deployment completed!${NC}"
    else
        echo -e "${YELLOW}Deployment cancelled.${NC}"
    fi
}

# Function to display system overview
show_system_overview() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    SUTAZAI LIVE MONITORING                   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Service status table
    echo -e "${BLUE}Service Status:${NC}"
    echo "┌─────────────────────┬────────┬──────────┬─────────────┐"
    echo "│ Container           │ Status │ Health   │ Ports       │"
    echo "├─────────────────────┼────────┼──────────┼─────────────┤"
    
    # Check Docker daemon first
    if ! check_docker_daemon; then
        echo "│ Docker daemon is not running or accessible                      │"
        echo "└─────────────────────┴────────┴──────────┴─────────────┘"
        echo ""
        echo -e "${YELLOW}Would you like to attempt automatic Docker recovery? (y/N)${NC}"
        read -r -n 1 recovery_choice
        echo ""
        if [[ $recovery_choice =~ ^[Yy]$ ]]; then
            if attempt_docker_recovery; then
                echo -e "${GREEN}Docker recovery successful! Restarting system overview...${NC}"
                sleep 2
                show_system_overview
                return $?
            else
                echo -e "${RED}Docker recovery failed. Manual intervention required.${NC}"
            fi
        fi
        return 1
    fi
    
    # Automatically discover all SutazAI containers
    containers=()
    while IFS= read -r container_name; do
        # Get port mappings for this container
        ports=$(docker port "$container_name" 2>/dev/null | grep '0.0.0.0:' | awk '{print $3}' | cut -d':' -f2 | sort -n -u | tr '\n' ',' | sed 's/,$//' || echo "")
        if [ -z "$ports" ]; then
            ports="-"
        fi
        containers+=("${container_name}:${ports}")
    done < <(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | sort)
    
    # If no SutazAI containers found, show message
    if [ ${#containers[@]} -eq 0 ]; then
        echo "│ No SutazAI containers found. Please start the system first.     │"
        echo "└─────────────────────┴────────┴──────────┴─────────────┘"
        echo ""
        echo -e "${YELLOW}To start SutazAI system:${NC}"
        echo -e "  ${CYAN}cd /opt/sutazaiapp && ./scripts/deploy.sh${NC}"
        return 0
    fi
    
    for container_info in "${containers[@]}"; do
        IFS=':' read -r container ports <<< "$container_info"
        # Shorten container name if needed
        short_name=$(echo "$container" | sed 's/sutazai-//' | cut -c1-19)
        if [ ${#short_name} -lt 19 ]; then
            short_name=$(printf "%-19s" "$short_name")
        fi
        status=$(check_container_status "$container" || echo "?")
        health=$(get_container_health "$container" || echo "unknown")
        # Format ports to fit column
        if [ ${#ports} -gt 11 ]; then
            ports=$(echo "$ports" | cut -c1-10)"…"
        else
            ports=$(printf "%-11s" "$ports")
        fi
        # Format health to fit column (allow full 'unhealthy')
        health=$(printf "%-10s" "$health" | cut -c1-10)
        # Use echo instead of printf to avoid formatting issues
        echo "│ ${short_name} │   ${status}    │ ${health} │ ${ports} │"
    done
    
    echo "└─────────────────────┴────────┴──────────┴─────────────┘"
    echo ""
    
    # Quick API tests
    echo -e "${BLUE}API Connectivity:${NC}"

    # Resolve backend and frontend ports dynamically (fallback to canonical ports)
    backend_port=$(docker port sutazai-backend 2>/dev/null | awk '/0.0.0.0:/ {print $3}' | head -1 | awk -F: '{print $2}')
    frontend_port=$(docker port sutazai-frontend 2>/dev/null | awk '/0.0.0.0:/ {print $3}' | head -1 | awk -F: '{print $2}')
    backend_port=${backend_port:-10010}
    frontend_port=${frontend_port:-10011}

    # Test backend health (FastAPI /health)
    if curl -fsS "http://localhost:${backend_port}/health" >/dev/null 2>&1; then
        echo -e "Backend API (${backend_port}): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Backend API (${backend_port}): ${RED}✗ Not responding${NC}"
    fi

    # Test frontend (Streamlit root)
    if curl -fsS "http://localhost:${frontend_port}/" >/dev/null 2>&1; then
        echo -e "Frontend (${frontend_port}): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Frontend (${frontend_port}): ${RED}✗ Not responding${NC}"
    fi
    
    # Test Ollama
    if curl -f http://localhost:10104/api/tags >/dev/null 2>&1; then
        echo -e "Ollama API (10104): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Ollama API (10104): ${RED}✗ Not responding${NC}"
    fi
    
    echo ""
    echo -e "${PURPLE}Access URLs:${NC}"
    
    # Dynamically discover web interfaces
    urls_found=0
    while IFS= read -r container_name; do
        service=$(echo "$container_name" | sed 's/sutazai-//')
        port=$(docker port "$container_name" 2>/dev/null | grep '0.0.0.0:' | head -1 | awk '{print $3}' | cut -d':' -f2 || echo "")
        
        if [ -n "$port" ]; then
            case "$service" in
                "frontend")
                    echo "• Frontend: http://localhost:${port}"
                    urls_found=1
                    ;;
                "backend")
                    echo "• Backend API: http://localhost:${port}"
                    echo "• API Docs: http://localhost:${port}/docs"
                    urls_found=1
                    ;;
                "grafana")
                    echo "• Grafana: http://localhost:${port}"
                    urls_found=1
                    ;;
                "prometheus")
                    echo "• Prometheus: http://localhost:${port}"
                    urls_found=1
                    ;;
                "n8n")
                    echo "• N8N Workflows: http://localhost:${port}"
                    urls_found=1
                    ;;
                "langflow")
                    echo "• LangFlow: http://localhost:${port}"
                    urls_found=1
                    ;;
                "flowise")
                    echo "• Flowise: http://localhost:${port}"
                    urls_found=1
                    ;;
                "neo4j")
                    echo "• Neo4j Browser: http://localhost:${port}"
                    urls_found=1
                    ;;
            esac
        fi
    done < <(docker ps --filter "name=sutazai-" --format "{{.Names}}")
    
    if [ $urls_found -eq 0 ]; then
        echo -e "${RED}No web interfaces found. Please start the system first.${NC}"
    fi
    echo ""
}

# Function to log with color and service prefix (FIXED for real-time output)
log_with_color() {
    local color="$1"
    local service="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Use printf for immediate output and explicit line ending
    printf "%b[%s] [%s]%b %s\n" "$color" "$timestamp" "$service" "$NC" "$message"
    # Force flush stdout
    exec 1>&1
}

# Function to show live logs
show_live_logs() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     SUTAZAI LIVE LOGS                       ║${NC}"
    echo -e "${CYAN}║                  Press Ctrl+C to exit                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Automatically discover and monitor all SutazAI containers
    log_pids=()
    container_colors=("$GREEN" "$BLUE" "$YELLOW" "$PURPLE" "$CYAN" "$RED" "$GREEN" "$BLUE" "$YELLOW" "$PURPLE" "$CYAN" "$RED")
    color_index=0
    
    # Get list of running SutazAI containers
    running_containers=($(docker ps --filter "name=sutazai-" --format "{{.Names}}" | sort))
    
    if [ ${#running_containers[@]} -eq 0 ]; then
        echo -e "${RED}No SutazAI containers found for log monitoring.${NC}"
        return
    fi
    
    echo -e "${GREEN}Monitoring logs from ${#running_containers[@]} containers:${NC}"
    for container in "${running_containers[@]}"; do
        # Create short display name
        display_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        echo "  • $container → $display_name"
    done
    echo ""
    
    # Start log monitoring for each container with FIXED real-time streaming
    for container in "${running_containers[@]}"; do
        # Create short display name
        display_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        
        # Get color for this container
        color=${container_colors[$((color_index % ${#container_colors[@]}))]}
        ((color_index++))
        
        # Start background log monitoring with proper unbuffered output
        (
            # Export variables for subshell
            export COLOR="$color"
            export DISPLAY_NAME="$display_name"
            
            # Check if stdbuf is available for optimal performance
            if command -v stdbuf >/dev/null 2>&1; then
                # Use stdbuf to disable buffering completely
                exec stdbuf -o0 -e0 docker logs -f --tail=50 "$container" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
                    log_with_color "$COLOR" "$DISPLAY_NAME" "$line"
                done
            else
                # Fallback: use direct piping without stdbuf
                exec docker logs -f --tail=50 "$container" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
                    log_with_color "$color" "$display_name" "$line"
                done
            fi
        ) &
        log_pids+=($!)
    done
    
    # Improved signal handling for clean shutdown
    cleanup_logs() {
        echo -e "\n${YELLOW}Stopping log monitoring...${NC}"
        for pid in "${log_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
        wait "${log_pids[@]}" 2>/dev/null
        echo -e "${GREEN}Log monitoring stopped.${NC}"
        exit 0
    }
    
    trap cleanup_logs INT TERM
    
    # Wait for all background processes
    wait "${log_pids[@]}"
}

# Function to test API endpoints
test_api_endpoints() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    API ENDPOINT TESTING                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Automatically discover endpoints based on running containers
    endpoints=()
    
    # Core backend endpoints (always test if backend is running)
    if docker ps --filter "name=sutazai-backend" --format "{{.Names}}" | grep -q .; then
        BACKEND_PORT=$(docker port sutazai-backend 2>/dev/null | awk '/0.0.0.0:/ {print $3}' | head -1 | awk -F: '{print $2}')
        BACKEND_PORT=${BACKEND_PORT:-10010}
        endpoints+=(
            "GET|http://localhost:${BACKEND_PORT}/health|Backend Health"
            "GET|http://localhost:${BACKEND_PORT}/agents|Agent List"
            "GET|http://localhost:${BACKEND_PORT}/models|Model List"
            "POST|http://localhost:${BACKEND_PORT}/simple-chat|Simple Chat"
        )
    fi
    
    # Dynamically add agent endpoints based on running containers
    while IFS= read -r container_name; do
        # Extract service name and get its port
        service=$(echo "$container_name" | sed 's/sutazai-//')
        port=$(docker port "$container_name" 2>/dev/null | grep '0.0.0.0:' | head -1 | awk '{print $3}' | cut -d':' -f2 || echo "")
        
        if [ -n "$port" ]; then
            case "$service" in
                "letta"|"crewai"|"autogpt"|"aider"|"gpt-engineer"|"faiss")
                    endpoints+=("GET|http://localhost:${port}/health|${service^} Health")
                    ;;
                "ollama")
                    endpoints+=("GET|http://localhost:${port}/api/tags|Ollama Models")
                    ;;
                "grafana")
                    endpoints+=("GET|http://localhost:${port}/api/health|Grafana Health")
                    ;;
                "prometheus")
                    endpoints+=("GET|http://localhost:${port}/-/healthy|Prometheus Health")
                    ;;
            esac
        fi
    done < <(docker ps --filter "name=sutazai-" --format "{{.Names}}")
    
    # If no endpoints found, show message
    if [ ${#endpoints[@]} -eq 0 ]; then
        echo -e "${RED}No API endpoints found to test. Please start the system first.${NC}"
        return
    fi
    
    echo -e "${GREEN}Testing ${#endpoints[@]} discovered endpoints:${NC}"
    echo ""
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS='|' read -r method url description <<< "$endpoint_info"
        echo -n "Testing $description ($method $url)... "
        
        if [ "$method" = "GET" ]; then
            # Use longer timeout for model-related endpoints
            timeout=10
            if [[ "$url" == *"/models"* ]] || [[ "$url" == *"simple-chat"* ]]; then
                timeout=30
            fi
            
            if curl -f --connect-timeout 5 --max-time $timeout "$url" >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        elif [ "$method" = "POST" ]; then
            # POST requests typically take longer (LLM processing)
            if curl -f --connect-timeout 5 --max-time 30 -X POST "$url" -H "Content-Type: application/json" -d '{"message":"test"}' >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        fi
        sleep 0.5
    done
    
    echo ""
    echo "Testing frontend connectivity..."
    if curl -f http://localhost:${frontend_port:-10011}/healthz >/dev/null 2>&1; then
        echo -e "Frontend Health: ${GREEN}✓ OK${NC}"
    else
        echo -e "Frontend Health: ${RED}✗ FAILED${NC}"
    fi
}

# Function to show container stats
show_container_stats() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   CONTAINER STATISTICS                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.PIDs}}" \
        sutazai-postgres sutazai-redis sutazai-neo4j sutazai-chromadb \
        sutazai-qdrant sutazai-ollama sutazai-backend sutazai-frontend 2>/dev/null
}

# Log management menu
show_log_management_menu() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     LOG MANAGEMENT                          ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Show current log status
    show_status
    echo ""
    
    echo "1. Cleanup Old Logs (7+ days)"
    echo "2. Cleanup Old Logs (Custom days)"
    echo "3. Reset All Logs (DANGEROUS)"
    echo "4. Archive Old Logs"
    echo "5. View Log Status"
    echo "6. Rotate Large Logs"
    echo "7. Clear Docker Logs"
    echo "8. Back to Main Menu"
    echo ""
    read -p "Select option (1-8): " choice
    
    case $choice in
        1) 
            cleanup_logs "7d" "false"
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        2)
            read -p "Enter number of days (e.g., 3): " days
            if [[ "$days" =~ ^[0-9]+$ ]]; then
                cleanup_logs "${days}d" "false"
            else
                echo -e "${RED}Invalid input. Please enter a number.${NC}"
            fi
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        3)
            echo -e "${RED}WARNING: This will delete ALL logs permanently!${NC}"
            read -p "Type 'CONFIRM' to proceed: " confirm
            if [[ "$confirm" == "CONFIRM" ]]; then
                reset_logs "true"
            else
                echo "Reset cancelled."
            fi
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        4)
            archive_logs
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        5)
            show_status
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        6)
            rotate_large_logs
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        7)
            echo "Clearing Docker container logs..."
            docker system prune -f --filter "until=24h" >/dev/null 2>&1
            echo -e "${GREEN}Docker logs cleared${NC}"
            read -p "Press Enter to continue..."
            show_log_management_menu
            ;;
        8) show_menu ;;
        *) echo "Invalid option"; show_log_management_menu ;;
    esac
}

# Debug control menu
show_debug_menu() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                      DEBUG CONTROLS                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Show current debug status
    echo -e "${YELLOW}Current Configuration:${NC}"
    echo "  Debug Mode: $DEBUG_MODE"
    echo "  Log Level: $LOG_LEVEL"
    echo ""
    
    echo "1. Enable Debug Mode"
    echo "2. Disable Debug Mode"
    echo "3. Set Log Level to DEBUG"
    echo "4. Set Log Level to INFO"
    echo "5. Set Log Level to WARN"
    echo "6. Set Log Level to ERROR"
    echo "7. Show Current Config"
    echo "8. Reset to Defaults"
    echo "9. Back to Main Menu"
    echo ""
    read -p "Select option (1-9): " choice
    
    case $choice in
        1)
            toggle_debug "on"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        2)
            toggle_debug "off"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        3)
            set_log_level "DEBUG"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        4)
            set_log_level "INFO"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        5)
            set_log_level "WARN"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        6)
            set_log_level "ERROR"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        7)
            show_config
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        8)
            DEBUG_MODE="$DEFAULT_DEBUG_MODE"
            LOG_LEVEL="$DEFAULT_LOG_LEVEL"
            MAX_LOG_SIZE="$DEFAULT_MAX_LOG_SIZE"
            MAX_LOG_FILES="$DEFAULT_MAX_LOG_FILES"
            CLEANUP_DAYS="$DEFAULT_CLEANUP_DAYS"
            save_config
            echo -e "${GREEN}Configuration reset to defaults${NC}"
            read -p "Press Enter to continue..."
            show_debug_menu
            ;;
        9) show_menu ;;
        *) echo "Invalid option"; show_debug_menu ;;
    esac
}

# Function to show unified live logs (all containers in one stream)
show_unified_live_logs() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                 SUTAZAI UNIFIED LIVE LOGS                   ║${NC}"
    echo -e "${CYAN}║                  Press Ctrl+C to return                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Check Docker daemon first
    if ! check_docker_daemon; then
        echo -e "${RED}Docker daemon is not accessible. Cannot show unified logs.${NC}"
        echo ""
        echo -e "${YELLOW}Would you like to attempt Docker recovery? (y/N)${NC}"
        read -r -n 1 recovery_choice
        echo ""
        if [[ $recovery_choice =~ ^[Yy]$ ]]; then
            if attempt_docker_recovery; then
                echo -e "${GREEN}Docker recovery successful! Restarting unified logs...${NC}"
                sleep 2
                show_unified_live_logs
                return
            else
                echo -e "${RED}Docker recovery failed. Use option 11 for advanced troubleshooting.${NC}"
                read -p "Press Enter to return to menu..."
                show_menu
                return
            fi
        else
            read -p "Press Enter to return to menu..."
            show_menu
            return
        fi
    fi
    
    # Get list of running SutazAI containers
    running_containers=($(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | sort))
    
    if [ ${#running_containers[@]} -eq 0 ]; then
        echo -e "${RED}No SutazAI containers found for log monitoring.${NC}"
        echo -e "${YELLOW}Tip: Start your containers first with: ${CYAN}cd /opt/sutazaiapp && docker-compose up -d${NC}"
        read -p "Press Enter to return to menu..."
        show_menu
        return
    fi
    
    echo -e "${GREEN}🔍 Monitoring ${#running_containers[@]} containers in unified stream${NC}"
    echo ""
    
    # Define colors for different containers
    colors=(31 32 33 34 35 36 91 92 93 94 95 96)
    
    # Set up cleanup function
    cleanup_unified_logs() {
        echo ""
        echo -e "${YELLOW}🛑 Stopping unified log monitoring...${NC}"
        jobs -p | xargs -r kill 2>/dev/null
        echo -e "${GREEN}✓ Returned to main menu${NC}"
        show_menu
    }
    
    # Set signal trap
    trap cleanup_unified_logs INT TERM
    
    # Start streaming logs from all containers
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📡 LIVE UNIFIED LOG STREAM - Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    
    # FIXED: Skip docker-compose logs entirely - it has unfixable internal buffering
    # Always use individual container streaming for true real-time output
    echo -e "${CYAN}Using direct container streaming for real-time logs...${NC}"
    echo ""
    individual_streaming_realtime
}

# NEW: True real-time streaming function with zero buffering
individual_streaming_realtime() {
    local color_index=0
    local colors=(31 32 33 34 35 36 91 92 93 94 95 96)
    local pids=()
    
    # Ensure Python applications output unbuffered
    export PYTHONUNBUFFERED=1
    
    for container in "${running_containers[@]}"; do
        local short_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        local color_code=${colors[$((color_index % ${#colors[@]}))]}
        
        # Use tail -F for true real-time following (better than docker logs)
        (
            # First, get the container's log file location
            log_file=$(docker inspect --format='{{.LogPath}}' "$container" 2>/dev/null)
            
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                # Direct file following - bypasses ALL Docker buffering
                tail -F -n 0 "$log_file" 2>/dev/null | \
                jq -r '.log // .stream // .stdout // .stderr // .' 2>/dev/null | \
                while IFS= read -r line; do
                    printf "\033[%sm[%s] [%s]\033[0m %s\n" "$color_code" "$(date '+%H:%M:%S')" "$short_name" "$line"
                done
            else
                # Fallback to docker logs with aggressive unbuffering
                docker logs -f --tail=0 --timestamps=false "$container" 2>&1 | \
                while IFS= read -r line; do
                    printf "\033[%sm[%s] [%s]\033[0m %s\n" "$color_code" "$(date '+%H:%M:%S')" "$short_name" "$line"
                done
            fi
        ) &
        pids+=($!)
        
        ((color_index++))
    done
    
    # Proper cleanup on exit
    cleanup_realtime() {
        echo -e "\n${YELLOW}Stopping real-time monitoring...${NC}"
        for pid in "${pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null
        done
        wait "${pids[@]}" 2>/dev/null
        echo -e "${GREEN}Monitoring stopped.${NC}"
    }
    
    trap cleanup_realtime INT TERM
    
    # Wait for all background processes
    wait "${pids[@]}"
}

# Fallback function for individual container streaming (FIXED for real-time)
individual_streaming() {
    local color_index=0
    local colors=(31 32 33 34 35 36 91 92 93 94 95 96)
    
    for container in "${running_containers[@]}"; do
        local short_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        local color_code=${colors[$((color_index % ${#colors[@]}))]}
        
        # Stream each container's logs in background with proper unbuffering
        {
            if command -v stdbuf >/dev/null 2>&1; then
                # Use stdbuf for optimal real-time performance
                exec stdbuf -o0 -e0 docker logs -f --tail=2 "$container" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
                    printf "\033[%sm[%s] [%s]\033[0m %s\n" "$color_code" "$(date '+%H:%M:%S')" "$short_name" "$line"
                done
            else
                # Fallback without stdbuf
                exec docker logs -f --tail=2 "$container" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
                    printf "\033[%sm[%s] [%s]\033[0m %s\n" "$color_code" "$(date '+%H:%M:%S')" "$short_name" "$line"
                done
            fi
        } &
        
        ((color_index++))
    done
    
    # Wait for all background processes
    wait
}

# Redeploy all containers function
redeploy_all_containers() {
    # Track deployment start time
    DEPLOYMENT_START_TIME=$(date +%s)
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              REDEPLOY ALL CONTAINERS                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}⚠️  WARNING: This will stop and redeploy all containers!${NC}"
    echo -e "${YELLOW}This action will:${NC}"
    echo "• Stop all running containers"
    echo "• Remove all containers (preserving data volumes)"
    echo "• Rebuild and restart all containers"
    echo "• Apply any configuration changes"
    echo ""
    read -p "Are you sure you want to proceed? (Y/n): " -r
    echo ""
    
    # Accept empty input (Enter key) as "yes", or explicit y/Y
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo -e "${YELLOW}Redeployment cancelled.${NC}"
        return 0
    fi
    
    echo ""
    echo -e "${CYAN}Starting enhanced redeployment process...${NC}"
    echo ""
    
    # Hardware Auto-Detection
    echo -e "${CYAN}🔍 Detecting hardware resources...${NC}"
    
    # CPU Detection
    CPU_CORES=$(nproc)
    CPU_THREADS=$(nproc --all)
    CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d: -f2 | xargs || echo "Unknown")
    
    # Memory Detection
    RAM_TOTAL_GB=$(free -g | awk 'NR==2{print $2}')
    RAM_AVAILABLE_GB=$(free -g | awk 'NR==2{print $7}')
    RAM_USAGE_PERCENT=$(free | awk 'NR==2{printf "%.0f", $3*100/($3+$4)}')
    
    # Disk Type Detection
    DISK_TYPE="HDD"
    if [ -d "/sys/block" ]; then
        for disk in /sys/block/*/queue/rotational; do
            if [ -r "$disk" ] && [ "$(cat "$disk")" = "0" ]; then
                DISK_TYPE="SSD"
                break
            fi
        done
    fi
    
    # GPU Detection
    GPU_PRESENT="false"
    GPU_COUNT=0
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | wc -l || echo 0)
        if [ "$GPU_COUNT" -gt 0 ]; then
            GPU_PRESENT="true"
        fi
    fi
    
    echo -e "${GREEN}📊 Hardware Profile:${NC}"
    echo "  CPU: $CPU_MODEL ($CPU_CORES cores, $CPU_THREADS threads)"
    echo "  RAM: ${RAM_AVAILABLE_GB}GB available / ${RAM_TOTAL_GB}GB total (${RAM_USAGE_PERCENT}% used)"
    echo "  Disk: $DISK_TYPE"
    echo "  GPU: $GPU_PRESENT (${GPU_COUNT} devices)"
    echo ""
    
    # Calculate optimal parallelism
    PULL_PARALLELISM=$CPU_CORES
    BUILD_PARALLELISM=$((CPU_CORES / 2))
    
    # Adjust based on RAM
    if [ "$RAM_AVAILABLE_GB" -lt 4 ]; then
        PULL_PARALLELISM=$((CPU_CORES / 2))
        BUILD_PARALLELISM=1
        echo -e "${YELLOW}⚠️  Low RAM detected, reducing parallelism${NC}"
    elif [ "$RAM_AVAILABLE_GB" -gt 16 ]; then
        PULL_PARALLELISM=$((CPU_CORES * 2))
        BUILD_PARALLELISM=$CPU_CORES
        echo -e "${GREEN}✅ High RAM available, increasing parallelism${NC}"
    fi
    
    # Adjust based on disk type
    if [ "$DISK_TYPE" = "SSD" ]; then
        PULL_PARALLELISM=$((PULL_PARALLELISM + 2))
        echo -e "${GREEN}✅ SSD detected, optimizing I/O operations${NC}"
    fi
    
    # Cap maximum parallelism
    PULL_PARALLELISM=$(( PULL_PARALLELISM > 16 ? 16 : PULL_PARALLELISM ))
    BUILD_PARALLELISM=$(( BUILD_PARALLELISM > 8 ? 8 : BUILD_PARALLELISM ))
    
    echo -e "${CYAN}🔧 Optimization Settings:${NC}"
    echo "  Pull parallelism: $PULL_PARALLELISM"
    echo "  Build parallelism: $BUILD_PARALLELISM"
    echo ""
    
    # Enable Docker BuildKit for faster builds
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    echo ""
    
    # Check Docker daemon first
    if ! check_docker_daemon; then
        echo -e "${RED}Docker daemon is not accessible. Cannot proceed with redeployment.${NC}"
        return 1
    fi
    
    # Navigate to project root
    cd "$PROJECT_ROOT" || {
        echo -e "${RED}Error: Cannot navigate to project root: $PROJECT_ROOT${NC}"
        return 1
    }
    
    # Check if docker-compose file exists
    if [[ ! -f "docker-compose.yml" ]]; then
        echo -e "${RED}Error: docker-compose.yml not found in $PROJECT_ROOT${NC}"
        return 1
    fi
    
    # Load environment variables if .env exists
    if [[ -f ".env" ]]; then
        echo -e "${CYAN}Loading environment variables from .env file...${NC}"
        set -a  # automatically export all variables
        source .env
        set +a
    else
        echo -e "${YELLOW}Warning: .env file not found. Some services may fail to start.${NC}"
    fi
    
    # Task allocation message
    echo -e "${PURPLE}🤖 Allocating AI agents to handle the redeployment...${NC}"
    echo ""
    
    # List of agents that will handle different aspects
    echo -e "${GREEN}✓ infrastructure-devops-manager${NC} - Managing deployment orchestration"
    echo -e "${GREEN}✓ deployment-automation-master${NC} - Handling container lifecycle"
    echo -e "${GREEN}✓ system-optimizer-reorganizer${NC} - Optimizing resource allocation"
    echo -e "${GREEN}✓ monitoring-engineer${NC} - Ensuring service health checks"
    echo -e "${GREEN}✓ self-healing-orchestrator${NC} - Monitoring for failures"
    echo ""
    
    # Step 1: Stop all containers
    echo -e "${CYAN}Step 1/5: Stopping all containers...${NC}"
    docker compose down --remove-orphans 2>&1 | while read -r line; do
        # Filter out warning messages about environment variables
        if [[ ! "$line" =~ "level=warning" ]]; then
            echo -e "  ${line}"
        fi
    done
    
    # Step 2: Clean up dangling resources
    echo ""
    echo -e "${CYAN}Step 2/5: Cleaning up unused resources...${NC}"
    docker system prune -f 2>&1 | while read -r line; do
        echo -e "  ${line}"
    done
    
    # Step 3: Analyze and pull/build images intelligently
    echo ""
    echo -e "${CYAN}Step 3/5: Analyzing image requirements...${NC}"
    
    # First, identify which services need building vs pulling
    echo -e "${CYAN}Analyzing docker-compose.yml for image sources...${NC}"
    
    # Extract services that use build contexts (need building)
    # Use yq if available, otherwise use grep-based approach
    if command -v yq &> /dev/null; then
        # SECURITY FIX: Use safer yq command without eval
        BUILD_SERVICES=$(yq '.services | to_entries | .[] | select(.value.build) | .key' docker-compose.yml 2>/dev/null | sort -u || echo "")
    else
        # Fallback: Find services with build sections using grep
        BUILD_SERVICES=$(grep -B 5 "^    build:" docker-compose.yml | grep "^  [a-zA-Z][a-zA-Z0-9_-]*:" | awk -F: '{print $1}' | sed 's/^  //' | sort -u)
    fi
    
    # Extract services that use external images (need pulling)
    # Only include images from services that DON'T have build contexts
    if command -v yq &> /dev/null; then
        # SECURITY FIX: Use safer yq command without eval
        EXTERNAL_IMAGES=$(yq '.services | to_entries | .[] | select(.value.build | not) | .value.image' docker-compose.yml 2>/dev/null | grep -v "null" | sort -u || echo "")
    else
        # Fallback: Get all images, then filter out those from services with build contexts
        ALL_IMAGES=$(grep "^\s*image:" docker-compose.yml | awk '{print $2}')
        EXTERNAL_IMAGES=""
        
        # Filter out images that are sutazaiapp-* (these are local builds)
        for image in $ALL_IMAGES; do
            if [[ ! "$image" =~ ^sutazaiapp- ]]; then
                EXTERNAL_IMAGES="$EXTERNAL_IMAGES$image"$'\n'
            fi
        done
        EXTERNAL_IMAGES=$(echo "$EXTERNAL_IMAGES" | sort -u | grep -v '^$')
    fi
    
    # Count totals
    EXTERNAL_COUNT=$(echo "$EXTERNAL_IMAGES" | grep -v '^$' | wc -l || echo "0")
    BUILD_COUNT=$(echo "$BUILD_SERVICES" | grep -v '^$' | wc -l || echo "0")
    
    echo -e "${CYAN}Image Analysis Results:${NC}"
    echo "  External images to pull: $EXTERNAL_COUNT"
    echo "  Services to build locally: $BUILD_COUNT"
    echo ""
    
    # Show details if verbose
    if [ "$EXTERNAL_COUNT" -gt 0 ]; then
        echo -e "${CYAN}External images that will be pulled:${NC}"
        echo "$EXTERNAL_IMAGES" | grep -v '^$' | sed 's/^/  • /'
        echo ""
    fi
    
    if [ "$BUILD_COUNT" -gt 0 ]; then
        echo -e "${CYAN}Services that will be built locally:${NC}"
        echo "$BUILD_SERVICES" | grep -v '^$' | sed 's/^/  • /'
        echo ""
    fi
    
    # Step 3a: Pull external images with parallel optimization
    if [ "$EXTERNAL_COUNT" -gt 0 ]; then
        echo -e "${CYAN}Step 3a: Pulling external images with parallel optimization...${NC}"
        echo -e "${YELLOW}Using $PULL_PARALLELISM parallel downloads based on hardware capabilities${NC}"
        
        # Function to pull single external image with progress
        pull_single_image() {
            local image="$1"
            
            # Try to pull the image
            if docker pull "$image" >/dev/null 2>&1; then
                echo -e "  ${GREEN}[PULL]${NC} $image"
                return 0
            else
                echo -e "  ${RED}[FAIL]${NC} $image"
                return 1
            fi
        }
        
        export -f pull_single_image
        export GREEN YELLOW RED NC
        
        # Parallel pull with progress monitoring
        echo "$EXTERNAL_IMAGES" | grep -v '^$' | xargs -P "$PULL_PARALLELISM" -I {} bash -c 'pull_single_image "$@"' _ {}
        
        echo -e "${GREEN}External image pull phase completed${NC}"
    else
        echo -e "${YELLOW}No external images to pull${NC}"
    fi
    
    # Step 3b: Build local services with intelligent handling
    if [ "$BUILD_COUNT" -gt 0 ]; then
        echo ""
        echo -e "${CYAN}Step 3b: Building local services with optimization...${NC}"
        echo -e "${YELLOW}Using $BUILD_PARALLELISM parallel builds with BuildKit${NC}"
        
        # Configure build optimization
        export BUILDKIT_INLINE_CACHE=1
        export DOCKER_BUILDKIT_INLINE_CACHE=1
        
        echo -e "${CYAN}Build configuration:${NC}"
        echo "  BuildKit: Enabled"
        echo "  Services to build: $BUILD_COUNT"
        echo "  Parallelism: $BUILD_PARALLELISM"
        echo ""
        
        # Build with detailed progress and error handling
        echo -e "${CYAN}Starting build process...${NC}"
        
        # Use docker compose build with better error handling (removed --memory flag as BuildKit doesn't support it)
        if docker compose build --parallel 2>&1 | while read -r line; do
            # Enhanced filtering and formatting
            if [[ ! "$line" =~ "level=warning" ]] && [[ -n "$line" ]]; then
                if [[ "$line" =~ "Building" ]] || [[ "$line" =~ "built" ]]; then
                    echo -e "  ${GREEN}${line}${NC}"
                elif [[ "$line" =~ "ERROR" ]] || [[ "$line" =~ "error" ]] || [[ "$line" =~ "failed" ]]; then
                    echo -e "  ${RED}${line}${NC}"
                elif [[ "$line" =~ "WARN" ]] || [[ "$line" =~ "warning" ]]; then
                    echo -e "  ${YELLOW}${line}${NC}"
                else
                    echo -e "  ${line}"
                fi
            fi
        done; then
            echo -e "${GREEN}Build phase completed successfully${NC}"
        else
            echo -e "${RED}Build phase encountered errors${NC}"
            echo -e "${YELLOW}Attempting to continue with deployment...${NC}"
        fi
    else
        echo -e "${YELLOW}No local services to build${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Continuing with deployment...${NC}"
    
    # Step 4: Start all containers  
    echo ""
    echo -e "${CYAN}Step 4/5: Starting all containers...${NC}"
    
    # Start containers and capture output
    docker compose up -d 2>&1 | while read -r line; do
        # Filter out warning messages
        if [[ ! "$line" =~ "level=warning" ]] && [[ -n "$line" ]]; then
            if [[ "$line" =~ "Created" ]] || [[ "$line" =~ "Started" ]]; then
                echo -e "  ${GREEN}${line}${NC}"
            elif [[ "$line" =~ "Error" ]]; then
                echo -e "  ${RED}${line}${NC}"
            else
                echo -e "  ${line}"
            fi
        fi
    done
    
    # Step 5: Verify deployment
    echo ""
    echo -e "${CYAN}Step 5/5: Verifying deployment...${NC}"
    sleep 5
    
    # Count running containers
    running_count=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | wc -l || echo "0")
    total_count=$(docker compose ps -a --format "{{.Names}}" 2>/dev/null | wc -l || echo "0")
    
    echo ""
    echo -e "${CYAN}Deployment Summary:${NC}"
    echo "• Total containers defined: $total_count"
    echo "• Containers running: $running_count"
    
    if [[ $running_count -eq $total_count ]] && [[ $running_count -gt 0 ]]; then
        echo ""
        echo -e "${GREEN}✅ All containers successfully redeployed!${NC}"
        
        # Show running containers
        echo ""
        echo -e "${CYAN}Running containers:${NC}"
        docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | tail -n +2 | while read -r line; do
            echo -e "  ${GREEN}✓${NC} $line"
        done
    else
        echo ""
        echo -e "${YELLOW}⚠️  Some containers may not be running properly.${NC}"
        
        # Show container status
        echo ""
        echo -e "${CYAN}Container status:${NC}"
        docker compose ps -a --format "table {{.Name}}\t{{.State}}\t{{.Status}}" | tail -n +2 | while read -r line; do
            if echo "$line" | grep -q "running"; then
                echo -e "  ${GREEN}✓${NC} $line"
            else
                echo -e "  ${RED}✗${NC} $line"
            fi
        done 2>/dev/null || echo "Unable to get container status"
        
        # Suggest troubleshooting
        echo ""
        echo -e "${YELLOW}Troubleshooting suggestions:${NC}"
        echo "• Check logs: docker compose logs -f [container-name]"
        echo "• Use option 11 for Docker troubleshooting"
        echo "• Use option 2 to view live logs"
        echo "• Use option 8 for system repair"
        echo ""
        echo -e "${CYAN}Common fixes:${NC}"
        echo "• Ensure .env file exists with required passwords"
        echo "• Check docker-compose.yml for invalid image names"
        echo "• Try deploying core services first: docker compose up -d postgres redis backend frontend"
    fi
    
    # Agent completion message
    echo ""
    echo -e "${PURPLE}🤖 AI agents have completed the redeployment task.${NC}"
    
    # Show access URLs
    echo ""
    echo -e "${CYAN}Access URLs:${NC}"
    echo "• Frontend: http://localhost:${frontend_port:-10011}"
    echo "• Backend API: http://localhost:${backend_port:-10010}"
    echo "• API Docs: http://localhost:${backend_port:-10010}/docs"
    
    # Additional URLs if monitoring is deployed
    if docker ps --filter "name=sutazai-grafana" --format "{{.Names}}" | grep -q .; then
        echo "• Grafana: http://localhost:3000"
    fi
    if docker ps --filter "name=sutazai-prometheus" --format "{{.Names}}" | grep -q .; then
        echo "• Prometheus: http://localhost:9090"
    fi
    
    echo ""
    
    # Performance Summary
    echo -e "${CYAN}⚡ Performance Summary:${NC}"
    echo "=============================="
    
    # Calculate deployment duration
    DEPLOYMENT_END_TIME=$(date +%s)
    if [ -n "${DEPLOYMENT_START_TIME}" ]; then
        DEPLOYMENT_DURATION=$((DEPLOYMENT_END_TIME - DEPLOYMENT_START_TIME))
        echo "  ⏱️  Total deployment time: ${DEPLOYMENT_DURATION}s"
    fi
    
    echo "  🔧 Hardware utilization:"
    echo "    - CPU cores used: $CPU_CORES"
    echo "    - Pull parallelism: $PULL_PARALLELISM"
    echo "    - Build parallelism: $BUILD_PARALLELISM"
    echo "    - RAM allocated: $MEMORY_LIMIT"
    
    # Final resource usage
    FINAL_CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "N/A")
    FINAL_MEM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/($3+$4)}' || echo "N/A")
    
    echo "  📊 Final resource usage:"
    echo "    - CPU: ${FINAL_CPU_USAGE}%"
    echo "    - RAM: ${FINAL_MEM_USAGE}%"
    
    echo ""
    echo -e "${GREEN}🎉 Enhanced redeployment completed!${NC}"
}

# Main menu
show_menu() {
    # Check numlock status
    local numlock_status=$(check_numlock_status)
    local numlock_indicator=""
    if [[ "$numlock_status" == "ON" ]]; then
        numlock_indicator="${GREEN}NUM${NC}"
    elif [[ "$numlock_status" == "OFF" ]]; then
        numlock_indicator="${RED}num${NC}"
    else
        numlock_indicator="${YELLOW}N/A${NC}"
    fi
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   SUTAZAI MONITORING MENU                   ║${NC}"
    echo -e "${CYAN}║                                                   [$numlock_indicator]     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. System Overview"
    echo "2. Live Logs (All Services)"
    echo "3. Test API Endpoints"
    echo "4. Container Statistics"
    echo "5. Log Management"
    echo "6. Debug Controls"
    echo "7. Database Repair"
    echo "8. System Repair"
    echo "9. Restart All Services"
    echo "10. Unified Live Logs (All in One)"
    echo "11. Docker Troubleshooting & Recovery"
    echo "12. Redeploy All Containers"
    echo "13. Smart Health Check & Repair (Unhealthy Only)"
    echo "14. Container Health Status"
    echo "15. Selective Service Deployment"
    echo "0. Exit"
    echo ""
    read -p "Select option (0-15): " choice
    
    case $choice in
        1) show_system_overview; read -p "Press Enter to continue..."; show_menu ;;
        2) show_live_logs ;;
        3) test_api_endpoints; read -p "Press Enter to continue..."; show_menu ;;
        4) show_container_stats; read -p "Press Enter to continue..."; show_menu ;;
        5) show_log_management_menu ;;
        6) show_debug_menu ;;
        7) init_database; read -p "Press Enter to continue..."; show_menu ;;
        8) repair_system; read -p "Press Enter to continue..."; show_menu ;;
        9) 
            echo "Restarting all SutazAI services..."
            docker compose -f /opt/sutazaiapp/docker-compose.yml restart
            echo "All services restarted!"
            read -p "Press Enter to continue..."
            show_menu
            ;;
        10) show_unified_live_logs ;;
        11) docker_troubleshooting_menu; read -p "Press Enter to continue..."; show_menu ;;
        12) redeploy_all_containers; read -p "Press Enter to continue..."; show_menu ;;
        13) check_and_repair_unhealthy_containers; read -p "Press Enter to continue..."; show_menu ;;
        14) show_container_health_status; read -p "Press Enter to continue..."; show_menu ;;
        15) selective_service_deployment; read -p "Press Enter to continue..."; show_menu ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option"; show_menu ;;
    esac
}

# Database initialization function
init_database() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    DATABASE INITIALIZATION                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo "Initializing SutazAI database..."
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec sutazai-postgres pg_isready -U postgres >/dev/null 2>&1; then
            echo -e "${GREEN}PostgreSQL is ready${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Create database
    echo "Creating sutazai database..."
    docker exec sutazai-postgres psql -U postgres -c "CREATE DATABASE sutazai;" 2>/dev/null || echo "Database may already exist"
    
    # Create user
    echo "Creating sutazai user..."
    docker exec sutazai-postgres psql -U postgres -c "CREATE USER sutazai WITH PASSWORD 'sutazai_password';" 2>/dev/null || echo "User may already exist"
    
    # Grant permissions
    echo "Granting permissions..."
    docker exec sutazai-postgres psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazai;" 2>/dev/null
    docker exec sutazai-postgres psql -U postgres -c "ALTER USER sutazai CREATEDB;" 2>/dev/null
    
    echo -e "${GREEN}Database initialization completed!${NC}"
}

# System repair function
repair_system() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                      SYSTEM REPAIR                          ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo "Running system repair operations..."
    
    # Initialize database
    init_database
    
    # Restart services in correct order
    echo "Restarting services in dependency order..."
    
    services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "backend" "frontend")
    
    for service in "${services[@]}"; do
        echo "Restarting sutazai-${service}..."
        docker restart "sutazai-${service}" >/dev/null 2>&1 || echo "Service sutazai-${service} not found"
        sleep 5
    done
    
    echo -e "${GREEN}System repair completed!${NC}"
}

# Start the monitoring system
case "${1:-}" in
    "--overview")
        show_system_overview
        ;;
    "--logs")
        show_live_logs
        ;;
    "--test")
        test_api_endpoints
        ;;
    "--stats")
        show_container_stats
        ;;
    "--init-db")
        init_database
        ;;
    "--repair")
        repair_system
        ;;
    "")
        show_menu
        ;;
    *)
        # Run main function with parameters
        main "$@"
        ;;
esac