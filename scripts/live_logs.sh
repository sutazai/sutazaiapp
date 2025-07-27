#!/bin/bash
# SutazAI Live Logs Management System
# Provides real-time log monitoring with cleanup and debugging controls

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/.logs_config"

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
DEBUG_MODE=${DEBUG_MODE}
LOG_LEVEL=${LOG_LEVEL}
MAX_LOG_SIZE=${MAX_LOG_SIZE}
MAX_LOG_FILES=${MAX_LOG_FILES}
CLEANUP_DAYS=${CLEANUP_DAYS}
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
            docker-compose -f "${PROJECT_ROOT}/${compose_file}" restart > /dev/null 2>&1 || true
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
            docker-compose restart > /dev/null 2>&1 || true
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
    "storage-driver": "overlay2"
}
EOF
            echo -e "${GREEN}✓ Minimal Docker configuration created${NC}"
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
        # Format health to fit column
        health=$(printf "%-8s" "$health" | cut -c1-8)
        # Use echo instead of printf to avoid formatting issues
        echo "│ ${short_name} │   ${status}    │ ${health} │ ${ports} │"
    done
    
    echo "└─────────────────────┴────────┴──────────┴─────────────┘"
    echo ""
    
    # Quick API tests
    echo -e "${BLUE}API Connectivity:${NC}"
    
    # Test backend health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "Backend API (8000): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Backend API (8000): ${RED}✗ Not responding${NC}"
    fi
    
    # Test frontend
    if curl -f http://localhost:8501/healthz >/dev/null 2>&1; then
        echo -e "Frontend (8501): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Frontend (8501): ${RED}✗ Not responding${NC}"
    fi
    
    # Test Ollama
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "Ollama API (11434): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Ollama API (11434): ${RED}✗ Not responding${NC}"
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
                "frontend-agi")
                    echo "• Frontend: http://localhost:${port}"
                    urls_found=1
                    ;;
                "backend-agi")
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

# Function to log with color and service prefix
log_with_color() {
    local color="$1"
    local service="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${color}[${timestamp}] [${service}]${NC} ${message}"
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
    
    # Start log monitoring for each container
    for container in "${running_containers[@]}"; do
        # Create short display name
        display_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        
        # Get color for this container
        color=${container_colors[$((color_index % ${#container_colors[@]}))]}
        ((color_index++))
        
        # Start background log monitoring
        (
            docker logs -f "$container" 2>&1 | while IFS= read -r line; do
                log_with_color "$color" "$display_name" "$line"
            done
        ) &
        log_pids+=($!)
    done
    
    # Wait for interrupt and kill all background processes
    trap "kill ${log_pids[*]} 2>/dev/null; exit 0" INT TERM
    wait
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
        endpoints+=(
            "GET|http://localhost:8000/health|Backend Health"
            "GET|http://localhost:8000/agents|Agent List"
            "GET|http://localhost:8000/models|Model List"
            "POST|http://localhost:8000/simple-chat|Simple Chat"
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
    if curl -f http://localhost:8501/healthz >/dev/null 2>&1; then
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
        sutazai-qdrant sutazai-ollama sutazai-backend-agi sutazai-frontend-agi 2>/dev/null
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
    
    # Use docker compose logs with follow for real unified streaming
    if command -v docker-compose &> /dev/null && [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
        # Use docker-compose logs for true unified streaming
        cd /opt/sutazaiapp
        docker-compose logs -f --tail=5 2>/dev/null || {
            # Fallback to individual container streaming
            echo -e "${YELLOW}Docker compose not available, using individual streams...${NC}"
            individual_streaming
        }
    else
        individual_streaming
    fi
}

# Fallback function for individual container streaming
individual_streaming() {
    local color_index=0
    local colors=(31 32 33 34 35 36 91 92 93 94 95 96)
    
    for container in "${running_containers[@]}"; do
        local short_name=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
        local color_code=${colors[$((color_index % ${#colors[@]}))]}
        
        # Stream each container's logs in background with color and formatting
        {
            docker logs -f --tail=2 "$container" 2>&1 | while IFS= read -r line; do
                printf "\033[%sm[%s] [%s]\033[0m %s\n" "$color_code" "$(date '+%H:%M:%S')" "$short_name" "$line"
            done
        } &
        
        ((color_index++))
    done
    
    # Wait for all background processes
    wait
}

# Main menu
show_menu() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   SUTAZAI MONITORING MENU                   ║${NC}"
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
    echo "0. Exit"
    echo ""
    read -p "Select option (0-11): " choice
    
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
            docker-compose -f /opt/sutazaiapp/docker-compose-consolidated.yml restart
            echo "All services restarted!"
            read -p "Press Enter to continue..."
            show_menu
            ;;
        10) show_unified_live_logs ;;
        11) docker_troubleshooting_menu; read -p "Press Enter to continue..."; show_menu ;;
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
    
    services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "backend-agi" "frontend-agi")
    
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