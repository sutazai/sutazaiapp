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
    echo -e "${YELLOW}Commands:${NC}"
    echo "  live                    - Start live log monitoring"
    echo "  follow [service]        - Follow specific service logs"
    echo "  cleanup                 - Clean old logs and reset"
    echo "  reset                   - Reset all logs (DANGEROUS)"
    echo "  debug [on|off]          - Toggle debugging mode"
    echo "  level [DEBUG|INFO|WARN] - Set log level"
    echo "  config                  - Show current configuration"
    echo "  status                  - Show log status and disk usage"
    echo "  archive                 - Archive old logs"
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
    echo "  $0 cleanup --older-than 3d"
    echo "  $0 debug on"
    echo "  $0 reset --confirm"
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
    
    echo -e "${GREEN}Starting live log monitoring...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Get all SutazAI containers
    local containers=($(docker ps --format "{{.Names}}" | grep "sutazai-" | sort))
    
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

# Execute main function
main "$@"
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
log_with_color() {
    local color=$1
    local service=$2
    local message=$3
    echo -e "${color}[$(date +'%H:%M:%S')] [${service}]${NC} ${message}"
}

# Function to check container status
check_container_status() {
    local container=$1
    if docker ps --filter "name=${container}" --format "{{.Names}}" | grep -q "${container}"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Function to get container health
get_container_health() {
    local container=$1
    local health=$(docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null)
    if [ "$health" = "healthy" ]; then
        echo -e "${GREEN}healthy${NC}"
    elif [ "$health" = "unhealthy" ]; then
        echo -e "${RED}unhealthy${NC}"
    elif [ "$health" = "starting" ]; then
        echo -e "${YELLOW}starting${NC}"
    else
        echo -e "${BLUE}no-check${NC}"
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
    
    containers=(
        "sutazai-postgres:5432"
        "sutazai-redis:6379"
        "sutazai-neo4j:7474,7687"
        "sutazai-chromadb:8001"
        "sutazai-qdrant:6333"
        "sutazai-ollama:11434"
        "sutazai-backend-agi:8000"
        "sutazai-frontend-agi:8501"
    )
    
    for container_info in "${containers[@]}"; do
        IFS=':' read -r container ports <<< "$container_info"
        status=$(check_container_status "$container")
        health=$(get_container_health "$container")
        printf "│ %-19s │ %-6s │ %-8s │ %-11s │\n" "$container" "$status" "$health" "$ports"
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
    echo "• Frontend: http://172.31.77.193:8501"
    echo "• Backend API: http://172.31.77.193:8000"
    echo "• API Docs: http://172.31.77.193:8000/docs"
    echo ""
}

# Function to show live logs
show_live_logs() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     SUTAZAI LIVE LOGS                       ║${NC}"
    echo -e "${CYAN}║                  Press Ctrl+C to exit                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Start background log monitoring
    (
        docker logs -f sutazai-backend-agi 2>&1 | while IFS= read -r line; do
            log_with_color "$GREEN" "BACKEND" "$line"
        done
    ) &
    BACKEND_PID=$!
    
    (
        docker logs -f sutazai-frontend-agi 2>&1 | while IFS= read -r line; do
            log_with_color "$BLUE" "FRONTEND" "$line"
        done
    ) &
    FRONTEND_PID=$!
    
    (
        docker logs -f sutazai-ollama 2>&1 | while IFS= read -r line; do
            log_with_color "$YELLOW" "OLLAMA" "$line"
        done
    ) &
    OLLAMA_PID=$!
    
    # Wait for interrupt
    trap 'kill $BACKEND_PID $FRONTEND_PID $OLLAMA_PID 2>/dev/null; exit 0' INT TERM
    wait
}

# Function to test API endpoints
test_api_endpoints() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    API ENDPOINT TESTING                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    endpoints=(
        "GET:http://172.31.77.193:8000/health:Health Check"
        "GET:http://172.31.77.193:8000/:System Info"
        "GET:http://172.31.77.193:8000/agents:Agent List"
        "GET:http://172.31.77.193:8000/models:Model List"
        "POST:http://172.31.77.193:8000/simple-chat:Simple Chat"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r method url description <<< "$endpoint_info"
        echo -n "Testing $description ($method $url)... "
        
        if [ "$method" = "GET" ]; then
            if curl -f "$url" >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        elif [ "$method" = "POST" ]; then
            if curl -f -X POST "$url" -H "Content-Type: application/json" -d '{"message":"test"}' >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        fi
        sleep 0.5
    done
    
    echo ""
    echo "Testing frontend connectivity..."
    if curl -f http://172.31.77.193:8501/healthz >/dev/null 2>&1; then
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
    echo "5. Restart All Services"
    echo "6. Exit"
    echo ""
    read -p "Select option (1-6): " choice
    
    case $choice in
        1) show_system_overview; read -p "Press Enter to continue..."; show_menu ;;
        2) show_live_logs ;;
        3) test_api_endpoints; read -p "Press Enter to continue..."; show_menu ;;
        4) show_container_stats; read -p "Press Enter to continue..."; show_menu ;;
        5) 
            echo "Restarting all SutazAI services..."
            docker-compose -f /opt/sutazaiapp/docker-compose-consolidated.yml restart
            echo "All services restarted!"
            read -p "Press Enter to continue..."
            show_menu
            ;;
        6) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option"; show_menu ;;
    esac
}

# Start the monitoring system
if [ "${1:-}" = "--overview" ]; then
    show_system_overview
elif [ "${1:-}" = "--logs" ]; then
    show_live_logs
elif [ "${1:-}" = "--test" ]; then
    test_api_endpoints
elif [ "${1:-}" = "--stats" ]; then
    show_container_stats
elif [ $# -eq 0 ]; then
    show_menu
else
    # Run main function with parameters
    main "$@"
fi