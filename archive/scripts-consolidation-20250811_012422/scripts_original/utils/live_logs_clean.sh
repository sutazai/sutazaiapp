#!/bin/bash
# SutazAI Live Logs Management System
# Enhanced with cleanup and debug controls

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

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
        "docker-compose-consolidated.yml"
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
            docker-compose -f "${PROJECT_ROOT}/docker-compose-consolidated.yml" restart > /dev/null 2>&1 || true
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
        
        printf "  %-20s Status: %-10s Log Size: %s\\n" "$container" "$status" "$log_size"
    done
    
    echo ""
    
    # System disk usage
    echo -e "${CYAN}System Disk Usage:${NC}"
    df -h / | tail -1 | awk '{printf "  Root: %s used, %s available (%s full)\\n", $3, $4, $5}'
    
    # Docker disk usage
    echo -e "${CYAN}Docker Disk Usage:${NC}"
    docker system df 2>/dev/null | tail -n +2 | while read -r line; do
        echo "  $line"
    done || echo "  Docker system df unavailable"
}

# Show usage
show_usage() {
    echo -e "${GREEN}Usage: $0 [COMMAND] [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}Primary Commands:${NC}"
    echo "  cleanup [days]          - Clean old logs (default: 7 days)"
    echo "  reset                   - Reset all logs (DANGEROUS)"
    echo "  debug [on|off]          - Toggle debugging mode"
    echo "  level [DEBUG|INFO|WARN] - Set log level"
    echo "  config                  - Show current configuration"
    echo "  status                  - Show log status and disk usage"
    echo ""
    echo -e "${YELLOW}System Management:${NC}"
    echo "  --init-db               - Initialize SutazAI database"
    echo "  --repair                - Complete system repair"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 cleanup 3            # Clean logs older than 3 days"  
    echo "  $0 debug on             # Enable debug mode"
    echo "  $0 --repair             # Fix database and restart services"
    echo "  $0 --init-db            # Initialize database only"
}

# Main execution
main() {
    load_config
    
    case "${1:-}" in
        "cleanup")
            if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                cleanup_logs "${2}d" "false"
            else
                cleanup_logs "${CLEANUP_DAYS}d" "false"
            fi
            ;;
        "reset")
            reset_logs "false"
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

# Execute based on command line arguments
if [[ "${1:-}" == "--init-db" ]]; then
    init_database
elif [[ "${1:-}" == "--repair" ]]; then
    repair_system
else
    main "$@"
fi