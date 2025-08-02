#!/bin/bash
# title        :monitor_system.sh
# description  :This script monitors system performance metrics and logs them
# author       :SutazAI Team
# version      :1.0
# usage        :sudo bash scripts/monitor_system.sh [--interval=N] [--output-file=PATH]
# notes        :Requires bash 4.0+ and standard Linux utilities

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Default configuration
INTERVAL=15  # seconds between checks
OUTPUT_FILE="$PROJECT_ROOT/logs/system_metrics.log"
DURATION=-1  # run indefinitely by default
VERBOSE=1    # verbose output by default

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --interval=*)
            INTERVAL="${arg#*=}"
            ;;
        --output-file=*)
            OUTPUT_FILE="${arg#*=}"
            ;;
        --duration=*)
            DURATION="${arg#*=}"
            ;;
        --quiet)
            VERBOSE=0
            ;;
        --help)
            echo "Usage: sudo bash scripts/monitor_system.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --interval=N        Check every N seconds (default: 15)"
            echo "  --output-file=PATH  Write metrics to this file (default: logs/system_metrics.log)"
            echo "  --duration=N        Run for N seconds, -1 for indefinitely (default: -1)"
            echo "  --quiet             Suppress terminal output"
            echo "  --help              Show this help message"
            exit 0
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Please run this script with sudo or as root"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Print to both console and log file
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$OUTPUT_FILE"
    
    # Print to console if verbose
    if [[ $VERBOSE -eq 1 ]]; then
        case $level in
            "INFO")
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            "SUCCESS")
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            "WARNING")
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            "ERROR")
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
            *)
                echo -e "$message"
                ;;
        esac
    fi
}

# Format bytes to human-readable format
format_bytes() {
    local bytes=$1
    if [[ $bytes -lt 1024 ]]; then
        echo "${bytes}B"
    elif [[ $bytes -lt 1048576 ]]; then
        echo "$(echo "scale=2; $bytes/1024" | bc)KB"
    elif [[ $bytes -lt 1073741824 ]]; then
        echo "$(echo "scale=2; $bytes/1048576" | bc)MB"
    else
        echo "$(echo "scale=2; $bytes/1073741824" | bc)GB"
    fi
}

# Function to collect system metrics
collect_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    local memory_total=$(free | grep Mem | awk '{print $2}')
    local memory_used=$(free | grep Mem | awk '{print $3}')
    local memory_percent=$(echo "scale=2; $memory_used*100/$memory_total" | bc)
    local disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | tr -d ',')
    local processes=$(ps aux | wc -l)
    local backend_processes=$(pgrep -fc "python.*backend" || echo "0")
    local webui_processes=$(pgrep -fc "node.*next" || echo "0")
    local vector_store_processes=$(pgrep -fc "qdrant_server" || echo "0")
    
    # API response time if backend is running
    local api_response_time="N/A"
    if [[ $backend_processes -gt 0 ]] && command -v curl >/dev/null 2>&1; then
        local start_time=$(date +%s.%N)
        if curl -s "http://localhost:8000/api/health" >/dev/null 2>&1; then
            local end_time=$(date +%s.%N)
            api_response_time=$(echo "$end_time - $start_time" | bc)
        else
            api_response_time="ERROR"
        fi
    fi
    
    # Network stats
    local network_rx=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
    local network_tx=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
    
    # If eth0 doesn't exist, try ens3 (common AWS interface)
    if [[ -z "$network_rx" ]]; then
        network_rx=$(cat /proc/net/dev | grep ens3 | awk '{print $2}')
        network_tx=$(cat /proc/net/dev | grep ens3 | awk '{print $10}')
    fi
    
    # If still empty, try any interface
    if [[ -z "$network_rx" ]]; then
        network_rx=$(cat /proc/net/dev | grep -v "lo:" | head -2 | tail -1 | awk '{print $2}')
        network_tx=$(cat /proc/net/dev | grep -v "lo:" | head -2 | tail -1 | awk '{print $10}')
    fi
    
    # Format network stats
    if [[ -n "$network_rx" ]]; then
        network_rx=$(format_bytes $network_rx)
        network_tx=$(format_bytes $network_tx)
    else
        network_rx="N/A"
        network_tx="N/A"
    fi
    
    # Check database size
    local db_type=$(grep "^DB_TYPE=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    local db_size="N/A"
    
    if [[ "$db_type" == "sqlite" ]]; then
        local db_path=$(grep "^SQLITE_PATH=" "$PROJECT_ROOT/.env" | cut -d= -f2)
        if [[ -f "$db_path" ]]; then
            db_size=$(du -h "$db_path" | cut -f1)
        fi
    fi
    
    # Current number of open files
    local open_files=$(lsof | wc -l)
    
    # Number of zombie processes
    local zombie_count=$(ps aux | grep -c 'Z')
    
    # Output metrics
    log "METRICS" "CPU Usage: ${cpu_usage}% | Memory: ${memory_percent}% | Disk: ${disk_usage}% | Load: ${load_avg} | API: ${api_response_time}s | Processes: ${processes} | DB: ${db_size} | Net RX/TX: ${network_rx}/${network_tx} | Open Files: ${open_files} | Zombies: ${zombie_count} | Backend: ${backend_processes} | WebUI: ${webui_processes} | VectorDB: ${vector_store_processes}"
    
    # Alert on high resource usage
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        log "WARNING" "Critical CPU usage: ${cpu_usage}%"
    elif (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log "WARNING" "High CPU usage: ${cpu_usage}%"
    fi
    
    if (( $(echo "$memory_percent > 90" | bc -l) )); then
        log "WARNING" "Critical memory usage: ${memory_percent}%"
    elif (( $(echo "$memory_percent > 80" | bc -l) )); then
        log "WARNING" "High memory usage: ${memory_percent}%"
    fi
    
    if [[ "$disk_usage" -gt 90 ]]; then
        log "WARNING" "Critical disk usage: ${disk_usage}%"
    elif [[ "$disk_usage" -gt 80 ]]; then
        log "WARNING" "High disk usage: ${disk_usage}%"
    fi
    
    if [[ "$api_response_time" != "N/A" ]] && [[ "$api_response_time" != "ERROR" ]] && (( $(echo "$api_response_time > 1.0" | bc -l) )); then
        log "WARNING" "Slow API response time: ${api_response_time}s"
    fi
    
    if [[ "$zombie_count" -gt 1 ]]; then
        log "WARNING" "Zombie processes detected: $((zombie_count-1))"
    fi
    
    if [[ $backend_processes -eq 0 ]]; then
        log "ERROR" "Backend is not running!"
    elif [[ $backend_processes -gt 1 ]]; then
        log "WARNING" "Multiple backend processes detected: $backend_processes"
    fi
    
    if [[ $webui_processes -eq 0 ]]; then
        log "ERROR" "Web UI is not running!"
    elif [[ $webui_processes -gt 1 ]]; then
        log "WARNING" "Multiple Web UI processes detected: $webui_processes"
    fi
    
    if [[ $vector_store_processes -eq 0 ]]; then
        log "ERROR" "Vector database is not running!"
    elif [[ $vector_store_processes -gt 1 ]]; then
        log "WARNING" "Multiple vector database processes detected: $vector_store_processes"
    fi
}

# Initial log
log "INFO" "Starting system monitoring (interval: ${INTERVAL}s, output: ${OUTPUT_FILE})"

# Start time
start_time=$(date +%s)

# Monitor loop
while true; do
    # Collect and log metrics
    collect_metrics
    
    # Break loop if duration is reached
    if [[ $DURATION -gt 0 ]]; then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [[ $elapsed -ge $DURATION ]]; then
            log "INFO" "Monitoring completed (duration: ${DURATION}s)"
            break
        fi
    fi
    
    # Wait for next interval
    sleep $INTERVAL
done

exit 0 