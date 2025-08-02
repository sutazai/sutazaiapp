#!/bin/bash
#
# monitor_dashboard.sh - Comprehensive real-time monitoring dashboard for SutazAI
# Displays system metrics, container health, API performance, and more
#

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Default configuration
UPDATE_INTERVAL=5  # seconds between updates
LOG_FILE="${PROJECT_ROOT}/logs/monitoring/dashboard_metrics_$(date +%Y%m%d_%H%M%S).log"
METRICS_HISTORY_FILE="${PROJECT_ROOT}/logs/monitoring/metrics_history.json"

# Create necessary directories
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$METRICS_HISTORY_FILE")"

# Color codes for terminal UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NORMAL='\033[22m'
NC='\033[0m' # No Color
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'

# Terminal control sequences
CLEAR_SCREEN='\033[2J'
MOVE_TO_TOP='\033[H'
HIDE_CURSOR='\033[?25l'
SHOW_CURSOR='\033[?25h'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

# Function to format bytes to human-readable
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

# Function to get color based on percentage
get_color_by_percent() {
    local percent=$1
    local critical=${2:-90}
    local warning=${3:-70}
    
    if (( $(echo "$percent >= $critical" | bc -l) )); then
        echo "$RED"
    elif (( $(echo "$percent >= $warning" | bc -l) )); then
        echo "$YELLOW"
    else
        echo "$GREEN"
    fi
}

# Function to create a progress bar
create_progress_bar() {
    local percent=$1
    local width=${2:-30}
    local label=${3:-""}
    
    local filled=$(echo "scale=0; $percent * $width / 100" | bc)
    local empty=$((width - filled))
    
    local color=$(get_color_by_percent $percent)
    
    printf "${label}"
    printf "${color}"
    printf '█%.0s' $(seq 1 $filled)
    printf "${DIM}"
    printf '░%.0s' $(seq 1 $empty)
    printf "${NC}"
    printf " %5.1f%%" $percent
}

# Function to draw a box around content
draw_box() {
    local title=$1
    local width=${2:-80}
    
    # Top border
    printf "${CYAN}╔"
    printf '═%.0s' $(seq 1 $((width - 2)))
    printf "╗${NC}\n"
    
    # Title
    printf "${CYAN}║${NC} ${BOLD}%-$((width - 4))s${NC} ${CYAN}║${NC}\n" "$title"
    
    # Separator
    printf "${CYAN}╠"
    printf '═%.0s' $(seq 1 $((width - 2)))
    printf "╣${NC}\n"
}

# Function to draw box footer
draw_box_footer() {
    local width=${1:-80}
    printf "${CYAN}╚"
    printf '═%.0s' $(seq 1 $((width - 2)))
    printf "╝${NC}\n"
}

# Function to check Docker container status
check_container_status() {
    local container_name=$1
    
    if ! command -v docker &> /dev/null; then
        echo "N/A"
        return
    fi
    
    local status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null || echo "not found")
    local health=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "")
    
    case "$status" in
        "running")
            if [[ "$health" == "healthy" ]]; then
                echo "${GREEN}●${NC} Running (Healthy)"
            elif [[ "$health" == "unhealthy" ]]; then
                echo "${RED}●${NC} Running (Unhealthy)"
            elif [[ -n "$health" ]]; then
                echo "${YELLOW}●${NC} Running (Starting)"
            else
                echo "${GREEN}●${NC} Running"
            fi
            ;;
        "exited")
            echo "${RED}●${NC} Stopped"
            ;;
        "paused")
            echo "${YELLOW}●${NC} Paused"
            ;;
        "restarting")
            echo "${YELLOW}●${NC} Restarting"
            ;;
        *)
            echo "${DIM}●${NC} Not Found"
            ;;
    esac
}

# Function to get container resource usage
get_container_stats() {
    local container_name=$1
    
    if ! command -v docker &> /dev/null; then
        echo "0|0|0|0"
        return
    fi
    
    local stats=$(docker stats --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}" "$container_name" 2>/dev/null || echo "0%|0B / 0B|0B / 0B|0B / 0B")
    echo "$stats"
}

# Function to check API response time
check_api_response() {
    local url=$1
    local timeout=${2:-2}
    
    if ! command -v curl &> /dev/null; then
        echo "N/A|0"
        return
    fi
    
    local start_time=$(date +%s.%N)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "$url" 2>/dev/null || echo "000")
    local end_time=$(date +%s.%N)
    
    local response_time=$(echo "$end_time - $start_time" | bc)
    echo "$http_code|$response_time"
}

# Function to get Ollama model information
get_ollama_info() {
    if ! command -v curl &> /dev/null; then
        echo "N/A|0|0"
        return
    fi
    
    local models=$(curl -s http://localhost:11434/api/tags 2>/dev/null | jq -r '.models | length' 2>/dev/null || echo "0")
    local running=$(curl -s http://localhost:11434/api/ps 2>/dev/null | jq -r '.models | length' 2>/dev/null || echo "0")
    
    # Get currently loaded model info
    local loaded_model="None"
    if [[ $running -gt 0 ]]; then
        loaded_model=$(curl -s http://localhost:11434/api/ps 2>/dev/null | jq -r '.models[0].name' 2>/dev/null || echo "Unknown")
    fi
    
    echo "$models|$running|$loaded_model"
}

# Function to get vector database stats
get_vector_db_stats() {
    local db_type=$1
    
    case "$db_type" in
        "chromadb")
            if command -v curl &> /dev/null; then
                local collections=$(curl -s http://localhost:8000/api/v1/collections 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")
                echo "ChromaDB|$collections collections"
            else
                echo "ChromaDB|N/A"
            fi
            ;;
        "qdrant")
            if command -v curl &> /dev/null; then
                local collections=$(curl -s http://localhost:6333/collections 2>/dev/null | jq -r '.result.collections | length' 2>/dev/null || echo "0")
                echo "Qdrant|$collections collections"
            else
                echo "Qdrant|N/A"
            fi
            ;;
        *)
            echo "Unknown|N/A"
            ;;
    esac
}

# Function to get active agent information
get_agent_info() {
    local agent_count=0
    local agent_list=""
    
    # Check for running Python processes that might be agents
    local python_agents=$(ps aux | grep -E "python.*agent|autogpt|letta|crewai" | grep -v grep | wc -l)
    agent_count=$((agent_count + python_agents))
    
    # Check Docker containers for agents
    if command -v docker &> /dev/null; then
        local docker_agents=$(docker ps --format "{{.Names}}" | grep -E "agent|autogpt|letta|crewai" | wc -l)
        agent_count=$((agent_count + docker_agents))
    fi
    
    echo "$agent_count"
}

# Function to display alerts
display_alerts() {
    local alerts=()
    
    # Check CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        alerts+=("${RED}[CRITICAL]${NC} CPU usage above 90%")
    elif (( $(echo "$cpu_usage > 80" | bc -l) )); then
        alerts+=("${YELLOW}[WARNING]${NC} CPU usage above 80%")
    fi
    
    # Check memory usage
    local mem_percent=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    if (( $(echo "$mem_percent > 90" | bc -l) )); then
        alerts+=("${RED}[CRITICAL]${NC} Memory usage above 90%")
    elif (( $(echo "$mem_percent > 80" | bc -l) )); then
        alerts+=("${YELLOW}[WARNING]${NC} Memory usage above 80%")
    fi
    
    # Check disk usage
    local disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
    if [[ $disk_usage -gt 90 ]]; then
        alerts+=("${RED}[CRITICAL]${NC} Disk usage above 90%")
    elif [[ $disk_usage -gt 80 ]]; then
        alerts+=("${YELLOW}[WARNING]${NC} Disk usage above 80%")
    fi
    
    # Display alerts
    if [[ ${#alerts[@]} -gt 0 ]]; then
        draw_box "⚠️  ALERTS" 80
        for alert in "${alerts[@]}"; do
            printf "${CYAN}║${NC} %-78s ${CYAN}║${NC}\n" "$alert"
        done
        draw_box_footer 80
        echo
    fi
}

# Function to log metrics to file
log_metrics() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local cpu_usage=$1
    local mem_usage=$2
    local disk_usage=$3
    local api_response_time=$4
    
    # Create JSON entry
    local json_entry=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "cpu_usage": $cpu_usage,
  "memory_usage": $mem_usage,
  "disk_usage": $disk_usage,
  "api_response_time": $api_response_time,
  "containers": {
EOF
)
    
    # Add container stats
    local containers=("sutazai-postgres" "sutazai-redis" "sutazai-ollama" "sutazai-chromadb" "sutazai-qdrant" "sutazai-backend" "sutazai-frontend")
    local first=true
    
    for container in "${containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^$container$" 2>/dev/null; then
            if [[ "$first" != "true" ]]; then
                json_entry+=","
            fi
            first=false
            
            local stats=$(docker stats --no-stream --format "{{.CPUPerc}}|{{.MemPerc}}" "$container" 2>/dev/null || echo "0%|0%")
            local cpu=$(echo "$stats" | cut -d'|' -f1 | tr -d '%')
            local mem=$(echo "$stats" | cut -d'|' -f2 | tr -d '%')
            
            json_entry+="
    \"$container\": {
      \"cpu_percent\": $cpu,
      \"memory_percent\": $mem
    }"
        fi
    done
    
    json_entry+="
  }
}"
    
    # Append to history file
    echo "$json_entry" >> "$METRICS_HISTORY_FILE"
    
    # Keep only last 24 hours of data (assuming 5-second intervals = 17280 entries)
    tail -n 17280 "$METRICS_HISTORY_FILE" > "$METRICS_HISTORY_FILE.tmp" && mv "$METRICS_HISTORY_FILE.tmp" "$METRICS_HISTORY_FILE"
}

# Main dashboard display function
display_dashboard() {
    # Clear screen and hide cursor
    printf "${CLEAR_SCREEN}${MOVE_TO_TOP}${HIDE_CURSOR}"
    
    # Header
    printf "${BOLD}${MAGENTA}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                       SutazAI System Monitoring Dashboard                     ║"
    echo "║                          $(date '+%Y-%m-%d %H:%M:%S')                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    printf "${NC}\n"
    
    # System Resources
    draw_box "💻 SYSTEM RESOURCES" 80
    
    # CPU Usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    printf "${CYAN}║${NC} CPU Usage:    "
    create_progress_bar $cpu_usage 40
    printf " ${CYAN}║${NC}\n"
    
    # Memory Usage
    local mem_total=$(free -b | grep Mem | awk '{print $2}')
    local mem_used=$(free -b | grep Mem | awk '{print $3}')
    local mem_percent=$(echo "scale=2; $mem_used*100/$mem_total" | bc)
    local mem_human=$(free -h | grep Mem | awk '{print $3 " / " $2}')
    printf "${CYAN}║${NC} Memory:       "
    create_progress_bar $mem_percent 40
    printf " %-13s ${CYAN}║${NC}\n" "($mem_human)"
    
    # Disk Usage
    local disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
    local disk_human=$(df -h / | tail -1 | awk '{print $3 " / " $2}')
    printf "${CYAN}║${NC} Disk:         "
    create_progress_bar $disk_usage 40
    printf " %-13s ${CYAN}║${NC}\n" "($disk_human)"
    
    # Load Average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}')
    local cpu_cores=$(nproc)
    printf "${CYAN}║${NC} Load Average: %-62s ${CYAN}║${NC}\n" "$load_avg (${cpu_cores} cores)"
    
    draw_box_footer 80
    echo
    
    # Container Health Status
    draw_box "🐳 CONTAINER HEALTH STATUS" 80
    
    local containers=(
        "sutazai-postgres|PostgreSQL Database"
        "sutazai-redis|Redis Cache"
        "sutazai-ollama|Ollama Model Server"
        "sutazai-chromadb|ChromaDB Vector Store"
        "sutazai-qdrant|Qdrant Vector Database"
        "sutazai-backend|Backend API"
        "sutazai-frontend|Frontend UI"
    )
    
    for container_info in "${containers[@]}"; do
        IFS='|' read -r container_name display_name <<< "$container_info"
        local status=$(check_container_status "$container_name")
        local stats=$(get_container_stats "$container_name")
        
        IFS='|' read -r cpu_perc mem_usage net_io block_io <<< "$stats"
        
        printf "${CYAN}║${NC} %-25s %s" "$display_name:" "$status"
        
        if [[ "$status" == *"Running"* ]]; then
            printf " ${DIM}[CPU: %-6s Mem: %-15s]${NC}" "$cpu_perc" "$mem_usage"
        fi
        
        printf "%*s${CYAN}║${NC}\n" $((27 - ${#status})) ""
    done
    
    draw_box_footer 80
    echo
    
    # API Performance
    draw_box "🚀 API PERFORMANCE" 80
    
    # Backend API
    IFS='|' read -r http_code response_time <<< $(check_api_response "http://localhost:8000/health")
    printf "${CYAN}║${NC} Backend API Health:     "
    if [[ "$http_code" == "200" ]]; then
        printf "${GREEN}✓ OK${NC} (${response_time}s)"
    else
        printf "${RED}✗ Failed${NC} (HTTP $http_code)"
    fi
    printf "%*s${CYAN}║${NC}\n" 45 ""
    
    # Frontend
    IFS='|' read -r http_code response_time <<< $(check_api_response "http://localhost:8501")
    printf "${CYAN}║${NC} Frontend Status:        "
    if [[ "$http_code" == "200" ]]; then
        printf "${GREEN}✓ OK${NC} (${response_time}s)"
    else
        printf "${RED}✗ Failed${NC} (HTTP $http_code)"
    fi
    printf "%*s${CYAN}║${NC}\n" 45 ""
    
    # API Endpoints
    local endpoints=(
        "http://localhost:8000/api/v1/agents/list|Agents API"
        "http://localhost:8000/api/v1/models/list|Models API"
        "http://localhost:11434/api/tags|Ollama API"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS='|' read -r url name <<< "$endpoint_info"
        IFS='|' read -r http_code response_time <<< $(check_api_response "$url")
        
        printf "${CYAN}║${NC} %-23s " "$name:"
        if [[ "$http_code" == "200" ]]; then
            printf "${GREEN}✓ OK${NC} (${response_time}s)"
        else
            printf "${RED}✗ Failed${NC} (HTTP $http_code)"
        fi
        printf "%*s${CYAN}║${NC}\n" 45 ""
    done
    
    draw_box_footer 80
    echo
    
    # Model & Agent Status
    draw_box "🤖 MODEL & AGENT STATUS" 80
    
    # Ollama Models
    IFS='|' read -r total_models running_models loaded_model <<< $(get_ollama_info)
    printf "${CYAN}║${NC} Ollama Models:          Total: %-8s Running: %-8s              ${CYAN}║${NC}\n" "$total_models" "$running_models"
    printf "${CYAN}║${NC} Currently Loaded:       %-53s ${CYAN}║${NC}\n" "$loaded_model"
    
    # Active Agents
    local agent_count=$(get_agent_info)
    printf "${CYAN}║${NC} Active AI Agents:       %-53s ${CYAN}║${NC}\n" "$agent_count agents running"
    
    draw_box_footer 80
    echo
    
    # Vector Database Statistics
    draw_box "🗄️  VECTOR DATABASE STATISTICS" 80
    
    IFS='|' read -r db_name db_stats <<< $(get_vector_db_stats "chromadb")
    printf "${CYAN}║${NC} %-25s %-51s ${CYAN}║${NC}\n" "$db_name:" "$db_stats"
    
    IFS='|' read -r db_name db_stats <<< $(get_vector_db_stats "qdrant")
    printf "${CYAN}║${NC} %-25s %-51s ${CYAN}║${NC}\n" "$db_name:" "$db_stats"
    
    draw_box_footer 80
    echo
    
    # Display any alerts
    display_alerts
    
    # Footer
    printf "${DIM}Press Ctrl+C to exit | Updates every ${UPDATE_INTERVAL}s | Logs: ${LOG_FILE}${NC}\n"
    
    # Log metrics
    log_metrics "$cpu_usage" "$mem_percent" "$disk_usage" "${response_time:-0}"
}

# Cleanup function
cleanup() {
    printf "${SHOW_CURSOR}"
    printf "${NC}\n"
    echo "Dashboard stopped. Metrics saved to: $METRICS_HISTORY_FILE"
    exit 0
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main loop
while true; do
    display_dashboard
    sleep $UPDATE_INTERVAL
done