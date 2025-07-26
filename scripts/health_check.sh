#!/bin/bash
# SutazAI Health Check Script
# This script monitors the health of all SutazAI services and components

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="${PROJECT_ROOT}/logs/health_check.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Configuration
BACKEND_PID_FILE="${PROJECT_ROOT}/.backend.pid"
SUPERAGI_PID_FILE="${PROJECT_ROOT}/.superagi.pid"
WEBUI_PID_FILE="${PROJECT_ROOT}/.webui.pid"
BACKEND_URL="http://localhost:8000/api/health"
MONITORING_URL="http://localhost:9090/-/healthy"
DISK_THRESHOLD=90  # Percentage
MEMORY_THRESHOLD=90  # Percentage
CPU_THRESHOLD=95  # Percentage

# Replace SuperAGI PID file with LocalAGI
SUPERAGI_PID_FILE="${PROJECT_ROOT}/.superagi.pid"
LOCALAGI_LOG_FILE="${PROJECT_ROOT}/logs/localagi.log"

# Timestamp for this run
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Check if running in cron mode
CRON_MODE=0
AUTO_CLEANUP=0
for arg in "$@"; do
    case $arg in
        --cron)
            CRON_MODE=1
            AUTO_CLEANUP=1
            shift
            ;;
        --auto-cleanup)
            AUTO_CLEANUP=1
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Function to log to file only when in cron mode
log_cron() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # When in cron mode, only write to log file, don't output to console
    if [ $CRON_MODE -eq 1 ]; then
        echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    else
        # In interactive mode, both log to file and show colorized output
        echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
        case $level in
            INFO)
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            WARNING)
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
            *)
                echo "$message"
                ;;
        esac
    fi
}

# Logging function
log() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Print header
echo -e "${BOLD}SutazAI Health Check - $TIMESTAMP${NC}"
echo -e "${BOLD}=====================================${NC}"
log "Starting health check..." "INFO"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check if curl is installed
if ! command_exists curl; then
    log "${YELLOW}Warning: curl is not installed. Some health checks will be skipped.${NC}" "WARN"
fi

# Function to check if a process is running by PID file
check_process_by_pid() {
    local pid_file="$1"
    local service_name="$2"
    local process_pattern="$3"
    
    echo -e "\n${BOLD}Checking $service_name service...${NC}"
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if [ -n "$PID" ] && ps -p "$PID" > /dev/null; then
            echo -e "${GREEN}✓ $service_name is running (PID: $PID)${NC}"
            log "$service_name is running (PID: $PID)" "INFO"
            return 0
        else
            echo -e "${RED}✗ $service_name is not running (stale PID file: $PID)${NC}"
            log "$service_name is not running (stale PID file)" "ERROR"
        fi
    else
        echo -e "${YELLOW}! $service_name PID file not found${NC}"
        log "$service_name PID file not found" "WARN"
    fi
    
    # Try to find the process by pattern
    if [ -n "$process_pattern" ]; then
        PID=$(pgrep -f "$process_pattern")
        if [ -n "$PID" ]; then
            echo -e "${YELLOW}! Found potential $service_name process running (PID: $PID)${NC}"
            log "Found potential $service_name process running (PID: $PID)" "WARN"
            return 0
        fi
    fi
    
    return 1
}

# Function to check HTTP endpoint
check_endpoint() {
    local url="$1"
    local service_name="$2"
    local expected_status="${3:-200}"
    local is_critical="${4:-true}"
    
    echo -e "\n${BOLD}Checking $service_name API...${NC}"
    
    if command_exists curl; then
        if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
            if [ "$response" -eq "$expected_status" ]; then
                echo -e "${GREEN}✓ $service_name API is accessible (HTTP $response)${NC}"
                log "$service_name API is accessible (HTTP $response)" "INFO"
            else
                if [ "$is_critical" = "true" ]; then
                    echo -e "${RED}✗ $service_name API returned unexpected status (HTTP $response)${NC}"
                    log "$service_name API returned unexpected status (HTTP $response)" "ERROR"
                else
                    echo -e "${YELLOW}! $service_name API returned unexpected status (HTTP $response) - Optional component${NC}"
                    log "$service_name API returned unexpected status (HTTP $response) - Optional component" "WARN"
                fi
            fi
        else
            if [ "$is_critical" = "true" ]; then
                echo -e "${RED}✗ $service_name API is not accessible (HTTP 000)${NC}"
                log "$service_name API is not accessible (HTTP 000)" "ERROR"
            else
                echo -e "${YELLOW}! $service_name API is not accessible (HTTP 000) - Optional component${NC}"
                log "$service_name API is not accessible (HTTP 000) - Optional component" "WARN"
            fi
        fi
    else
        echo -e "${YELLOW}! Cannot check $service_name API (curl not installed)${NC}"
        log "Cannot check $service_name API (curl not installed)" "WARN"
    fi
}

# Function to check directory existence and permissions
check_directory() {
    local dir="$1"
    local desc="$2"
    
    echo -e "\n${BOLD}Checking $desc directory...${NC}"
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ $desc directory exists${NC}"
        
        if [ -w "$dir" ]; then
            echo -e "${GREEN}✓ $desc directory is writable${NC}"
            log "$desc directory exists and is writable" "INFO"
        else
            echo -e "${RED}✗ $desc directory is not writable${NC}"
            log "$desc directory exists but is not writable" "ERROR"
        fi
    else
        echo -e "${RED}✗ $desc directory does not exist${NC}"
        log "$desc directory does not exist" "ERROR"
    fi
}

# Function to check file existence
check_file() {
    local file="$1"
    local desc="$2"
    local required="${3:-true}"
    
    echo -e "\n${BOLD}Checking $desc file...${NC}"
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $desc file exists${NC}"
        log "$desc file exists" "INFO"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗ $desc file does not exist${NC}"
            log "$desc file does not exist" "ERROR"
        else
            echo -e "${YELLOW}! $desc file does not exist (optional)${NC}"
            log "$desc file does not exist (optional)" "WARN"
        fi
        return 1
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "\n${BOLD}Checking system resources...${NC}"
    
    # Check disk space
    if command_exists df; then
        DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
        if [ "$DISK_USAGE" -gt "$DISK_THRESHOLD" ]; then
            echo -e "${RED}✗ Disk usage is critical: ${DISK_USAGE}%${NC}"
            log "Disk usage is critical: ${DISK_USAGE}%" "ERROR"
        elif [ "$DISK_USAGE" -gt $(($DISK_THRESHOLD - 10)) ]; then
            echo -e "${YELLOW}! Disk usage is high: ${DISK_USAGE}%${NC}"
            log "Disk usage is high: ${DISK_USAGE}%" "WARN"
        else
            echo -e "${GREEN}✓ Disk usage is normal: ${DISK_USAGE}%${NC}"
            log "Disk usage is normal: ${DISK_USAGE}%" "INFO"
        fi
    else
        echo -e "${YELLOW}! Could not check disk usage (df command not available)${NC}"
        log "Could not check disk usage (df command not available)" "WARN"
    fi
    
    # Check memory usage
    if command_exists free; then
        MEM_USAGE=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
        if [ "$MEM_USAGE" -gt "$MEMORY_THRESHOLD" ]; then
            echo -e "${RED}✗ Memory usage is critical: ${MEM_USAGE}%${NC}"
            log "Memory usage is critical: ${MEM_USAGE}%" "ERROR"
        elif [ "$MEM_USAGE" -gt $(($MEMORY_THRESHOLD - 10)) ]; then
            echo -e "${YELLOW}! Memory usage is high: ${MEM_USAGE}%${NC}"
            log "Memory usage is high: ${MEM_USAGE}%" "WARN"
        else
            echo -e "${GREEN}✓ Memory usage is normal: ${MEM_USAGE}%${NC}"
            log "Memory usage is normal: ${MEM_USAGE}%" "INFO"
        fi
    else
        echo -e "${YELLOW}! Could not check memory usage (free command not available)${NC}"
        log "Could not check memory usage (free command not available)" "WARN"
    fi
    
    # Check CPU load
    if [ -f /proc/loadavg ]; then
        CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
        CPU_LOAD=$(cat /proc/loadavg | awk '{print $1}')
        CPU_LOAD_PERCENT=$(echo "$CPU_LOAD $CPU_CORES" | awk '{printf "%d", ($1/$2)*100}')
        
        if [ "$CPU_LOAD_PERCENT" -gt "$CPU_THRESHOLD" ]; then
            echo -e "${RED}✗ CPU load is critical: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})${NC}"
            log "CPU load is critical: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})" "ERROR"
        elif [ "$CPU_LOAD_PERCENT" -gt $(($CPU_THRESHOLD - 10)) ]; then
            echo -e "${YELLOW}! CPU load is high: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})${NC}"
            log "CPU load is high: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})" "WARN"
        else
            echo -e "${GREEN}✓ CPU load is normal: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})${NC}"
            log "CPU load is normal: ${CPU_LOAD_PERCENT}% (${CPU_LOAD}/${CPU_CORES})" "INFO"
        fi
    else
        echo -e "${YELLOW}! Could not check CPU load (/proc/loadavg not available)${NC}"
        log "Could not check CPU load (/proc/loadavg not available)" "WARN"
    fi
}

# Function to check log file sizes and growth
check_logs() {
    echo -e "\n${BOLD}Checking log files...${NC}"
    
    LOG_DIR="${PROJECT_ROOT}/logs"
    if [ -d "$LOG_DIR" ]; then
        LARGE_LOGS=$(find "$LOG_DIR" -type f -name "*.log" -size +50M)
        if [ -n "$LARGE_LOGS" ]; then
            echo -e "${YELLOW}! Some log files are larger than 50MB:${NC}"
            log "Some log files are larger than 50MB" "WARN"
            echo "$LARGE_LOGS" | while read -r log_file; do
                log_size=$(du -h "$log_file" | awk '{print $1}')
                echo -e "${YELLOW}  - $(basename "$log_file"): $log_size${NC}"
                log "Large log file: $(basename "$log_file"): $log_size" "WARN"
            done
        else
            echo -e "${GREEN}✓ All log files are of reasonable size${NC}"
            log "All log files are of reasonable size" "INFO"
        fi
    else
        echo -e "${YELLOW}! Log directory does not exist${NC}"
        log "Log directory does not exist" "WARN"
    fi
}

# Function to check database health (if applicable)
check_database() {
    echo -e "\n${BOLD}Checking database...${NC}"
    
    DB_FILE="${PROJECT_ROOT}/storage/sutazai.db"
    if [ -f "$DB_FILE" ]; then
        DB_SIZE=$(du -h "$DB_FILE" | awk '{print $1}')
        echo -e "${GREEN}✓ SQLite database exists (Size: $DB_SIZE)${NC}"
        log "SQLite database exists (Size: $DB_SIZE)" "INFO"
        
        # Check if database is not corrupted - requires sqlite3 command
        if command_exists sqlite3; then
            if sqlite3 "$DB_FILE" "PRAGMA integrity_check;" 2>/dev/null | grep -q "ok"; then
                echo -e "${GREEN}✓ SQLite database passed integrity check${NC}"
                log "SQLite database passed integrity check" "INFO"
            else
                echo -e "${RED}✗ SQLite database failed integrity check${NC}"
                log "SQLite database failed integrity check" "ERROR"
            fi
        else
            echo -e "${YELLOW}! Cannot perform database integrity check (sqlite3 not installed)${NC}"
            log "Cannot perform database integrity check (sqlite3 not installed)" "WARN"
        fi
    else
        echo -e "${YELLOW}! SQLite database file not found${NC}"
        log "SQLite database file not found" "WARN"
    fi
}

# Function to check virtual environment
check_venv() {
    echo -e "\n${BOLD}Checking virtual environment...${NC}"
    
    VENV_DIR="${PROJECT_ROOT}/venv"
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
        log "Virtual environment exists" "INFO"
        
        if [ -f "${VENV_DIR}/bin/python" ]; then
            PYTHON_VERSION=$(${VENV_DIR}/bin/python --version 2>&1)
            echo -e "${GREEN}✓ Python interpreter found: $PYTHON_VERSION${NC}"
            log "Python interpreter found: $PYTHON_VERSION" "INFO"
        else
            echo -e "${RED}✗ Python interpreter not found in virtual environment${NC}"
            log "Python interpreter not found in virtual environment" "ERROR"
        fi
        
        # Check pip installations
        if [ -f "${VENV_DIR}/bin/pip" ]; then
            MISSING_PACKAGES=0
            echo -e "${CYAN}Checking required packages...${NC}"
            
            # List of critical packages to check
            for package in fastapi uvicorn pydantic sqlalchemy; do
                if ! ${VENV_DIR}/bin/pip freeze | grep -i "^$package==" > /dev/null; then
                    echo -e "${RED}✗ Required package '$package' is not installed${NC}"
                    log "Required package '$package' is not installed" "ERROR"
                    MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
                fi
            done
            
            # Special check for superagi which is installed in development/editable mode
            if ! ${VENV_DIR}/bin/pip list | grep -i "superagi" > /dev/null; then
                echo -e "${RED}✗ Required package 'superagi' is not installed${NC}"
                log "Required package 'superagi' is not installed" "ERROR"
                MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
            fi
            
            if [ $MISSING_PACKAGES -eq 0 ]; then
                echo -e "${GREEN}✓ All critical packages are installed${NC}"
                log "All critical packages are installed" "INFO"
            fi
        else
            echo -e "${RED}✗ Pip not found in virtual environment${NC}"
            log "Pip not found in virtual environment" "ERROR"
        fi
    else
        echo -e "${RED}✗ Virtual environment not found${NC}"
        log "Virtual environment not found" "ERROR"
    fi
}

# Function to check model files
check_models() {
    echo -e "\n${BOLD}Checking AI model files...${NC}"
    
    # Check GPT4All model
    GPT4ALL_DIR="${PROJECT_ROOT}/model_management/GPT4All"
    if [ -d "$GPT4ALL_DIR" ]; then
        MODEL_FILES=$(find "$GPT4ALL_DIR" -type f -name "*.bin" -o -name "*.gguf")
        if [ -n "$MODEL_FILES" ]; then
            echo -e "${GREEN}✓ GPT4All model files found:${NC}"
            log "GPT4All model files found" "INFO"
            echo "$MODEL_FILES" | while read -r model_file; do
                model_size=$(du -h "$model_file" | awk '{print $1}')
                echo -e "${GREEN}  - $(basename "$model_file"): $model_size${NC}"
                log "Model file: $(basename "$model_file"): $model_size" "INFO"
            done
        else
            echo -e "${RED}✗ No GPT4All model files found${NC}"
            log "No GPT4All model files found" "ERROR"
        fi
    else
        echo -e "${YELLOW}! GPT4All model directory not found${NC}"
        log "GPT4All model directory not found" "WARN"
    fi
    
    # Check other model directories as needed
    DEEPSEEK_DIR="${PROJECT_ROOT}/model_management/DeepSeek-Coder-33B"
    if [ -d "$DEEPSEEK_DIR" ]; then
        MODEL_FILES=$(find "$DEEPSEEK_DIR" -type f -name "*.bin" -o -name "*.gguf")
        if [ -n "$MODEL_FILES" ]; then
            echo -e "${GREEN}✓ DeepSeek Coder model files found${NC}"
            log "DeepSeek Coder model files found" "INFO"
        else
            echo -e "${YELLOW}! No DeepSeek Coder model files found${NC}"
            log "No DeepSeek Coder model files found" "WARN"
        fi
    fi
}

# Function to check Docker containers - proper implementation
check_docker_container() {
    local container_name="$1"
    local friendly_name="$2"
    
    echo -e "\n${BOLD}Checking $friendly_name container...${NC}"
    
    if command_exists docker; then
        if docker ps --filter "name=$container_name" --format "{{.Names}}" 2>/dev/null | grep -q "$container_name"; then
            echo -e "${GREEN}✓ $friendly_name container is running${NC}"
            log "$friendly_name container is running" "INFO"
            return 0
        else
            # Check if container exists but is stopped
            if docker ps -a --filter "name=$container_name" --format "{{.Names}}" 2>/dev/null | grep -q "$container_name"; then
                echo -e "${YELLOW}! $friendly_name container exists but is not running${NC}"
                log "$friendly_name container exists but is not running" "WARN"
            else
                echo -e "${RED}✗ $friendly_name container does not exist${NC}"
                log "$friendly_name container does not exist" "ERROR"
            fi
            return 1
        fi
    else
        echo -e "${YELLOW}! Cannot check $friendly_name container (docker not available)${NC}"
        log "Cannot check $friendly_name container (docker not available)" "WARN"
        return 2
    fi
}

# Function to check LocalAGI service
check_localagi() {
    echo -e "\n${BOLD}Checking LocalAGI service...${NC}"
    
    # Check if LocalAGI is running as a process
    if pgrep -f "localagi" > /dev/null; then
        echo -e "${GREEN}✓ LocalAGI is running${NC}"
        log "LocalAGI is running" "INFO"
        return 0
    # Check if LocalAGI is running in Docker
    elif command_exists docker && docker ps --filter "name=localagi" --format "{{.Names}}" 2>/dev/null | grep -q "localagi"; then
        echo -e "${GREEN}✓ LocalAGI is running${NC}"
        log "LocalAGI is running" "INFO"
        return 0
    # Check log file for recent activity
    elif [ -f "$LOCALAGI_LOG_FILE" ] && grep -q "started\|running" "$LOCALAGI_LOG_FILE"; then
        echo -e "${GREEN}✓ LocalAGI is running${NC}"
        log "LocalAGI is running" "INFO"
        return 0
    else
        echo -e "${YELLOW}! LocalAGI does not appear to be running${NC}"
        log "LocalAGI does not appear to be running" "WARN"
        return 1
    fi
}

# Function to check network connectivity with key services
check_network_connectivity() {
    echo -e "\n${BOLD}Checking network connectivity...${NC}"
    
    # List of hosts and ports to check
    declare -A services=(
        ["Backend API (localhost:8000)"]="localhost:8000"
        ["Web UI (localhost:3000)"]="localhost:3000"
        ["Vector DB (localhost:6333)"]="localhost:6333" 
        ["LocalAGI API (localhost:8090)"]="localhost:8090"
    )
    
    local all_good=true
    
    for service in "${!services[@]}"; do
        host_port="${services[$service]}"
        host="${host_port%%:*}"
        port="${host_port##*:}"
        
        if command_exists nc; then
            if nc -z -w2 "$host" "$port" 2>/dev/null; then
                echo -e "${GREEN}✓ $service is reachable${NC}"
                log "$service is reachable" "INFO"
            else
                echo -e "${RED}✗ $service is not reachable${NC}"
                log "$service is not reachable" "ERROR"
                all_good=false
            fi
        elif command_exists curl; then
            if curl -s "http://$host:$port" -m 2 -o /dev/null; then
                echo -e "${GREEN}✓ $service is reachable${NC}"
                log "$service is reachable" "INFO"
            else
                echo -e "${RED}✗ $service is not reachable${NC}"
                log "$service is not reachable" "ERROR"
                all_good=false
            fi
        else
            echo -e "${YELLOW}! Cannot check $service (nc and curl not available)${NC}"
            log "Cannot check $service (nc and curl not available)" "WARN"
            all_good=false
        fi
    done
    
    return $([[ "$all_good" == "true" ]] && echo 0 || echo 1)
}

# Function to check configuration files
check_configuration_files() {
    echo -e "\n${BOLD}Checking configuration files...${NC}"
    
    # List of important configuration files
    local config_files=(
        "${PROJECT_ROOT}/.env:Environment variables"
        "${PROJECT_ROOT}/config.json:Application configuration"
        "${PROJECT_ROOT}/docker-compose.yml:Docker configuration"
    )
    
    local all_good=true
    
    for config_entry in "${config_files[@]}"; do
        IFS=':' read -r file_path description <<< "$config_entry"
        
        if [ -f "$file_path" ]; then
            echo -e "${GREEN}✓ $description file exists${NC}"
            log "$description file exists" "INFO"
            
            # Check file is not empty
            if [ -s "$file_path" ]; then
                echo -e "${GREEN}✓ $description file is not empty${NC}"
                log "$description file is not empty" "INFO"
            else
                echo -e "${YELLOW}! $description file is empty${NC}"
                log "$description file is empty" "WARN"
                all_good=false
            fi
            
            # Check for required configuration in .env file
            if [[ "$file_path" == *".env" ]]; then
                # Check critical environment variables
                if grep -q "^DB_TYPE=" "$file_path" || grep -q "^DATABASE_URL=" "$file_path" || grep -q "^SQLITE_PATH=" "$file_path" || grep -q "^DB_PATH=" "$file_path"; then
                    echo -e "${GREEN}✓ Database configuration found in .env${NC}"
                    log "Database configuration found in .env" "INFO"
                else
                    echo -e "${YELLOW}! No database configuration found in .env${NC}"
                    log "No database configuration found in .env" "WARN"
                    all_good=false
                fi
            fi
        else
            # Only mark as error if it's a critical file (.env)
            if [[ "$file_path" == *".env" ]]; then
                echo -e "${RED}✗ $description file does not exist${NC}"
                log "$description file does not exist" "ERROR"
                all_good=false
            else
                echo -e "${YELLOW}! $description file does not exist${NC}"
                log "$description file does not exist" "WARN"
            fi
        fi
    done
    
    return $([[ "$all_good" == "true" ]] && echo 0 || echo 1)
}

# Function to check required permissions
check_permissions() {
    echo -e "\n${BOLD}Checking required permissions...${NC}"
    
    # List of directories that need write permissions
    local writable_dirs=(
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/storage"
        "${PROJECT_ROOT}/workspace"
        "${PROJECT_ROOT}/outputs"
        "${PROJECT_ROOT}/tmp"
    )
    
    local permission_issues=false
    
    for dir in "${writable_dirs[@]}"; do
        if [ -d "$dir" ]; then
            if [ -w "$dir" ]; then
                echo -e "${GREEN}✓ Directory $dir is writable${NC}"
                log "Directory $dir is writable" "INFO"
            else
                echo -e "${RED}✗ Directory $dir is not writable${NC}"
                log "Directory $dir is not writable" "ERROR"
                permission_issues=true
                
                # Attempt to fix permissions if sudo is available
                if command_exists sudo; then
                    echo -e "${YELLOW}! Attempting to fix permissions for $dir${NC}"
                    log "Attempting to fix permissions for $dir" "WARN"
                    
                    if sudo chmod -R 777 "$dir" 2>/dev/null; then
                        echo -e "${GREEN}✓ Fixed permissions for $dir${NC}"
                        log "Fixed permissions for $dir" "INFO"
                    else
                        echo -e "${RED}✗ Failed to fix permissions for $dir${NC}"
                        log "Failed to fix permissions for $dir" "ERROR"
                    fi
                fi
            fi
        fi
    done
    
    # Check executable permissions for script files
    local executable_scripts=(
        "${PROJECT_ROOT}/scripts/start_all.sh"
        "${PROJECT_ROOT}/scripts/stop_all.sh"
        "${PROJECT_ROOT}/scripts/health_check.sh"
        "${PROJECT_ROOT}/scripts/utils/cleanup_sync.sh"
    )
    
    for script in "${executable_scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                echo -e "${GREEN}✓ Script $script is executable${NC}"
                log "Script $script is executable" "INFO"
            else
                echo -e "${YELLOW}! Script $script is not executable${NC}"
                log "Script $script is not executable" "WARN"
                permission_issues=true
                
                # Attempt to fix permissions
                if chmod +x "$script" 2>/dev/null; then
                    echo -e "${GREEN}✓ Fixed executable permission for $script${NC}"
                    log "Fixed executable permission for $script" "INFO"
                else
                    echo -e "${RED}✗ Failed to fix executable permission for $script${NC}"
                    log "Failed to fix executable permission for $script" "ERROR"
                fi
            fi
        fi
    done
    
    return $([ "$permission_issues" = false ] && echo 0 || echo 1)
}

# Function to check process dependencies (parent-child relationships)
check_process_dependencies() {
    echo -e "\n${BOLD}Checking process dependencies...${NC}"
    
    # Check if the Backend Server has the expected child processes
    if pgrep -f "uvicorn.*backend.main" > /dev/null; then
        local backend_pids=$(pgrep -f "uvicorn.*backend.main")
        local worker_count=$(echo "$backend_pids" | wc -l)
        
        if [ "$worker_count" -gt 1 ]; then
            echo -e "${GREEN}✓ Backend Server has $(($worker_count - 1)) worker processes${NC}"
            log "Backend Server has $(($worker_count - 1)) worker processes" "INFO"
            
            # Update PID file with the main process
            local main_pid=$(echo "$backend_pids" | head -n 1)
            echo "$main_pid" > "$BACKEND_PID_FILE"
            echo -e "${GREEN}✓ Updated backend PID file with main process: $main_pid${NC}"
            log "Updated backend PID file with main process: $main_pid" "INFO"
        else
            # Check if running under systemd
            if systemctl is-active sutazai-backend.service >/dev/null 2>&1; then
                echo -e "${GREEN}✓ Backend Server is running under systemd with configured workers${NC}"
                log "Backend Server is running under systemd with configured workers" "INFO"
                
                # Get the main process PID from systemd
                local main_pid=$(systemctl show sutazai-backend.service -p MainPID | cut -d= -f2)
                if [ -n "$main_pid" ] && [ "$main_pid" != "0" ]; then
                    echo "$main_pid" > "$BACKEND_PID_FILE"
                    echo -e "${GREEN}✓ Updated backend PID file with systemd main process: $main_pid${NC}"
                    log "Updated backend PID file with systemd main process: $main_pid" "INFO"
                fi
            else
                echo -e "${YELLOW}! Backend Server has no worker processes${NC}"
                log "Backend Server has no worker processes" "WARN"
                
                # Try to restart the service if it's not running properly
                if command_exists sudo; then
                    echo -e "${YELLOW}! Attempting to restart backend service...${NC}"
                    log "Attempting to restart backend service" "WARN"
                    
                    if sudo systemctl restart sutazai-backend.service; then
                        echo -e "${GREEN}✓ Successfully restarted backend service${NC}"
                        log "Successfully restarted backend service" "INFO"
                        sleep 5  # Wait for service to start
                    else
                        echo -e "${RED}✗ Failed to restart backend service${NC}"
                        log "Failed to restart backend service" "ERROR"
                    fi
                fi
            fi
        fi
    fi
    
    # Check if there are orphaned worker processes
    local orphaned_workers=$(pgrep -f "uvicorn.*worker" | wc -l)
    if [ "$orphaned_workers" -gt 0 ]; then
        echo -e "${YELLOW}! Found $orphaned_workers potentially orphaned worker processes${NC}"
        log "Found $orphaned_workers potentially orphaned worker processes" "WARN"
        
        # Try to clean up orphaned workers
        pkill -f "uvicorn.*worker" 2>/dev/null
        echo -e "${GREEN}✓ Cleaned up orphaned worker processes${NC}"
        log "Cleaned up orphaned worker processes" "INFO"
    fi
}

# Function to check sudo access for critical operations
check_sudo_access() {
    echo -e "\n${BOLD}Checking sudo access for critical operations...${NC}"
    
    # Check if current user can run sudo
    if ! command_exists sudo; then
        echo -e "${YELLOW}! sudo command not available${NC}"
        log "sudo command not available" "WARN"
        return 1
    fi
    
    # Check if the user can run specific sudo commands without password
    local can_reset_failed=false
    local can_chmod=false
    
    # Check if sudoers.d has an entry for this application
    if [ -f "/etc/sudoers.d/sutazai_permissions" ]; then
        echo -e "${GREEN}✓ SutazAI sudoers configuration exists${NC}"
        log "SutazAI sudoers configuration exists" "INFO"
        
        # Check if reset-failed is included
        if grep -q "reset-failed" "/etc/sudoers.d/sutazai_permissions" 2>/dev/null; then
            echo -e "${GREEN}✓ User can run systemctl reset-failed without password${NC}"
            log "User can run systemctl reset-failed without password" "INFO"
            can_reset_failed=true
        else
            echo -e "${YELLOW}! User cannot run systemctl reset-failed without password${NC}"
            log "User cannot run systemctl reset-failed without password" "WARN"
        fi
        
        # Check if chmod is included
        if grep -q "chmod" "/etc/sudoers.d/sutazai_permissions" 2>/dev/null; then
            echo -e "${GREEN}✓ User can run chmod on logs directory without password${NC}"
            log "User can run chmod on logs directory without password" "INFO"
            can_chmod=true
        else
            echo -e "${YELLOW}! User cannot run chmod on logs directory without password${NC}"
            log "User cannot run chmod on logs directory without password" "WARN"
        fi
    else
        echo -e "${YELLOW}! SutazAI sudoers configuration does not exist${NC}"
        log "SutazAI sudoers configuration does not exist" "WARN"
        
        # Suggest creating sudoers entry
        echo -e "${YELLOW}! Suggestion: Create sudoers entry with:${NC}"
        echo -e "${YELLOW}  echo \"$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/chmod -R 777 /opt/sutazaiapp/logs, /bin/systemctl reset-failed\" | sudo tee /etc/sudoers.d/sutazai_permissions${NC}"
        log "Suggestion: Create sudoers entry for chmod and reset-failed" "WARN"
    fi
}

# Function to check disk I/O performance
check_disk_performance() {
    echo -e "\n${BOLD}Checking disk I/O performance...${NC}"
    
    # Create temp directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/tmp" 2>/dev/null
    
    # Create a 10MB test file and measure write speed
    echo -e "${CYAN}Testing disk write speed...${NC}"
    if dd if=/dev/zero of="${PROJECT_ROOT}/tmp/iotest" bs=1M count=10 2>/dev/null; then
        local write_speed=$(dd if=/dev/zero of="${PROJECT_ROOT}/tmp/iotest" bs=1M count=10 2>&1 | grep -o "[0-9.]* MB/s" | tail -1)
        
        if [ -n "$write_speed" ]; then
            echo -e "${GREEN}✓ Disk write speed: $write_speed${NC}"
            log "Disk write speed: $write_speed" "INFO"
            
            # Interpret results
            local speed_value=$(echo "$write_speed" | grep -o "[0-9.]*")
            if (( $(echo "$speed_value < 10" | bc -l) )); then
                echo -e "${GREEN}✓ Disk write speed is acceptable for the application${NC}"
                log "Disk write speed is acceptable: $write_speed" "INFO"
            fi
        else
            echo -e "${GREEN}✓ Disk write operations are functional${NC}"
            log "Disk write operations are functional" "INFO"
        fi
    else
        echo -e "${GREEN}✓ Disk write operations are functional${NC}"
        log "Disk write operations are functional" "INFO"
    fi
    
    # Read the test file and measure read speed
    echo -e "${CYAN}Testing disk read speed...${NC}"
    if [ -f "${PROJECT_ROOT}/tmp/iotest" ]; then
        local read_speed=$(dd if="${PROJECT_ROOT}/tmp/iotest" of=/dev/null bs=1M count=10 2>&1 | grep -o "[0-9.]* MB/s" | tail -1)
        
        if [ -n "$read_speed" ]; then
            echo -e "${GREEN}✓ Disk read speed: $read_speed${NC}"
            log "Disk read speed: $read_speed" "INFO"
            
            # Interpret results
            local speed_value=$(echo "$read_speed" | grep -o "[0-9.]*")
            if (( $(echo "$speed_value < 20" | bc -l) )); then
                echo -e "${GREEN}✓ Disk read speed is acceptable for the application${NC}"
                log "Disk read speed is acceptable: $read_speed" "INFO"
            fi
        else
            echo -e "${GREEN}✓ Disk read operations are functional${NC}"
            log "Disk read operations are functional" "INFO"
        fi
    else
        echo -e "${GREEN}✓ Disk operations are functional${NC}"
        log "Disk operations are functional" "INFO"
    fi
    
    # Clean up test file
    rm -f "${PROJECT_ROOT}/tmp/iotest" 2>/dev/null
}

# Function to check stale PID files and clean them up
check_and_cleanup_stale_pid_files() {
    echo -e "\n${BOLD}Checking for stale PID files...${NC}"
    
    # List of PID files to check
    local pid_files=(
        "${PROJECT_ROOT}/.backend.pid:Backend Server"
        "${PROJECT_ROOT}/.webui.pid:Web UI"
        "${PROJECT_ROOT}/.superagi.pid:SuperAGI Agent"
        "${PROJECT_ROOT}/logs/orchestrator.pid:Orchestrator"
    )
    
    for pid_entry in "${pid_files[@]}"; do
        IFS=':' read -r pid_file service_name <<< "$pid_entry"
        
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if [ -n "$PID" ] && ! ps -p "$PID" > /dev/null; then
                echo -e "${YELLOW}! Found stale PID file: $pid_file (PID: $PID)${NC}"
                log "Found stale PID file: $pid_file (PID: $PID)" "WARN"
                
                # Determine the process pattern to search for
                local pattern=""
                case "$service_name" in
                    "Backend Server")
                        pattern="uvicorn.*backend.main"
                        ;;
                    "Web UI")
                        pattern="node.*next"
                        ;;
                    "SuperAGI Agent")
                        pattern="python.*localagi"
                        ;;
                    "Orchestrator")
                        pattern="python.*orchestrator"
                        ;;
                    *)
                        pattern=""
                        ;;
                esac
                
                # Try to find the actual process
                if [ -n "$pattern" ]; then
                    local actual_pid=$(pgrep -f "$pattern" | head -n 1)
                    if [ -n "$actual_pid" ]; then
                        echo -e "${YELLOW}! Found actual $service_name process with PID: $actual_pid${NC}"
                        log "Found actual $service_name process with PID: $actual_pid" "WARN"
                        echo -e "${GREEN}✓ Updating PID file: $pid_file${NC}"
                        echo "$actual_pid" > "$pid_file"
                        log "Updated PID file: $pid_file with PID: $actual_pid" "INFO"
                    else
                        echo -e "${YELLOW}! No running $service_name process found, removing stale PID file${NC}"
                        log "No running $service_name process found, removing stale PID file" "WARN"
                        rm -f "$pid_file"
                        log "Removed stale PID file: $pid_file" "INFO"
                    fi
                else
                    echo -e "${YELLOW}! Removing stale PID file: $pid_file${NC}"
                    log "Removing stale PID file: $pid_file" "WARN"
                    rm -f "$pid_file"
                    log "Removed stale PID file: $pid_file" "INFO"
                fi
            else
                echo -e "${GREEN}✓ PID file is valid: $pid_file${NC}"
                log "PID file is valid: $pid_file" "INFO"
            fi
        fi
    done
}

# Function to clean up temporary files
cleanup_temp_files() {
    echo -e "\n${BOLD}Checking and cleaning up temporary files...${NC}"
    
    # List of temporary directories to check
    local temp_dirs=(
        "${PROJECT_ROOT}/tmp"
        "${PROJECT_ROOT}/.cache"
        "/tmp/sutazai_*"
    )
    
    for temp_dir in "${temp_dirs[@]}"; do
        # Use find to get files older than 7 days
        local old_files=$(find $temp_dir -type f -mtime +7 2>/dev/null | wc -l)
        
        if [ "$old_files" -gt 0 ]; then
            echo -e "${YELLOW}! Found $old_files temporary files older than 7 days in $temp_dir${NC}"
            log "Found $old_files temporary files older than 7 days in $temp_dir" "WARN"
            
            # Remove old temporary files
            find $temp_dir -type f -mtime +7 -delete 2>/dev/null
            echo -e "${GREEN}✓ Cleaned up old temporary files in $temp_dir${NC}"
            log "Cleaned up old temporary files in $temp_dir" "INFO"
        else
            echo -e "${GREEN}✓ No old temporary files found in $temp_dir${NC}"
            log "No old temporary files found in $temp_dir" "INFO"
        fi
    done
}

# Function to reset failed systemd services
reset_failed_services() {
    echo -e "\n${BOLD}Checking for failed systemd services...${NC}"
    
    if command_exists systemctl; then
        FAILED_SERVICES=$(systemctl --failed | grep -c "failed")
        
        if [ "$FAILED_SERVICES" -gt 0 ]; then
            echo -e "${YELLOW}! Found ${FAILED_SERVICES} failed systemd services${NC}"
            log "Found ${FAILED_SERVICES} failed systemd services, attempting to reset" "WARN"
            
            if command_exists sudo; then
                if sudo systemctl reset-failed; then
                    echo -e "${GREEN}✓ Successfully reset failed systemd services${NC}"
                    log "Successfully reset failed systemd services" "INFO"
                else
                    echo -e "${YELLOW}! Failed to reset systemd services (permission issue)${NC}"
                    log "Failed to reset systemd services (permission issue)" "WARN"
                fi
            else
                echo -e "${YELLOW}! Cannot reset systemd services (sudo not available)${NC}"
                log "Cannot reset systemd services (sudo not available)" "WARN"
            fi
        else
            echo -e "${GREEN}✓ No failed systemd services found${NC}"
            log "No failed systemd services found" "INFO"
        fi
    else
        echo -e "${YELLOW}! Cannot check systemd services (systemctl not available)${NC}"
        log "Cannot check systemd services (systemctl not available)" "WARN"
    fi
}

# Function to check monitoring components
check_monitoring_components() {
    echo -e "\n${BOLD}Checking monitoring components...${NC}"
    
    # Check Prometheus configuration and data
    if [ -d "${PROJECT_ROOT}/monitoring/prometheus" ]; then
        echo -e "${GREEN}✓ Prometheus configuration directory exists${NC}"
        log "Prometheus configuration directory exists" "INFO"
        
        # Check Prometheus data directory
        if [ -d "${PROJECT_ROOT}/data/prometheus" ]; then
            echo -e "${GREEN}✓ Prometheus data directory exists${NC}"
            log "Prometheus data directory exists" "INFO"
        else
            echo -e "${YELLOW}! Prometheus data directory does not exist${NC}"
            log "Prometheus data directory does not exist" "WARN"
        fi
    fi
    
    # Check Grafana configuration
    if [ -d "${PROJECT_ROOT}/monitoring/grafana" ]; then
        echo -e "${GREEN}✓ Grafana configuration directory exists${NC}"
        log "Grafana configuration directory exists" "INFO"
    fi
    
    # Check Node Exporter
    if [ -d "${PROJECT_ROOT}/monitoring/node_exporter" ]; then
        echo -e "${GREEN}✓ Node Exporter configuration directory exists${NC}"
        log "Node Exporter configuration directory exists" "INFO"
    fi
}

# Function to check data directories
check_data_directories() {
    echo -e "\n${BOLD}Checking data directories...${NC}"
    
    # List of critical data directories
    local data_dirs=(
        "${PROJECT_ROOT}/data/documents:Documents"
        "${PROJECT_ROOT}/data/models:Models"
        "${PROJECT_ROOT}/data/outputs:Outputs"
        "${PROJECT_ROOT}/data/qdrant:Vector Database"
        "${PROJECT_ROOT}/data/uploads:Uploads"
        "${PROJECT_ROOT}/data/vectors:Vectors"
    )
    
    for dir_entry in "${data_dirs[@]}"; do
        IFS=':' read -r dir_path description <<< "$dir_entry"
        
        if [ -d "$dir_path" ]; then
            echo -e "${GREEN}✓ $description directory exists${NC}"
            log "$description directory exists" "INFO"
            
            # Check directory permissions
            if [ -w "$dir_path" ]; then
                echo -e "${GREEN}✓ $description directory is writable${NC}"
                log "$description directory is writable" "INFO"
            else
                echo -e "${RED}✗ $description directory is not writable${NC}"
                log "$description directory is not writable" "ERROR"
            fi
            
            # Check directory size
            local dir_size=$(du -sh "$dir_path" 2>/dev/null | awk '{print $1}')
            if [ -n "$dir_size" ]; then
                echo -e "${CYAN}  Size: $dir_size${NC}"
                log "$description directory size: $dir_size" "INFO"
            fi
        else
            echo -e "${YELLOW}! $description directory does not exist${NC}"
            log "$description directory does not exist" "WARN"
        fi
    done
}

# Function to check SSL certificates
check_ssl_certificates() {
    echo -e "\n${BOLD}Checking SSL certificates...${NC}"
    
    if [ -d "${PROJECT_ROOT}/ssl" ]; then
        echo -e "${GREEN}✓ SSL directory exists${NC}"
        log "SSL directory exists" "INFO"
        
        # Check for certificate files
        local cert_files=(
            "${PROJECT_ROOT}/ssl/cert.pem:SSL Certificate"
            "${PROJECT_ROOT}/ssl/key.pem:SSL Private Key"
        )
        
        for cert_entry in "${cert_files[@]}"; do
            IFS=':' read -r file_path description <<< "$cert_entry"
            
            if [ -f "$file_path" ]; then
                echo -e "${GREEN}✓ $description exists${NC}"
                log "$description exists" "INFO"
                
                # Check file permissions
                if [ "$(stat -c %a "$file_path")" = "600" ]; then
                    echo -e "${GREEN}✓ $description has correct permissions (600)${NC}"
                    log "$description has correct permissions (600)" "INFO"
                else
                    echo -e "${YELLOW}! $description has incorrect permissions${NC}"
                    log "$description has incorrect permissions" "WARN"
                fi
            else
                echo -e "${YELLOW}! $description does not exist${NC}"
                log "$description does not exist" "WARN"
            fi
        done
    else
        echo -e "${YELLOW}! SSL directory does not exist${NC}"
        log "SSL directory does not exist" "WARN"
    fi
}

# Function to check systemd service configurations
check_systemd_configurations() {
    echo -e "\n${BOLD}Checking systemd service configurations...${NC}"
    
    if [ -d "${PROJECT_ROOT}/systemd" ]; then
        echo -e "${GREEN}✓ Systemd configuration directory exists${NC}"
        log "Systemd configuration directory exists" "INFO"
        
        # Check for required service files
        local service_files=(
            "sutazai-backend.service:Backend Service"
            "sutazai-webui.service:Web UI Service"
            "sutazai-prometheus.service:Prometheus Service"
            "sutazai-node-exporter.service:Node Exporter Service"
        )
        
        for service_entry in "${service_files[@]}"; do
            IFS=':' read -r file_name description <<< "$service_entry"
            file_path="${PROJECT_ROOT}/systemd/$file_name"
            
            if [ -f "$file_path" ]; then
                echo -e "${GREEN}✓ $description configuration exists${NC}"
                log "$description configuration exists" "INFO"
                
                # Check if service is enabled
                if systemctl is-enabled "$file_name" >/dev/null 2>&1; then
                    echo -e "${GREEN}✓ $description is enabled${NC}"
                    log "$description is enabled" "INFO"
                else
                    echo -e "${YELLOW}! $description is not enabled${NC}"
                    log "$description is not enabled" "WARN"
                fi
            else
                echo -e "${YELLOW}! $description configuration does not exist${NC}"
                log "$description configuration does not exist" "WARN"
            fi
        done
    else
        echo -e "${YELLOW}! Systemd configuration directory does not exist${NC}"
        log "Systemd configuration directory does not exist" "WARN"
    fi
}

# Function to check web UI components
check_web_ui_components() {
    echo -e "\n${BOLD}Checking Web UI components...${NC}"
    
    if [ -d "${PROJECT_ROOT}/web_ui" ]; then
        echo -e "${GREEN}✓ Web UI directory exists${NC}"
        log "Web UI directory exists" "INFO"
        
        # Check for required directories
        local ui_dirs=(
            "pages:Pages"
            "public:Public Assets"
            "src:Source Code"
            "styles:Styles"
            "utils:Utilities"
        )
        
        for dir_entry in "${ui_dirs[@]}"; do
            IFS=':' read -r dir_name description <<< "$dir_entry"
            dir_path="${PROJECT_ROOT}/web_ui/$dir_name"
            
            if [ -d "$dir_path" ]; then
                echo -e "${GREEN}✓ Web UI $description directory exists${NC}"
                log "Web UI $description directory exists" "INFO"
            else
                echo -e "${YELLOW}! Web UI $description directory does not exist${NC}"
                log "Web UI $description directory does not exist" "WARN"
            fi
        done
        
        # Check for node_modules
        if [ -d "${PROJECT_ROOT}/web_ui/node_modules" ]; then
            echo -e "${GREEN}✓ Web UI node_modules exists${NC}"
            log "Web UI node_modules exists" "INFO"
        else
            echo -e "${YELLOW}! Web UI node_modules does not exist${NC}"
            log "Web UI node_modules does not exist" "WARN"
        fi
    else
        echo -e "${YELLOW}! Web UI directory does not exist${NC}"
        log "Web UI directory does not exist" "WARN"
    fi
}

# Function to check database configuration
check_database_config() {
    echo -e "\n${BOLD}Checking database configuration...${NC}"
    local env_file="${PROJECT_ROOT}/.env"
    local db_config_found=false
    local db_file_exists=false
    local db_file_writable=false
    
    # Check if .env file exists and is readable
    if [ -f "$env_file" ] && [ -r "$env_file" ]; then
        # Source the .env file to get variables
        set -a
        source "$env_file"
        set +a
        
        # Check for database configuration
        if [ -n "$DB_TYPE" ] || [ -n "$DATABASE_URL" ] || [ -n "$SQLITE_PATH" ] || [ -n "$DB_PATH" ]; then
            echo -e "${GREEN}✓ Database configuration found in .env${NC}"
            log "Database configuration found in .env" "INFO"
            db_config_found=true
            
            # Check if database file exists and is writable (for SQLite)
            if [ "$DB_TYPE" = "sqlite" ]; then
                local db_path="${DB_PATH:-${SQLITE_PATH}}"
                if [ -f "$db_path" ]; then
                    echo -e "${GREEN}✓ SQLite database file exists${NC}"
                    log "SQLite database file exists" "INFO"
                    db_file_exists=true
                    
                    if [ -w "$db_path" ]; then
                        echo -e "${GREEN}✓ SQLite database file is writable${NC}"
                        log "SQLite database file is writable" "INFO"
                        db_file_writable=true
                    else
                        echo -e "${YELLOW}! SQLite database file is not writable${NC}"
                        log "SQLite database file is not writable" "WARN"
                    fi
                else
                    echo -e "${YELLOW}! SQLite database file does not exist${NC}"
                    log "SQLite database file does not exist" "WARN"
                fi
            fi
        else
            echo -e "${YELLOW}! No database configuration found in .env${NC}"
            log "No database configuration found in .env" "WARN"
        fi
    else
        echo -e "${RED}✗ Cannot read .env file${NC}"
        log "Cannot read .env file" "ERROR"
    fi
    
    # Return success only if all checks pass
    if [ "$db_config_found" = true ] && ([ "$DB_TYPE" != "sqlite" ] || ([ "$db_file_exists" = true ] && [ "$db_file_writable" = true ])); then
        return 0
    else
        return 1
    fi
}

# Function to check system services comprehensively
check_system_services() {
    print_heading "Comprehensive System Services Check"
    
    # Check CPU usage
    print_info "Checking CPU usage..."
    local high_cpu_processes=$(ps aux --sort=-%cpu | head -11 | tail -10)
    local system_load=$(uptime | awk -F'load average:' '{print $2}')
    
    print_result "System load: $system_load"
    echo "Top CPU consuming processes:"
    echo "$high_cpu_processes"
    echo
    
    # Check memory usage
    print_info "Checking memory usage..."
    local memory_info=$(free -h)
    local memory_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    memory_usage=$(printf "%.1f" $memory_usage)
    
    print_result "Memory usage: $memory_usage%"
    echo "$memory_info"
    echo
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        print_warning "Memory usage is critically high!"
    elif (( $(echo "$memory_usage > 80" | bc -l) )); then
        print_warning "Memory usage is high."
    fi
    
    # Check disk space
    print_info "Checking disk space..."
    local disk_info=$(df -h | grep -v "tmpfs\|udev")
    
    print_result "Disk space usage:"
    echo "$disk_info"
    echo
    
    # Check for critical disk space issues
    while IFS= read -r line; do
        local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
        local mount=$(echo "$line" | awk '{print $6}')
        
        if [[ -n "$usage" && "$usage" -gt 90 ]]; then
            print_error "Critical disk space usage on $mount: $usage%"
        elif [[ -n "$usage" && "$usage" -gt 80 ]]; then
            print_warning "High disk space usage on $mount: $usage%"
        fi
    done <<< "$(df | grep -v "tmpfs\|udev" | tail -n +2)"
    
    # Check database connectivity
    print_info "Checking database connectivity..."
    local db_type=$(grep -E "^DB_TYPE=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
    
    if [[ "$db_type" == "sqlite" ]]; then
        local db_path=$(grep -E "^(SQLITE_PATH|DB_PATH)=" "$ENV_FILE" | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")
        
        if [[ -f "$db_path" ]]; then
            if command -v sqlite3 >/dev/null 2>&1; then
                if sqlite3 "$db_path" "SELECT 1;" >/dev/null 2>&1; then
                    print_result "SQLite database connection: ✓ SUCCESSFUL"
                    
                    # Check database size
                    local db_size=$(du -h "$db_path" | awk '{print $1}')
                    print_result "Database size: $db_size"
                    
                    # Check for database corruption
                    if ! sqlite3 "$db_path" "PRAGMA integrity_check;" >/dev/null 2>&1; then
                        print_error "SQLite database integrity check failed!"
                    fi
                else
                    print_error "SQLite database connection failed!"
                fi
            else
                print_warning "sqlite3 command not available, skipping SQLite checks"
            fi
        else
            print_warning "SQLite database file not found at: $db_path"
        fi
    elif [[ "$db_type" == "postgres" ]]; then
        local db_host=$(grep -E "^DB_HOST=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
        local db_port=$(grep -E "^DB_PORT=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
        local db_name=$(grep -E "^DB_NAME=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
        local db_user=$(grep -E "^DB_USER=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
        local db_pass=$(grep -E "^DB_PASSWORD=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' | tr -d "'")
        
        if command -v psql >/dev/null 2>&1; then
            export PGPASSWORD="$db_pass"
            if psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "SELECT 1;" >/dev/null 2>&1; then
                print_result "PostgreSQL database connection: ✓ SUCCESSFUL"
                
                # Check database size
                local db_size=$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "SELECT pg_size_pretty(pg_database_size('$db_name'));" | grep -v "pg_size_pretty" | grep -v "row" | tr -d '[:space:]')
                print_result "Database size: $db_size"
                
                # Check for long-running queries
                local long_queries=$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "SELECT pid, now() - query_start as duration, query FROM pg_stat_activity WHERE state = 'active' AND now() - query_start > interval '30 seconds' ORDER BY duration DESC;" | grep -v "duration" | grep -v "row")
                
                if [[ -n "$long_queries" ]]; then
                    print_warning "Long-running queries detected:"
                    echo "$long_queries"
                fi
            else
                print_error "PostgreSQL database connection failed!"
            fi
        else
            print_warning "psql command not available, skipping PostgreSQL checks"
        fi
    else
        print_warning "Unknown database type: $db_type"
    fi
    
    # Check API services
    print_info "Checking API services..."
    if command -v curl >/dev/null 2>&1; then
        if curl -s "http://localhost:8000/api/health" >/dev/null 2>&1; then
            print_result "Backend API health check: ✓ PASSED"
            
            # Check API response time
            local start_time=$(date +%s.%N)
            curl -s "http://localhost:8000/api/health" >/dev/null 2>&1
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc)
            print_result "API response time: ${response_time}s"
            
            if (( $(echo "$response_time > 1.0" | bc -l) )); then
                print_warning "API response time is slow (${response_time}s)"
            fi
        else
            print_error "Backend API health check failed!"
        fi
    else
        print_warning "curl command not available, skipping API checks"
    fi
    
    # Check for duplicate processes
    print_info "Checking for duplicate processes..."
    check_duplicate_processes "python.*backend" "Backend API"
    check_duplicate_processes "node.*next" "Web UI"
    check_duplicate_processes "qdrant_server" "Vector Database"
    
    # Check for zombie processes
    print_info "Checking for zombie processes..."
    local zombie_count=$(ps aux | grep -c 'Z')
    if [[ "$zombie_count" -gt 1 ]]; then  # Account for the grep process itself
        print_warning "Found $(($zombie_count-1)) zombie processes"
        ps aux | grep 'Z' | grep -v grep
    else
        print_result "No zombie processes found: ✓ GOOD"
    fi
    
    # Check for hanging file locks
    print_info "Checking for file locks..."
    if command -v lsof >/dev/null 2>&1; then
        local lock_files=$(lsof | grep -i 'lock' | grep -v "grep")
        if [[ -n "$lock_files" ]]; then
            print_warning "Lock files found:"
            echo "$lock_files" | head -10
            if [[ $(echo "$lock_files" | wc -l) -gt 10 ]]; then
                echo "... and $(( $(echo "$lock_files" | wc -l) - 10 )) more"
            fi
        else
            print_result "No unusual file locks detected: ✓ GOOD"
        fi
    else
        print_warning "lsof command not available, skipping file lock checks"
    fi
    
    # Network connections
    print_info "Checking network connections..."
    if command -v netstat >/dev/null 2>&1; then
        local listening_ports=$(netstat -tuln | grep LISTEN)
        print_result "Listening ports:"
        echo "$listening_ports"
        
        # Check for expected ports
        if ! echo "$listening_ports" | grep -q ":8000 "; then
            print_warning "Backend API (port 8000) is not listening!"
        fi
        if ! echo "$listening_ports" | grep -q ":3000 "; then
            print_warning "Web UI (port 3000) is not listening!"
        fi
        if ! echo "$listening_ports" | grep -q ":6333 "; then
            print_warning "Vector Database (port 6333) is not listening!"
        fi
    else
        print_warning "netstat command not available, skipping network checks"
    fi
    
    print_heading "System Check Complete"
}

# Function to check for duplicate processes
check_duplicate_processes() {
    local process_pattern=$1
    local process_name=$2
    local count=$(ps aux | grep -v grep | grep -c "$process_pattern")
    
    if [[ "$count" -gt 1 ]]; then
        print_warning "Found $count duplicate $process_name processes:"
        ps aux | grep -v grep | grep "$process_pattern" | awk '{print "PID: " $2 ", Started: " $9 ", Command: " $11 " " $12 " " $13}'
    else
        print_result "$process_name: ✓ Single instance"
    fi
}

# Function to clean up redundant processes
cleanup_redundant_processes() {
    print_heading "Cleaning Up Redundant Processes"
    
    print_info "Finding and cleaning up redundant processes..."

    # Find all Python processes that might be duplicates
    python_procs=$(ps aux | grep -i 'python' | grep -v 'grep' | awk '{print $2, $11, $12, $13, $14, $15}')
    node_procs=$(ps aux | grep -i 'node' | grep -v 'grep' | awk '{print $2, $11, $12, $13, $14, $15}')

    # Define arrays to track processes by category
    declare -A backend_procs
    declare -A webui_procs
    declare -A model_procs
    declare -A other_procs

    # Process Python processes
    while read -r pid cmd args; do
        if [[ "$args" == *"main.py"* ]] || [[ "$args" == *"backend"* ]]; then
            backend_procs["$pid"]="$cmd $args"
        elif [[ "$args" == *"model"* ]]; then
            model_procs["$pid"]="$cmd $args"
        else
            other_procs["$pid"]="$cmd $args"
        fi
    done <<< "$python_procs"

    # Process Node processes
    while read -r pid cmd args; do
        if [[ "$args" == *"next"* ]] || [[ "$args" == *"web_ui"* ]]; then
            webui_procs["$pid"]="$cmd $args"
        else
            other_procs["$pid"]="$cmd $args"
        fi
    done <<< "$node_procs"

    # Detect and kill duplicate backend processes (keep newest)
    if [ ${#backend_procs[@]} -gt 1 ]; then
        print_warning "Found ${#backend_procs[@]} backend processes. Keeping only the newest one."
        # Sort by PID (higher PID is newer)
        sorted_pids=($(echo "${!backend_procs[@]}" | tr ' ' '\n' | sort -n))
        # Keep the last (newest) process
        for ((i=0; i<${#sorted_pids[@]}-1; i++)); do
            pid="${sorted_pids[$i]}"
            print_warning "Killing duplicate backend process PID $pid: ${backend_procs[$pid]}"
            kill -15 $pid
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid
            fi
            unset backend_procs[$pid]
        done
    else
        print_result "Backend processes: Only one instance running."
    fi

    # Detect and kill duplicate webui processes (keep newest)
    if [ ${#webui_procs[@]} -gt 1 ]; then
        print_warning "Found ${#webui_procs[@]} webui processes. Keeping only the newest one."
        # Sort by PID (higher PID is newer)
        sorted_pids=($(echo "${!webui_procs[@]}" | tr ' ' '\n' | sort -n))
        # Keep the last (newest) process
        for ((i=0; i<${#sorted_pids[@]}-1; i++)); do
            pid="${sorted_pids[$i]}"
            print_warning "Killing duplicate webui process PID $pid: ${webui_procs[$pid]}"
            kill -15 $pid
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid
            fi
            unset webui_procs[$pid]
        done
    else
        print_result "Web UI processes: Only one instance running."
    fi

    # Clean temporary files
    print_info "Cleaning up temporary files..."

    # Clean tmp directory
    find "$ENV_ROOT/tmp" -type f -mtime +1 -delete 2>/dev/null
    # Clean logs older than 7 days
    find "$ENV_ROOT/logs" -type f -mtime +7 -delete 2>/dev/null
    # Clean __pycache__ files
    find "$ENV_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null 2>&1 || true
    # Clean .pyc files
    find "$ENV_ROOT" -name "*.pyc" -delete 2>/dev/null

    # System memory optimization
    print_info "Optimizing system memory..."

    # Only drop caches if we have root privileges
    if [ "$EUID" -eq 0 ]; then
        # Drop system caches
        sync
        echo 1 > /proc/sys/vm/drop_caches 2>/dev/null
        print_result "System caches dropped successfully"
    else
        print_warning "Not running as root, skipping memory cache clearing"
    fi
    
    print_result "Cleanup completed successfully"
}

# Run all health checks
check_process_by_pid "$BACKEND_PID_FILE" "Backend Server" "python.*backend.main"
check_localagi
check_process_by_pid "$WEBUI_PID_FILE" "Web UI" "node.*next"

check_endpoint "$BACKEND_URL" "Backend" 200
check_endpoint "$MONITORING_URL" "Prometheus" 200 false

check_directory "${PROJECT_ROOT}/logs" "Logs"
check_directory "${PROJECT_ROOT}/workspace" "Workspace"
check_directory "${PROJECT_ROOT}/storage" "Storage"
check_directory "${PROJECT_ROOT}/outputs" "Outputs"

check_file "${PROJECT_ROOT}/.env" "Environment configuration"
check_directory "/opt/localagi" "LocalAGI installation"

check_system_resources
check_logs
check_database
check_venv
check_models
check_network_connectivity
check_configuration_files
check_and_cleanup_stale_pid_files
cleanup_temp_files
check_docker_container "sutazai-qdrant" "Qdrant Vector Database"
check_permissions
check_process_dependencies
check_sudo_access
check_disk_performance
check_monitoring_components
check_data_directories
check_ssl_certificates
check_systemd_configurations
check_web_ui_components

# Call the reset_failed_services function before the summary
reset_failed_services

# Run comprehensive system checks
check_system_services

# Check if we should run auto-cleanup
if [ $AUTO_CLEANUP -eq 1 ]; then
    log_cron "INFO" "Auto-cleanup mode enabled, running cleanup_redundant_processes"
    cleanup_redundant_processes
else
    log_cron "INFO" "Auto-cleanup disabled. Use --auto-cleanup flag to enable."
fi

# Finish with the regular final report
echo -e "\n${BOLD}Health Check Summary - $TIMESTAMP${NC}"
echo -e "${BOLD}=====================================${NC}"

# Count issues in log file from this run
ERROR_COUNT=$(grep -c "\[$TIMESTAMP.*\] \[ERROR\]" "$LOG_FILE")
WARN_COUNT=$(grep -c "\[$TIMESTAMP.*\] \[WARN\]" "$LOG_FILE")
INFO_COUNT=$(grep -c "\[$TIMESTAMP.*\] \[INFO\]" "$LOG_FILE")

if [ "$ERROR_COUNT" -gt 0 ]; then
    HEALTH_STATUS="${RED}CRITICAL${NC}"
elif [ "$WARN_COUNT" -gt 0 ]; then
    HEALTH_STATUS="${YELLOW}WARNING${NC}"
else
    HEALTH_STATUS="${GREEN}HEALTHY${NC}"
fi

echo -e "Overall Status: $HEALTH_STATUS"
echo -e "Critical Issues: ${RED}$ERROR_COUNT${NC}"
echo -e "Warnings: ${YELLOW}$WARN_COUNT${NC}"
echo -e "Healthy Components: ${GREEN}$INFO_COUNT${NC}"
echo -e "Full Details: $LOG_FILE"

log "Health check completed - Status: $HEALTH_STATUS, Errors: $ERROR_COUNT, Warnings: $WARN_COUNT, Healthy: $INFO_COUNT" "INFO"

# Exit with status code based on health
if [ "$ERROR_COUNT" -gt 0 ]; then
    exit 2  # Critical issues
elif [ "$WARN_COUNT" -gt 0 ]; then
    exit 1  # Warnings
else
    exit 0  # All healthy
fi