#!/bin/bash
# title        :cleanup_redundancies.sh
# description  :This script performs a comprehensive cleanup of redundant processes and optimizes system performance
# author       :SutazAI Team
# version      :1.0
# usage        :sudo bash scripts/cleanup_redundancies.sh
# notes        :Requires bash 4.0+ and standard Linux utilities

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo or as root"
    exit 1
fi

print_status "Starting comprehensive system cleanup and optimization"

# Secure process management initialization
init_process_management() {
    # Create secure shared memory directory
    SHMEM_DIR="/dev/shm/sutazai-$RANDOM"
    mkdir -p "$SHMEM_DIR"
    chmod 1700 "$SHMEM_DIR"
    chattr +i "$SHMEM_DIR"
    
    # Create protected service lock directory
    SERVICE_LOCK_DIR="$SHMEM_DIR/.service_locks"
    mkdir -p "$SERVICE_LOCK_DIR"
    chmod 1700 "$SERVICE_LOCK_DIR"
    
    # Load AI model securely
    AI_MODEL_DIR="/etc/sutazai/models/process_optimizer"
    if [ ! -d "$AI_MODEL_DIR" ]; then
        print_error "Security-critical AI models missing from $AI_MODEL_DIR"
        exit 1
    fi
    
    # Initialize resource arbitration database
    RESOURCE_DB="$SHMEM_DIR/resources.db"
    sqlite3 "$RESOURCE_DB" "CREATE TABLE IF NOT EXISTS resource_alloc (
        service TEXT PRIMARY KEY,
        cpu REAL,
        memory INTEGER,
        priority INTEGER,
        last_used INTEGER
    );"
    
    # Load security policies
    SECURITY_PROFILES=(
        "backend:cpu=30,mem=4096,cap=NET_BIND_SERVICE"
        "webui:cpu=15,mem=2048,cap=CHOWN"
        "qdrant:cpu=40,mem=8192,cap=SYS_NICE"
    )
}

# Enhanced service locking with resource limits
service_lock() {
    local service_name=$1
    local lock_file="$SERVICE_LOCK_DIR/${service_name}.lock"
    local timeout=15  # Increased timeout for resource negotiation
    local max_retries=3
    local retry_count=0
    
    # Get security profile for service
    local profile=$(printf '%s\n' "${SECURITY_PROFILES[@]}" | grep "^$service_name:")
    local cpu_limit=$(echo "$profile" | cut -d: -f2 | cut -d= -f2)
    local mem_limit=$(echo "$profile" | cut -d: -f3 | cut -d= -f2)
    local capabilities=$(echo "$profile" | cut -d: -f4 | cut -d= -f2)

    while [ $retry_count -lt $max_retries ]; do
        # Create lock file with process metadata
        exec 200>"$lock_file"
        flock -w $timeout -x 200 && {
            # Write process metadata to lock file
            echo "PID=$$" > "$lock_file"
            echo "TIMESTAMP=$(date +%s)" >> "$lock_file"
            echo "RESOURCE_ALLOC=CPU:$cpu_limit,MEM:$mem_limit" >> "$lock_file"
            
            # Check resource availability
            local alloc_status=$(sqlite3 "$RESOURCE_DB" \
                "INSERT INTO resource_alloc VALUES('$service_name', $cpu_limit, $mem_limit, 0, $(date +%s))
                ON CONFLICT(service) DO UPDATE SET last_used=$(date +%s)
                RETURNING 1;")
                
            if [ "$alloc_status" -eq 1 ]; then
                # Set Linux capabilities
                if [ -n "$capabilities" ]; then
                    /sbin/setcap "$capabilities"+ep $$ || {
                        print_error "Failed to set capabilities for $service_name"
                        return 2
                    }
                fi
                
                # Start heartbeat monitor
                local heartbeat_file="$SERVICE_LOCK_DIR/${service_name}.heartbeat"
                ( while true; do
                    echo "ALIVE $(date +%s)" > "$heartbeat_file"
                    sleep 5
                done ) &
                
                return 0
            fi
            
            print_warning "Resource constraints violated for $service_name"
            flock -u 200
            rm -f "$lock_file"
            retry_count=$((retry_count+1))
            sleep 1
        } || {
            print_error "Lock acquisition failed for $service_name (attempt $((retry_count+1)))"
            retry_count=$((retry_count+1))
        }
    done
    
    print_error "Failed to acquire lock after $max_retries attempts"
    return 1
}

service_unlock() {
    flock -u 200
    rm -f "$lock_file"
}

# AI-powered process analysis
analyze_process_health() {
    local pid=$1
    local service_type=$2
    
    # Get detailed process metrics
    local metrics=$(ps -p $pid -o %cpu,%mem,rss,vsz,etime,pcpu,pmem --no-headers)
    local open_files=$(lsof -p $pid 2>/dev/null | wc -l)
    local threads=$(ps -L -p $pid | wc -l)
    
    # Use lightweight AI model for anomaly detection
    local anomaly_score=$(
        python3 -c "import sys; from ai_process_optimizer import evaluate_process; \
        print(evaluate_process('$metrics',$open_files,$threads,'$service_type'))" 2>/dev/null
    )
    
    # Normalize score between 0-1
    anomaly_score=$(echo "$anomaly_score" | awk '{print $1+0}')
    
    if (( $(echo "$anomaly_score > 0.85" | bc -l) )); then
        print_warning "High anomaly score ($anomaly_score) detected for PID $pid"
        return 1
    fi
    return 0
}

# Modern process cleanup with AI analysis
cleanup_redundant_processes() {
    init_process_management
    
    # Get services from systemd
    local services=($(systemctl list-units --type=service --state=running \
                    | grep -E 'sutazai-.*\.service' | awk '{print $1}'))
    
    for service in "${services[@]}"; do
        if service_lock "$service"; then
            local main_pid=$(systemctl show -p MainPID "$service" | awk -F= '{print $2}')
            local service_type=$(echo "$service" | sed 's/sutazai-\(.*\)\.service/\1/')
            
            if ! analyze_process_health "$main_pid" "$service_type"; then
                print_status "Restarting problematic service: $service"
                systemctl restart "$service"
            fi
            
            # Clean duplicate instances
            pids=($(pgrep -f "$service_type" | grep -v "$main_pid"))
            for pid in "${pids[@]}"; do
                if analyze_process_health "$pid" "$service_type"; then
                    print_status "Found duplicate $service_type process (PID $pid)"
                    kill -TERM "$pid"
                    sleep 0.5
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid"
                    fi
                fi
            done
            
            service_unlock
        fi
    done
}

# Step 1: AI-powered process optimization
print_status "Starting AI-enhanced process optimization..."
cleanup_redundant_processes

# Detect and kill duplicate webui processes (keep newest)
if [ ${#webui_procs[@]} -gt 1 ]; then
    print_warning "Found ${#webui_procs[@]} webui processes. Keeping only the newest one."
    # Sort by PID (higher PID is newer)
    sorted_pids=($(echo "${!webui_procs[@]}" | tr ' ' '\n' | sort -n))
    # Keep the last (newest) process
    for ((i=0; i<${#sorted_pids[@]}-1; i++)); do
        pid="${sorted_pids[$i]}"
        print_status "Killing duplicate webui process PID $pid: ${webui_procs[$pid]}"
        kill -15 $pid
        sleep 1
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
        fi
        unset webui_procs[$pid]
    done
fi

print_success "Process cleanup completed"

# Step 2: Clean up temporary files
print_status "Cleaning up temporary files..."

# Clean tmp directory
find "$PROJECT_ROOT/tmp" -type f -mtime +1 -delete 2>/dev/null
# Clean logs older than 7 days
find "$PROJECT_ROOT/logs" -type f -mtime +7 -delete 2>/dev/null
# Clean __pycache__ files
find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
# Clean .pyc files
find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null

print_success "Temporary file cleanup completed"

# Step 3: Database optimization
print_status "Optimizing database..."

# Check database type from .env
DB_TYPE=$(grep "^DB_TYPE=" "$PROJECT_ROOT/.env" | cut -d= -f2)

if [[ "$DB_TYPE" == "sqlite" ]]; then
    DB_PATH=$(grep "^SQLITE_PATH=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    if [[ -f "$DB_PATH" ]]; then
        print_status "Optimizing SQLite database at $DB_PATH"
        # Create temporary SQL file
        TMP_SQL="$PROJECT_ROOT/tmp/optimize.sql"
        echo "VACUUM;" > "$TMP_SQL"
        echo "ANALYZE;" >> "$TMP_SQL"
        # Run optimization
        sqlite3 "$DB_PATH" < "$TMP_SQL"
        rm -f "$TMP_SQL"
        print_success "SQLite database optimized"
    else
        print_warning "SQLite database file not found at $DB_PATH"
    fi
elif [[ "$DB_TYPE" == "postgres" ]]; then
    DB_HOST=$(grep "^DB_HOST=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    DB_PORT=$(grep "^DB_PORT=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    DB_NAME=$(grep "^DB_NAME=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    DB_USER=$(grep "^DB_USER=" "$PROJECT_ROOT/.env" | cut -d= -f2)
    
    if [[ -n "$DB_HOST" && -n "$DB_PORT" && -n "$DB_NAME" && -n "$DB_USER" ]]; then
        print_status "Optimizing PostgreSQL database $DB_NAME on $DB_HOST:$DB_PORT"
        # Check if pg_dump is available
        if command -v psql >/dev/null 2>&1; then
            export PGPASSWORD=$(grep "^DB_PASSWORD=" "$PROJECT_ROOT/.env" | cut -d= -f2)
            psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;"
            print_success "PostgreSQL database optimized"
        else
            print_warning "PostgreSQL client tools not installed, skipping database optimization"
        fi
    else
        print_warning "Incomplete PostgreSQL configuration, skipping database optimization"
    fi
else
    print_warning "Unknown database type: $DB_TYPE, skipping database optimization"
fi

# Step 4: System memory optimization
print_status "Optimizing system memory..."

# Drop system caches (requires sudo)
sync
echo 3 > /proc/sys/vm/drop_caches

# Set memory cleanup interval
if ! grep -q "vm.dirty_ratio" /etc/sysctl.conf; then
    echo "vm.dirty_ratio = 10" >> /etc/sysctl.conf
    echo "vm.dirty_background_ratio = 5" >> /etc/sysctl.conf
    sysctl -p
fi

print_success "System memory optimized"

# Step 5: Check for and fix conflicting services
print_status "Checking for conflicting services..."

# Check ports in use
check_port() {
    local port=$1
    local service=$2
    local force_terminate=${3:-no}  # Optional third parameter to force termination without prompting
    
    netstat -tuln | grep -q ":$port "
    if [ $? -eq 0 ]; then
        local pid=$(lsof -i :$port -t 2>/dev/null)
        if [ -z "$pid" ]; then
            print_warning "Port $port is in use but could not determine the process"
            return
        fi
        
        local process=$(ps -p $pid -o comm= 2>/dev/null)
        
        if [[ "$process" != *"$service"* ]]; then
            print_warning "Port $port is used by '$process' (PID $pid) instead of '$service'"
            
            if [[ "$force_terminate" == "yes" ]]; then
                print_status "Force terminating process on port $port (PID: $pid)"
                kill -15 $pid 2>/dev/null
                sleep 2
                if kill -0 $pid 2>/dev/null; then
                    kill -9 $pid 2>/dev/null
                fi
                print_success "Process on port $port terminated"
            else
                # Check if we're running in a terminal
                if [ -t 1 ]; then
                    read -p "Do you want to stop this process? (y/n): " choice
                    if [[ "$choice" == "y" ]]; then
                        kill -15 $pid 2>/dev/null
                        sleep 2
                        if kill -0 $pid 2>/dev/null; then
                            kill -9 $pid 2>/dev/null
                        fi
                        print_success "Process on port $port terminated"
                    fi
                else
                    # Not in a terminal, just log the issue
                    print_warning "Not terminating conflicting process (non-interactive mode)"
                fi
            fi
        else
            print_status "Port $port is correctly used by '$service'"
        fi
    fi
}

# Function to add for cron job checks
is_cron_job() {
    # Check if this script is being run from cron by checking parent process
    if ps -o comm= -p $PPID | grep -q "cron"; then
        return 0  # True, this is a cron job
    else
        return 1  # False, not a cron job
    fi
}

# Check if this is a cron job
if is_cron_job; then
    # Set non-interactive mode
    print_status "Running in cron job mode (non-interactive)"
    IS_CRON=1
else
    IS_CRON=0
fi

# Check main service ports
check_port 8000 "python" $([[ $IS_CRON -eq 1 ]] && echo "yes" || echo "no")  # Backend API
check_port 3000 "node" $([[ $IS_CRON -eq 1 ]] && echo "yes" || echo "no")    # Web UI
check_port 6333 "qdrant" $([[ $IS_CRON -eq 1 ]] && echo "yes" || echo "no")  # Vector store

print_success "Service conflict check completed"

# Step 6: Create a status report
print_status "Creating system status report..."

REPORT_FILE="$PROJECT_ROOT/system_status_report.txt"

{
    echo "SutazAI System Status Report"
    echo "============================="
    echo "Date: $(date)"
    echo ""
    
    echo "1. Running Processes"
    echo "-------------------"
    echo "Backend processes: ${#backend_procs[@]}"
    for pid in "${!backend_procs[@]}"; do
        echo "  - PID $pid: ${backend_procs[$pid]}"
    done
    echo "WebUI processes: ${#webui_procs[@]}"
    for pid in "${!webui_procs[@]}"; do
        echo "  - PID $pid: ${webui_procs[$pid]}"
    done
    echo "Model processes: ${#model_procs[@]}"
    for pid in "${!model_procs[@]}"; do
        echo "  - PID $pid: ${model_procs[$pid]}"
    done
    echo ""
    
    echo "2. System Resources"
    echo "------------------"
    echo "Memory usage:"
    free -h
    echo ""
    echo "Disk usage:"
    df -h | grep -v "tmpfs"
    echo ""
    
    echo "3. Service Status"
    echo "----------------"
    echo "Backend API (Port 8000): $(netstat -tuln | grep -q ":8000 " && echo "RUNNING" || echo "STOPPED")"
    echo "Web UI (Port 3000): $(netstat -tuln | grep -q ":3000 " && echo "RUNNING" || echo "STOPPED")"
    echo "Vector Store (Port 6333): $(netstat -tuln | grep -q ":6333 " && echo "RUNNING" || echo "STOPPED")"
    echo ""
    
    echo "4. Database"
    echo "-----------"
    echo "Type: $DB_TYPE"
    if [[ "$DB_TYPE" == "sqlite" ]]; then
        echo "Path: $DB_PATH"
        echo "Size: $(du -h "$DB_PATH" 2>/dev/null | cut -f1)"
    elif [[ "$DB_TYPE" == "postgres" ]]; then
        echo "Host: $DB_HOST:$DB_PORT"
        echo "Database: $DB_NAME"
    fi
    echo ""
    
} > "$REPORT_FILE"

print_success "System status report created at $REPORT_FILE"

# Final success message
print_success "System cleanup and optimization completed successfully!"
echo -e "${BOLD}The system has been optimized for better performance.${NC}"
echo -e "You can view the system status report at: $REPORT_FILE"

exit 0 