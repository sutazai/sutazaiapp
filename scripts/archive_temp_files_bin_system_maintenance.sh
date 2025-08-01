#!/bin/bash
# SutazAI System Maintenance Script
# This script performs routine maintenance tasks for the SutazAI system

APP_ROOT="/opt/sutazaiapp"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
BACKUP_DIR="$APP_ROOT/backups"
MAX_LOG_SIZE=50000000  # ~50MB
MAX_LOG_FILES=10
MAINTENANCE_LOG="$LOGS_DIR/maintenance.log"

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"
mkdir -p "$BACKUP_DIR"

# Function to log messages
log_message() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" >> "$MAINTENANCE_LOG"
    echo "[$timestamp] $1"
}

# Function to colorize output
print_message() {
    local color_code="\033[0;32m"  # Green
    local reset_code="\033[0m"
    
    if [ "$2" = "error" ]; then
        color_code="\033[0;31m"  # Red
    elif [ "$2" = "warning" ]; then
        color_code="\033[0;33m"  # Yellow
    elif [ "$2" = "info" ]; then
        color_code="\033[0;34m"  # Blue
    fi
    
    echo -e "${color_code}$1${reset_code}"
    log_message "$1"
}

# Start maintenance
print_message "Starting SutazAI system maintenance..." "info"

# 1. Check system resources
print_message "Checking system resources..." "info"
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
memory_usage=$(free -m | awk '/Mem/{printf "%.2f", $3*100/$2}')
disk_usage=$(df -h $APP_ROOT | awk 'NR==2 {print $5}' | sed 's/%//')

if (( $(echo "$cpu_usage > 90" | bc -l) )); then
    print_message "WARNING: High CPU usage: ${cpu_usage}%" "warning"
fi

if (( $(echo "$memory_usage > 90" | bc -l) )); then
    print_message "WARNING: High memory usage: ${memory_usage}%" "warning"
fi

if (( disk_usage > 85 )); then
    print_message "WARNING: High disk usage: ${disk_usage}%" "warning"
    
    # Clean up old log files if disk space is low
    if (( disk_usage > 90 )); then
        print_message "Critical disk space - cleaning old logs" "warning"
        find "$LOGS_DIR" -name "*.log.*" -type f -mtime +7 -delete
        find "$LOGS_DIR" -name "*.log" -type f -size +${MAX_LOG_SIZE}c -exec truncate -s 1M {} \;
    fi
fi

# 2. Check and clean stale PID files
print_message "Checking for stale PID files..." "info"
for pid_file in "$PIDS_DIR"/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        service_name=$(basename "$pid_file" .pid)
        
        if ! ps -p "$pid" > /dev/null; then
            print_message "Removing stale PID file for $service_name (PID: $pid)" "warning"
            rm "$pid_file"
        else
            print_message "$service_name is running (PID: $pid)" "info"
        fi
    fi
done

# 3. Rotate log files
print_message "Rotating log files..." "info"
for log_file in "$LOGS_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        log_name=$(basename "$log_file")
        log_size=$(stat -c%s "$log_file")
        
        if [ "$log_size" -gt "$MAX_LOG_SIZE" ]; then
            timestamp=$(date "+%Y%m%d-%H%M%S")
            print_message "Rotating log file: $log_name (Size: $(numfmt --to=iec-i --suffix=B $log_size))" "info"
            
            # Remove oldest log if we have too many
            old_logs=$(ls -t "${log_file}."* 2>/dev/null | tail -n +$MAX_LOG_FILES)
            for old_log in $old_logs; do
                print_message "Removing old log: $(basename "$old_log")" "info"
                rm "$old_log"
            done
            
            # Create new rotated log
            mv "$log_file" "${log_file}.${timestamp}"
            touch "$log_file"
            chmod 644 "$log_file"
        fi
    fi
done

# 4. Check service health
print_message "Checking service health..." "info"

# Check vector database (Qdrant)
if pgrep -f "qdrant" > /dev/null || docker ps --filter name=qdrant -q > /dev/null 2>&1; then
    print_message "Vector database (Qdrant) is running" "info"
    
    # Test connection
    if command -v curl > /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/health > /dev/null 2>&1; then
            print_message "Vector database API is accessible" "info"
        else
            print_message "WARNING: Vector database API is not responding" "warning"
        fi
    fi
else
    print_message "Vector database (Qdrant) is not running" "warning"
fi

# Check LocalAGI
if pgrep -f "localagi" > /dev/null; then
    print_message "LocalAGI is running" "info"
    
    # Test connection
    if command -v curl > /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8090/health > /dev/null 2>&1; then
            print_message "LocalAGI API is accessible" "info"
        else
            print_message "WARNING: LocalAGI API is not responding" "warning"
        fi
    fi
else
    print_message "LocalAGI is not running" "warning"
fi

# Check Backend API
if pgrep -f "backend.main" > /dev/null; then
    print_message "Backend API is running" "info"
    
    # Test connection
    if command -v curl > /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health > /dev/null 2>&1; then
            print_message "Backend API is accessible" "info"
        else
            print_message "WARNING: Backend API is not responding" "warning"
        fi
    fi
else
    print_message "Backend API is not running" "warning"
fi

# Check Web UI
if pgrep -f "webui\|web_ui\|npm run start" > /dev/null; then
    print_message "Web UI is running" "info"
    
    # Test connection
    if command -v curl > /dev/null; then
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 > /dev/null 2>&1; then
            print_message "Web UI is accessible" "info"
        else
            print_message "WARNING: Web UI is not responding" "warning"
        fi
    fi
else
    print_message "Web UI is not running" "warning"
fi

# 5. Create system backup
timestamp=$(date "+%Y%m%d-%H%M%S")
backup_file="$BACKUP_DIR/sutazai_config_backup_${timestamp}.tar.gz"

print_message "Creating system configuration backup..." "info"
tar -czf "$backup_file" \
    -C "$APP_ROOT" config/ \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="*.log" \
    --exclude="*.pid" \
    --exclude="node_modules" \
    --exclude=".git" 2>/dev/null

if [ -f "$backup_file" ]; then
    backup_size=$(stat -c%s "$backup_file")
    print_message "Backup created: $(basename "$backup_file") (Size: $(numfmt --to=iec-i --suffix=B $backup_size))" "info"
    
    # Clean up old backups, keep last 10
    old_backups=$(ls -t "$BACKUP_DIR"/sutazai_config_backup_*.tar.gz 2>/dev/null | tail -n +11)
    for old_backup in $old_backups; do
        print_message "Removing old backup: $(basename "$old_backup")" "info"
        rm "$old_backup"
    done
else
    print_message "WARNING: Backup creation failed" "warning"
fi

# 6. Check Python dependencies
if [ -f "$APP_ROOT/requirements.txt" ]; then
    print_message "Checking Python dependencies..." "info"
    missing_deps=$(python3 -c "
import pkg_resources, sys
with open('$APP_ROOT/requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
for req in requirements:
    try:
        pkg_resources.require(req)
    except Exception as e:
        print(req)
" 2>/dev/null)

    if [ -n "$missing_deps" ]; then
        print_message "WARNING: Missing or incompatible Python dependencies found:" "warning"
        for dep in $missing_deps; do
            print_message " - $dep" "warning"
        done
    else
        print_message "All Python dependencies are satisfied" "info"
    fi
fi

# 7. Optimize Database if applicable
if [ -d "$APP_ROOT/database" ]; then
    print_message "Checking database..." "info"
    
    # For SQLite databases
    sqlite_dbs=$(find "$APP_ROOT/database" -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" 2>/dev/null)
    if [ -n "$sqlite_dbs" ]; then
        if command -v sqlite3 > /dev/null; then
            for db in $sqlite_dbs; do
                print_message "Optimizing SQLite database: $(basename "$db")" "info"
                sqlite3 "$db" "VACUUM; ANALYZE;" > /dev/null 2>&1
            done
        else
            print_message "SQLite databases found, but sqlite3 command is not available" "warning"
        fi
    fi
fi

# 8. Run system optimizer if it exists
if [ -f "$APP_ROOT/bin/system_optimizer.py" ]; then
    print_message "Running system optimizer..." "info"
    python3 "$APP_ROOT/bin/system_optimizer.py" >> "$MAINTENANCE_LOG" 2>&1
    if [ $? -eq 0 ]; then
        print_message "System optimizer completed successfully" "info"
    else
        print_message "WARNING: System optimizer reported errors" "warning"
    fi
fi

# 9. Check for software updates
print_message "Checking for updates..." "info"

# Check for Git repositories
if command -v git > /dev/null; then
    git_repos=$(find "$APP_ROOT" -name ".git" -type d -maxdepth 3 2>/dev/null)
    for repo_dir in $git_repos; do
        repo_path=$(dirname "$repo_dir")
        repo_name=$(basename "$repo_path")
        print_message "Checking for updates in repository: $repo_name" "info"
        
        (cd "$repo_path" && git fetch -q)
        local_rev=$(cd "$repo_path" && git rev-parse HEAD)
        remote_rev=$(cd "$repo_path" && git rev-parse @{u} 2>/dev/null)
        
        if [ $? -eq 0 ] && [ "$local_rev" != "$remote_rev" ]; then
            print_message "Updates available for $repo_name" "info"
        fi
    done
fi

# 10. Restart services that need it based on health checks
if [ "$1" = "--auto-restart" ]; then
    print_message "Checking if services need to be restarted..." "info"
    
    # Vector DB check and restart
    if pgrep -f "qdrant" > /dev/null || docker ps --filter name=qdrant -q > /dev/null 2>&1; then
        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/health > /dev/null 2>&1; then
            print_message "Restarting Vector Database service..." "warning"
            if [ -f "$APP_ROOT/bin/stop_all.sh" ] && [ -f "$APP_ROOT/bin/start_vector_db.sh" ]; then
                "$APP_ROOT/bin/stop_all.sh" > /dev/null 2>&1
                sleep 2
                "$APP_ROOT/bin/start_vector_db.sh" > /dev/null 2>&1
            fi
        fi
    fi
    
    # Only restart services that are found to be running but not responsive
    for service in "localagi" "backend" "webui"; do
        pid_file="$PIDS_DIR/${service}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null; then
                # Service is running, check if responsive
                case "$service" in
                    "localagi")
                        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8090/health > /dev/null 2>&1; then
                            print_message "Restarting LocalAGI service..." "warning"
                            kill "$pid" > /dev/null 2>&1
                            sleep 2
                            if [ -f "$APP_ROOT/bin/start_localagi.sh" ]; then
                                "$APP_ROOT/bin/start_localagi.sh" > /dev/null 2>&1
                            fi
                        fi
                        ;;
                    "backend")
                        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health > /dev/null 2>&1; then
                            print_message "Restarting Backend API service..." "warning"
                            kill "$pid" > /dev/null 2>&1
                            sleep 2
                            if [ -f "$APP_ROOT/bin/start_backend.sh" ]; then
                                "$APP_ROOT/bin/start_backend.sh" > /dev/null 2>&1
                            fi
                        fi
                        ;;
                    "webui")
                        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 > /dev/null 2>&1; then
                            print_message "Restarting Web UI service..." "warning"
                            kill "$pid" > /dev/null 2>&1
                            sleep 2
                            if [ -f "$APP_ROOT/bin/start_webui.sh" ]; then
                                "$APP_ROOT/bin/start_webui.sh" > /dev/null 2>&1
                            fi
                        fi
                        ;;
                esac
            fi
        fi
    done
fi

print_message "System maintenance completed successfully" "info"

# Output system status summary
print_message "\nSystem Status Summary:" "info"
print_message "CPU Usage: ${cpu_usage}%" "info"
print_message "Memory Usage: ${memory_usage}%" "info"
print_message "Disk Usage: ${disk_usage}%" "info"
print_message "Active Services:" "info"

running_services=0
if pgrep -f "qdrant" > /dev/null || docker ps --filter name=qdrant -q > /dev/null 2>&1; then
    print_message " - Vector Database (Qdrant)" "info"
    running_services=$((running_services + 1))
fi
if pgrep -f "localagi\|minimal_server" > /dev/null; then
    print_message " - LocalAGI" "info"
    running_services=$((running_services + 1))
fi
if pgrep -f "backend.main" > /dev/null; then
    print_message " - Backend API" "info"
    running_services=$((running_services + 1))
fi
if pgrep -f "webui\|web_ui\|npm run start" > /dev/null; then
    print_message " - Web UI" "info"
    running_services=$((running_services + 1))
fi

print_message "Total running services: $running_services/4" "info"
print_message "System maintenance log: $MAINTENANCE_LOG" "info" 