#!/bin/bash
# title        :stop_sutazai.sh
# description  :This script stops all SutazAI services
# author       :SutazAI Team
# version      :2.2
# usage        :bash scripts/stop_sutazai.sh [--force]

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Parse command-line arguments
FORCE=false
VERBOSE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Function to get PID of a process by pattern
get_process_pid() {
    local pattern="$1"
    pgrep -f "$pattern" | grep -v "$$" | tr '\n' ' '
}

# Function to stop a service
stop_service() {
    local service_name="$1"
    local pattern="$2"
    local pid_file="pids/${service_name}.pid"
    local pids=""
    
    # First, check if the PID file exists
    if [ -f "$pid_file" ]; then
        local file_pid=$(cat "$pid_file")
        if [ -z "$file_pid" ]; then
            log "Empty PID file found for $service_name"
            rm -f "$pid_file"
        elif ps -p "$file_pid" > /dev/null 2>&1; then
            [ "$VERBOSE" = true ] && log "Found PID $file_pid from file for $service_name"
            pids="$file_pid"
        else
            log "$service_name is not running (stale PID file)"
            rm -f "$pid_file"
        fi
    else
        [ "$VERBOSE" = true ] && log "No PID file found for $service_name"
    fi
    
    # Also find processes by pattern
    if [ ! -z "$pattern" ]; then
        local pattern_pids=$(get_process_pid "$pattern")
        if [ ! -z "$pattern_pids" ]; then
            [ "$VERBOSE" = true ] && log "Found PIDs by pattern for $service_name: $pattern_pids"
            # Merge PIDs, avoiding duplicates
            for pid in $pattern_pids; do
                if [[ ! " $pids " =~ " $pid " ]]; then
                    pids="$pids $pid"
                fi
            done
            pids=$(echo $pids | xargs) # Trim
        fi
    fi
    
    # If we don't have any PIDs, service is not running
    if [ -z "$pids" ]; then
        log "$service_name is not running"
        return 0
    fi
    
    # Stop each process
    log "Stopping $service_name (PIDs: $pids)..."
    for pid in $pids; do
        # Send SIGTERM first
        kill $pid 2>/dev/null
        
        # Wait for process to terminate
        local count=0
        while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
            [ "$VERBOSE" = true ] && log "Waiting for PID $pid to terminate ($count/10)..."
        done
        
        # If process is still running and --force is specified, use SIGKILL
        if ps -p $pid > /dev/null 2>&1; then
            if [ "$FORCE" = true ]; then
                log "Force stopping PID $pid..."
                kill -9 $pid 2>/dev/null
                sleep 1
            else
                log "WARNING: PID $pid is still running. Use --force to force stop."
            fi
        fi
        
        if ! ps -p $pid > /dev/null 2>&1; then
            log "PID $pid stopped successfully"
        fi
    done
    
    # Check if any processes are still running
    local remaining_pids=""
    for pid in $pids; do
        if ps -p $pid > /dev/null 2>&1; then
            remaining_pids="$remaining_pids $pid"
        fi
    done
    
    # Clean up PID file regardless
    [ -f "$pid_file" ] && rm -f "$pid_file"
    
    if [ -z "$remaining_pids" ]; then
        log "$service_name stopped successfully"
        return 0
    else
        log "WARNING: Some $service_name processes are still running (PIDs:$remaining_pids)"
        return 1
    fi
}

log "Stopping SutazAI services..."

# Stop services in reverse order of startup
log "Stopping Web UI..."
stop_service "webui" "streamlit run web_ui/app.py"

log "Stopping Backend API..."
stop_service "backend" "python.*backend.main"

log "Stopping Vector Database..."
stop_service "vector-db" "python.*vector_db"

# Also check for child processes that might need to be stopped
log "Cleaning up child processes..."
for pid_file in pids/*.pid; do
    if [ -f "$pid_file" ]; then
        pkill -P $(cat "$pid_file") 2>/dev/null || true
    fi
done

# Make sure no processes are left running (safety net)
# Only do this if --force is specified
if [ "$FORCE" = true ]; then
    log "Checking for remaining processes..."
    
    # Kill any leftover streamlit processes
    pkill -f "streamlit run web_ui/app.py" 2>/dev/null || true
    
    # Kill any leftover backend processes
    pkill -f "python -m backend.main" 2>/dev/null || true
    
    # Kill any leftover vector database processes
    pkill -f "python.*vector_db" 2>/dev/null || true
    
    log "Force cleanup complete"
fi

log "All SutazAI services stopped"
log "To restart services, run: ./scripts/start_sutazai.sh"
exit 0 