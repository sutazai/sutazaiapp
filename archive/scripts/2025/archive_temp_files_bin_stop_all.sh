#!/bin/bash
# SutazAI Complete System Shutdown
# This script stops all components of the SutazAI system

# Determine the absolute path of the script itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assume the script is in 'bin', so the root is one level up
APP_ROOT="$(dirname "$SCRIPT_DIR")"

# APP_ROOT="/opt/sutazaiapp" # REMOVED Hardcoded path
PIDS_DIR="$APP_ROOT/pids"
LOGS_DIR="$APP_ROOT/logs"

# Function to print colorized messages
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
}

# Function to stop a service
stop_service() {
    local service_name=$1
    # Use dynamic PIDS_DIR
    local pid_file="$PIDS_DIR/$service_name.pid"
    
    print_message "Stopping $service_name..." "info"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null; then
            print_message "Stopping $service_name (PID: $pid)" "info"
            # Consider removing sudo if not strictly necessary or handle permissions properly
            kill "$pid" # Attempt graceful kill
            sleep 2
            
            # Check if still running
            if ps -p "$pid" > /dev/null; then
                print_message "Force killing $service_name (PID: $pid)" "warning"
                kill -9 "$pid" # Force kill
                sleep 1
            fi
            
            if ! ps -p "$pid" > /dev/null; then
                print_message "$service_name stopped successfully" "info"
            else
                print_message "Failed to stop $service_name" "error"
            fi
        else
            print_message "No running process found for $service_name (PID: $pid)" "warning"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        print_message "No PID file found for $service_name at $pid_file" "warning" # Updated message
        
        # Try to find and kill by pattern (patterns might need adjustment based on actual process names)
        local pids
        case "$service_name" in
            "backend")
                # Update pattern if necessary to match how the backend is actually started (e.g., by start_backend.sh)
                pids=$(pgrep -f "(uvicorn|gunicorn).*$APP_ROOT/backend/main:app")
                ;;
            "webui")
                # Update pattern for Node.js/Next.js if needed
                pids=$(pgrep -f "node .*next dev.*$APP_ROOT/backend/ui")
                ;;
            "vector-db")
                pids=$(pgrep -f "qdrant/qdrant") # Keep this if Qdrant runs as a separate binary
                # Add pgrep for python script if it runs that way
                pids+=$(pgrep -f "python.*run_qdrant.py")
                ;;
            "localagi")
                # Update pattern if necessary
                pids=$(pgrep -f "python.*minimal_server.py.*$APP_ROOT") 
                ;;
             # Add cases for other services stopped by this script if applicable
        esac
        
        if [ -n "$pids" ]; then
            print_message "Found potential $service_name process(es) by pattern: $pids" "info"
            for pid in $pids; do
                print_message "Stopping $service_name process (PID: $pid)" "info"
                kill "$pid" 2>/dev/null
                sleep 1
                if ps -p "$pid" > /dev/null; then
                    kill -9 "$pid" 2>/dev/null
                fi
            done
            print_message "$service_name processes stopped via pattern matching" "info"
        else
            print_message "No running processes found for $service_name via pattern matching" "warning"
        fi
    fi
}

print_message "Stopping all SutazAI services..." "info"

# Stop services in reverse order (4->1)
# 4. Stop Web UI
stop_service "webui"

# 3. Stop Backend API
stop_service "backend"

# 2. Stop LocalAGI
stop_service "localagi"

# Stop all Ollama model runner processes managed by start_all.sh
print_message "Stopping Ollama model processes..." "info"

# --- New Graceful Stop Logic ---
if command -v ollama &> /dev/null; then
    # Get list of currently loaded models from ollama ps (NAME column)
    # Skip the header line (NAME...), extract first field (model name before :) 
    mapfile -t loaded_models < <(ollama ps 2>/dev/null | awk 'NR>1 {split($1,a,":"); print a[1]}' | sort -u)
    
    if [ ${#loaded_models[@]} -gt 0 ]; then
        print_message "Found loaded Ollama models: ${loaded_models[*]}" "info"
        print_message "Attempting graceful stop using 'ollama stop'..." "info"
        for model_name in "${loaded_models[@]}"; do
            if [[ -n "$model_name" ]]; then # Ensure model name is not empty
                print_message "  Stopping: ollama stop $model_name" "info"
                ollama stop "$model_name" 
                # No need for long sleep, stop should be relatively quick
            fi
        done
        print_message "Ollama graceful stop commands issued." "info"
        sleep 5 # Give a few seconds for models to unload
    else
        print_message "No models reported as loaded by 'ollama ps'." "info"
    fi
else
    print_message "Ollama command not found. Skipping graceful Ollama stop." "warning"
fi
# --- End of New Logic ---

# Fallback: Kill any remaining 'ollama run' processes just in case
print_message "Checking for remaining 'ollama run' processes (fallback)..." "info"
pids=$(pgrep -f "ollama run")
if [ -n "$pids" ]; then
    print_message "Found remaining 'ollama run' processes via pgrep. Attempting to stop..." "warning"
    for pid in $pids; do
        print_message "   Stopping Ollama run process (PID: $pid)" "info"
        kill "$pid" 2>/dev/null
        sleep 1
        if ps -p "$pid" > /dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    print_message "   Remaining Ollama run process(es) stopped." "info"
else
    print_message "No remaining 'ollama run' processes found via pgrep." "info"
fi

# 1. Stop Vector Database
stop_service "vector-db"

# Check for Docker containers
if command -v docker &> /dev/null; then
    if docker ps --filter name=qdrant -q &>/dev/null; then
        print_message "Stopping Qdrant Docker container..." "info"
        docker stop qdrant
        print_message "Qdrant Docker container stopped" "info"
    fi
fi

# Final verification that all processes are stopped
print_message "Verifying all services are stopped..." "info"
sleep 2

# Use updated pattern for backend verification
backends_running=$(pgrep -f "(uvicorn|gunicorn).*backend.main:app" | wc -l)
webui_running=$(pgrep -f "webui\\|web_ui\\|npm run start" | wc -l)
vectordb_running=$(pgrep -f "qdrant\\|vector-db" | wc -l)
localagi_running=$(pgrep -f "localagi\\|minimal_server" | wc -l)

total_running=$((backends_running + webui_running + vectordb_running + localagi_running))

if [ $total_running -eq 0 ]; then
    print_message "All SutazAI services successfully stopped." "info"
else
    print_message "Warning: $total_running service processes still running:" "warning"
    
    if [ $backends_running -gt 0 ]; then
        print_message " - Backend API: $backends_running process(es)" "warning"
    fi
    
    if [ $webui_running -gt 0 ]; then
        print_message " - Web UI: $webui_running process(es)" "warning"
    fi
    
    if [ $vectordb_running -gt 0 ]; then
        print_message " - Vector Database: $vectordb_running process(es)" "warning"
    fi
    
    if [ $localagi_running -gt 0 ]; then
        print_message " - LocalAGI: $localagi_running process(es)" "warning"
    fi
    
    print_message "You may need to manually kill these processes." "warning"
fi

print_message "System shutdown complete." "info" 