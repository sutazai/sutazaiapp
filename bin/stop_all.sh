#!/bin/bash
# SutazAI Complete System Shutdown
# This script stops all components of the SutazAI system

APP_ROOT="/opt/sutazaiapp"
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
    local pid_file="$PIDS_DIR/$service_name.pid"
    
    print_message "Stopping $service_name..." "info"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null; then
            print_message "Stopping $service_name (PID: $pid)" "info"
            sudo kill "$pid"
            sleep 2
            
            # Check if still running
            if ps -p "$pid" > /dev/null; then
                print_message "Force killing $service_name (PID: $pid)" "warning"
                sudo kill -9 "$pid"
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
        print_message "No PID file found for $service_name" "warning"
        
        # Try to find and kill by pattern
        local pids
        case "$service_name" in
            "backend")
                # Find both uvicorn and gunicorn processes running the backend app
                pids=$(pgrep -f "(uvicorn|gunicorn).*backend.main:app")
                ;;
            "webui")
                pids=$(pgrep -f "webui\\|web_ui\\|npm run start")
                ;;
            "vector-db")
                pids=$(pgrep -f "qdrant\\|vector-db")
                ;;
            "localagi")
                pids=$(pgrep -f "localagi\\|minimal_server")
                ;;
        esac
        
        if [ -n "$pids" ]; then
            for pid in $pids; do
                print_message "Stopping $service_name process (PID: $pid)" "info"
                sudo kill "$pid" 2>/dev/null
                sleep 1
                if ps -p "$pid" > /dev/null; then
                    sudo kill -9 "$pid" 2>/dev/null
                fi
            done
            print_message "$service_name processes stopped" "info"
        else
            print_message "No running processes found for $service_name" "warning"
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

# Find all ollama model PID files
find "$PIDS_DIR" -name 'ollama_model_*.pid' -print0 | while IFS= read -r -d $'\0' pid_file; do
    if [ -f "$pid_file" ]; then
        model_pid=$(cat "$pid_file")
        model_name=$(basename "$pid_file" | sed -e 's/ollama_model_//' -e 's/.pid//') # Extract sanitized name
        print_message "-- Found PID file for model $model_name ($pid_file)" "info"
        if ps -p "$model_pid" > /dev/null; then
            print_message "   Stopping process (PID: $model_pid)" "info"
            kill "$model_pid" # Try graceful first
            sleep 2
            if ps -p "$model_pid" > /dev/null; then
                print_message "   Force killing process (PID: $model_pid)" "warning"
                kill -9 "$model_pid"
            fi
            print_message "   Process stopped." "info"
        else
            print_message "   Process (PID: $model_pid from file) not found." "warning"
        fi
        rm -f "$pid_file"
    fi
done

# Fallback: Kill any remaining 'ollama run' processes just in case
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