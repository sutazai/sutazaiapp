#!/bin/bash
# SutazAI Complete System Startup
# This script starts all components of the SutazAI system

APP_ROOT="/opt/sutazaiapp"
BIN_DIR="$APP_ROOT/bin"
SCRIPTS_DIR="$APP_ROOT/scripts"
LOGS_DIR="$APP_ROOT/logs"
STATUS_FILE="$LOGS_DIR/system_status.json"
PIDS_DIR="$APP_ROOT/pids"
BACKEND_LOG="/home/sutazaidev/sutazai_backend.log"
WEBUI_LOG="$LOGS_DIR/webui.log"
OPTIMIZER_LOG="$LOGS_DIR/optimizer.log"
SYSTEM_STATUS_FILE="$LOGS_DIR/system_status.json"

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"

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

# Check if we need to run as sudo (only for Docker)
if command -v docker &> /dev/null && ! docker info &>/dev/null; then
    print_message "Warning: Docker detected but may require elevated privileges." "warning"
    print_message "Running with sudo may be required for Docker operations." "warning"
fi

# Run system optimizer first
print_message "Running system optimizer..." "info"
if [ -f "$SCRIPTS_DIR/system_optimizer.py" ]; then
    # Activate the virtual environment
    source "$APP_ROOT/venv-sutazaiapp/bin/activate"
    cd "$APP_ROOT" && python3 "$SCRIPTS_DIR/system_optimizer.py"
    # Deactivate after use (optional, but good practice)
    # deactivate # <-- REMOVED: Keep venv active for subsequent scripts
else
    print_message "System optimizer not found. Skipping optimization." "warning"
fi

# Start services in the correct order (Ensure they run in VENV)
print_message "Starting SutazAI services..." "info"

# # 1. Start Vector Database (Qdrant) first - TEMPORARILY DISABLED
# print_message "1. Starting Vector Database (Qdrant)..." "info"
# if [ -f "$BIN_DIR/start_vector_db.sh" ]; then
#     # Ensure script runs in venv if it uses Python internally
#     source "$APP_ROOT/venv-sutazaiapp/bin/activate" && bash "$BIN_DIR/start_vector_db.sh"
# else
#     print_message "Vector database startup script not found. Skipping." "error"
# fi
# sleep 5 # Wait for DB
# print_message "Vector Database startup completed." "info"
# 
# sleep 2 # Add delay before next service

# # 2. Start LocalAGI - TEMPORARILY DISABLED
# print_message "2. Starting LocalAGI service..." "info"
# if [ -f "$BIN_DIR/start_localagi.sh" ]; then
#     source "$APP_ROOT/venv-sutazaiapp/bin/activate" && bash "$BIN_DIR/start_localagi.sh"
# else
#     print_message "LocalAGI startup script not found. Skipping." "error"
# fi
# 
# sleep 2 # Add delay before next service

# 3. Start Backend API Directly
print_message "3. Starting Backend API Directly..." "info"
cd "$APP_ROOT"
BACKEND_PID_FILE="$PIDS_DIR/backend.pid"

# Stop existing backend process if PID file exists (Keep this part)
if [ -f "$BACKEND_PID_FILE" ]; then
    pid_to_kill=$(cat "$BACKEND_PID_FILE")
    if ps -p "$pid_to_kill" > /dev/null; then
        print_message "Stopping existing backend process (PID: $pid_to_kill)" "warning"
        # Try graceful kill first, then force kill
        sudo kill "$pid_to_kill" 2>/dev/null || sudo kill -9 "$pid_to_kill" 2>/dev/null
        sleep 1
    fi
    rm -f "$BACKEND_PID_FILE"
fi

# Activate venv (needed for start_backend.sh)
source "$APP_ROOT/venv-sutazaiapp/bin/activate"

# --- REMOVED Gunicorn Start & PID Handling --- 

# --- Call start_backend.sh and check its exit status --- 
# Ensure start_backend.sh is executable and run it
if [ -x "$BIN_DIR/start_backend.sh" ]; then
    print_message "Executing start_backend.sh..." "info"
    bash "$BIN_DIR/start_backend.sh" # Run the correct script
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        print_message "Error: start_backend.sh reported an error (exit code $EXIT_CODE). Check logs." "error"
        # Optional: uncomment to stop if backend fails
        # exit 1 
    else
        print_message "start_backend.sh completed successfully." "info"
        # --- Start Ollama Models (Added Section) ---
        print_message "Attempting to pre-load Ollama models..." "info"
        if command -v ollama &> /dev/null; then
            # Wait a moment for ollama serve (started within start_backend.sh or elsewhere) to be ready
            sleep 5 
            
            # Check if ollama serve is actually running (optional but recommended)
            if pgrep -f "ollama serve" > /dev/null; then
                print_message "Ollama service detected. Getting list of local models..." "info"
                
                # Get list of models (extract only the name part before the colon)
                mapfile -t local_models < <(ollama list | awk -F: '{print $1}' | grep -v '^NAME' | sort -u)
                
                if [ ${#local_models[@]} -gt 0 ]; then
                    print_message "Found local models: ${local_models[*]}" "info"
                    print_message "Starting 'ollama run' for each model in the background..." "warning"
                    print_message "(This may consume significant RAM/VRAM)" "warning"
                    
                    for model_name in "${local_models[@]}"; do
                        if [[ -n "$model_name" ]]; then # Ensure model name is not empty
                            print_message "  Starting: ollama run $model_name \"\" &" "info"
                            # Run the model with an empty prompt to load it, in the background
                            # Redirect stdout and stderr to /dev/null to suppress output
                            ollama run "$model_name" "" > /dev/null 2>&1 &
                            sleep 2 # Small delay between starting each model
                        fi
                    done
                    print_message "Ollama model pre-loading initiated." "info"
                else
                    print_message "No local Ollama models found via 'ollama list'." "warning"
                fi
            else
                print_message "Ollama service (ollama serve) does not appear to be running. Skipping model pre-loading." "error"
            fi
        else
            print_message "Ollama command not found. Skipping Ollama model pre-loading." "warning"
        fi
        # --- End of Added Section ---
    fi
else
    print_message "Error: start_backend.sh not found or not executable!" "error"
    # Optional: uncomment to stop if script is missing
    # exit 1 
fi

sleep 2 # Keep delay before next service

# 4. Start Web UI
print_message "4. Starting Web UI..." "info"
if [ -f "$BIN_DIR/start_webui.sh" ]; then
    # WebUI is Node.js, doesn't need venv activation
    bash "$BIN_DIR/start_webui.sh"
else
    print_message "Web UI startup script not found. Skipping." "error"
fi

# Check all services are running
print_message "Verifying all services..." "info"
sleep 5

# Create a status report
cat > "$STATUS_FILE" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "services": {
    "backend": {
      "running": $(if [ -f "$PIDS_DIR/backend.pid" ] && ps -p $(cat "$PIDS_DIR/backend.pid") > /dev/null; then echo "true"; else echo "false"; fi),
      "port": 8000,
      "url": "http://localhost:8000",
      "health_endpoint": "/health"
    },
    "webui": {
      "running": $(if pgrep -f "next dev" > /dev/null; then echo "true"; else echo "false"; fi),
      "port": 8501,
      "url": "http://localhost:8501"
    },
    "vector_db": {
      "running": $(if pgrep -f "qdrant/qdrant" > /dev/null || pgrep -f "run_qdrant.py" > /dev/null; then echo "true"; else echo "false"; fi),
      "port": 6333,
      "url": "http://localhost:6333"
    },
    "localagi": {
      "running": $(if pgrep -f "minimal_server.py" > /dev/null; then echo "true"; else echo "false"; fi),
      "port": 8090,
      "url": "http://localhost:8090",
      "health_endpoint": "/health"
    }
  },
  "system": {
    "hostname": "$(hostname)",
    "cpu_usage": "$(top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}')%",
    "memory_usage": "$(free -m | grep Mem | awk '{print int($3*100/$2)}')%",
    "disk_usage": "$(df -h / | grep / | awk '{print $5}')"
  }
}
EOF

# Output summary
print_message "\n=== SutazAI System Status ===" "info"
print_message "Backend API:      $(if [ -f "$PIDS_DIR/backend.pid" ] && ps -p $(cat "$PIDS_DIR/backend.pid") > /dev/null; then echo "[RUNNING]"; else echo "[STOPPED]"; fi)" "$(if [ -f "$PIDS_DIR/backend.pid" ] && ps -p $(cat "$PIDS_DIR/backend.pid") > /dev/null; then echo "info"; else echo "error"; fi)"
print_message "Web UI:           $(if pgrep -f "next dev" > /dev/null; then echo "[RUNNING]"; else echo "[STOPPED]"; fi)" "$(if pgrep -f "next dev" > /dev/null; then echo "info"; else echo "error"; fi)"
print_message "Vector Database:  $(if pgrep -f "qdrant/qdrant" > /dev/null || pgrep -f "run_qdrant.py" > /dev/null; then echo "[RUNNING]"; else echo "[STOPPED]"; fi)" "$(if pgrep -f "qdrant/qdrant" > /dev/null || pgrep -f "run_qdrant.py" > /dev/null; then echo "info"; else echo "error"; fi)"
print_message "LocalAGI:         $(if pgrep -f "minimal_server.py" > /dev/null; then echo "[RUNNING]"; else echo "[STOPPED]"; fi)" "$(if pgrep -f "minimal_server.py" > /dev/null; then echo "info"; else echo "error"; fi)"
print_message "===========================" "info"
print_message "\nAccess the web interface at: http://localhost:8501" "info"
print_message "API endpoints available at: http://localhost:8000" "info"
print_message "System status saved to: $STATUS_FILE" "info"
print_message "\nTo check individual service logs, see the files in: $LOGS_DIR and $BACKEND_LOG" "info"

# Make all scripts executable
chmod +x "$BIN_DIR"/*.sh 