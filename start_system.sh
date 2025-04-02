#!/bin/bash
# SutazAI System Starter Script

APP_ROOT="/opt/sutazaiapp"
LOG_DIR="$APP_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=== SutazAI System Starter ==="
echo "$(date) - Starting system check and initialization"

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

# Kill any stale processes
print_message "Checking and killing stale processes..." "info"
for port in 3000 8000 6333 8090; do
  pid=$(lsof -ti :$port)
  if [ -n "$pid" ]; then
    print_message "Found process using port $port (PID: $pid), killing it..." "warning"
    kill -9 $pid 2>/dev/null
  fi
done

# Ensure the virtual environment is activated
print_message "Activating virtual environment..." "info"
source /opt/venv-sutazaiapp/bin/activate

# Rebuild the Web UI if needed
print_message "Checking and rebuilding Web UI if needed..." "info"
cd "$APP_ROOT/web_ui"
if [ ! -d ".next" ] || [ ! -f ".next/server/pages/_app.js" ]; then
  print_message "Web UI needs rebuilding..." "warning"
  npm run build
  if [ $? -ne 0 ]; then
    print_message "Failed to build Web UI" "error"
  else
    print_message "Web UI built successfully" "info"
  fi
fi

# Return to app root
cd "$APP_ROOT"

# Start the system using the official script
print_message "Starting all SutazAI services..." "info"
bash "$APP_ROOT/bin/start_all.sh"

# Verify services
print_message "Verifying services..." "info"
sleep 5

check_service() {
  local name=$1
  local port=$2
  local pid=$(lsof -ti :$port)
  
  if [ -n "$pid" ]; then
    print_message "$name is running (PID: $pid)" "info"
    return 0
  else
    print_message "$name is NOT running" "error"
    return 1
  fi
}

# Check each service
check_service "Backend API" 8000
check_service "Web UI" 3000
check_service "Vector Database" 6333
check_service "LocalAGI" 8090

print_message "System check completed" "info"
print_message "Access the web interface at: http://localhost:3000" "info"
print_message "API endpoints available at: http://localhost:8000" "info"

echo "=== SutazAI System Starter Complete ===" 