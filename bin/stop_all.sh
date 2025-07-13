#!/bin/bash
# SutazAI Complete System Shutdown Script
# Gracefully stops all components

set -e

# Project configuration
PROJECT_ROOT="/opt/sutazaiapp"
PIDS_DIR="$PROJECT_ROOT/run"
LOGS_DIR="$PROJECT_ROOT/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log "🛑 Stopping SutazAI Complete System"
log "====================================="

# Function to stop service by PID file
stop_service() {
    local service_name=$1
    local pid_file="$PIDS_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log "Stopping $service_name (PID: $pid)..."
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait up to 10 seconds for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing $service_name..."
                kill -KILL "$pid" 2>/dev/null || true
            fi
            
            log "✅ $service_name stopped"
        else
            warn "⚠️ $service_name PID not found or already stopped"
        fi
        rm -f "$pid_file"
    else
        warn "⚠️ No PID file for $service_name"
    fi
}

# Function to stop service by process name
stop_by_process() {
    local process_pattern=$1
    local service_name=$2
    
    local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Stopping $service_name..."
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        local remaining_pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            warn "Force killing $service_name..."
            echo "$remaining_pids" | xargs kill -KILL 2>/dev/null || true
        fi
        log "✅ $service_name stopped"
    else
        log "✅ $service_name not running"
    fi
}

# Create PIDs directory if it doesn't exist
mkdir -p "$PIDS_DIR"

# 1. Stop Web UI
log "🎨 Stopping Web UI..."
stop_service "webui"
stop_by_process "python.*-m http.server.*3000" "Web UI"

# 2. Stop Backend
log "🌐 Stopping FastAPI Backend..."
stop_service "backend"
stop_by_process "uvicorn.*main:app" "Backend"

# 3. Stop Redis
log "🗄️ Stopping Redis..."
stop_service "redis"
stop_by_process "redis-server" "Redis"

# 4. Stop Ollama
log "🤖 Stopping Ollama..."
stop_service "ollama"
stop_by_process "ollama serve" "Ollama"

# 5. Clean up any remaining Python processes
log "🧹 Cleaning up remaining processes..."
stop_by_process "python.*sutazai" "SutazAI Python processes"

# 6. Clean up PID directory
log "🗂️ Cleaning up PID files..."
rm -rf "$PIDS_DIR"/*.pid 2>/dev/null || true

# 7. Final verification
log "🔍 Verifying shutdown..."
remaining_processes=$(pgrep -f "(ollama|uvicorn.*main|redis-server|python.*sutazai)" 2>/dev/null || true)
if [ -n "$remaining_processes" ]; then
    warn "⚠️ Some processes may still be running:"
    echo "$remaining_processes" | while read pid; do
        ps -p "$pid" -o pid,comm,args --no-headers 2>/dev/null || true
    done
else
    log "✅ All processes stopped successfully"
fi

log "====================================="
log "🎯 SutazAI System Shutdown Complete!"
log "====================================="
log "📄 Logs preserved in: $LOGS_DIR"
log "🚀 To restart: $PROJECT_ROOT/bin/start_all.sh"
log "====================================="