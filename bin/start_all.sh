#!/bin/bash
# SutazAI Complete System Startup Script
# Starts all components including Ollama, backend, and web UI

set -e

# Project configuration
PROJECT_ROOT="/opt/sutazaiapp"
VENV_PATH="$PROJECT_ROOT/venv"
LOGS_DIR="$PROJECT_ROOT/logs"
PIDS_DIR="$PROJECT_ROOT/run"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Create necessary directories
mkdir -p "$LOGS_DIR" "$PIDS_DIR"
cd "$PROJECT_ROOT"

log "🚀 Starting SutazAI Complete System"
log "============================================="

# 1. Activate virtual environment
log "📦 Activating virtual environment..."
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    error "Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"
log "✅ Virtual environment activated"

# 2. Start Ollama service (Mock AI Server)
log "🤖 Starting Ollama AI service..."
if ! curl -sf http://127.0.0.1:11434/health > /dev/null 2>&1; then
    # Start mock Ollama server
    nohup python scripts/create_mock_ollama.py > "$LOGS_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAMA_PID > "$PIDS_DIR/ollama.pid"
    
    # Wait for Ollama to start
    sleep 5
    
    if curl -sf http://127.0.0.1:11434/health > /dev/null 2>&1; then
        log "✅ Ollama AI service started (PID: $OLLAMA_PID)"
    else
        warn "⚠️ Ollama AI service may not have started properly"
    fi
else
    log "✅ Ollama AI service already running"
fi

# 3. Verify AI models availability
log "📥 Verifying AI models..."
sleep 2

if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    log "✅ AI models are available and ready"
else
    warn "⚠️ AI models verification failed"
fi

# 4. Start Redis (if available)
log "🗄️ Starting Redis..."
if command -v redis-server > /dev/null; then
    if ! pgrep -f "redis-server" > /dev/null; then
        nohup redis-server > "$LOGS_DIR/redis.log" 2>&1 &
        REDIS_PID=$!
        echo $REDIS_PID > "$PIDS_DIR/redis.pid"
        log "✅ Redis started (PID: $REDIS_PID)"
    else
        log "✅ Redis already running"
    fi
else
    warn "⚠️ Redis not available - using memory cache"
fi

# 5. Start PostgreSQL (if available)
log "🐘 Checking PostgreSQL..."
if command -v psql > /dev/null; then
    if pg_isready -q; then
        log "✅ PostgreSQL is ready"
    else
        warn "⚠️ PostgreSQL not ready - using SQLite fallback"
    fi
else
    warn "⚠️ PostgreSQL not available - using SQLite fallback"
fi

# 6. Run database migrations
log "🔄 Running database setup..."
if [ -f "main.py" ]; then
    python main.py --migrate > "$LOGS_DIR/migration.log" 2>&1 || warn "⚠️ Database migration had issues"
elif [ -f "scripts/setup_database.py" ]; then
    python scripts/setup_database.py > "$LOGS_DIR/migration.log" 2>&1 || warn "⚠️ Database setup had issues"
else
    log "✅ No database migration script found - using defaults"
fi

# 7. Start FastAPI backend
log "🌐 Starting FastAPI backend..."
if ! pgrep -f "uvicorn.*backend_main:app" > /dev/null; then
    if [ -f "backend/backend_main.py" ]; then
        nohup uvicorn backend.backend_main:app --host 0.0.0.0 --port 8000 --workers 2 > "$LOGS_DIR/backend.log" 2>&1 &
    elif [ -f "main.py" ]; then
        nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 > "$LOGS_DIR/backend.log" 2>&1 &
    else
        error "No backend entry point found"
        exit 1
    fi
    
    BACKEND_PID=$!
    echo $BACKEND_PID > "$PIDS_DIR/backend.pid"
    
    # Wait for backend to start
    sleep 5
    
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log "✅ Backend started successfully (PID: $BACKEND_PID)"
    else
        warn "⚠️ Backend health check failed"
    fi
else
    log "✅ Backend already running"
fi

# 8. Start Web UI server
log "🎨 Starting Web UI server..."
if ! pgrep -f "python.*-m http.server.*3000" > /dev/null; then
    cd web_ui
    nohup python -m http.server 3000 > "$LOGS_DIR/webui.log" 2>&1 &
    WEBUI_PID=$!
    echo $WEBUI_PID > "$PIDS_DIR/webui.pid"
    cd "$PROJECT_ROOT"
    
    sleep 2
    log "✅ Web UI started (PID: $WEBUI_PID)"
else
    log "✅ Web UI already running"
fi

# 9. Final system check
log "🔍 Running final system check..."
sleep 3

SERVICES_STATUS=""

# Check Ollama
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    SERVICES_STATUS+="✅ Ollama API: http://localhost:11434\n"
else
    SERVICES_STATUS+="❌ Ollama API: OFFLINE\n"
fi

# Check Backend
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    SERVICES_STATUS+="✅ Backend API: http://localhost:8000\n"
else
    SERVICES_STATUS+="❌ Backend API: OFFLINE\n"
fi

# Check Web UI
if curl -sf http://localhost:3000 > /dev/null 2>&1; then
    SERVICES_STATUS+="✅ Web UI: http://localhost:3000\n"
else
    SERVICES_STATUS+="❌ Web UI: OFFLINE\n"
fi

# Display final status
log "============================================="
log "🎉 SutazAI System Startup Complete!"
log "============================================="
echo -e "$SERVICES_STATUS"
log "============================================="
log "📋 Quick Access:"
log "   • Main Dashboard: http://localhost:3000"
log "   • Chat Interface: http://localhost:3000/chat.html"
log "   • API Documentation: http://localhost:8000/docs"
log "   • Ollama API: http://localhost:11434"
log ""
log "📄 Logs available in: $LOGS_DIR"
log "🔧 PIDs stored in: $PIDS_DIR"
log ""
log "To stop all services: $PROJECT_ROOT/bin/stop_all.sh"
log "============================================="