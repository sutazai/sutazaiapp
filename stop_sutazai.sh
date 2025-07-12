#!/bin/bash

# SutazAI - Stop Script
# Gracefully stops all SutazAI services

set -e

SUTAZAI_ROOT="/opt/sutazaiapp"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log "🛑 Stopping SutazAI services..."

# Stop FastAPI backend
if [ -f "$SUTAZAI_ROOT/fastapi.pid" ]; then
    FASTAPI_PID=$(cat "$SUTAZAI_ROOT/fastapi.pid")
    if ps -p $FASTAPI_PID > /dev/null; then
        kill $FASTAPI_PID
        success "FastAPI backend stopped (PID: $FASTAPI_PID)"
    else
        warning "FastAPI backend was not running"
    fi
    rm -f "$SUTAZAI_ROOT/fastapi.pid"
else
    warning "FastAPI PID file not found"
fi

# Stop Streamlit frontend
if [ -f "$SUTAZAI_ROOT/streamlit.pid" ]; then
    STREAMLIT_PID=$(cat "$SUTAZAI_ROOT/streamlit.pid")
    if ps -p $STREAMLIT_PID > /dev/null; then
        kill $STREAMLIT_PID
        success "Streamlit frontend stopped (PID: $STREAMLIT_PID)"
    else
        warning "Streamlit frontend was not running"
    fi
    rm -f "$SUTAZAI_ROOT/streamlit.pid"
else
    warning "Streamlit PID file not found"
fi

# Clean up any remaining processes
pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true
pkill -f "uvicorn api.main:app" 2>/dev/null || true

# Remove status file
rm -f "$SUTAZAI_ROOT/service_status.json"

success "🎉 All SutazAI services stopped successfully!"

echo ""
echo "📊 Service Status: STOPPED"
echo "⏰ Stopped at: $(date)"
echo ""
echo "To restart SutazAI, run: ./start_sutazai.sh"