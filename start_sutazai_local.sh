#!/bin/bash
# SutazAI Local Startup Script
# Starts the system without Docker for immediate use

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="logs/startup_$(date +%Y%m%d_%H%M%S).log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

# Setup
setup_directories() {
    mkdir -p logs data/{chromadb,qdrant,faiss} monitoring/{prometheus,grafana}
    log "Directories created"
}

# Check if Python virtual environment exists
setup_python_env() {
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    log "Virtual environment activated"
    
    # Install basic requirements if needed
    pip install -q --upgrade pip
    
    # Check for core packages
    python3 -c "
import sys
try:
    import streamlit, fastapi, requests, pandas, plotly
    print('Core packages available')
except ImportError as e:
    print(f'Installing missing packages: {e}')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit', 'fastapi[all]', 'uvicorn', 'requests', 'pandas', 'plotly', 'psutil'])
"
}

# Start services
start_services() {
    log_info "Starting SutazAI services..."
    
    # Start Ollama if available
    if command -v ollama >/dev/null 2>&1; then
        if ! pgrep -f "ollama serve" > /dev/null; then
            log "Starting Ollama service..."
            nohup ollama serve > logs/ollama.log 2>&1 &
            sleep 3
        else
            log "Ollama already running"
        fi
    else
        log_warn "Ollama not installed. AI chat will use simulated responses."
    fi
    
    # Start backend if it exists
    if [ -f "backend/app/working_main.py" ]; then
        if ! pgrep -f "working_main.py" > /dev/null; then
            log "Starting backend service..."
            cd backend
            nohup python3 app/working_main.py > ../logs/backend.log 2>&1 &
            cd ..
            sleep 5
        else
            log "Backend already running"
        fi
    else
        log_warn "Backend not found. Some features may not work."
    fi
    
    # Start frontend
    log "Starting frontend interface..."
    if [ -f "optimized_sutazai_app.py" ]; then
        streamlit run optimized_sutazai_app.py --server.port=8501 --server.address=0.0.0.0 &
        STREAMLIT_PID=$!
        log "Frontend started on http://localhost:8501"
    else
        log_error "Frontend application not found!"
        exit 1
    fi
}

# Health checks
check_services() {
    log_info "Checking service health..."
    
    # Check Streamlit
    sleep 5
    if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        log "✅ Frontend (Streamlit) - http://localhost:8501"
    else
        log_warn "Frontend may still be starting..."
    fi
    
    # Check backend if running
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        log "✅ Backend API - http://localhost:8000"
    else
        log_warn "Backend not accessible (this is normal if not started)"
    fi
    
    # Check Ollama
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log "✅ Ollama LLM Server - http://localhost:11434"
    else
        log_warn "Ollama not accessible (AI chat will use fallback responses)"
    fi
}

# Main execution
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "🧠 SutazAI Enterprise AGI/ASI System"
    echo "🚀 Local Startup - No Docker Required"
    echo "=================================================================="
    echo -e "${NC}"
}

main() {
    print_banner
    
    cd "$SCRIPT_DIR"
    
    setup_directories
    setup_python_env
    start_services
    check_services
    
    echo -e "\n${GREEN}🎉 SutazAI System Started Successfully! 🎉${NC}"
    echo -e "${BLUE}📊 Access Points:${NC}"
    echo -e "${YELLOW}   🖥️  Frontend: http://localhost:8501${NC}"
    echo -e "${YELLOW}   🔧 Backend:  http://localhost:8000 (if available)${NC}"
    echo -e "${YELLOW}   📡 Ollama:   http://localhost:11434 (if available)${NC}"
    echo -e "\n${BLUE}📝 Logs: $LOG_FILE${NC}"
    echo -e "${BLUE}🛑 To stop: Press Ctrl+C${NC}\n"
    
    # Wait for Streamlit to finish
    wait $STREAMLIT_PID
}

# Cleanup on exit
cleanup() {
    log_info "Shutting down services..."
    pkill -f "streamlit run" || true
    pkill -f "working_main.py" || true
    log "Services stopped"
}

trap cleanup EXIT

# Run main function
main "$@"