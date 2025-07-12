#!/bin/bash

# SutazAI - Comprehensive E2E Autonomous AGI/ASI System
# Startup Script

set -e

echo "🤖 Starting SutazAI - Comprehensive E2E Autonomous AGI/ASI System"
echo "================================================================="

# Configuration
SUTAZAI_ROOT="/opt/sutazaiapp"
VENV_PATH="$SUTAZAI_ROOT/venv"
LOG_DIR="$SUTAZAI_ROOT/logs"
DATA_DIR="$SUTAZAI_ROOT/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$SUTAZAI_ROOT/temp"
    mkdir -p "$SUTAZAI_ROOT/uploads"
    mkdir -p "$SUTAZAI_ROOT/exports"
    
    success "Directories created successfully"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log "Found Python: $PYTHON_VERSION"
    
    # Check pip
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        error "pip is not installed. Please install pip."
        exit 1
    fi
    
    success "System requirements check passed"
}

# Setup virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_PATH" ]; then
        $PYTHON_CMD -m venv "$VENV_PATH"
        success "Virtual environment created"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    success "Virtual environment setup complete"
}

# Install dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    source "$VENV_PATH/bin/activate"
    
    if [ -f "$SUTAZAI_ROOT/requirements.txt" ]; then
        pip install -r "$SUTAZAI_ROOT/requirements.txt"
        success "Dependencies installed successfully"
    else
        warning "requirements.txt not found. Installing basic dependencies..."
        pip install streamlit fastapi uvicorn aiosqlite pandas numpy
    fi
}

# Initialize database
initialize_database() {
    log "Initializing database..."
    
    source "$VENV_PATH/bin/activate"
    cd "$SUTAZAI_ROOT"
    
    # Create database initialization script
    cat > init_db.py << 'EOF'
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def init_database():
    try:
        from core.database import init_database
        await init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    return True

async def init_vector_memory():
    try:
        from core.vector_memory import init_vector_memory
        await init_vector_memory()
        print("✅ Vector memory initialized successfully")
    except Exception as e:
        print(f"❌ Vector memory initialization failed: {e}")
        return False
    return True

async def main():
    print("🔧 Initializing SutazAI components...")
    
    db_success = await init_database()
    vm_success = await init_vector_memory()
    
    if db_success and vm_success:
        print("🎉 All components initialized successfully!")
        return True
    else:
        print("⚠️ Some components failed to initialize")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF
    
    $PYTHON_CMD init_db.py
    rm init_db.py
    
    success "Database initialization complete"
}

# Start services
start_services() {
    log "Starting SutazAI services..."
    
    source "$VENV_PATH/bin/activate"
    cd "$SUTAZAI_ROOT"
    
    # Start FastAPI backend in background
    log "Starting FastAPI backend..."
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > "$LOG_DIR/fastapi.log" 2>&1 &
    FASTAPI_PID=$!
    echo $FASTAPI_PID > "$SUTAZAI_ROOT/fastapi.pid"
    
    # Wait a moment for FastAPI to start
    sleep 3
    
    # Check if FastAPI is running
    if ps -p $FASTAPI_PID > /dev/null; then
        success "FastAPI backend started (PID: $FASTAPI_PID)"
    else
        error "Failed to start FastAPI backend"
        exit 1
    fi
    
    # Start Streamlit frontend
    log "Starting Streamlit frontend..."
    export STREAMLIT_SERVER_PORT=8501
    export STREAMLIT_SERVER_ADDRESS=0.0.0.0
    
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > "$SUTAZAI_ROOT/streamlit.pid"
    
    success "Streamlit frontend started (PID: $STREAMLIT_PID)"
    
    # Create status file
    cat > "$SUTAZAI_ROOT/service_status.json" << EOF
{
    "status": "running",
    "fastapi_pid": $FASTAPI_PID,
    "streamlit_pid": $STREAMLIT_PID,
    "started_at": "$(date -Iseconds)",
    "fastapi_url": "http://localhost:8000",
    "streamlit_url": "http://localhost:8501"
}
EOF
    
    success "All services started successfully!"
}

# Display service information
show_service_info() {
    echo ""
    echo "🎉 SutazAI System Started Successfully!"
    echo "====================================="
    echo ""
    echo "📡 Services:"
    echo "  • FastAPI Backend:  http://localhost:8000"
    echo "  • Streamlit UI:     http://localhost:8501"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo ""
    echo "📁 Important Paths:"
    echo "  • Application Root: $SUTAZAI_ROOT"
    echo "  • Logs Directory:   $LOG_DIR"
    echo "  • Data Directory:   $DATA_DIR"
    echo ""
    echo "🛠️ Management Commands:"
    echo "  • Stop Services:    ./stop_sutazai.sh"
    echo "  • View Logs:        tail -f $LOG_DIR/fastapi.log"
    echo "  • System Status:    cat $SUTAZAI_ROOT/service_status.json"
    echo ""
    echo "📖 Documentation:"
    echo "  • Visit http://localhost:8501 for the main interface"
    echo "  • Visit http://localhost:8000/docs for API documentation"
    echo ""
}

# Main execution
main() {
    log "Starting SutazAI initialization..."
    
    create_directories
    check_requirements
    setup_venv
    install_dependencies
    initialize_database
    start_services
    show_service_info
    
    success "SutazAI system is now running!"
    
    # Keep script running to show logs
    log "Monitoring services... Press Ctrl+C to stop"
    
    trap 'echo ""; log "Shutting down SutazAI..."; kill $(cat "$SUTAZAI_ROOT/fastapi.pid" 2>/dev/null || echo "") $(cat "$SUTAZAI_ROOT/streamlit.pid" 2>/dev/null || echo "") 2>/dev/null; exit 0' INT
    
    # Monitor services
    while true do
        if [ -f "$SUTAZAI_ROOT/fastapi.pid" ] && [ -f "$SUTAZAI_ROOT/streamlit.pid" ]; then
            FASTAPI_PID=$(cat "$SUTAZAI_ROOT/fastapi.pid")
            STREAMLIT_PID=$(cat "$SUTAZAI_ROOT/streamlit.pid")
            
            if ! ps -p $FASTAPI_PID > /dev/null || ! ps -p $STREAMLIT_PID > /dev/null; then
                error "One or more services have stopped unexpectedly"
                break
            fi
        fi
        
        sleep 10
    done
}

# Run main function
main "$@"